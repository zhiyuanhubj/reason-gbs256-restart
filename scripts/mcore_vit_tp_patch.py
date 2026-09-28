#!/usr/bin/env python3
"""Shard packed ViT images across tensor-parallel ranks.

Stock mcore-bridge keeps a replicated HF ViT: every TP rank encodes every
image. This plugin splits the packed image list (not raw patches) across the
TP group, all-gathers merger outputs with autograd, and SUM-reduces visual
grads so the optimizer update matches the replicated-ViT math.

Loaded only from training-vittp via ``--external_plugins`` when ``VIT_TP=true``.
"""

from __future__ import annotations

import importlib
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import parallel_state
from megatron.core.utils import get_attr_wrapped_model, get_tensor_model_parallel_group_if_none

# `from megatron.core.distributed import finalize_model_grads` is the function.
# The sibling submodule has the same name, so use importlib to get the module.
finalize_mod = importlib.import_module('megatron.core.distributed.finalize_model_grads')
from mcore_bridge.model.mm_gpts.utils import HuggingFaceVit
from swift.utils import get_logger


logger = get_logger()

_ORIG_HF_GET_INPUTS_EMBEDS = HuggingFaceVit._hf_get_inputs_embeds
_ORIG_ALLREDUCE_NON_TP = finalize_mod._allreduce_non_tensor_model_parallel_grads
_LOG_REMAINING = 3


def _unwrap_visual_output(output):
    if hasattr(output, 'pooler_output'):
        return output.pooler_output
    return output


def _assign_images(patch_counts: Sequence[int], tp_size: int) -> List[int]:
    loads = [0] * tp_size
    assign: List[int] = []
    for count in patch_counts:
        rank = min(range(tp_size), key=lambda i: (loads[i], i))
        assign.append(rank)
        loads[rank] += int(count)
    return assign


def _slice_images(pixel_values, grid_thw, patches, indices: Sequence[int]):
    if not indices:
        return pixel_values[:0], grid_thw[:0]
    starts = torch.cumsum(patches, dim=0) - patches
    chunks = []
    for idx in indices:
        start = int(starts[idx].item())
        length = int(patches[idx].item())
        chunks.append(pixel_values[start:start + length])
    index_tensor = torch.tensor(indices, device=grid_thw.device, dtype=torch.long)
    return torch.cat(chunks, dim=0), grid_thw.index_select(0, index_tensor)


def _keep_unused_visual_params(visual) -> Optional[torch.Tensor]:
    keep = None
    for param in visual.parameters():
        if not param.requires_grad:
            continue
        term = param.view(-1)[:1] * 0
        keep = term if keep is None else keep + term
    return keep


def _mark_visual_params(visual) -> None:
    if getattr(visual, '_vit_tp_params_marked', False):
        return
    for param in visual.parameters():
        setattr(param, 'vit_tp_image_split', True)
    visual._vit_tp_params_marked = True


class _AllGatherEqual(torch.autograd.Function):
    """All-gather equal-shaped tensors; backward returns the local slice."""

    @staticmethod
    def forward(ctx, local: torch.Tensor):
        group = parallel_state.get_tensor_model_parallel_group()
        ctx.rank = parallel_state.get_tensor_model_parallel_rank()
        world = dist.get_world_size(group)
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local.contiguous(), group=group)
        return torch.stack(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_stacked: torch.Tensor):
        return grad_stacked[ctx.rank]


def _all_gather_tp(local: torch.Tensor) -> torch.Tensor:
    return _AllGatherEqual.apply(local)


def _encode_visual_tp(visual, pixel_values, grid_thw, vision_config):
    _mark_visual_params(visual)
    if not parallel_state.is_initialized():
        return _unwrap_visual_output(visual(pixel_values, grid_thw=grid_thw))
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return _unwrap_visual_output(visual(pixel_values, grid_thw=grid_thw))

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    patches = grid_thw.prod(dim=-1)
    patch_counts = [int(v) for v in patches.detach().cpu().tolist()]
    assign = _assign_images(patch_counts, tp_size)
    local_indices = [i for i, owner in enumerate(assign) if owner == tp_rank]

    merge_length = int(vision_config.spatial_merge_size) ** 2
    tokens_per_img = [count // merge_length for count in patch_counts]
    out_h = int(getattr(vision_config, 'out_hidden_size', None) or vision_config.hidden_size)

    local_pixels, local_grid = _slice_images(pixel_values, grid_thw, patches, local_indices)
    keep = None
    if local_pixels.numel() == 0:
        my_emb = pixel_values.new_zeros((0, out_h))
        keep = _keep_unused_visual_params(visual)
    else:
        my_emb = _unwrap_visual_output(visual(local_pixels, grid_thw=local_grid))
        if my_emb.dim() != 2:
            raise RuntimeError(f'ViT TP expected 2D merger output, got {tuple(my_emb.shape)}')

    local_tokens = int(my_emb.shape[0])
    expected_tokens = sum(tokens_per_img[i] for i in local_indices)
    if local_tokens != expected_tokens:
        raise RuntimeError(
            f'ViT TP token count mismatch on TP rank {tp_rank}: '
            f'got {local_tokens}, expected {expected_tokens} '
            f'(images={len(local_indices)}, merge_length={merge_length})'
        )

    counts = torch.zeros(tp_size, dtype=torch.long, device=pixel_values.device)
    counts[tp_rank] = local_tokens
    dist.all_reduce(counts, group=parallel_state.get_tensor_model_parallel_group())
    token_counts = [int(v) for v in counts.tolist()]
    max_n = max(token_counts) if token_counts else 0
    if max_n == 0:
        mixed = pixel_values.new_zeros((0, out_h))
        return mixed if keep is None else mixed + keep

    if local_tokens == 0:
        padded = pixel_values.new_zeros((max_n, out_h))
    else:
        if my_emb.shape[-1] != out_h:
            out_h = int(my_emb.shape[-1])
        pad_n = max_n - local_tokens
        padded = my_emb if pad_n == 0 else torch.cat(
            [my_emb, my_emb.new_zeros((pad_n, my_emb.shape[-1]))], dim=0
        )

    gathered = _all_gather_tp(padded)
    rank_pos = [0] * tp_size
    chunks = []
    for image_idx, owner in enumerate(assign):
        n_tok = tokens_per_img[image_idx]
        start = rank_pos[owner]
        chunks.append(gathered[owner, start:start + n_tok])
        rank_pos[owner] += n_tok
    mixed = torch.cat(chunks, dim=0) if chunks else padded[:0]

    global _LOG_REMAINING
    if _LOG_REMAINING > 0:
        _LOG_REMAINING -= 1
        logger.info(
            'ViT TP image-split: images=%s tp=%s/%s local_images=%s local_patches=%s local_tokens=%s',
            len(assign),
            tp_rank,
            tp_size,
            len(local_indices),
            sum(patch_counts[i] for i in local_indices),
            local_tokens,
        )
    if keep is not None:
        mixed = mixed + keep.to(device=mixed.device, dtype=mixed.dtype)
    return mixed


@staticmethod
def _hf_get_inputs_embeds_vit_tp(inputs_embeds, inputs, visual, hf_config):
    input_ids = inputs['input_ids']
    pixel_values = inputs.get('pixel_values')
    pixel_values_videos = inputs.get('pixel_values_videos')
    image_grid_thw = inputs.get('image_grid_thw')
    video_grid_thw = inputs.get('video_grid_thw')
    dtype = visual.dtype
    vision_config = HuggingFaceVit._get_vision_config(hf_config)
    if pixel_values is None and pixel_values_videos is None:
        hidden_size = (
            vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
        )
        pixel_values = torch.zeros(16 * 16, hidden_size, dtype=dtype, device=input_ids.device)
        image_grid_thw = input_ids.new_tensor([[1, 16, 16]])
        image_embeds = visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = _unwrap_visual_output(image_embeds)
        inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
    else:
        if pixel_values is None:
            pixel_values_mixed = pixel_values_videos
            grid_thw = video_grid_thw
        elif pixel_values_videos is None:
            pixel_values_mixed = pixel_values
            grid_thw = image_grid_thw
        else:
            pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
            grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
        pixel_values_mixed = pixel_values_mixed.type(dtype)
        mixed_embeds = _encode_visual_tp(visual, pixel_values_mixed, grid_thw, vision_config)
        if pixel_values is None:
            image_embeds = None
            video_embeds = mixed_embeds
        elif pixel_values_videos is None:
            image_embeds = mixed_embeds
            video_embeds = None
        else:
            merge_length = vision_config.spatial_merge_size**2
            image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
            image_embeds = mixed_embeds[:image_tokens]
            video_embeds = mixed_embeds[image_tokens:]

        if image_embeds is not None:
            image_mask = (input_ids == hf_config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = image_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if video_embeds is not None:
            video_mask = (input_ids == hf_config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = video_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    return inputs_embeds


def _collect_grad(param, ddp_config):
    grad_attr = finalize_mod._get_main_grad_attr(param)
    grad = getattr(param, grad_attr)
    if grad is None:
        zeros = torch.zeros_like(param)
        setattr(param, grad_attr, zeros)
        grad = zeros
    if ddp_config.use_megatron_fsdp:
        return param, grad._local_tensor.data
    return param, finalize_mod._unshard_if_dtensor(grad).data


def _allreduce_non_tensor_model_parallel_grads_vit_tp(model, config, tp_group=None):
    tp_group = get_tensor_model_parallel_group_if_none(tp_group)
    if tp_group.size() <= 1:
        return

    params_sum = []
    grads_sum = []
    params_avg = []
    grads_avg = []
    params_vit = []
    grads_vit = []
    ddp_config = None

    for model_chunk in model:
        ddp_config = model_chunk.ddp_config
        for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
            if not param.requires_grad:
                continue
            if getattr(param, 'vit_tp_image_split', False):
                p, g = _collect_grad(param, ddp_config)
                params_vit.append(p)
                grads_vit.append(g)
            elif getattr(param, 'average_gradients_across_tp_domain', False):
                grad_attr = finalize_mod._get_main_grad_attr(param)
                grad = getattr(param, grad_attr)
                if grad is None:
                    continue
                params_avg.append(param)
                if ddp_config.use_megatron_fsdp:
                    grads_avg.append(grad._local_tensor.data)
                else:
                    grads_avg.append(finalize_mod._unshard_if_dtensor(grad).data)
            elif (config.sequence_parallel and getattr(param, 'sequence_parallel', False)) or (
                config.qk_layernorm and ('q_layernorm' in name or 'k_layernorm' in name)
            ):
                grad_attr = finalize_mod._get_main_grad_attr(param)
                grad = getattr(param, grad_attr)
                if grad is None:
                    continue
                params_sum.append(param)
                if ddp_config.use_megatron_fsdp:
                    grads_sum.append(grad._local_tensor.data)
                else:
                    grads_sum.append(finalize_mod._unshard_if_dtensor(grad).data)

    for params, grads, all_reduce_op in (
        (params_sum, grads_sum, dist.ReduceOp.SUM),
        (params_vit, grads_vit, dist.ReduceOp.SUM),
        (params_avg, grads_avg, dist.ReduceOp.AVG),
    ):
        if not grads:
            continue
        coalesced = _flatten_dense_tensors(grads)
        dist.all_reduce(coalesced, op=all_reduce_op, group=tp_group)
        for param, buf, synced in zip(
            params, grads, _unflatten_dense_tensors(coalesced, grads)
        ):
            buf.copy_(synced)
            grad_attr = finalize_mod._get_main_grad_attr(param)
            orig_grad = getattr(param, grad_attr)
            if ddp_config is not None and ddp_config.use_megatron_fsdp:
                setattr(param, grad_attr, orig_grad)
            else:
                setattr(param, grad_attr, finalize_mod._reshard_if_dtensor(buf, orig_grad))


HuggingFaceVit._hf_get_inputs_embeds = _hf_get_inputs_embeds_vit_tp
finalize_mod._allreduce_non_tensor_model_parallel_grads = (
    _allreduce_non_tensor_model_parallel_grads_vit_tp
)
finalize_mod._allreduce_layernorm_grads = _allreduce_non_tensor_model_parallel_grads_vit_tp
logger.info('Installed ViT TP image-split patch (shard images across TP, SUM visual grads).')
