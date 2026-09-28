"""Finite, restartable validation for Megatron SFT (PP=CP=1).

Enable with FULL_VAL_EVAL=true and load this file through --external_plugins.
Every evaluation drains the finite DP-sharded validation loader from its start.
Uneven DP tails replay a cached batch with zero reporting weight, so no real
validation batch is dropped or counted twice. Training is not modified.
"""

import copy
import os

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map


def _cpu_copy(batch):
    return tree_map(lambda x: x.detach().cpu().clone() if isinstance(x, torch.Tensor)
                    else copy.deepcopy(x), batch)


def _broadcast_batch(batch, source, device):
    """Bootstrap an empty DP shard without sending pixel tensors via pickle."""
    rank = dist.get_rank()
    leaves, spec = tree_flatten(batch) if rank == source else (None, None)
    metadata = None
    if rank == source:
        metadata = (spec, [(True, tuple(x.shape), x.dtype) if isinstance(x, torch.Tensor)
                           else (False, x) for x in leaves])
    payload = [metadata]
    dist.broadcast_object_list(payload, src=source, device=device)
    spec, descriptors = payload[0]
    received = []
    for i, descriptor in enumerate(descriptors):
        if not descriptor[0]:
            received.append(descriptor[1])
            continue
        tensor = (leaves[i].to(device).contiguous() if rank == source else
                  torch.empty(descriptor[1], dtype=descriptor[2], device=device))
        dist.broadcast(tensor, src=source)
        received.append(tensor.cpu())
    return tree_unflatten(received, spec)


class FiniteValidationIterator:
    """All ranks take the same number of rounds, ending at the longest shard."""

    def __init__(self, iterable, device):
        self.iterator = iter(iterable)
        self.device = device
        self.cached = None
        self.exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        batch, error = None, None
        if not self.exhausted:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.exhausted = True
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        status = torch.tensor([int(batch is not None), int(error is not None)],
                              device=self.device, dtype=torch.int64)
        dist.all_reduce(status)
        if status[1].item():
            source = torch.tensor(dist.get_rank() if error else dist.get_world_size(),
                                  device=self.device, dtype=torch.int64)
            dist.all_reduce(source, op=dist.ReduceOp.MIN)
            payload = [error]
            dist.broadcast_object_list(payload, src=int(source.item()), device=self.device)
            raise RuntimeError(f'Full validation data loader failed: {payload[0]}')
        if not status[0].item():
            raise StopIteration

        if self.cached is None and batch is not None:
            self.cached = _cpu_copy(batch)
        needs_cache = torch.tensor(int(self.cached is None), device=self.device)
        dist.all_reduce(needs_cache, op=dist.ReduceOp.MAX)
        if needs_cache.item():
            source = torch.tensor(dist.get_rank() if batch is not None else dist.get_world_size(),
                                  device=self.device, dtype=torch.int64)
            dist.all_reduce(source, op=dist.ReduceOp.MIN)
            bootstrap = _broadcast_batch(batch, int(source.item()), self.device)
            if self.cached is None:
                self.cached = bootstrap
        if batch is None:
            return _cpu_copy(self.cached), True
        return batch, False


def _install():
    from megatron.core import mpu
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from swift.megatron.trainers.base import BaseMegatronTrainer, _TensorParallelBroadcastIterator
    from swift.megatron.trainers.trainer import MegatronTrainer
    from swift.utils import get_current_device, get_logger

    if getattr(BaseMegatronTrainer, '_full_val_installed', False):
        return
    logger = get_logger()
    original_loader = BaseMegatronTrainer._prepare_dataloader
    original_loss = MegatronTrainer.loss_func

    def prepare_dataloader(self, train_dataset, val_dataset=None):
        args = self.args
        if (not args.streaming or not args.streaming_shard_by_dp or
                args.pipeline_model_parallel_size != 1 or args.context_parallel_size != 1 or
                args.virtual_pipeline_model_parallel_size is not None or
                args.dataloader_num_workers != 0 or args.task_type != 'causal_lm'):
            raise ValueError('FULL_VAL_EVAL requires streaming_shard_by_dp=true, streaming=true, '
                             'PP=CP=1, no virtual pipeline, dataloader_num_workers=0, causal_lm')
        if args.val_dataset_shuffle:
            raise ValueError('FULL_VAL_EVAL requires val_dataset_shuffle=false')
        if getattr(val_dataset, 'cyclic', False):
            raise ValueError('FULL_VAL_EVAL requires a finite validation dataset (cyclic=false)')
        loaders = original_loader(self, train_dataset, val_dataset)
        self._full_val_loader = loaders[1]
        logger.info('Full validation enabled: restart and drain validation at every eval; '
                    'eval_iters is ignored, uneven DP tails have zero loss weight.')
        return loaders

    def loss_func(self, output_tensor, *, labels, **kwargs):
        # Preserve real labels during model forward (including MTP kernels),
        # then exclude dummy batches from BOTH numerator and denominator.
        if getattr(self, '_full_val_padding', False):
            labels = torch.full_like(labels, -100)
        return original_loss(self, output_tensor, labels=labels, **kwargs)

    def evaluate(self, _unused_val_iterator):
        loader = self._full_val_loader
        if loader is None:
            raise ValueError('Full validation requested without a validation dataset')
        # A fresh DataLoader iterator restarts the finite, unshuffled validation
        # source. Unlike the training packer, validation has no pass-aware cursor.
        iterator = iter(loader)
        if os.environ.get('TP_DATALOADER_BROADCAST', 'false').lower() in {'1', 'true', 'yes', 'on'}:
            iterator = _TensorParallelBroadcastIterator(iterator)
        iterator = FiniteValidationIterator(iterator, get_current_device())
        forward = get_forward_backward_func()
        metrics = {}
        counts = torch.zeros(2, dtype=torch.int64, device=get_current_device())
        model_modes = [m.training for m in self.wrapped_models]
        old_eval_iters = self.args.eval_iters
        self.args.eval_iters = None  # tqdm uses an unknown total while draining.
        try:
            for model in self.wrapped_models:
                model.eval()
            self.call_event('on_eval_begin')
            with torch.no_grad():
                for batch, padding in iterator:
                    self._full_val_padding = padding
                    if not padding:
                        counts[0] += batch['labels'].shape[0]
                        counts[1] += (batch['labels'] != -100).sum().to(counts.device)
                    micro_metrics = forward(
                        forward_step_func=self.forward_step,
                        data_iterator=iter([batch]), model=self.wrapped_models,
                        num_microbatches=1, seq_length=self.args.seq_length,
                        micro_batch_size=self.args.micro_batch_size, forward_only=True)
                    # Training's optional input-token counter does not mask
                    # dummy batches and is not an eval loss denominator.
                    for item in micro_metrics:
                        for key in list(item):
                            if key.startswith('model_input_tokens_'):
                                del item[key]
                    self._aggregated_metrics(micro_metrics, metrics)
                    self.call_event('on_eval_step')
            if 'loss' not in metrics:
                raise ValueError('Full validation produced no supervised tokens')
            dist.all_reduce(counts, group=mpu.get_data_parallel_group())
            metrics['full_val_batches'] = metrics['n_steps']
            metrics['full_val_sequences'] = int(counts[0].item())
            metrics['full_val_supervised_tokens'] = int(counts[1].item())
            self.compute_eval_metrics(metrics)
            self.on_log(logs=metrics, prefix='eval_')
            self.call_event('on_eval_end')
            return metrics
        finally:
            self._full_val_padding = False
            self.args.eval_iters = old_eval_iters
            for model, training in zip(self.wrapped_models, model_modes):
                model.train(training)

    BaseMegatronTrainer._prepare_dataloader = prepare_dataloader
    BaseMegatronTrainer.evaluate = evaluate
    MegatronTrainer.loss_func = loss_func
    BaseMegatronTrainer._full_val_installed = True


if os.environ.get('FULL_VAL_EVAL', 'false').lower() in {'1', 'true', 'yes', 'on'}:
    _install()
