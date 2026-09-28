#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Megatron-SWIFT plugin: log cumulative model-input tokens per channel.

The loss denominator only counts labels that participate in SFT loss, which
omits prompt/context tokens.  For padding-free packed batches this plugin uses
the final packed sequence boundaries instead.  Those boundaries are produced
after multimodal preprocessing, so image/video placeholders have already been
expanded into the visual tokens that are actually sent to the model.

Counts are reduced across data/context-parallel ranks and exposed as cumulative
``trained_tokens_<channel>`` metrics in the terminal and ``logging.jsonl``.  In
WandB they are emitted as ``trained_tokens/<channel>`` so they get their own
workspace section instead of being mixed into ``train/*``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from types import MethodType

import torch
import torch.distributed as dist
from megatron.core import mpu

from swift.megatron.callbacks import MegatronCallback, megatron_callbacks_map
from swift.megatron.callbacks.utils import rewrite_logs
from swift.utils import get_logger

logger = get_logger()


class ChannelTokenCounterCallback(MegatronCallback):
    """Count complete model-input tokens, grouped by dataset channel."""

    metric_prefix = "trained_tokens_"
    internal_metric_prefix = "model_input_tokens_"
    aggregate_metrics = {"total", "total_token_used", "step_total"}

    def __init__(self, trainer):
        super().__init__(trainer)
        self.cumulative = defaultdict(int)
        self._patched = False

    def _restore_from_log(self):
        # A normal fresh run has iteration 0 and must not inherit an old file
        # accidentally left in output_dir.  Resume runs restore the last totals.
        if self.state.iteration <= 0:
            return
        path = os.path.join(self.args.output_dir, "logging.jsonl")
        if not os.path.isfile(path):
            return
        last_totals = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    for key, value in row.items():
                        metric_name = key[len(self.metric_prefix):]
                        if key.startswith(self.metric_prefix) and metric_name not in self.aggregate_metrics:
                            try:
                                last_totals[metric_name] = int(value)
                            except (TypeError, ValueError):
                                continue
        except OSError as exc:
            logger.warning(f"Cannot restore channel token counters from {path}: {exc}")
            return
        self.cumulative.update(last_totals)
        if last_totals:
            logger.info(f"Restored channel token counters: {dict(sorted(last_totals.items()))}")

    @staticmethod
    def _token_count(metric):
        if not isinstance(metric, torch.Tensor) or metric.numel() != 2:
            return None
        return int(metric.reshape(-1)[1].item())

    def _patch_wandb_callback(self):
        """Keep JSON/console names stable while giving WandB a separate section."""
        for wandb_callback in self.trainer.callbacks:
            if wandb_callback.__class__.__name__ != "WandbCallback":
                continue
            if getattr(wandb_callback, "_channel_tokens_grouped", False):
                return

            metric_prefix = self.metric_prefix

            def on_log_with_grouped_tokens(wandb_self, logs):
                normal_logs = {
                    key: value for key, value in logs.items()
                    if not key.startswith(metric_prefix)
                }
                wandb_logs = rewrite_logs(normal_logs)
                for key, value in logs.items():
                    if key.startswith(metric_prefix) and not isinstance(value, str):
                        metric_name = key[len(metric_prefix):]
                        wandb_logs[f"trained_tokens/{metric_name}"] = value
                # WandbCallback only creates a writer on the reporting rank.
                if wandb_self.writer is not None:
                    wandb_self.writer.log(wandb_logs, step=wandb_self.state.iteration)

            wandb_callback.on_log = MethodType(on_log_with_grouped_tokens, wandb_callback)
            wandb_callback._channel_tokens_grouped = True
            logger.info("WandB channel token metrics grouped under trained_tokens/*.")
            return

    def _compute_input_token_metrics(self, losses, channels, packed_seq_params):
        """Build ``[0, token_count]`` metrics compatible with SWIFT aggregation.

        Context parallelism splits each packed sequence across ranks.  We count
        the local slice here and reduce over the same DP+CP group used by the
        channel-loss metrics, yielding each global token exactly once.
        """
        metrics = defaultdict(
            lambda: torch.tensor([0.0, 0.0], dtype=torch.float32, device=losses.device))

        if self.args.padding_free:
            if packed_seq_params is None:
                raise RuntimeError("padding_free input token counting requires packed_seq_params")
            num_samples = packed_seq_params.seq_lens.shape[0]
            cu_seqlens = packed_seq_params.cu_seqlens_q[:num_samples + 1]
            cu_seqlens = cu_seqlens // self.args.context_parallel_size
            for i in range(num_samples):
                channel = None if channels is None else channels[i]
                token_count = cu_seqlens[i + 1] - cu_seqlens[i]
                metrics[f"{self.internal_metric_prefix}{channel}"][1] += token_count
        else:
            # Non-padding-free batches do not expose their attention mask to
            # loss_func.  The current OneVision training path is padding-free;
            # keep a conservative fallback for dynamically padded text runs.
            for i in range(losses.shape[0]):
                channel = None if channels is None else channels[i]
                metrics[f"{self.internal_metric_prefix}{channel}"][1] += losses.shape[1]

        dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
        all_keys = [None] * torch.distributed.get_world_size(group=dp_cp_group)
        dist.all_gather_object(all_keys, list(metrics.keys()), group=dp_cp_group)
        reduced = {}
        for key in sorted(set().union(*all_keys)):
            reduced[key] = metrics[key]
        return self.trainer._all_reduce_metric(
            reduced, torch.distributed.ReduceOp.SUM, group=dp_cp_group)

    def on_train_begin(self):
        if self._patched:
            return
        if not self.args.enable_channel_loss:
            logger.warning("channel_token_counter is enabled but enable_channel_loss=false; no per-channel tokens.")
            return

        self._restore_from_log()
        trainer = self.trainer
        self._patch_wandb_callback()
        original_compute_channel_loss = trainer._compute_channel_loss
        original_on_log = trainer.on_log
        callback = self

        def compute_channel_loss_with_input_tokens(
                trainer_self, losses, loss_mask, channels, packed_seq_params=None):
            metrics = original_compute_channel_loss(
                losses, loss_mask, channels, packed_seq_params)
            metrics.update(callback._compute_input_token_metrics(
                losses, channels, packed_seq_params))
            return metrics

        trainer._compute_channel_loss = MethodType(
            compute_channel_loss_with_input_tokens, trainer)

        def on_log_with_channel_tokens(trainer_self, logs, prefix=""):
            # Evaluation uses the same loss_<channel> representation.  Only an
            # empty prefix denotes training logs, so eval tokens are excluded.
            if not prefix:
                interval_total = 0
                for key, metric in list(logs.items()):
                    if not key.startswith(callback.internal_metric_prefix):
                        continue
                    # Internal aggregation carrier; do not expose its dummy
                    # numerator as an additional zero-valued log metric.
                    logs.pop(key)
                    count = callback._token_count(metric)
                    if count is None:
                        continue
                    channel = key[len(callback.internal_metric_prefix):]
                    callback.cumulative[channel] += count
                    interval_total += count

                for channel, count in sorted(callback.cumulative.items()):
                    logs[f"{callback.metric_prefix}{channel}"] = count
                total_token_used = sum(callback.cumulative.values())
                logs[f"{callback.metric_prefix}total"] = total_token_used
                logs[f"{callback.metric_prefix}total_token_used"] = total_token_used
                logs["trained_tokens_step_total"] = interval_total

            return original_on_log(logs, prefix)

        trainer.on_log = MethodType(on_log_with_channel_tokens, trainer)
        self._patched = True
        logger.info(
            "Channel token counter enabled: trained_tokens_<channel> counts cumulative full model-input tokens "
            "after multimodal token expansion.")


megatron_callbacks_map["channel_token_counter"] = ChannelTokenCounterCallback
