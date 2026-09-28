#!/usr/bin/env python
"""Short-run timing probe for Megatron-SWIFT SFT jobs.

Reports synchronized wall time spent in get_batch versus the rest of each
optimizer step. Load only in diagnostic jobs via EXTRA_EXTERNAL_PLUGINS.
"""

from __future__ import annotations

import time
import os

import torch
import torch.distributed as dist

from swift.megatron.trainers.base import BaseMegatronTrainer
from swift.megatron.callbacks.default_flow import DefaultFlowCallback


_ORIGINAL_GET_BATCH = BaseMegatronTrainer.get_batch
_ORIGINAL_TRAIN_STEP = BaseMegatronTrainer.train_step


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_get_batch(self, data_iterator, vp_stage=None):
    _synchronize()
    start = time.perf_counter()
    batch = _ORIGINAL_GET_BATCH(self, data_iterator, vp_stage)
    _synchronize()
    self._timing_probe_batch_s = getattr(self, "_timing_probe_batch_s", 0.0) + (time.perf_counter() - start)
    self._timing_probe_batch_calls = getattr(self, "_timing_probe_batch_calls", 0) + 1
    return batch


def _timed_train_step(self, train_data_iterator):
    self._timing_probe_batch_s = 0.0
    self._timing_probe_batch_calls = 0
    _synchronize()
    start = time.perf_counter()
    result = _ORIGINAL_TRAIN_STEP(self, train_data_iterator)
    _synchronize()
    total_s = time.perf_counter() - start

    values = torch.tensor(
        [total_s, self._timing_probe_batch_s],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    total_max_s, batch_max_s = values.tolist()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        non_batch_s = max(0.0, total_max_s - batch_max_s)
        fraction = 100.0 * batch_max_s / total_max_s if total_max_s else 0.0
        print(
            "[timing-probe] "
            f"batch_calls={self._timing_probe_batch_calls} "
            f"step_max_ms={total_max_s * 1000.0:.1f} "
            f"get_batch_max_ms={batch_max_s * 1000.0:.1f} "
            f"compute_comm_optim_ms={non_batch_s * 1000.0:.1f} "
            f"get_batch_fraction={fraction:.2f}%",
            flush=True,
        )
    return result


BaseMegatronTrainer.get_batch = _timed_get_batch
BaseMegatronTrainer.train_step = _timed_train_step


# Diagnostic runs do not need a final 100+ GiB distributed checkpoint. Keep
# this opt-in so loading the probe alone never changes normal save behavior.
if os.environ.get("SWIFT_TIMING_PROBE_DISABLE_SAVE", "0") == "1":
    _ORIGINAL_ON_STEP_END = DefaultFlowCallback.on_step_end

    def _on_step_end_without_artifacts(self):
        _ORIGINAL_ON_STEP_END(self)
        self.state.should_save = False
        if self.state.iteration >= self.args.train_iters:
            self.state.should_eval = False

    DefaultFlowCallback.on_step_end = _on_step_end_without_artifacts

