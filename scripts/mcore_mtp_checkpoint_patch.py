#!/usr/bin/env python3
"""Fix replicated MTP checkpoint metadata in the pinned mcore-bridge.

The bridge leaves ``MultiTokenPredictionLayer.tp_group`` as ``None``. When
Megatron wraps replicated norm parameters for distributed checkpointing, that
makes every TP rank look like TP rank zero, so all four copies are marked as
main replicas and sharding validation fails. Resolve the actual TP process
group after model-parallel initialization.
"""

from __future__ import annotations

from megatron.core import parallel_state
from mcore_bridge.model.modules.mtp_layer import MultiTokenPredictionLayer
from swift.utils import get_logger


logger = get_logger()
_ORIGINAL_INIT = MultiTokenPredictionLayer.__init__
_ORIGINAL_CHECKPOINTED_FORWARD = MultiTokenPredictionLayer._checkpointed_forward


def _init_with_tp_group(self, *args, **kwargs):
    _ORIGINAL_INIT(self, *args, **kwargs)
    if getattr(self, "tp_group", None) is None:
        self.tp_group = parallel_state.get_tensor_model_parallel_group()


def _checkpointed_forward_with_block_support(self, *args, **kwargs):
    """Keep the single MTP layer checkpointed when decoder layers use block mode.

    The pinned bridge skips MTP recompute for ``recompute_method=block`` even
    though MTP is a single layer.  Reuse its supported uniform/1 path only for
    the duration of the MTP call; the decoder keeps its configured block count.
    """
    config = self.config
    if config.recompute_method != "block":
        return _ORIGINAL_CHECKPOINTED_FORWARD(self, *args, **kwargs)

    recompute_method = config.recompute_method
    recompute_num_layers = config.recompute_num_layers
    config.recompute_method = "uniform"
    config.recompute_num_layers = 1
    try:
        return _ORIGINAL_CHECKPOINTED_FORWARD(self, *args, **kwargs)
    finally:
        config.recompute_method = recompute_method
        config.recompute_num_layers = recompute_num_layers


MultiTokenPredictionLayer.__init__ = _init_with_tp_group
MultiTokenPredictionLayer._checkpointed_forward = _checkpointed_forward_with_block_support
logger.info("Installed MTP checkpoint metadata and block-recompute compatibility patch.")
