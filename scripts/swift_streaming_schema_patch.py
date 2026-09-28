#!/usr/bin/env python
"""Normalize multimodal feature schemas before streaming interleave.

Hugging Face datasets infers features from the first rows of each iterable
dataset.  That turns an empty image list into ``List(Value("null"))`` and an
empty objects column into ``Value("null")``.  Genshin additionally stores each
image as bare bytes, while LLaVA stores ``{"bytes", "path"}`` structs.  A mixed
text/image/grounding mixture then fails before training because those features
cannot be aligned.  Normalize them at SWIFT's interleave boundary.
"""

from __future__ import annotations

import torch
from datasets import Features, IterableDataset, Json, List, Value
from pathlib import Path

from swift.dataset.dataset_meta import BaseDatasetLoader
from swift.template.templates.qwen import Qwen2VLTemplate
from swift.utils import get_logger

logger = get_logger()

_ORIGINAL_INTERLEAVE = BaseDatasetLoader.interleave_datasets
_ORIGINAL_QWEN_PACKING_ROW = Qwen2VLTemplate.packing_row


def _normalize_multimodal_features(dataset):
    if not isinstance(dataset, IterableDataset):
        return dataset
    if dataset.features is None:
        dataset = dataset._resolve_features()

    features = Features(dict(dataset.features))
    image_feature = features.get("images")
    image_item = getattr(image_feature, "feature", None)
    bare_binary_images = isinstance(image_item, Value) and image_item.dtype in {
        "binary",
        "large_binary",
    }
    changed = False
    if "images" in features:
        features["images"] = List(
            {
                "bytes": Value("binary"),
                "path": Value("string"),
            }
        )
        changed = True
    if "objects" in features:
        features["objects"] = Json()
        changed = True
    if "channel" not in features:
        features["channel"] = Value("string")
        changed = True

    if not changed:
        return dataset

    def normalize_row(row):
        updates = {}
        if bare_binary_images and row.get("images") is not None:
            updates["images"] = [
                {"bytes": bytes(image), "path": None} if image is not None else None
                for image in row["images"]
            ]
        if not row.get("channel"):
            source = str(row.get("dataset") or "unknown").rstrip("/")
            name = Path(source).name or "unknown"
            if "genshin" in source.lower() or "yuanshen" in source.lower():
                name = "genshin"
            updates["channel"] = name
        return updates

    return dataset.map(normalize_row, features=features)


def _interleave_with_normalized_features(datasets, *args, **kwargs):
    normalized = [_normalize_multimodal_features(dataset) for dataset in datasets]
    return _ORIGINAL_INTERLEAVE(normalized, *args, **kwargs)


BaseDatasetLoader.interleave_datasets = staticmethod(_interleave_with_normalized_features)


def _qwen_packing_row_allow_text_only(self, row):
    """Represent a text-only row with an all-zero multimodal token mask.

    Megatron uses the padding-free collator even when ``packing=false``.
    Qwen2VLTemplate.packing_row previously indexed ``mm_token_type_ids`` when
    the key existed with a ``None`` value, which is how text-only rows emerge
    from a unified text/image streaming schema. Qwen3.5 requires the argument
    in ``get_rope_index``, so the correct text-only value is a zero mask.
    """
    for record in row:
        if record.get("mm_token_type_ids") is None:
            record["mm_token_type_ids"] = torch.zeros(
                len(record["input_ids"]), dtype=torch.int64
            )
    return _ORIGINAL_QWEN_PACKING_ROW(self, row)


Qwen2VLTemplate.packing_row = _qwen_packing_row_allow_text_only
logger.info(
    "Streaming multimodal schema normalization and Qwen text-only padding-free collator patch enabled."
)
