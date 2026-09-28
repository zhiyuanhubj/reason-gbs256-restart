#!/usr/bin/env python
"""Count exact SWIFT/Qwen multimodal input tokens for sharded JSONL files."""

import argparse
import json
import os
import time
from pathlib import Path


GENSHIN_SPECIAL_TOKENS = [
    "<|action_start|>",
    "<|action_end|>",
    "<|action_sep|>",
    "<|thought_start|>",
    "<|thought_end|>",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-length", type=int, default=131072)
    parser.add_argument("jsonl", nargs="+")
    args = parser.parse_args()

    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world = int(os.environ.get("SLURM_NTASKS", "1"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from swift import get_processor, get_template

    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    processor = get_processor(args.model, new_special_tokens=GENSHIN_SPECIAL_TOKENS)
    template = get_template(
        processor,
        max_length=10**9,
        truncation_strategy="raise",
        padding_free=True,
        sequence_parallel_size=4,
        add_non_thinking_prefix=False,
    )
    template.set_mode("train")
    image_token_id = template.image_token_id

    totals = {}
    started = time.time()
    for path_str in args.jsonl:
        path = Path(path_str)
        stats = {
            "rows": 0,
            "tokens": 0,
            "visual_tokens": 0,
            "text_tokens": 0,
            "train_rows": 0,
            "train_tokens": 0,
            "train_visual_tokens": 0,
            "overlength_rows": 0,
            "overlength_tokens": 0,
            "max_length": 0,
            "errors": [],
        }
        with path.open(encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle):
                if line_idx % world != rank or not line.strip():
                    continue
                try:
                    encoded = template.encode(json.loads(line))
                    input_ids = encoded["input_ids"]
                    token_count = len(input_ids)
                    visual_count = input_ids.count(image_token_id)
                    grid_count = sum(
                        int(grid.prod()) // processor.image_processor.merge_size**2
                        for grid in encoded.get("image_grid_thw", [])
                    )
                    if visual_count != grid_count:
                        raise RuntimeError(
                            f"visual count mismatch: input_ids={visual_count}, grids={grid_count}"
                        )
                    stats["rows"] += 1
                    stats["tokens"] += token_count
                    stats["visual_tokens"] += visual_count
                    stats["text_tokens"] += token_count - visual_count
                    stats["max_length"] = max(stats["max_length"], token_count)
                    if token_count <= args.max_train_length:
                        stats["train_rows"] += 1
                        stats["train_tokens"] += token_count
                        stats["train_visual_tokens"] += visual_count
                    else:
                        stats["overlength_rows"] += 1
                        stats["overlength_tokens"] += token_count
                    del encoded
                except Exception as exc:  # preserve the complete shard and report failures
                    stats["errors"].append({"line": line_idx + 1, "error": repr(exc)})
        totals[path.name] = stats

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rank": rank,
        "world": world,
        "elapsed_seconds": time.time() - started,
        "datasets": totals,
    }
    destination = output_dir / f"rank-{rank:04d}.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rank": rank, "elapsed_seconds": payload["elapsed_seconds"], "output": str(destination)}))


if __name__ == "__main__":
    main()
