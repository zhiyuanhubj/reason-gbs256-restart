#!/usr/bin/env python3
"""Resolve and validate a resumable Megatron-Core checkpoint.

SWIFT writes one ``checkpoint-N`` directory per save.  A usable directory has
its own iteration tracker and an ``iter_N/common.pt`` file.  For a lossless
resume, the checkpoint must also have been written with optimizer and RNG
saving enabled.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def _load_common(path: Path) -> dict[str, Any]:
    import torch

    common = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(common, dict):
        raise ValueError(f"invalid common state in {path}")
    return common


def inspect_checkpoint(path: Path, require_full_state: bool = False) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")

    tracker = path / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise ValueError(f"missing iteration tracker: {tracker}")
    try:
        iteration = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise ValueError(f"invalid iteration tracker: {tracker}") from exc

    match = CHECKPOINT_RE.search(path.name)
    if match and int(match.group(1)) != iteration:
        raise ValueError(
            f"checkpoint name/tracker mismatch: {path.name} vs iteration {iteration}"
        )

    iteration_dir = path / f"iter_{iteration:07d}"
    common_path = iteration_dir / "common.pt"
    if not common_path.is_file():
        raise ValueError(f"incomplete checkpoint; missing {common_path}")

    state = _load_common(common_path)
    saved_args = state.get("args")
    no_save_optim = bool(getattr(saved_args, "no_save_optim", False))
    no_save_rng = bool(getattr(saved_args, "no_save_rng", False))
    global_batch_size = int(getattr(saved_args, "global_batch_size", 0) or 0)
    consumed_samples = int(
        getattr(saved_args, "consumed_train_samples", 0)
        or iteration * global_batch_size
    )

    if require_full_state:
        missing = []
        if no_save_optim:
            missing.append("optimizer")
        if no_save_rng:
            missing.append("RNG")
        if missing:
            raise ValueError(
                f"checkpoint cannot provide a lossless resume; missing {', '.join(missing)} "
                f"state: {path}"
            )

    return {
        "path": str(path),
        "iteration": iteration,
        "consumed_samples": consumed_samples,
        "global_batch_size": global_batch_size,
        "tensor_model_parallel_size": int(
            getattr(saved_args, "tensor_model_parallel_size", 0) or 0
        ),
        "pipeline_model_parallel_size": int(
            getattr(saved_args, "pipeline_model_parallel_size", 0) or 0
        ),
        "no_save_optim": no_save_optim,
        "no_save_rng": no_save_rng,
    }


def latest_checkpoint(output_dir: Path, require_full_state: bool = False) -> dict[str, Any]:
    candidates = []
    # ``--add_version true`` in older launchers nested checkpoints under
    # vN-<timestamp>; recurse so those runs remain recoverable.
    for path in output_dir.expanduser().rglob("checkpoint-*"):
        match = CHECKPOINT_RE.search(path.name)
        if match and path.is_dir():
            candidates.append((int(match.group(1)), path))
    errors = []
    for _iteration, path in sorted(candidates, reverse=True):
        try:
            return inspect_checkpoint(path, require_full_state=require_full_state)
        except ValueError as exc:
            errors.append(str(exc))
    detail = f" ({'; '.join(errors[:3])})" if errors else ""
    raise ValueError(f"no valid checkpoint found under {output_dir}{detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--latest-under")
    parser.add_argument("--require-full-state", action="store_true")
    parser.add_argument(
        "--field",
        choices=(
            "path",
            "iteration",
            "consumed_samples",
            "global_batch_size",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "json",
        ),
        default="path",
    )
    args = parser.parse_args()

    try:
        if args.checkpoint:
            info = inspect_checkpoint(
                Path(args.checkpoint), require_full_state=args.require_full_state
            )
        else:
            info = latest_checkpoint(
                Path(args.latest_under), require_full_state=args.require_full_state
            )
    except ValueError as exc:
        parser.error(str(exc))

    if args.field == "json":
        print(json.dumps(info, sort_keys=True))
    else:
        print(info[args.field])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
