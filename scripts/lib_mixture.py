#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按每个数据集的总训练倍率生成 ms-swift ``path#N`` 规格和 phase manifest。

配置模板见 ``data_configs/mixture.example.yaml``。

推荐字段：
  - passes: 0.6  随机无重复使用数据集的 60%
  - passes: 1.0  完整训练一遍
  - passes: 2.3  完整训练两遍，再随机使用 30%

兼容旧字段：
  - multiplier: 与 passes 含义相同
  - sample:     直接指定绝对采样条数

混训数据已经在这里展开为整个训练任务所需的最终配额，因此训练器必须只
运行一个 epoch。各数据集合并后由 ms-swift 全局 shuffle。

JSONL mixture 直接统计单文件或子集目录内所有 ``*.jsonl`` shard 的行数。
Parquet mixture 从 root 下的 canonical_index.json 读取每个 canonical
子数据集的行数。Mixed mixture 允许每个 dataset 通过 ``format``
选择 JSONL 或 Parquet，并通过 ``default_dataset_format`` 设置默认格式。
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import struct
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


_JSONL_INDEX_MAGIC = b"JSLIDX01"
_JSONL_INDEX_HEADER = struct.Struct("<8sQQQ")


def _jsonl_index_cache_path(path: Path) -> Path:
    cache_root = os.environ.get("JSONL_INDEX_CACHE_DIR")
    if cache_root:
        root = Path(os.path.expandvars(cache_root))
    else:
        project_root = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
        root = project_root / "data" / ".jsonl_indices"
    digest = hashlib.sha256(os.fspath(path.resolve()).encode()).hexdigest()[:20]
    return root / f"{path.name}.{digest}.lineidx"


def _read_jsonl_index_header(index_path: Path) -> tuple[int, int, int] | None:
    try:
        with index_path.open("rb") as f:
            payload = f.read(_JSONL_INDEX_HEADER.size)
        magic, file_size, mtime_ns, rows = _JSONL_INDEX_HEADER.unpack(payload)
    except (FileNotFoundError, OSError, struct.error):
        return None
    if magic != _JSONL_INDEX_MAGIC:
        return None
    return file_size, mtime_ns, rows


def ensure_jsonl_line_index(path: Path) -> tuple[Path, int]:
    """Build a compact seek index once, then reuse it for early DP sharding."""
    path = path.resolve()
    stat = path.stat()
    index_path = _jsonl_index_cache_path(path)
    cached = _read_jsonl_index_header(index_path)
    if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
        expected_size = _JSONL_INDEX_HEADER.size + (cached[2] + 1) * 8
        if index_path.stat().st_size == expected_size:
            return index_path, cached[2]

    offsets = array.array("Q", [0])
    position = 0
    with path.open("rb") as f:
        for line in f:
            position += len(line)
            offsets.append(position)
    if position != stat.st_size:
        raise RuntimeError(f"JSONL index size mismatch: {path} read={position} stat={stat.st_size}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(_JSONL_INDEX_HEADER.pack(
            _JSONL_INDEX_MAGIC, stat.st_size, stat.st_mtime_ns, len(offsets) - 1))
        offsets.tofile(f)
    tmp.replace(index_path)
    return index_path, len(offsets) - 1


def inspect_jsonl_path(path: Path) -> tuple[int, list[dict]]:
    """Count rows and record seek indexes for one JSONL file or shard directory."""
    if path.is_file():
        shards = [path]
    elif path.is_dir():
        shards = sorted(path.rglob("*.jsonl"))
        if not shards:
            raise ValueError(f"JSONL 数据集目录中没有 .jsonl 文件: {path}")
    else:
        raise ValueError(f"数据集路径不存在: {path}")
    details = []
    for shard in shards:
        index_path, rows = ensure_jsonl_line_index(shard)
        details.append({
            "path": os.fspath(shard.resolve()),
            "line_index": os.fspath(index_path.resolve()),
            "rows": rows,
        })
    return sum(item["rows"] for item in details), details


def _jsonl_root(cfg: dict, override: str | None = None) -> Path | None:
    value = override or cfg.get("root")
    if not value:
        return None
    root = Path(os.path.expandvars(os.fspath(value)))
    if not root.is_absolute() and os.environ.get("NFS_DIR"):
        root = Path(os.environ["NFS_DIR"]) / "data" / root
    return root


def load_yaml(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        sys.exit("[ERR] 需要 pyyaml: pip install pyyaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML 顶层必须是 mapping")
    return cfg


def _positive_decimal(value, field: str, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"数据集 {name} 的 {field} 不是有效数字: {value!r}") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"数据集 {name} 的 {field} 必须是有限正数")
    return number


def _parquet_root(cfg: dict, override: str | None = None) -> Path:
    value = override or cfg.get("root")
    if not isinstance(value, str) or not value:
        raise ValueError("parquet mixture requires root or --data-root")
    root = Path(os.path.expandvars(value))
    if not root.is_absolute() and os.environ.get("NFS_DIR"):
        root = Path(os.environ["NFS_DIR"]) / "data" / root
    return root


def _parquet_sizes(cfg: dict, override: str | None = None) -> tuple[Path, dict[str, int]]:
    root = _parquet_root(cfg, override)
    index_path = root / "canonical_index.json"
    if not index_path.is_file():
        raise ValueError(f"canonical parquet index does not exist: {index_path}")
    with index_path.open(encoding="utf-8") as f:
        index = json.load(f)
    sizes = {}
    for item in index.get("datasets") or []:
        name = item.get("canonical_name")
        rows = (item.get("stats") or {}).get("output_rows")
        if isinstance(name, str) and isinstance(rows, int) and rows > 0:
            sizes[name] = rows
    if not sizes:
        raise ValueError(f"canonical parquet index contains no dataset row counts: {index_path}")
    return root, sizes


def compute(cfg: dict, data_root: str | None = None) -> dict:
    if "weight" in cfg or "total_samples" in cfg:
        raise ValueError("不再支持 weight/total_samples；请为每个数据集设置 passes")
    if cfg.get("datasets") is not None and cfg.get("games") is not None:
        raise ValueError("YAML 不能同时包含 datasets 和 games")

    datasets = cfg.get("datasets")
    if datasets is None:
        datasets = cfg.get("games")  # 兼容旧配置名
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("YAML 中必须包含非空的 datasets mapping")

    data_format = cfg.get("format", "jsonl")
    if data_format not in {"jsonl", "parquet", "mixed"}:
        raise ValueError("format must be jsonl, parquet, or mixed")
    default_dataset_format = cfg.get(
        "default_dataset_format", "jsonl" if data_format == "mixed" else data_format
    )
    if default_dataset_format not in {"jsonl", "parquet"}:
        raise ValueError("default_dataset_format must be jsonl or parquet")
    parquet_root = None
    parquet_sizes = None
    if data_format == "parquet" or (
        data_format == "mixed"
        and (
            default_dataset_format == "parquet"
            or any(
                isinstance(item, dict) and item.get("format") == "parquet"
                for item in datasets.values()
            )
        )
    ):
        parquet_root, parquet_sizes = _parquet_sizes(cfg, data_root)

    info = {}
    for name, item in datasets.items():
        if not isinstance(item, dict):
            raise ValueError(f"数据集 {name} 的配置必须是 mapping")
        if "weight" in item:
            raise ValueError(f"数据集 {name} 仍在使用 weight；请改为 passes")
        item_format = item.get("format", default_dataset_format)
        if item_format not in {"jsonl", "parquet"}:
            raise ValueError(f"数据集 {name} 的 format 必须是 jsonl 或 parquet")
        controls = [key for key in ("passes", "multiplier", "sample") if key in item]
        if len(controls) > 1:
            raise ValueError(
                f"数据集 {name} 只能设置 passes/multiplier/sample 中的一项，"
                f"当前设置了: {', '.join(controls)}"
            )
        if "path" not in item and item_format == "jsonl" and _jsonl_root(cfg, data_root) is None:
            raise ValueError(f"JSONL 数据集 {name} 缺少 path，且 YAML 未设置 root")

        if item_format == "parquet":
            assert parquet_root is not None and parquet_sizes is not None
            relative = os.fspath(item.get("path", name))
            path = os.fspath(parquet_root / relative)
            if name not in parquet_sizes:
                raise ValueError(f"数据集 {name} 不在 canonical_index.json 中")
            if not os.path.isdir(path):
                raise ValueError(f"数据集 {name} 的目录不存在: {path}")
            size = parquet_sizes[name]
            spec_path = f"{path}/."
        else:
            root = _jsonl_root(cfg, data_root)
            relative = os.fspath(item.get("path", name))
            path_value = Path(os.path.expandvars(relative))
            if not path_value.is_absolute() and root is not None:
                path_value = root / path_value
            if "path" not in item and not path_value.exists():
                jsonl_candidate = path_value.with_suffix(".jsonl")
                if jsonl_candidate.is_file():
                    path_value = jsonl_candidate
            path = os.fspath(path_value)
            size, jsonl_shards = inspect_jsonl_path(path_value)
            if size <= 0:
                raise ValueError(f"数据集 {name} 没有有效 JSONL 行: {path}")
            spec_path = path

        if "sample" in item:
            sample = int(item["sample"])
            if sample <= 0:
                raise ValueError(f"数据集 {name} 的 sample 必须 > 0")
            passes = Decimal(sample) / Decimal(size)
            mode = "sample"
        else:
            field = "passes" if "passes" in item else "multiplier"
            passes = _positive_decimal(item.get(field, 1), field, name)
            sample = int(
                (Decimal(size) * passes).to_integral_value(rounding=ROUND_HALF_UP)
            )
            mode = field if field in item else "default"

        info[name] = {
            "path": path,
            "spec_path": spec_path,
            "size": size,
            "passes": passes,
            "n": sample,
            "mode": mode,
            "format": item_format,
        }
        if item_format == "jsonl":
            info[name]["jsonl_shards"] = jsonl_shards
    return info


def build_pass_plan(info: dict, seed: int) -> dict:
    """Return a compact plan for deterministic, uniformly phased exposure.

    Each dataset owns one logical occurrence stream. Occurrences ``0..size``
    form pass zero, the next ``size`` occurrences form pass one, and so on.
    Fractional datasets are spread across all global phases, while a 3-pass
    anchor has exactly one complete pass in each of three phases.
    """
    if not info:
        raise ValueError("cannot build a pass-aware plan from an empty mixture")

    datasets: dict[str, dict] = {}
    for name, item in info.items():
        size, target = int(item["size"]), int(item["n"])
        if size < 1 or target < 1:
            raise ValueError(
                f"invalid pass-aware dataset size/quota: {name} size={size} target={target}"
            )
        full_passes, remainder = divmod(target, size)
        datasets[name] = {
            "path": item["path"],
            "spec_path": item.get("spec_path", item["path"]),
            "format": item.get("format", "jsonl"),
            "size": size,
            "target_samples": target,
            "full_passes": full_passes,
            "fractional_samples": remainder,
        }
        if item.get("jsonl_shards"):
            datasets[name]["jsonl_shards"] = item["jsonl_shards"]

    phase_count = max(
        (item["target_samples"] + item["size"] - 1) // item["size"]
        for item in datasets.values()
    )
    phase_list = []
    for phase_id in range(phase_count):
        quotas = {}
        for name, item in datasets.items():
            target = item["target_samples"]
            start = target * phase_id // phase_count
            end = target * (phase_id + 1) // phase_count
            if end > start:
                quotas[name] = {"start": start, "count": end - start}
        phase_list.append({
            "pass_id": phase_id,
            "datasets": quotas,
            "total_samples": sum(quota["count"] for quota in quotas.values()),
        })
    manifest = {
        "version": 2,
        "seed": int(seed),
        "phase_count": phase_count,
        "dataset_order": list(datasets),
        "total_samples": sum(item["target_samples"] for item in datasets.values()),
        "datasets": datasets,
        "phases": phase_list,
    }
    # Lets launchers detect a stale or partially replaced shared manifest.
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["digest"] = hashlib.sha256(payload.encode()).hexdigest()
    return manifest


def write_pass_plan(path: str, info: dict, seed: int) -> dict:
    manifest = build_pass_plan(info, seed)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return manifest


def print_audit(info: dict, stream=sys.stdout) -> None:
    total = sum(item["n"] for item in info.values())
    print(
        f"{'数据集':<20}{'原始条数':>12}{'passes':>12}"
        f"{'采样条数':>12}{'实际概率':>12}{'模式':>12}",
        file=stream,
    )
    print("-" * 80, file=stream)
    for name, item in info.items():
        probability = item["n"] / total if total else 0
        print(
            f"{name:<20}{item['size']:>12}{str(item['passes']):>12}"
            f"{item['n']:>12}{probability:>11.4%}{item['mode']:>12}",
            file=stream,
        )
    print("-" * 80, file=stream)
    print(
        f"{'合计':<20}{sum(item['size'] for item in info.values()):>12}"
        f"{'':>12}{total:>12}{100:>11.4f}%",
        file=stream,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml")
    parser.add_argument("--print-spec", action="store_true")
    parser.add_argument(
        "--print-source-spec",
        action="store_true",
        help="print each source exactly once for the pass-aware streaming loader",
    )
    parser.add_argument(
        "--print-interleave-prob",
        action="store_true",
        help="print sample-count-normalized interleave probabilities in dataset order",
    )
    parser.add_argument(
        "--print-total-samples",
        action="store_true",
        help="print the final sample exposure after applying every dataset pass multiplier",
    )
    parser.add_argument("--print-val", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--data-root", help="override the mixture data root")
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed stored in a pass-aware plan")
    parser.add_argument("--write-plan", help="write a versioned pass-aware manifest")
    args = parser.parse_args()

    cfg = load_yaml(args.yaml)
    if args.print_val:
        # One path per line preserves spaces and supports multiple validation sets.
        values = cfg.get("val", "") or ""
        if not isinstance(values, list):
            values = [values]
        for value in values:
            val = os.path.expandvars(os.fspath(value))
            if cfg.get("format") == "parquet" and val and not Path(val).is_absolute():
                val = _parquet_root(cfg, args.data_root).parent / val
            print(val)
        return

    info = compute(cfg, args.data_root)
    manifest = None
    if args.write_plan:
        manifest = write_pass_plan(args.write_plan, info, args.seed)
    if args.print_source_spec:
        print_audit(info, stream=sys.stderr)
        if manifest is not None:
            print(
                f"[mixture] wrote pass-aware plan: {args.write_plan} "
                f"({len(manifest['phases'])} phases, {manifest['total_samples']} samples)",
                file=sys.stderr,
            )
        print(" ".join(f"{item['spec_path']}#{item['size']}" for item in info.values()))
        return
    if args.print_interleave_prob:
        total = sum(item["n"] for item in info.values())
        if total <= 0:
            raise ValueError("所有数据集的最终采样条数都是 0")
        print(" ".join(str(item["n"] / total) for item in info.values()))
        return
    if args.print_total_samples:
        print(sum(item["n"] for item in info.values()))
        return
    active = [(name, item) for name, item in info.items() if item["n"] > 0]
    skipped = [name for name, item in info.items() if item["n"] == 0]
    if not active:
        raise ValueError("所有数据集的最终采样条数都是 0")

    if args.print_spec:
        print_audit(info, stream=sys.stderr)
        if skipped:
            print(
                "[mixture][WARN] 配额取整为 0，未加入训练: " + ", ".join(skipped),
                file=sys.stderr,
            )
        print(" ".join(f"{item['spec_path']}#{item['n']}" for _, item in active))
        return

    if manifest is not None:
        print(f"wrote pass-aware plan: {args.write_plan} ({len(manifest['phases'])} phase(s), "
              f"{manifest['total_samples']} samples)")
        return

    print_audit(info)
    print(f"\nval: {cfg.get('val', '(无)')}")
    print(
        "DATASET_SPEC = "
        + " ".join(f"{item['spec_path']}#{item['n']}" for _, item in active)
    )


if __name__ == "__main__":
    main()
