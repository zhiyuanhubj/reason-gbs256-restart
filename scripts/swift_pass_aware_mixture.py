#!/usr/bin/env python
"""Exact pass-aware streaming mixture integration for Megatron-SWIFT.

The launcher loads every source exactly once and lets SWIFT apply its normal
column mapping and row preprocessing. This plugin then performs an exact quota
interleave and applies one deterministic bounded shuffle to each mixed phase.
Hugging Face ``all_exhausted`` is never used for quota ownership, so an
exhausted source is never restarted accidentally.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import struct
from bisect import bisect_right
from copy import copy, deepcopy
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from datasets import Features, IterableDataset, Json, List, Value
from datasets.iterable_dataset import ExamplesIterable
from swift.dataset.dataset_meta import BaseDatasetLoader
from swift.dataset.loader import DatasetLoader
from swift.utils import disable_safe_ddp_context_use_barrier, get_logger

logger = get_logger()

_JSONL_INDEX_HEADER = struct.Struct("<8sQQQ")
_JSONL_INDEX_MAGIC = b"JSLIDX01"
_PLAN_SOURCE_PATHS: dict[str, dict[str, Any]] = {}


class _IndexedRawSource:
    """Rank-local physical rows with O(1) JSONL/row-group resume seeks."""

    def __init__(self, dataset: IterableDataset, chunks: Sequence[dict[str, Any]], size: int, source_format: str):
        self.dataset = dataset
        self.chunks = tuple(chunks)
        self.size = int(size)
        self.source_format = source_format
        self._ends = tuple(int(chunk["local_end"]) for chunk in self.chunks)
        if (self.chunks and self._ends[-1] != self.size) or (not self.chunks and self.size):
            raise RuntimeError(f"rank-local {source_format} chunk index does not cover {self.size} rows")

    def _chunk_for(self, index: int) -> tuple[dict[str, Any], int]:
        if not 0 <= index < self.size:
            raise IndexError(f"rank-local row index out of range: {index}/{self.size}")
        chunk_idx = bisect_right(self._ends, index)
        chunk = self.chunks[chunk_idx]
        return chunk, index - int(chunk["local_start"])

    def _iter_jsonl_from(self, start: int):
        key = start
        chunk_idx = bisect_right(self._ends, start) if start < self.size else len(self.chunks)
        for idx in range(chunk_idx, len(self.chunks)):
            chunk = self.chunks[idx]
            within = start - int(chunk["local_start"]) if idx == chunk_idx else 0
            shard_row = int(chunk["shard_start_row"]) + within
            start_byte = _read_jsonl_offsets(chunk["line_index"], (shard_row, ))[0]
            with open(chunk["path"], "rb") as f:
                f.seek(start_byte)
                end_byte = int(chunk["end_byte"])
                while f.tell() < end_byte:
                    line = f.readline()
                    if not line:
                        break
                    yield key, json.loads(line)
                    key += 1

    def _iter_parquet_from(self, start: int):
        import pyarrow.parquet as pq

        key = start
        chunk_idx = bisect_right(self._ends, start) if start < self.size else len(self.chunks)
        for idx in range(chunk_idx, len(self.chunks)):
            chunk = self.chunks[idx]
            within = start - int(chunk["local_start"]) if idx == chunk_idx else 0
            parquet = pq.ParquetFile(chunk["path"])
            skipped = 0
            for batch in parquet.iter_batches(row_groups=[int(chunk["row_group"])]):
                rows = batch.to_pylist()
                if within > skipped:
                    cut = min(len(rows), within - skipped)
                    rows = rows[cut:]
                    skipped += cut
                for row in rows:
                    yield key, row
                    key += 1

    def iter_raw_from(self, start: int = 0):
        if not 0 <= start <= self.size:
            raise IndexError(f"rank-local resume index out of range: {start}/{self.size}")
        if self.source_format == "jsonl":
            yield from self._iter_jsonl_from(start)
        else:
            yield from self._iter_parquet_from(start)

    def processed_from(self, start: int = 0) -> IterableDataset:
        reader = ExamplesIterable(self.iter_raw_from, {"start": int(start)})
        return _clone_with_reader(self.dataset, reader)

    def _read_jsonl_indices(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        rows = []
        handles = {}
        try:
            for index in indices:
                chunk, within = self._chunk_for(index)
                shard_row = int(chunk["shard_start_row"]) + within
                offset = _read_jsonl_offsets(chunk["line_index"], (shard_row, ))[0]
                handle = handles.setdefault(chunk["path"], open(chunk["path"], "rb"))
                handle.seek(offset)
                rows.append(json.loads(handle.readline()))
        finally:
            for handle in handles.values():
                handle.close()
        return rows

    def _read_parquet_indices(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq

        requests: dict[tuple[str, int], list[tuple[int, int]]] = {}
        for output_pos, index in enumerate(indices):
            chunk, within = self._chunk_for(index)
            key = (chunk["path"], int(chunk["row_group"]))
            requests.setdefault(key, []).append((output_pos, within))
        rows: list[dict[str, Any] | None] = [None] * len(indices)
        for (path, row_group), positions in requests.items():
            table = pq.ParquetFile(path).read_row_group(row_group)
            for output_pos, within in positions:
                rows[output_pos] = table.slice(within, 1).to_pylist()[0]
        return rows  # type: ignore[return-value]

    def processed_indices(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        if not indices:
            return []
        raw_rows = (self._read_jsonl_indices(indices) if self.source_format == "jsonl"
                    else self._read_parquet_indices(indices))

        def generate(rows):
            for key, row in enumerate(rows):
                yield key, row

        reader = ExamplesIterable(generate, {"rows": tuple(raw_rows)})
        rows = list(_clone_with_reader(self.dataset, reader))
        if len(rows) != len(indices):
            raise RuntimeError(f"indexed preprocessing changed row count: {len(rows)}!={len(indices)}")
        return rows


def _seed(base: int, dataset: str, logical_pass: int) -> int:
    value = f"{base}:{dataset}:{logical_pass}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def load_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        plan = json.load(f)
    if plan.get("version") != 2:
        raise RuntimeError(f"unsupported pass-aware manifest version: {plan.get('version')!r}")
    if not isinstance(plan.get("seed"), int):
        raise RuntimeError("invalid pass-aware manifest: integer seed is required")

    datasets = plan.get("datasets")
    order = plan.get("dataset_order")
    phases = plan.get("phases")
    if not isinstance(datasets, dict) or not datasets:
        raise RuntimeError("invalid pass-aware manifest: datasets are required")
    if not isinstance(order, list) or len(order) != len(set(order)) or set(order) != set(datasets):
        raise RuntimeError("invalid pass-aware manifest: dataset_order must match datasets exactly")
    if not isinstance(phases, list) or len(phases) != plan.get("phase_count") or not phases:
        raise RuntimeError("invalid pass-aware manifest: phases/phase_count mismatch")

    cursors = {name: 0 for name in order}
    expected_total = 0
    for expected_phase_id, phase in enumerate(phases):
        if not isinstance(phase, dict) or phase.get("pass_id") != expected_phase_id:
            raise RuntimeError("invalid pass-aware manifest: phase ids must be contiguous")
        quotas = phase.get("datasets")
        if not isinstance(quotas, dict):
            raise RuntimeError("invalid pass-aware manifest phase datasets")
        phase_total = 0
        for name, quota in quotas.items():
            if name not in datasets or not isinstance(quota, dict):
                raise RuntimeError(f"invalid pass-aware quota for {name!r}")
            start, count = quota.get("start"), quota.get("count")
            if not isinstance(start, int) or not isinstance(count, int) or count < 1:
                raise RuntimeError(f"invalid pass-aware quota: {name}={quota!r}")
            if start != cursors[name]:
                raise RuntimeError(f"non-contiguous occurrence range for {name}: {start}!={cursors[name]}")
            cursors[name] += count
            phase_total += count
        if phase_total != phase.get("total_samples"):
            raise RuntimeError("pass-aware phase total_samples mismatch")
        expected_total += phase_total

    for name, spec in datasets.items():
        size, target = spec.get("size"), spec.get("target_samples")
        if not isinstance(size, int) or size < 1 or not isinstance(target, int) or target < 1:
            raise RuntimeError(f"invalid pass-aware dataset metadata for {name}")
        if cursors[name] != target:
            raise RuntimeError(f"pass-aware target mismatch for {name}: {cursors[name]}!={target}")
    if expected_total != plan.get("total_samples"):
        raise RuntimeError("pass-aware manifest total_samples does not match phases")
    digest = plan.get("digest")
    unsigned = dict(plan)
    unsigned.pop("digest", None)
    expected_digest = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != expected_digest:
        raise RuntimeError("pass-aware manifest digest mismatch")
    return plan


def _normalize_multimodal_features(dataset):
    """Keep the schema normalization that the exact interleave replaces."""
    if not isinstance(dataset, IterableDataset):
        return dataset
    if dataset.features is None:
        dataset = dataset._resolve_features()
    features = Features(dict(dataset.features))
    changed = False
    if "images" in features:
        features["images"] = List({"bytes": Value("binary"), "path": Value("string")})
        changed = True
    if "objects" in features:
        features["objects"] = Json()
        changed = True
    return dataset.cast(features) if changed else dataset


def _affine_permutation_parameters(size: int, seed: int) -> tuple[int, int, int]:
    """Return ``a, b, a_inv`` for a deterministic bijection modulo ``size``."""
    if size == 1:
        return 0, 0, 0
    rng = random.Random(seed)
    a = rng.randrange(1, size)
    while math.gcd(a, size) != 1:
        a = (a + 1) % size
        if a == 0:
            a = 1
    b = rng.randrange(size)
    return a, b, pow(a, -1, size)


def _fractional_subset(dataset, size: int, count: int, seed: int) -> Iterator[dict[str, Any]]:
    """Select exactly ``count`` rows without replacement across the source."""
    if not 0 < count < size:
        raise RuntimeError(f"fractional subset requires 0<count<size, got {count}/{size}")
    _a, b, a_inv = _affine_permutation_parameters(size, seed)
    seen = 0
    emitted = 0
    for original_index, row in enumerate(dataset):
        if original_index >= size:
            break
        seen += 1
        position = (a_inv * (original_index - b)) % size
        if position < count:
            emitted += 1
            yield row
    if seen != size or emitted != count:
        raise RuntimeError(
            f"fractional source size changed: saw={seen}/{size}, selected={emitted}/{count}"
        )


def _buffer_shuffle(rows: Iterator[dict[str, Any]], seed: int, buffer_size: int) -> Iterator[dict[str, Any]]:
    """Deterministically shuffle a finite iterator with bounded memory."""
    rng = random.Random(seed)
    iterator = iter(rows)
    buffer = list(itertools.islice(iterator, max(1, buffer_size)))
    for row in iterator:
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row
    rng.shuffle(buffer)
    yield from buffer


def _dataset_occurrences(plan: dict[str, Any], name: str, dataset) -> Iterator[dict[str, Any]]:
    """Yield one continuous occurrence stream without a per-source row buffer.

    Source selection is already randomized by the quota interleave below, and
    the resulting mixed phase is shuffled by one shared bounded buffer. Keeping
    another buffer here would retain ``source_count * buffer_size`` multimodal
    rows (and TP replicas of them) for the lifetime of the training iterator.
    """
    spec = plan["datasets"][name]
    size = int(spec["size"])
    remaining = int(spec["target_samples"])
    logical_pass = 0
    while remaining:
        count = min(size, remaining)
        pass_seed = _seed(plan["seed"], name, logical_pass)
        if count == size:
            rows = iter(dataset)
        else:
            rows = _fractional_subset(dataset, size, count, pass_seed)
        emitted = 0
        for row in itertools.islice(iter(rows), count):
            emitted += 1
            # The manifest dataset name is the source identity used for
            # pass/quota accounting, so use the same identity for per-channel
            # loss and token metrics.  Do not inherit a row-level `channel`
            # (for example one mapped from `game`), because multiple source
            # datasets may legitimately share that value.
            row = dict(row)
            row["channel"] = name
            yield row
        if emitted != count:
            raise RuntimeError(
                f"pass-aware source ended early: {name} pass={logical_pass} "
                f"emitted={emitted} expected={count}"
            )
        remaining -= count
        logical_pass += 1


def _iter_phase_rows(plan, phase, streams, cursors) -> Iterator[dict[str, Any]]:
    """Interleave one phase exactly; buffering is owned by the caller."""
    phase_id = phase["pass_id"]
    remaining = {}
    for name, quota in phase["datasets"].items():
        if quota["start"] != cursors[name]:
            raise RuntimeError(f"runtime occurrence cursor mismatch for {name}")
        remaining[name] = int(quota["count"])
    rng = random.Random(_seed(plan["seed"], "__interleave__", phase_id))
    while remaining:
        names = sorted(remaining)
        # Sampling by remaining quota produces a weighted interleave while
        # only ever consuming the head of each per-dataset pass stream.
        choice = rng.randrange(sum(remaining.values()))
        selected = names[-1]
        for name in names:
            choice -= remaining[name]
            if choice < 0:
                selected = name
                break
        yield next(streams[selected])
        cursors[selected] += 1
        remaining[selected] -= 1
        if remaining[selected] == 0:
            del remaining[selected]


def iter_plan_rows(
    plan: dict[str, Any], source_datasets: Sequence, shuffle_buffer_size: int
) -> Iterator[dict[str, Any]]:
    """Yield exact phase quotas, weighted without replacement inside a phase."""
    order = plan["dataset_order"]
    if len(source_datasets) != len(order):
        raise RuntimeError(
            f"pass-aware source count mismatch: got {len(source_datasets)}, expected {len(order)}"
        )
    streams = {name: _dataset_occurrences(plan, name, dataset) for name, dataset in zip(order, source_datasets)}
    cursors = {name: 0 for name in order}
    for phase in plan["phases"]:
        phase_id = phase["pass_id"]
        phase_rows = _iter_phase_rows(plan, phase, streams, cursors)
        # One buffer per mixed phase bounds retained rows independently of the
        # number of source datasets. Flushing at the phase boundary preserves
        # exact pass/quota ownership.
        yield from _buffer_shuffle(
            phase_rows,
            _seed(plan["seed"], "__global_shuffle__", phase_id),
            shuffle_buffer_size,
        )


def _data_parallel_rank() -> tuple[int, int]:
    """Return the pure DP rank; TP/PP/CP replicas must see the same samples."""
    try:
        import torch.distributed as dist
        from megatron.core import mpu
        if not dist.is_available() or not dist.is_initialized():
            return 0, 1
        # Do not include context parallelism here. Every CP rank collaborates on
        # one sequence and therefore must consume identical rows and packed
        # sequence metadata. Using the combined DP+CP group shards different
        # samples onto CP peers and deadlocks their shape-sensitive collectives.
        return mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    except Exception as exc:  # pragma: no cover - exercised in a real launcher
        raise RuntimeError("cannot resolve Megatron data-parallel group for pass-aware mixture") from exc


def _is_tensor_parallel_data_source() -> bool:
    try:
        import torch.distributed as dist
        from megatron.core import mpu
        return not dist.is_available() or not dist.is_initialized() or mpu.get_tensor_model_parallel_rank() == 0
    except Exception as exc:  # pragma: no cover - exercised in a real launcher
        raise RuntimeError("cannot resolve tensor-parallel rank for pass-aware mixture") from exc


def _empty_rows() -> Iterator[dict[str, Any]]:
    if False:
        yield {}


def _replace_ex_iterable_leaf(ex_iterable, replacement):
    """Clone an HF iterable wrapper chain and replace its physical reader."""
    child = getattr(ex_iterable, "ex_iterable", None)
    if child is None:
        return replacement
    cloned = copy(ex_iterable)
    cloned.ex_iterable = _replace_ex_iterable_leaf(child, replacement)
    return cloned


def _clone_with_reader(dataset: IterableDataset, reader) -> IterableDataset:
    return IterableDataset(
        ex_iterable=_replace_ex_iterable_leaf(dataset._ex_iterable, reader),
        info=dataset._info.copy(),
        split=dataset._split,
        formatting=dataset._formatting,
        distributed=deepcopy(dataset._distributed),
        token_per_repo_id=dataset._token_per_repo_id,
    )


def _leaf_ex_iterable(dataset: IterableDataset):
    current = dataset._ex_iterable
    while getattr(current, "ex_iterable", None) is not None:
        current = current.ex_iterable
    return current


def _read_jsonl_offsets(index_path: str, indices: Sequence[int]) -> list[int]:
    values = []
    with open(index_path, "rb") as f:
        payload = f.read(_JSONL_INDEX_HEADER.size)
        magic, _file_size, _mtime_ns, rows = _JSONL_INDEX_HEADER.unpack(payload)
        if magic != _JSONL_INDEX_MAGIC:
            raise RuntimeError(f"invalid JSONL line index: {index_path}")
        for index in indices:
            if not 0 <= index <= rows:
                raise RuntimeError(f"JSONL line index out of range: {index}/{rows}")
            f.seek(_JSONL_INDEX_HEADER.size + index * 8)
            values.append(struct.unpack("<Q", f.read(8))[0])
    return values


def _generate_jsonl_segments(segments: Sequence[dict[str, Any]]):
    key = 0
    for segment in segments:
        with open(segment["path"], "rb") as f:
            f.seek(segment["start_byte"])
            while f.tell() < segment["end_byte"]:
                line = f.readline()
                if not line:
                    break
                yield key, json.loads(line)
                key += 1


def _shard_jsonl_source(dataset, spec, rank: int, world_size: int):
    shards = spec.get("jsonl_shards") or []
    if not shards:
        raise RuntimeError(f"JSONL source lacks line indexes: {spec['path']}")
    total = sum(int(shard["rows"]) for shard in shards)
    if total != int(spec["size"]):
        raise RuntimeError(f"JSONL indexed size mismatch: {spec['path']} {total}!={spec['size']}")

    rank_sizes = [total * (r + 1) // world_size - total * r // world_size for r in range(world_size)]
    global_start = total * rank // world_size
    global_end = total * (rank + 1) // world_size
    segments = []
    indexed_chunks = []
    cursor = 0
    local_cursor = 0
    for shard in shards:
        shard_rows = int(shard["rows"])
        overlap_start = max(global_start, cursor)
        overlap_end = min(global_end, cursor + shard_rows)
        if overlap_end > overlap_start:
            local_start = overlap_start - cursor
            local_end = overlap_end - cursor
            start_byte, end_byte = _read_jsonl_offsets(
                shard["line_index"], (local_start, local_end))
            segments.append({
                "path": shard["path"],
                "start_byte": start_byte,
                "end_byte": end_byte,
            })
            rows = local_end - local_start
            indexed_chunks.append({
                "path": shard["path"],
                "line_index": shard["line_index"],
                "shard_start_row": local_start,
                "end_byte": end_byte,
                "local_start": local_cursor,
                "local_end": local_cursor + rows,
            })
            local_cursor += rows
        cursor += shard_rows
    reader = ExamplesIterable(_generate_jsonl_segments, {"segments": tuple(segments)})
    local_dataset = _clone_with_reader(dataset, reader)
    accessor = _IndexedRawSource(local_dataset, indexed_chunks, rank_sizes[rank], "jsonl")
    return local_dataset, rank_sizes, accessor


def _shard_parquet_source(dataset, spec, rank: int, world_size: int):
    import pyarrow.parquet as pq

    leaf = _leaf_ex_iterable(dataset)
    kwargs = getattr(leaf, "kwargs", None)
    if not isinstance(kwargs, dict) or "files" not in kwargs or "row_groups_list" not in kwargs:
        raise RuntimeError(f"cannot locate Parquet reader for early DP sharding: {spec['path']}")
    files = list(kwargs["files"])
    requested_groups = list(kwargs["row_groups_list"])
    rank_sizes = [0] * world_size
    local_files = []
    local_groups = []
    indexed_chunks = []
    local_cursor = 0
    group_index = 0
    total = 0
    for file, requested in zip(files, requested_groups):
        metadata = pq.ParquetFile(file).metadata
        groups = range(metadata.num_row_groups) if requested is None else requested
        for row_group in groups:
            rows = int(metadata.row_group(int(row_group)).num_rows)
            owner = group_index % world_size
            rank_sizes[owner] += rows
            total += rows
            if owner == rank:
                local_files.append(file)
                local_groups.append((int(row_group), ))
                indexed_chunks.append({
                    "path": file,
                    "row_group": int(row_group),
                    "local_start": local_cursor,
                    "local_end": local_cursor + rows,
                })
                local_cursor += rows
            group_index += 1
    if total != int(spec["size"]):
        raise RuntimeError(f"Parquet indexed size mismatch: {spec['path']} {total}!={spec['size']}")
    local_leaf = copy(leaf)
    local_leaf.kwargs = dict(kwargs, files=local_files, row_groups_list=local_groups)
    local_dataset = _clone_with_reader(dataset, local_leaf)
    accessor = _IndexedRawSource(local_dataset, indexed_chunks, rank_sizes[rank], "parquet")
    return local_dataset, rank_sizes, accessor


def _allocate_quota(count: int, rank_sizes: Sequence[int], total_size: int) -> list[int]:
    """Hamilton apportionment: exact global quota, proportional local quotas."""
    numerators = [count * size for size in rank_sizes]
    quotas = [value // total_size for value in numerators]
    remaining = count - sum(quotas)
    order = sorted(range(len(rank_sizes)), key=lambda r: (-(numerators[r] % total_size), r))
    for r in order[:remaining]:
        quotas[r] += 1
    return quotas


def _build_rank_local_plan(plan, rank_sizes_by_name, rank: int) -> dict[str, Any]:
    local = deepcopy(plan)
    local.pop("digest", None)
    local["source_digest"] = plan.get("digest")
    active = []
    cursors = {}
    for name in plan["dataset_order"]:
        local_size = int(rank_sizes_by_name[name][rank])
        local["datasets"][name]["size"] = local_size
        local["datasets"][name]["target_samples"] = 0
        local["datasets"][name]["full_passes"] = 0
        local["datasets"][name]["fractional_samples"] = 0
        cursors[name] = 0

    for phase in local["phases"]:
        local_quotas = {}
        for name, quota in phase["datasets"].items():
            global_size = int(plan["datasets"][name]["size"])
            counts = _allocate_quota(int(quota["count"]), rank_sizes_by_name[name], global_size)
            count = counts[rank]
            if count:
                local_quotas[name] = {"start": cursors[name], "count": count}
                cursors[name] += count
        phase["datasets"] = local_quotas
        phase["total_samples"] = sum(item["count"] for item in local_quotas.values())

    for name in plan["dataset_order"]:
        spec = local["datasets"][name]
        target = cursors[name]
        size = int(spec["size"])
        spec["target_samples"] = target
        if size and target:
            spec["full_passes"], spec["fractional_samples"] = divmod(target, size)
            active.append(name)
    local["dataset_order"] = active
    local["datasets"] = {name: local["datasets"][name] for name in active}
    local["total_samples"] = sum(cursors.values())
    return local


def _prepare_rank_local_sources(plan, source_datasets, rank: int, world_size: int):
    datasets = {}
    accessors = {}
    rank_sizes = {}
    for name, dataset in zip(plan["dataset_order"], source_datasets):
        spec = plan["datasets"][name]
        if spec.get("format") == "jsonl" or str(spec["path"]).endswith((".jsonl", ".json")):
            local_dataset, sizes, accessor = _shard_jsonl_source(dataset, spec, rank, world_size)
        else:
            local_dataset, sizes, accessor = _shard_parquet_source(dataset, spec, rank, world_size)
        datasets[name] = local_dataset
        accessors[name] = accessor
        rank_sizes[name] = sizes
    local_plan = _build_rank_local_plan(plan, rank_sizes, rank)
    return (
        local_plan,
        [datasets[name] for name in local_plan["dataset_order"]],
        {name: accessors[name] for name in local_plan["dataset_order"]},
    )


class _SourceRuntime:
    """One deterministic occurrence stream with a directly seekable raw cursor."""

    def __init__(self, plan: dict[str, Any], name: str, dataset: IterableDataset, accessor: _IndexedRawSource):
        self.plan = plan
        self.name = name
        self.dataset = dataset
        self.accessor = accessor
        self.size = int(plan["datasets"][name]["size"])
        self.target = int(plan["datasets"][name]["target_samples"])
        self.logical_pass = 0
        self.occurrence_cursor = 0
        self.emitted_in_pass = 0
        self.raw_next = 0
        self._iterator = None

    def _pass_count(self) -> int:
        return min(self.size, self.target - (self.occurrence_cursor - self.emitted_in_pass))

    def _ensure_iterator(self):
        if self._iterator is None:
            self._iterator = iter(self.accessor.processed_from(self.raw_next))

    def _selected(self, raw_index: int, count: int) -> bool:
        if count == self.size:
            return True
        _a, b, a_inv = _affine_permutation_parameters(
            self.size, _seed(self.plan["seed"], self.name, self.logical_pass))
        return (a_inv * (raw_index - b)) % self.size < count

    def _advance_pass(self):
        count = self._pass_count()
        if self.emitted_in_pass != count:
            raise RuntimeError(
                f"source pass ended early: {self.name} pass={self.logical_pass} "
                f"emitted={self.emitted_in_pass}/{count}")
        self.logical_pass += 1
        self.emitted_in_pass = 0
        self.raw_next = 0
        self._iterator = None

    def next(self) -> tuple[dict[str, Any], dict[str, Any]]:
        while self.occurrence_cursor < self.target:
            count = self._pass_count()
            if self.emitted_in_pass >= count:
                self._advance_pass()
                continue
            self._ensure_iterator()
            try:
                row = next(self._iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    f"indexed source ended early: {self.name} raw={self.raw_next}/{self.size}") from exc
            raw_index = self.raw_next
            self.raw_next += 1
            if not self._selected(raw_index, count):
                continue
            ref = {"dataset": self.name, "pass": self.logical_pass, "index": raw_index}
            self.emitted_in_pass += 1
            self.occurrence_cursor += 1
            row = dict(row)
            row["channel"] = self.name
            row["_stream_ref"] = ref
            return ref, row
        raise StopIteration

    def state_dict(self) -> dict[str, Any]:
        return {
            "logical_pass": self.logical_pass,
            "occurrence_cursor": self.occurrence_cursor,
            "emitted_in_pass": self.emitted_in_pass,
            "raw_next": self.raw_next,
        }

    def load_state_dict(self, state: dict[str, Any]):
        self.logical_pass = int(state["logical_pass"])
        self.occurrence_cursor = int(state["occurrence_cursor"])
        self.emitted_in_pass = int(state["emitted_in_pass"])
        self.raw_next = int(state["raw_next"])
        if not 0 <= self.occurrence_cursor <= self.target:
            raise RuntimeError(f"invalid occurrence cursor for {self.name}: {self.occurrence_cursor}/{self.target}")
        if not 0 <= self.raw_next <= self.size:
            raise RuntimeError(f"invalid raw cursor for {self.name}: {self.raw_next}/{self.size}")
        self._iterator = None


class _StatefulPassAwareMixer:
    """Stateful exact interleave whose checkpoint contains references, not rows."""

    VERSION = 1

    def __init__(self, plan, source_datasets, accessors, shuffle_buffer_size, dp_rank, dp_size):
        self.plan = plan
        self.order = tuple(plan["dataset_order"])
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.dp_rank = int(dp_rank)
        self.dp_size = int(dp_size)
        self.sources = {
            name: _SourceRuntime(plan, name, dataset, accessors[name])
            for name, dataset in zip(self.order, source_datasets)
        }
        self.phase_idx = 0
        self.phase_remaining = None
        self.interleave_rng = None
        self.shuffle_rng = None
        self.mode = "fill"
        self.buffer: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.emitted = 0
        self._loaded = False

    def _start_phase(self):
        phase = self.plan["phases"][self.phase_idx]
        self.phase_remaining = {name: int(item["count"]) for name, item in phase["datasets"].items()}
        self.interleave_rng = random.Random(_seed(self.plan["seed"], "__interleave__", self.phase_idx))
        self.shuffle_rng = random.Random(_seed(self.plan["seed"], "__global_shuffle__", self.phase_idx))
        self.mode = "fill"
        self.buffer = []

    def _next_phase_row(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.phase_remaining:
            raise StopIteration
        names = sorted(self.phase_remaining)
        choice = self.interleave_rng.randrange(sum(self.phase_remaining.values()))
        selected = names[-1]
        for name in names:
            choice -= self.phase_remaining[name]
            if choice < 0:
                selected = name
                break
        ref, row = self.sources[selected].next()
        self.phase_remaining[selected] -= 1
        if self.phase_remaining[selected] == 0:
            del self.phase_remaining[selected]
        return ref, row

    def _finish_phase(self):
        self.phase_idx += 1
        self.phase_remaining = None
        self.interleave_rng = None
        self.shuffle_rng = None
        self.mode = "fill"
        self.buffer = []

    def iter_rows(self):
        while self.phase_idx < len(self.plan["phases"]):
            if self.phase_remaining is None:
                self._start_phase()

            if self.mode == "fill":
                while self.phase_remaining and len(self.buffer) < self.shuffle_buffer_size:
                    self.buffer.append(self._next_phase_row())
                self.mode = "replace" if self.phase_remaining else "drain_prepare"

            if self.mode == "replace":
                while self.phase_remaining:
                    incoming = self._next_phase_row()
                    index = self.shuffle_rng.randrange(len(self.buffer))
                    outgoing = self.buffer[index]
                    self.buffer[index] = incoming
                    self.emitted += 1
                    yield outgoing[1]
                self.mode = "drain_prepare"

            if self.mode == "drain_prepare":
                self.shuffle_rng.shuffle(self.buffer)
                self.mode = "drain"

            while self.mode == "drain" and self.buffer:
                _ref, row = self.buffer.pop(0)
                self.emitted += 1
                yield row
            if self.mode == "drain" and not self.buffer:
                self._finish_phase()

    def iter_key_rows(self):
        for key, row in enumerate(self.iter_rows(), start=self.emitted):
            yield key, row

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "plan_digest": self.plan.get("source_digest", self.plan.get("digest")),
            "dp_rank": self.dp_rank,
            "dp_size": self.dp_size,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "phase_idx": self.phase_idx,
            "phase_remaining": deepcopy(self.phase_remaining),
            "interleave_rng_state": self.interleave_rng.getstate() if self.interleave_rng else None,
            "shuffle_rng_state": self.shuffle_rng.getstate() if self.shuffle_rng else None,
            "mode": self.mode,
            "buffer_refs": [deepcopy(ref) for ref, _row in self.buffer],
            "sources": {name: source.state_dict() for name, source in self.sources.items()},
            "emitted": self.emitted,
        }

    def _fetch_refs(self, refs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for pos, ref in enumerate(refs):
            grouped.setdefault(ref["dataset"], []).append((pos, ref))
        rows: list[dict[str, Any] | None] = [None] * len(refs)
        for name, items in grouped.items():
            indices = [int(ref["index"]) for _pos, ref in items]
            processed = self.sources[name].accessor.processed_indices(indices)
            for (pos, ref), row in zip(items, processed):
                row = dict(row)
                row["channel"] = name
                row["_stream_ref"] = deepcopy(ref)
                rows[pos] = row
        return rows  # type: ignore[return-value]

    def fetch_refs(self, refs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._fetch_refs(refs)

    def load_state_dict(self, state: dict[str, Any]):
        if int(state.get("version", -1)) != self.VERSION:
            raise RuntimeError(f"unsupported pass-aware state version: {state.get('version')}")
        expected = (self.plan.get("source_digest", self.plan.get("digest")), self.dp_rank, self.dp_size,
                    self.shuffle_buffer_size)
        actual = (state.get("plan_digest"), int(state["dp_rank"]), int(state["dp_size"]),
                  int(state["shuffle_buffer_size"]))
        if actual != expected:
            raise RuntimeError(f"pass-aware state identity mismatch: {actual}!={expected}")
        self.phase_idx = int(state["phase_idx"])
        self.phase_remaining = deepcopy(state["phase_remaining"])
        self.mode = state["mode"]
        self.emitted = int(state["emitted"])
        for name, source_state in state["sources"].items():
            self.sources[name].load_state_dict(source_state)
        self.interleave_rng = random.Random()
        self.shuffle_rng = random.Random()
        if state["interleave_rng_state"] is None:
            self.interleave_rng = None
        else:
            self.interleave_rng.setstate(state["interleave_rng_state"])
        if state["shuffle_rng_state"] is None:
            self.shuffle_rng = None
        else:
            self.shuffle_rng.setstate(state["shuffle_rng_state"])
        refs = state["buffer_refs"]
        rows = self._fetch_refs(refs)
        self.buffer = [(deepcopy(ref), row) for ref, row in zip(refs, rows)]
        self._loaded = True


_ORIGINAL_INTERLEAVE = BaseDatasetLoader.interleave_datasets
_ORIGINAL_SHUFFLE_DATASET = BaseDatasetLoader.shuffle_dataset
_ORIGINAL_DATASET_LOAD = DatasetLoader.load


def _plan_source_spec(dataset_syntax):
    dataset_path = getattr(dataset_syntax, "dataset", None)
    if not dataset_path:
        return None
    try:
        resolved = os.fspath(Path(dataset_path).resolve())
    except OSError:
        return None
    return _PLAN_SOURCE_PATHS.get(resolved)


def _tp_source_only_load(self, *args, **kwargs):
    """Open physical datasets only on tensor-parallel rank zero.

    The trainer broadcasts the source rank's collated batches to the remaining
    TP ranks.  Returning a lazy empty placeholder here keeps the normal SWIFT
    preprocessing graph intact on consumers without opening Parquet metadata,
    JSONL streams, images, or packing workers there.
    """
    dataset_syntax = args[0] if args else kwargs.get("dataset_syntax")
    source_spec = _plan_source_spec(dataset_syntax)
    # Explicit validation datasets are not part of the mixture manifest. Keep
    # their original all-rank loading path so SWIFT's internal cache barriers
    # remain symmetric. The production validation JSONL has only 64 rows.
    if source_spec is None:
        return _ORIGINAL_DATASET_LOAD(self, *args, **kwargs)
    if _is_tensor_parallel_data_source():
        # DatasetLoader and RowPreprocessor normally serialize every WORLD rank
        # through cache-preparation barriers. Only TP source ranks enter these
        # calls now, so disable those barriers for every manifest source.
        # Streaming maps are lazy and do not materialize Arrow caches here; all
        # JSONL and Parquet access is read-only.
        with disable_safe_ddp_context_use_barrier():
            return _ORIGINAL_DATASET_LOAD(self, *args, **kwargs)
    dataset = IterableDataset.from_generator(_empty_rows)
    dataset._tp_data_consumer = True
    logger.info_once(
        "Pass-aware TP data consumer: physical dataset loading is disabled; "
        "collated batches will be received from TP rank zero."
    )
    return dataset


def _pass_aware_interleave(_datasets, *args, **kwargs):
    if not _datasets:
        return _ORIGINAL_INTERLEAVE(_datasets, *args, **kwargs)
    expected = len(_PLAN["dataset_order"])
    if len(_datasets) != expected:
        raise RuntimeError(
            f"pass-aware train source count mismatch: got {len(_datasets)}, expected {expected}"
        )
    # All TP ranks need the same packed microbatch, but only TP rank zero owns
    # physical readers and packing workers. The trainer broadcasts its collated
    # batch to the other TP ranks before model input preparation.
    if not _is_tensor_parallel_data_source():
        dataset = IterableDataset.from_generator(_empty_rows)
        dataset._pass_aware_phase_shuffled = True
        dataset._tp_data_consumer = True
        logger.info("Pass-aware TP data consumer: physical train readers disabled.")
        return dataset

    dp_rank, dp_size = _data_parallel_rank()
    local_plan, local_sources, accessors = _prepare_rank_local_sources(_PLAN, _datasets, dp_rank, dp_size)
    controller = _StatefulPassAwareMixer(
        local_plan, local_sources, accessors, _SHUFFLE_BUFFER_SIZE, dp_rank, dp_size)
    # Construct the HF wrapper directly. from_generator() fingerprints and
    # serializes gen_kwargs, which would duplicate the 135 live source graphs.
    dataset = IterableDataset(
        ex_iterable=ExamplesIterable(controller.iter_key_rows, {}),
        info=local_sources[0]._info.copy(),
        split=local_sources[0]._split,
    )
    # load_dataset() normally applies another streaming shuffle after the
    # interleave hook. The generator already owns the phase-bounded global
    # shuffle, so mark it to avoid retaining a second multimodal row buffer.
    dataset._pass_aware_phase_shuffled = True
    dataset._tp_data_source = True
    dataset._stream_state_controller = controller
    logger.info(
        "Pass-aware early DP shard: rank %s/%s owns %s sample occurrence(s) "
        "from %s active source(s).",
        dp_rank,
        dp_size,
        local_plan["total_samples"],
        len(local_plan["dataset_order"]),
    )
    return dataset


def _pass_aware_shuffle_dataset(dataset, *args, **kwargs):
    if getattr(dataset, "_pass_aware_phase_shuffled", False):
        return dataset
    return _ORIGINAL_SHUFFLE_DATASET(dataset, *args, **kwargs)


def _install() -> None:
    global _PLAN, _PLAN_SOURCE_PATHS, _SHUFFLE_BUFFER_SIZE
    from swift.pipelines import SwiftSft

    original_shard_streaming_dataset_by_dp = SwiftSft._shard_streaming_dataset_by_dp

    plan_path = os.environ.get("MIXTURE_PLAN")
    if not plan_path:
        raise RuntimeError("PASS_AWARE_MIXTURE=true requires MIXTURE_PLAN")
    _PLAN = load_plan(plan_path)
    _PLAN_SOURCE_PATHS = {
        os.fspath(Path(spec["path"]).resolve()): spec
        for spec in _PLAN["datasets"].values()
    }
    try:
        _SHUFFLE_BUFFER_SIZE = int(os.environ.get("SHUFFLE_BUFFER_SIZE", "10000"))
    except ValueError as exc:
        raise RuntimeError("SHUFFLE_BUFFER_SIZE must be an integer") from exc
    if _SHUFFLE_BUFFER_SIZE < 1:
        raise RuntimeError("SHUFFLE_BUFFER_SIZE must be positive")
    if not hasattr(BaseDatasetLoader, "interleave_datasets"):
        raise RuntimeError("installed SWIFT lacks BaseDatasetLoader.interleave_datasets")
    if not hasattr(IterableDataset, "from_generator"):
        raise RuntimeError("installed datasets lacks IterableDataset.from_generator")
    BaseDatasetLoader.interleave_datasets = staticmethod(_pass_aware_interleave)
    BaseDatasetLoader.shuffle_dataset = staticmethod(_pass_aware_shuffle_dataset)
    DatasetLoader.load = _tp_source_only_load
    # Each DP rank builds its own training batch in parallel. The generator
    # above already owns exact train sharding, so bypass SwiftSft's later modulo
    # filter only for train. Validation still needs Swift's normal DP sharding;
    # otherwise every DP rank evaluates duplicate copies of the same prefix.
    def shard_streaming_dataset_by_dp(dataset, name):
        if name == "train":
            return dataset
        return original_shard_streaming_dataset_by_dp(dataset, name)

    SwiftSft._shard_streaming_dataset_by_dp = staticmethod(shard_streaming_dataset_by_dp)
    logger.info(
        "Pass-aware mixture enabled: %s phase(s), %s exact samples, "
        "one global shuffle buffer of %s row(s), manifest=%s",
        len(_PLAN["phases"]), _PLAN["total_samples"], _SHUFFLE_BUFFER_SIZE, plan_path)


if os.environ.get("PASS_AWARE_MIXTURE", "false").lower() in {"1", "true", "yes", "on"}:
    _install()
