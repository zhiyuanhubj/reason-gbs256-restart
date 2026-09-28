#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 zhiyuanhucs/delta-force-bc-sft-shuffled 数据集转换成 ms-swift 训练格式。

输入  : data/delta-force-bc-sft-shuffled/*.parquet   (ShareGPT 格式 + binary images)
输出  : data/bc_sft_processed/
        ├── images/<chunk_idx>/<row_idx>/<img_idx>.jpg     # 75940 张图
        ├── train.jsonl                                     # 17 chunks (~3400 traj)
        └── val.jsonl                                       # 2 chunks (~400 traj)

每个 jsonl 行格式（ms-swift messages 标准）:
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "<image>"},   # 1
        {"role": "assistant", "content": "<|action_start|>...<|action_end|>"},
        ... (20 轮)
      ],
      "images": ["abs/path/0.jpg", "abs/path/1.jpg", ..., "abs/path/19.jpg"],
      "game": "delta_force"
    }

用法:
    python scripts/06_prepare_bc_dataset.py
    python scripts/06_prepare_bc_dataset.py --val_chunks 2 --num_workers 4
    python scripts/06_prepare_bc_dataset.py --skip_images   # 仅校验/重写 jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))

DEFAULT_INPUT = PROJECT_ROOT / "data" / "delta-force-bc-sft-shuffled"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "bc_sft_processed"

ROLE_MAP = {"system": "system", "human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}


def process_one_parquet(args) -> dict:
    """处理一个 parquet 文件，返回 (out_jsonl_lines, n_rows, n_imgs, n_skipped)."""
    parquet_path: Path
    out_image_dir: Path
    chunk_idx: int
    skip_images: bool
    parquet_path, out_image_dir, chunk_idx, skip_images, rel_to, abs_paths, game = args

    table = pq.read_table(str(parquet_path))
    rows = table.to_pylist()

    out_lines = []
    n_imgs = 0
    n_skipped = 0

    for row_idx, row in enumerate(rows):
        images_bytes = row.get("images") or []
        convs = row.get("conversations") or []

        if len(images_bytes) == 0 or len(convs) == 0:
            n_skipped += 1
            continue

        row_img_dir = out_image_dir / f"chunk_{chunk_idx:03d}" / f"row_{row_idx:04d}"

        # 写图像文件
        image_paths: list[str] = []
        if not skip_images:
            row_img_dir.mkdir(parents=True, exist_ok=True)
        for i, img_bytes in enumerate(images_bytes):
            img_path = row_img_dir / f"{i:02d}.jpg"
            if not skip_images:
                img_path.write_bytes(img_bytes)
            # 默认写相对路径（相对 rel_to，一般是输出目录），便于整目录搬移；
            # 绝对路径换机器/换路径就失效了。
            if abs_paths:
                image_paths.append(str(img_path))
            else:
                image_paths.append(os.path.relpath(img_path, rel_to))
        n_imgs += len(image_paths)

        # ShareGPT (from/value) → messages (role/content)
        messages = []
        for conv in convs:
            role_raw = conv.get("from") or conv.get("role") or ""
            content = conv.get("value") or conv.get("content") or ""
            role = ROLE_MAP.get(role_raw)
            if role is None:
                n_skipped += 1
                continue
            messages.append({"role": role, "content": content})

        if not messages or not any(m["role"] == "assistant" and m["content"] for m in messages):
            n_skipped += 1
            continue

        out_lines.append(json.dumps({
            "messages": messages,
            "images": image_paths,
            "game": game,
        }, ensure_ascii=False))

    return {
        "chunk_idx": chunk_idx,
        "lines": out_lines,
        "n_rows": len(rows),
        "n_imgs": n_imgs,
        "n_skipped": n_skipped,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--val_chunks",
        type=int,
        default=2,
        help="last N parquet 当 val；设为 0 表示全部用于 train、不生成 val.jsonl",
    )
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--skip_images", action="store_true", help="仅重写 jsonl，不重写图片")
    ap.add_argument("--game", type=str, default="delta_force", help="写进 jsonl 的 game 标签，如 genshin")
    ap.add_argument("--rel_to", type=Path, default=None,
                    help="图片相对路径基准目录，默认 = --output_dir（整目录可整体搬走）")
    ap.add_argument("--abs_paths", action="store_true", help="写绝对路径（默认相对路径）")
    args = ap.parse_args()

    parquet_files = sorted(args.input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[ERR ] 没在 {args.input_dir} 找到 parquet 文件。先跑 scripts/05_download_dataset.py")
        return 1

    if args.val_chunks < 0:
        ap.error("--val_chunks must be >= 0")

    n_total = len(parquet_files)
    n_val = 0 if args.val_chunks == 0 else max(1, min(args.val_chunks, n_total // 2))
    train_files = parquet_files if n_val == 0 else parquet_files[:-n_val]
    val_files = [] if n_val == 0 else parquet_files[-n_val:]

    print(f"[prep] input  : {args.input_dir}  ({n_total} parquet files)")
    print(f"[prep] output : {args.output_dir}")
    print(f"[prep] split  : train={len(train_files)} chunks, val={len(val_files)} chunks")
    print(f"[prep] workers: {args.num_workers}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rel_to = (args.rel_to or args.output_dir).resolve()
    print(f"[prep] game   : {args.game}")
    if args.abs_paths:
        print("[prep] paths  : ABSOLUTE (--abs_paths)")
    else:
        print(f"[prep] rel_to : {rel_to}  (图片以相对此目录的路径写进 jsonl)")

    t0 = time.time()
    splits = [("train", train_files)]
    if val_files:
        splits.append(("val", val_files))
    for split, files in splits:
        out_jsonl = args.output_dir / f"{split}.jsonl"
        print(f"\n[prep] === processing {split} ({len(files)} chunks) -> {out_jsonl} ===")

        tasks = [
            (p, image_dir, i, args.skip_images, str(rel_to), args.abs_paths, args.game)
            for i, p in enumerate(files, start=(0 if split == "train" else len(train_files)))
        ]

        n_rows_total = n_imgs_total = n_skipped_total = 0
        with out_jsonl.open("w", encoding="utf-8") as fout:
            if args.num_workers > 1:
                with mp.Pool(args.num_workers) as pool:
                    for r in pool.imap_unordered(process_one_parquet, tasks):
                        for ln in r["lines"]:
                            fout.write(ln + "\n")
                        n_rows_total += r["n_rows"]
                        n_imgs_total += r["n_imgs"]
                        n_skipped_total += r["n_skipped"]
                        elapsed = time.time() - t0
                        print(f"  [chunk {r['chunk_idx']:>3}] rows={r['n_rows']} imgs={r['n_imgs']} "
                              f"skipped={r['n_skipped']} | total {n_rows_total}/{n_imgs_total} "
                              f"({elapsed:.0f}s)")
            else:
                for task in tasks:
                    r = process_one_parquet(task)
                    for ln in r["lines"]:
                        fout.write(ln + "\n")
                    n_rows_total += r["n_rows"]
                    n_imgs_total += r["n_imgs"]
                    n_skipped_total += r["n_skipped"]
                    elapsed = time.time() - t0
                    print(f"  [chunk {r['chunk_idx']:>3}] rows={r['n_rows']} imgs={r['n_imgs']} "
                          f"skipped={r['n_skipped']} | total {n_rows_total}/{n_imgs_total} "
                          f"({elapsed:.0f}s)")

        print(f"[prep] {split}: {n_rows_total} rows, {n_imgs_total} images, {n_skipped_total} skipped")
        print(f"[prep] -> {out_jsonl}  ({out_jsonl.stat().st_size / 1024:.1f} KB)")

    elapsed = time.time() - t0
    print(f"\n[prep] all done in {elapsed:.0f}s.")
    print(f"[prep] verify with:")
    print(f"  python {PROJECT_ROOT}/scripts/02_prepare_data.py {args.output_dir}/train.jsonl --stats")
    return 0


if __name__ == "__main__":
    sys.exit(main())
