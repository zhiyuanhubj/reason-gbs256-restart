#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 yuanshen_sample30 下多个玩家目录的 train.jsonl 合并 + 完全 shuffle + 切 val。

每个玩家目录结构:
    <player>/train.jsonl                 # messages / images(相对路径) / game
    <player>/images/chunk_xxxx/...       # 图片

关键点:
  - 每个玩家 jsonl 里的 images 是**相对该玩家目录**的相对路径 (images/chunk_0000/...)，
    多个玩家合并时相对路径会冲突，所以这里统一改写成**绝对路径**。
  - 完全 shuffle（所有玩家所有轨迹混在一起打乱），固定 seed 可复现。
  - 末尾切出一个小 val 集给训练时 eval 用。

用法:
    python scripts/07_merge_shuffle_genshin.py \
        --input_dir /fsx/home/zhiyuan/yuanshen_sample30 \
        --output_dir /fsx/home/zhiyuan/yuanshen_sample30/_merged \
        --val_size 64 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=Path("/fsx/home/zhiyuan/yuanshen_sample30"))
    ap.add_argument("--output_dir", type=Path, default=Path("/fsx/home/zhiyuan/yuanshen_sample30/_merged"))
    ap.add_argument("--val_size", type=int, default=64, help="末尾切出多少条做 val")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check_images", action="store_true", help="逐条校验图片存在(慢)")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 找到所有玩家 train.jsonl（排除输出目录本身，以及 *_sample / logs 等非正式玩家目录）
    jsonl_files = sorted(
        p for p in input_dir.glob("*/train.jsonl")
        if out_dir not in p.parents
        and not p.parent.name.endswith("_sample")
        and p.parent.name not in ("logs", "_merged")
    )
    if not jsonl_files:
        print(f"[ERR] 没在 {input_dir} 找到 */train.jsonl")
        return 1

    print(f"[merge] input : {input_dir}")
    print(f"[merge] output: {out_dir}")
    print(f"[merge] 找到 {len(jsonl_files)} 个玩家 train.jsonl")

    all_records: list[dict] = []
    n_missing_img = 0
    per_player: dict[str, int] = {}

    for fp in jsonl_files:
        player_dir = fp.parent
        player = player_dir.name
        n = 0
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                # 相对路径 -> 绝对路径（相对该玩家目录）
                # 注意: 用纯字符串拼接(os.path.join/normpath)，不要用 Path.resolve()，
                # 后者会对每张图做 stat 解析软链，74820 张在 Lustre 上极慢。
                abs_imgs = []
                for img in rec.get("images", []):
                    if os.path.isabs(img):
                        abs_path = img
                    else:
                        abs_path = os.path.normpath(os.path.join(str(player_dir), img))
                    if args.check_images and not os.path.exists(abs_path):
                        n_missing_img += 1
                    abs_imgs.append(abs_path)
                rec["images"] = abs_imgs
                # 只保留 ms-swift 标准字段(messages/images/game)，避免额外列引发兼容问题。
                # 来源玩家仅用于打印统计，不写进 jsonl。
                rec = {k: rec[k] for k in ("messages", "images", "game") if k in rec}
                all_records.append(rec)
                n += 1
        per_player[player] = n
        print(f"  [{player:12s}] {n:5d} 条")

    print(f"[merge] 合计 {len(all_records)} 条轨迹")
    if args.check_images and n_missing_img:
        print(f"[WARN] 有 {n_missing_img} 张图片路径不存在！")

    # 完全 shuffle
    rng = random.Random(args.seed)
    rng.shuffle(all_records)
    print(f"[merge] 已用 seed={args.seed} 完全 shuffle")

    # 切 val（从打乱后的末尾切，等价于随机切）
    val_size = min(args.val_size, len(all_records) // 10)
    val_records = all_records[:val_size]
    train_records = all_records[val_size:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with val_path.open("w", encoding="utf-8") as f:
        for rec in val_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[merge] train: {len(train_records)} 条 -> {train_path}")
    print(f"[merge] val  : {len(val_records)} 条 -> {val_path}")
    print("[merge] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
