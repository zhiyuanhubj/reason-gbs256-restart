#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给 jsonl 每行加 channel 字段(=游戏名)，用于按游戏分别追踪 loss(--enable_channel_loss)。

默认用每行已有的 "game" 字段作为 channel；也可用 --channel 强制指定。
就地改写或写到新文件。

用法:
  # 用已有 game 字段作 channel, 就地改写
  python 12_add_channel.py /path/train.jsonl --inplace
  # 强制指定 channel=honkai, 写到新文件
  python 12_add_channel.py in.jsonl --channel honkai --out out.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--channel", default=None, help="强制 channel 值；默认用每行的 game 字段")
    ap.add_argument("--out", default=None)
    ap.add_argument("--inplace", action="store_true")
    args = ap.parse_args()

    if not args.inplace and not args.out:
        sys.exit("[ERR] 需要 --out 或 --inplace")
    out = args.input + ".tmp" if args.inplace else args.out

    n = 0
    miss_game = 0
    chans = {}
    with open(args.input, encoding="utf-8") as fin, open(out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ch = args.channel or r.get("game")
            if ch is None:
                miss_game += 1
                ch = "unknown"
            r["channel"] = ch
            chans[ch] = chans.get(ch, 0) + 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1

    if args.inplace:
        os.replace(out, args.input)
        out = args.input
    print(f"[add_channel] {n} 行 -> {out}")
    print(f"[add_channel] channel 分布: {chans}")
    if miss_game:
        print(f"[WARN] {miss_game} 行没有 game 字段, channel 置为 'unknown'")


if __name__ == "__main__":
    sys.exit(main())
