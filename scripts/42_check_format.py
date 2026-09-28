#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验训练后的 ckpt 是否学会了 BC SFT 的输出格式。

用法:
    python scripts/42_check_format.py outputs/<run>/checkpoint-<step>
    python scripts/42_check_format.py outputs/<run>/checkpoint-<step> --num_samples 10
    python scripts/42_check_format.py outputs/<run>/checkpoint-<step> --gpu 2

校验维度（每个样本计算一项指标，最后聚合）:
  1. ContainsStart    : 输出中是否含 <|action_start|>
  2. ContainsEnd      : 输出中是否含 <|action_end|>
  3. WellFormed       : <|action_start|>...<|action_end|> 配对且非空
  4. ParsesXYZ        : 解析出的 dx dy wheel 段是 3 个数字
  5. HasActionSep     : 输出中含 <|action_sep|> 分隔符（按键组）。注意分号 ; 现在是合法字符，不能再当分隔符
  6. ExactMatch       : 与 ground-truth assistant 文本完全相同（很难）
  7. EditDistance     : 与 GT 的字符编辑距离（越小越好）

也会把 (input -> predicted vs GT) 写进 outputs/<run>/format_check_<step>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))


ACTION_RE = re.compile(r"<\|action_start\|>(.*?)<\|action_end\|>", re.DOTALL)


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def analyze_one(pred: str, gt: str) -> dict:
    has_start = "<|action_start|>" in pred
    has_end = "<|action_end|>" in pred
    matches = ACTION_RE.findall(pred)
    well_formed = bool(matches and matches[0].strip())
    parses_xyz = False
    has_action_sep = False
    if matches:
        body = matches[0]
        # 第一段应该是 "dx dy wheel"，按 <|action_sep|> 切分
        first_seg = body.split("<|action_sep|>")[0].strip()
        nums = first_seg.split()
        parses_xyz = len(nums) == 3 and all(re.fullmatch(r"-?\d+", n) for n in nums)
        has_action_sep = "<|action_sep|>" in body

    return {
        "contains_start": has_start,
        "contains_end": has_end,
        "well_formed": well_formed,
        "parses_xyz": parses_xyz,
        "has_action_sep": has_action_sep,
        "exact_match": pred.strip() == gt.strip(),
        "edit_distance": edit_distance(pred.strip(), gt.strip()),
        "pred_len": len(pred),
        "gt_len": len(gt),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="ckpt 目录或 base model 目录")
    ap.add_argument("--val_file", type=Path,
                    default=PROJECT_ROOT / "data" / "bc_sft_processed" / "val.jsonl")
    ap.add_argument("--num_samples", type=int, default=20, help="抽多少样本验证")
    ap.add_argument("--gpu", type=str, default="2")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None, help="结果写入 jsonl，默认 ckpt 同级")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        print(f"[ERR] checkpoint 不存在: {args.checkpoint}")
        return 1
    if not args.val_file.exists():
        print(f"[ERR] val 数据不存在: {args.val_file}")
        return 1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")

    # lazy import
    import random
    import torch
    from swift.infer_engine import PtEngine, RequestConfig, InferRequest

    random.seed(args.seed)

    # 加载 val 样本
    with args.val_file.open("r", encoding="utf-8") as f:
        all_samples = [json.loads(ln) for ln in f if ln.strip()]
    print(f"[check] val total: {len(all_samples)} samples")

    # 我们对每条 trajectory 只抽 1 个对话轮（第 N 个 user→assistant），
    # 抽满 args.num_samples 个 (sample, turn_idx) 对
    eval_pairs = []
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    for sample in all_samples:
        if len(eval_pairs) >= args.num_samples:
            break
        msgs = sample["messages"]
        # 找所有 (user, assistant) 对
        ua_pairs = []
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                ua_pairs.append(i)
        if not ua_pairs:
            continue
        turn = rng.choice(ua_pairs)
        eval_pairs.append((sample, turn))

    print(f"[check] sampled {len(eval_pairs)} (traj, turn) pairs")
    print(f"[check] loading model from {args.checkpoint} on GPU {args.gpu} ...")

    t0 = time.time()
    engine = PtEngine(model_id_or_path=str(args.checkpoint), torch_dtype=torch.bfloat16)
    print(f"[check] model loaded in {time.time() - t0:.1f}s")

    # 推理
    rc = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0 if args.temperature == 0 else 0.9,
        stream=False,
    )

    out_path = args.out or (args.checkpoint.parent / f"format_check_{args.checkpoint.name}.jsonl")
    print(f"[check] writing per-sample results to: {out_path}")

    aggregates = {
        "contains_start": 0, "contains_end": 0, "well_formed": 0,
        "parses_xyz": 0, "has_action_sep": 0, "exact_match": 0,
        "edit_distance_sum": 0, "n": 0, "errors": 0,
    }

    with out_path.open("w", encoding="utf-8") as fout:
        for idx, (sample, turn) in enumerate(eval_pairs):
            msgs = sample["messages"]
            images = sample.get("images", [])
            # 切到当前 turn 之前 + 当前 user 消息
            ctx_msgs = msgs[: turn + 1]
            gt = msgs[turn + 1]["content"]

            # 拷贝时去掉 assistant
            req = InferRequest(messages=ctx_msgs, images=images)

            try:
                t1 = time.time()
                resp = engine.infer([req], request_config=rc)
                gen = resp[0].choices[0].message.content
                gen_time = time.time() - t1
            except Exception as e:
                print(f"[check]  #{idx}: ERROR {type(e).__name__}: {e}")
                aggregates["errors"] += 1
                continue

            metrics = analyze_one(gen, gt)
            aggregates["n"] += 1
            for k in ("contains_start", "contains_end", "well_formed",
                      "parses_xyz", "has_action_sep", "exact_match"):
                if metrics[k]:
                    aggregates[k] += 1
            aggregates["edit_distance_sum"] += metrics["edit_distance"]

            # 摘要打印
            if idx < 5 or metrics["exact_match"]:
                print(f"\n[check]  #{idx}  turn={turn}  gen_time={gen_time:.2f}s")
                print(f"  GT  : {gt[:160]}{'...' if len(gt) > 160 else ''}")
                print(f"  PRED: {gen[:160]}{'...' if len(gen) > 160 else ''}")
                print(f"  metrics: {metrics}")
            else:
                print(f"[check]  #{idx}  turn={turn}  well_formed={metrics['well_formed']} "
                      f"edit_dist={metrics['edit_distance']:>4} ({gen_time:.2f}s)")

            fout.write(json.dumps({
                "idx": idx,
                "turn": turn,
                "gt": gt,
                "pred": gen,
                "metrics": metrics,
            }, ensure_ascii=False) + "\n")

    # 聚合打印
    n = aggregates["n"]
    print("\n=================== FORMAT CHECK SUMMARY ===================")
    print(f"  ckpt              : {args.checkpoint}")
    print(f"  total inferred    : {n} (errors: {aggregates['errors']})")
    if n > 0:
        print(f"  contains_start    : {aggregates['contains_start']:>4}/{n}  ({100*aggregates['contains_start']/n:.1f}%)")
        print(f"  contains_end      : {aggregates['contains_end']:>4}/{n}  ({100*aggregates['contains_end']/n:.1f}%)")
        print(f"  well_formed (pair): {aggregates['well_formed']:>4}/{n}  ({100*aggregates['well_formed']/n:.1f}%)")
        print(f"  parses_xyz        : {aggregates['parses_xyz']:>4}/{n}  ({100*aggregates['parses_xyz']/n:.1f}%)")
        print(f"  has_action_sep    : {aggregates['has_action_sep']:>4}/{n}  ({100*aggregates['has_action_sep']/n:.1f}%)")
        print(f"  exact_match       : {aggregates['exact_match']:>4}/{n}  ({100*aggregates['exact_match']/n:.1f}%)")
        print(f"  avg edit distance : {aggregates['edit_distance_sum']/n:.1f}")
    print(f"  per-sample jsonl  : {out_path}")
    print("=============================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
