#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 ckpt 在 val 上推理，保存【完整输入+输出+GT】并量化"动作幅度"指标。
专门用于排查"模型输出动作不动/幅度偏小"的问题。

对每条选中的 val 轨迹，按 turn 逐步推理（context = system + 历史图+历史GT动作 + 当前图），
预测当前 step 的动作，与 GT 对比。

指标：
  - 格式：contains_start/end, well_formed, parses_xyz, has_action_sep
  - 幅度：解析 pred 与 GT 的 dx,dy；统计 |dx|+|dy| 的均值、pred/GT 比值；
          GT 动了但 pred 没动(0 0 0) 的比例（== "动作不动"失败率）
完整结果写 jsonl：每条含 full_input(文本形式,图用路径) / gt / pred / 解析的 dx dy。

用法(在 GPU 节点)：
  python scripts/42b_eval_actions.py <ckpt> --val_file <val.jsonl> \
      --num_traj 12 --max_turns 30 --gpu 0 --out <out.jsonl>
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path

ACTION_RE = re.compile(r"<\|action_start\|>(.*?)<\|action_end\|>", re.DOTALL)
FIRST_SEG_RE = re.compile(r"<\|action_start\|>\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)")


def parse_mouse(text):
    m = FIRST_SEG_RE.search(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def analyze(pred, gt):
    matches = ACTION_RE.findall(pred)
    body = matches[0] if matches else ""
    pm = parse_mouse(pred)
    gm = parse_mouse(gt)
    d = {
        "contains_start": "<|action_start|>" in pred,
        "contains_end": "<|action_end|>" in pred,
        "well_formed": bool(matches and matches[0].strip()),
        "has_action_sep": "<|action_sep|>" in body,
        "parses_xyz": pm is not None,
        "pred_mouse": pm,
        "gt_mouse": gm,
    }
    if pm and gm:
        d["pred_mag"] = abs(pm[0]) + abs(pm[1])
        d["gt_mag"] = abs(gm[0]) + abs(gm[1])
        d["gt_moved_pred_still"] = (d["gt_mag"] > 10 and d["pred_mag"] == 0)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--val_file", type=Path,
                    default=Path("/fsx/home/zhiyuan/yuanshen_processed/_merged/val.jsonl"))
    ap.add_argument("--num_traj", type=int, default=12, help="评测多少条轨迹")
    ap.add_argument("--max_turns", type=int, default=30, help="每条轨迹评测前多少个 turn")
    ap.add_argument("--gpu", type=str, default="0")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--shard_idx", type=int, default=0, help="本 worker 处理第几片")
    ap.add_argument("--num_shards", type=int, default=1, help="总片数(=并行 GPU 数)")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")

    import random, torch
    from swift import TransformersEngine, RequestConfig, InferRequest

    rng = random.Random(args.seed)
    samples = [json.loads(l) for l in open(args.val_file, encoding="utf-8") if l.strip()]
    rng.shuffle(samples)
    samples = samples[: args.num_traj]
    # 分片: 每个 worker 用全局索引取自己那一份(保留全局 traj 编号便于合并)
    indexed = list(enumerate(samples))
    if args.num_shards > 1:
        indexed = [(gi, s) for (gi, s) in indexed if gi % args.num_shards == args.shard_idx]
    print(f"[eval] val={args.val_file}  本片 {len(indexed)} 条(shard {args.shard_idx}/{args.num_shards}), 每条前 {args.max_turns} turn")
    print(f"[eval] loading {args.checkpoint} on GPU {args.gpu} ...")
    t0 = time.time()
    engine = TransformersEngine(str(args.checkpoint), torch_dtype=torch.bfloat16)
    print(f"[eval] loaded in {time.time()-t0:.1f}s")
    rc = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature,
                       top_p=1.0 if args.temperature == 0 else 0.9, stream=False)

    out_path = args.out or (args.checkpoint.parent / f"eval_actions_{args.checkpoint.name}.jsonl")
    agg = {"n": 0, "errors": 0, "well_formed": 0, "parses_xyz": 0, "has_action_sep": 0,
           "pred_mag_sum": 0.0, "gt_mag_sum": 0.0, "gt_moved": 0, "gt_moved_pred_still": 0,
           "pred_still": 0}
    fout = open(out_path, "w", encoding="utf-8")

    for ti, sample in indexed:
        msgs = sample["messages"]
        images = sample.get("images", [])
        # user turn 在 msgs 里的位置；第 k 个 user 对应第 k 张图
        ua = [i for i in range(len(msgs) - 1)
              if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"]
        n_img_used = 0
        for turn_idx, ui in enumerate(ua[: args.max_turns]):
            ctx = msgs[: ui + 1]
            n_img_used = sum(1 for m in ctx if m["role"] == "user")
            imgs = images[:n_img_used]
            gt = msgs[ui + 1]["content"]
            try:
                resp = engine.infer([InferRequest(messages=ctx, images=imgs)], request_config=rc)
                pred = resp[0].choices[0].message.content
            except Exception as e:
                agg["errors"] += 1
                print(f"[eval] traj{ti} turn{turn_idx} ERROR {type(e).__name__}: {e}")
                continue
            met = analyze(pred, gt)
            agg["n"] += 1
            for k in ("well_formed", "parses_xyz", "has_action_sep"):
                agg[k] += int(met[k])
            print(f"[eval] traj{ti} turn{turn_idx} imgs={len(imgs)} "
                  f"GT={met.get('gt_mouse')} PRED={met.get('pred_mouse')}", flush=True)
            if "pred_mag" in met:
                agg["pred_mag_sum"] += met["pred_mag"]
                agg["gt_mag_sum"] += met["gt_mag"]
                if met["pred_mag"] == 0:
                    agg["pred_still"] += 1
                if met["gt_mag"] > 10:
                    agg["gt_moved"] += 1
                    if met["pred_mag"] == 0:
                        agg["gt_moved_pred_still"] += 1
            fout.write(json.dumps({
                "traj": ti, "turn": turn_idx, "n_images": len(imgs),
                "full_input": [{"role": m["role"], "content": m["content"]} for m in ctx],
                "images": imgs,
                "gt": gt, "pred": pred, "metrics": met,
            }, ensure_ascii=False) + "\n")
            fout.flush()
        print(f"[eval] traj global#{ti} done ({turn_idx+1} turns)", flush=True)

    fout.close()
    n = max(agg["n"], 1)
    gtm = max(agg["gt_moved"], 1)
    print("\n=================== ACTION EVAL SUMMARY ===================")
    print(f"  ckpt            : {args.checkpoint}")
    print(f"  inferred        : {agg['n']} (errors {agg['errors']})")
    print(f"  well_formed     : {agg['well_formed']}/{n} ({100*agg['well_formed']/n:.1f}%)")
    print(f"  parses_xyz      : {agg['parses_xyz']}/{n} ({100*agg['parses_xyz']/n:.1f}%)")
    print(f"  has_action_sep  : {agg['has_action_sep']}/{n} ({100*agg['has_action_sep']/n:.1f}%)")
    print(f"  --- 幅度(关键) ---")
    print(f"  pred 平均|dx|+|dy| : {agg['pred_mag_sum']/n:.1f}")
    print(f"  GT   平均|dx|+|dy| : {agg['gt_mag_sum']/n:.1f}")
    print(f"  pred/GT 幅度比     : {agg['pred_mag_sum']/max(agg['gt_mag_sum'],1):.3f}  (<1 = 模型动得比GT小)")
    print(f"  pred 完全不动(0 0)占比 : {100*agg['pred_still']/n:.1f}%")
    print(f"  GT动了但pred不动   : {agg['gt_moved_pred_still']}/{gtm} ({100*agg['gt_moved_pred_still']/gtm:.1f}%)  <- '动作不动'失败率")
    print(f"  完整结果 jsonl   : {out_path}")
    print("==========================================================")


if __name__ == "__main__":
    sys.exit(main())
