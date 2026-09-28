#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""汇总 42b 产出的逐条 jsonl(可多片合并)的动作指标。用法: 42c_summarize.py a.jsonl [b.jsonl ...]"""
import json, sys

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(rows)
    wf = xyz = sep = 0
    pmag = gmag = 0.0
    pstill = gmoved = gmoved_pstill = 0
    mag_pairs = 0
    for r in rows:
        m = r["metrics"]
        wf += int(m.get("well_formed", False))
        xyz += int(m.get("parses_xyz", False))
        sep += int(m.get("has_action_sep", False))
        if "pred_mag" in m and "gt_mag" in m:
            mag_pairs += 1
            pmag += m["pred_mag"]; gmag += m["gt_mag"]
            if m["pred_mag"] == 0:
                pstill += 1
            if m["gt_mag"] > 10:
                gmoved += 1
                if m["pred_mag"] == 0:
                    gmoved_pstill += 1
    nn = max(n, 1); mp = max(mag_pairs, 1); gm = max(gmoved, 1)
    print(f"\n==== {path} ====")
    print(f"  样本数: {n}")
    print(f"  well_formed: {wf}/{n} ({100*wf/nn:.1f}%)  parses_xyz: {100*xyz/nn:.1f}%  has_action_sep: {100*sep/nn:.1f}%")
    print(f"  pred 平均|dx|+|dy|: {pmag/mp:.1f}")
    print(f"  GT   平均|dx|+|dy|: {gmag/mp:.1f}")
    print(f"  pred/GT 幅度比: {pmag/max(gmag,1):.3f}")
    print(f"  pred 完全不动占比: {100*pstill/mp:.1f}%")
    print(f"  GT动了但pred不动: {gmoved_pstill}/{gm} ({100*gmoved_pstill/gm:.1f}%)")
