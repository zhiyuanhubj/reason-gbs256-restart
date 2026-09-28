#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""画前 N step 的训练曲线(loss / mtp / grad_norm)。用法: 10b_plot_first_n.py [N] [out.png]"""
import glob, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
out = sys.argv[2] if len(sys.argv) > 2 else \
    f"/fsx/home/zhiyuan/game/extracted/training-main/outputs/loss_first{N}.png"
run = sorted(glob.glob("/fsx/home/zhiyuan/game/extracted/training-main/outputs/sft_*/v0-*"))[-1]
lj = os.path.join(run, "logging.jsonl")
rows = [json.loads(l) for l in open(lj) if l.strip()]

def step(r):
    it = r.get("iteration")
    return int(it.split("/")[0]) if isinstance(it, str) and "/" in it else None

s, l, ms, ml, gs, gl = [], [], [], [], [], []
for r in rows:
    st = step(r)
    if st is None or st > N:
        continue
    if r.get("loss") is not None: s.append(st); l.append(r["loss"])
    if r.get("mtp_1_loss") is not None: ms.append(st); ml.append(r["mtp_1_loss"])
    if r.get("grad_norm") is not None: gs.append(st); gl.append(r["grad_norm"])

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"SFT first {N} steps (job 11772)", fontsize=14)
ax[0].plot(s, l, "o-", ms=3, lw=1, color="tab:blue", label="train loss")
ax[0].plot(ms, ml, "s-", ms=3, lw=1, color="tab:green", alpha=0.7, label="mtp_1_loss")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(s, l, "o-", ms=3, lw=1, color="tab:blue")
ax[1].set_yscale("log"); ax[1].set_xlabel("step"); ax[1].set_ylabel("loss (log)")
ax[1].set_title("Loss (log)"); ax[1].grid(alpha=0.3, which="both")
ax[2].plot(gs, gl, "o-", ms=3, lw=1, color="tab:orange")
ax[2].set_xlabel("step"); ax[2].set_ylabel("grad norm"); ax[2].set_title("Grad norm"); ax[2].grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(out, dpi=110)
print(f"[plot] first {N} steps, points={len(s)}")
if l: print(f"[plot] loss: step1={l[0]:.3f} step{s[-1]}={l[-1]:.3f} min={min(l):.3f}")
print(f"[plot] saved -> {out}")
