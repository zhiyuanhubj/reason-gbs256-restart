#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""根据 logging.jsonl 画训练曲线(train loss / eval loss / mtp loss / grad_norm)。"""
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run_glob):
    runs = sorted(glob.glob(run_glob))
    lj = sorted(glob.glob(os.path.join(runs[-1], "logging.jsonl")))[-1]
    rows = [json.loads(l) for l in open(lj) if l.strip()]
    return lj, rows


def step_of(r):
    it = r.get("iteration")
    if isinstance(it, str) and "/" in it:
        return int(it.split("/")[0])
    return r.get("global_step") or r.get("step")


def main():
    run_glob = sys.argv[1] if len(sys.argv) > 1 else \
        "/fsx/home/zhiyuan/game/extracted/training-main/outputs/sft_*/v0-*"
    out = sys.argv[2] if len(sys.argv) > 2 else \
        "/fsx/home/zhiyuan/game/extracted/training-main/outputs/loss_curve.png"
    lj, rows = load(run_glob)

    tr_s, tr_l, mtp_s, mtp_l, gn_s, gn_l = [], [], [], [], [], []
    ev_s, ev_l = [], []
    for r in rows:
        s = step_of(r)
        if s is None:
            continue
        if r.get("eval_loss") is not None:
            ev_s.append(s); ev_l.append(r["eval_loss"])
        if r.get("loss") is not None:
            tr_s.append(s); tr_l.append(r["loss"])
        if r.get("mtp_1_loss") is not None:
            mtp_s.append(s); mtp_l.append(r["mtp_1_loss"])
        if r.get("grad_norm") is not None:
            gn_s.append(s); gn_l.append(r["grad_norm"])

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"SFT training curves  (job 11772, {len(tr_s)} steps logged)", fontsize=14)

    # 1) train loss + eval loss
    a = ax[0][0]
    a.plot(tr_s, tr_l, lw=0.7, alpha=0.6, color="tab:blue", label="train loss")
    # 移动平均
    if len(tr_l) > 20:
        w = 20
        ma = [sum(tr_l[max(0,i-w):i+1]) / len(tr_l[max(0,i-w):i+1]) for i in range(len(tr_l))]
        a.plot(tr_s, ma, lw=1.8, color="tab:blue", label="train loss (MA20)")
    if ev_s:
        a.plot(ev_s, ev_l, "o-", color="tab:red", ms=5, label="eval loss")
    a.set_xlabel("step"); a.set_ylabel("loss"); a.set_title("Loss (train vs eval)")
    a.legend(); a.grid(alpha=0.3)

    # 2) train loss log scale (看早期下降)
    a = ax[0][1]
    a.plot(tr_s, tr_l, lw=0.7, color="tab:blue")
    a.set_yscale("log"); a.set_xlabel("step"); a.set_ylabel("loss (log)")
    a.set_title("Train loss (log scale)"); a.grid(alpha=0.3, which="both")

    # 3) mtp loss
    a = ax[1][0]
    if mtp_s:
        a.plot(mtp_s, mtp_l, lw=0.7, color="tab:green", label="mtp_1_loss")
    a.set_xlabel("step"); a.set_ylabel("loss"); a.set_title("MTP head loss")
    a.legend(); a.grid(alpha=0.3)

    # 4) grad norm
    a = ax[1][1]
    if gn_s:
        a.plot(gn_s, gn_l, lw=0.6, color="tab:orange")
    a.set_xlabel("step"); a.set_ylabel("grad norm"); a.set_title("Grad norm")
    a.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out, dpi=110)
    print(f"[plot] src: {lj}")
    print(f"[plot] steps: train={len(tr_s)}, eval={len(ev_s)}")
    if tr_l:
        print(f"[plot] train loss: first={tr_l[0]:.3f}  last={tr_l[-1]:.3f}  min={min(tr_l):.3f}")
    if ev_l:
        print(f"[plot] eval  loss: {[round(x,3) for x in ev_l]}")
    print(f"[plot] saved -> {out}")


if __name__ == "__main__":
    main()
