#!/bin/bash
# 在一个 8 卡节点上并行跑完整 val: 把 128 条轨迹按 GPU 分 8 片, 每卡一片。
# 跑两个温度(0.0 贪心 / 1.0 采样)。每片写各自 jsonl, 最后由外部合并。
set -uo pipefail
source /fsx/home/zhiyuan/miniconda3/etc/profile.d/conda.sh
conda activate megatron-sft
source /fsx/home/zhiyuan/game/extracted/training-main/env.sh >/dev/null 2>&1
cd /fsx/home/zhiyuan/game/extracted/training-main

CKPT="${EVAL_CKPT:-$(ls -dt outputs/sft_*j11772/v0-*/checkpoint-* | head -1)}"
NUM_TRAJ="${EVAL_NUM_TRAJ:-128}"
MAX_TURNS="${EVAL_MAX_TURNS:-30}"
VAL=/fsx/home/zhiyuan/yuanshen_processed/_merged/val.jsonl
CKPT_NAME=$(basename "$CKPT")
OUTDIR="outputs/val_full_${CKPT_NAME}"
mkdir -p "$OUTDIR"
echo "[full_eval] ckpt=$CKPT num_traj=$NUM_TRAJ max_turns=$MAX_TURNS out=$OUTDIR host=$(hostname)"

for TEMP in 0.0 1.0; do
    echo "[full_eval] ===== temperature=$TEMP ====="
    pids=()
    for g in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$g python scripts/42b_eval_actions.py "$CKPT" \
            --val_file "$VAL" --num_traj "$NUM_TRAJ" --max_turns "$MAX_TURNS" \
            --gpu 0 --temperature "$TEMP" \
            --shard_idx "$g" --num_shards 8 \
            --out "$OUTDIR/T${TEMP}_shard${g}.jsonl" \
            > "$OUTDIR/log_T${TEMP}_shard${g}.log" 2>&1 &
        pids+=($!)
    done
    echo "[full_eval] 8 workers started for T=$TEMP: ${pids[*]}"
    wait "${pids[@]}"
    cat "$OUTDIR"/T${TEMP}_shard*.jsonl > "$OUTDIR/T${TEMP}_all.jsonl"
    echo "[full_eval] T=$TEMP 完成, 合并 $(wc -l < "$OUTDIR/T${TEMP}_all.jsonl") 条 -> $OUTDIR/T${TEMP}_all.jsonl"
done
echo "[full_eval] ALL DONE -> $OUTDIR"
