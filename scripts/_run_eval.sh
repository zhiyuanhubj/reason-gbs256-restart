#!/bin/bash
set -uo pipefail
source /fsx/home/zhiyuan/miniconda3/etc/profile.d/conda.sh
conda activate megatron-sft
source /fsx/home/zhiyuan/game/extracted/training-main/env.sh >/dev/null 2>&1
cd /fsx/home/zhiyuan/game/extracted/training-main
CKPT="${EVAL_CKPT:-$(ls -dt outputs/sft_*j11772/v0-*/checkpoint-* | head -1)}"
EVAL_TEMP="${EVAL_TEMP:-0.0}"
EVAL_NUM_TRAJ="${EVAL_NUM_TRAJ:-12}"
EVAL_MAX_TURNS="${EVAL_MAX_TURNS:-30}"
echo "[run_eval] ckpt=$CKPT temp=$EVAL_TEMP num_traj=$EVAL_NUM_TRAJ max_turns=$EVAL_MAX_TURNS"
python scripts/42b_eval_actions.py "$CKPT" \
    --val_file /fsx/home/zhiyuan/yuanshen_processed/_merged/val.jsonl \
    --num_traj "$EVAL_NUM_TRAJ" --max_turns "$EVAL_MAX_TURNS" \
    --gpu 0 --temperature "$EVAL_TEMP" \
    --out "outputs/eval_actions_$(basename "$CKPT")_T${EVAL_TEMP}.jsonl"
