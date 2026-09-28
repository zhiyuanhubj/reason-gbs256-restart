#!/usr/bin/env bash
# Real-data smoke test: raw Genshin + canonical LLaVA, 128K packing, save/resume.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if ! command -v megatron >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /fsx/home/zhiyuan/miniconda3/etc/profile.d/conda.sh
    conda activate megatron-sft
fi

DATA_ROOT="${DATA_ROOT:-/opt/dlami/nvme/${USER}/genshin-training}"
GENSHIN_ROOT="${GENSHIN_ROOT:-${DATA_ROOT}/genshin-parquet}"
LLAVA_ROOT="${LLAVA_ROOT:-${DATA_ROOT}/llava-onevision}"
MODEL_PATH="${MODEL_PATH:-/fsx/home/zhiyuan/nfs/models/Qwen3.5-9B}"
SMOKE_ROOT="${SMOKE_ROOT:-${DATA_ROOT}/real-mixed-resume-${SLURM_JOB_ID:-local}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
OUTPUT_DIR="${SMOKE_ROOT}/output"
RESOLVER="${PROJECT_ROOT}/scripts/resolve_mcore_checkpoint.py"

GENSHIN_FILE="$(python - "$GENSHIN_ROOT" <<'PY'
import sys
from pathlib import Path
files = sorted(Path(sys.argv[1]).glob("*.parquet"))
if not files:
    raise SystemExit("no Genshin parquet files")
print(files[0])
PY
)"
LLAVA_FILE="${LLAVA_ROOT}/gsm8k/train-00000-of-00001.qwen35.parquet"
[[ -f "$LLAVA_FILE" ]] || { echo "missing completed LLaVA shard: $LLAVA_FILE" >&2; exit 1; }

export PROJECT_ROOT MODEL_PATH OUTPUT_DIR
export TRAIN_FILE="$GENSHIN_FILE" VAL_FILE="$LLAVA_FILE"
export DATASET_SPEC="${GENSHIN_FILE}#4 ${LLAVA_FILE}#64"
export INTERLEAVE_PROB="0.25 0.75" STOPPING_STRATEGY=all_exhausted_without_replacement
export CHANNEL_COLUMNS_JSON='{"conversations":"messages","data_source":"channel"}'
export ENABLE_CHANNEL_LOSS=false ADD_GENSHIN_SPECIAL_TOKENS=true
export GAME_NAME=real_genshin_llava_resume

export GPUS=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 TP=4 PP=1
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29681}"
export MICRO_BS=1 GLOBAL_BS=2
export MAX_LEN=131072 IMG_MAX_TOK="${SMOKE_IMG_MAX_TOK:-1024}"
export PACKING=true PACKING_LENGTH=131072 PACKING_INTERVAL=8
export PACKING_STRATEGY=binpack DATASET_NUM_PROC=4
export STREAMING=true DATASET_SHUFFLE=true SHUFFLE_BUFFER_SIZE=64
export TRAIN_DATALOADER_SHUFFLE=false PASS_AWARE_MIXTURE=false
export MTP_LAYERS="${SMOKE_MTP_LAYERS:-0}" RECOMPUTE="${SMOKE_RECOMPUTE:-full}" RECOMPUTE_NUM_LAYERS=1
export CROSS_ENTROPY_LOSS_FUSION=true CROSS_ENTROPY_FUSION_IMPL="${SMOKE_CE_IMPL:-te}"
export EVAL_ITERS=0 EVAL_STEPS=0 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=2
export LOGGING_STEPS=1 NO_SAVE_OPTIM=false NO_SAVE_RNG=false
export NO_LOAD_OPTIM=false NO_LOAD_RNG=false
export SAVE_SAFETENSORS=false ADD_VERSION=false
export DISABLE_TORCH_COMPILE=true
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_PROJECT=opensima

if [[ "${BASELINE_ONLY:-false}" == "true" ]]; then
    echo "[real-mixed-resume] uninterrupted baseline: iterations 1 and 2"
    export RUN_NAME=real_mixed_baseline TRAIN_ITERS=2 AUTO_RESUME=false
    unset RESUME_FROM
    bash "${PROJECT_ROOT}/scripts/41_train_bc_full_megatron.sh"
    BASELINE="$(
        python "$RESOLVER" --latest-under "$OUTPUT_DIR" \
            --require-full-state --field path
    )"
    [[ "$(python "$RESOLVER" --checkpoint "$BASELINE" --field iteration)" == "2" ]]
    echo "[real-mixed-resume] BASELINE PASS: $BASELINE"
    exit 0
fi

echo "[real-mixed-resume] phase 1: start the two-step schedule, then crash after checkpoint-1"
export RUN_NAME=real_mixed_initial TRAIN_ITERS=2 AUTO_RESUME=false
unset RESUME_FROM
setsid bash "${PROJECT_ROOT}/scripts/41_train_bc_full_megatron.sh" &
PHASE1_PID=$!

FIRST=""
for _ in $(seq 1 360); do
    if candidate="$(
        python "$RESOLVER" --latest-under "$OUTPUT_DIR" \
            --require-full-state --field path 2>/dev/null
    )" && [[ "$(python "$RESOLVER" --checkpoint "$candidate" --field iteration)" == "1" ]]; then
        FIRST="$candidate"
        break
    fi
    sleep 5
done
[[ -n "$FIRST" ]] || {
    echo "[real-mixed-resume] checkpoint-1 did not become complete" >&2
    kill -TERM -- "-$PHASE1_PID" 2>/dev/null || true
    exit 1
}

echo "[real-mixed-resume] checkpoint-1 complete; simulating process loss"
kill -TERM -- "-$PHASE1_PID" 2>/dev/null || true
for _ in $(seq 1 20); do
    kill -0 -- "-$PHASE1_PID" 2>/dev/null || break
    sleep 1
done
kill -KILL -- "-$PHASE1_PID" 2>/dev/null || true
wait "$PHASE1_PID" 2>/dev/null || true
# IterablePackingDataset encoding workers create their own process groups. A
# hard parent loss can orphan them while they still hold CUDA/UVM handles.
GPU_WORKERS="$(fuser /dev/nvidia-uvm 2>/dev/null || true)"
[[ -z "$GPU_WORKERS" ]] || kill -KILL $GPU_WORKERS 2>/dev/null || true
sleep 3

[[ "$(python "$RESOLVER" --checkpoint "$FIRST" --field iteration)" == "1" ]]
[[ "$(python "$RESOLVER" --checkpoint "$FIRST" --field consumed_samples)" == "2" ]]

echo "[real-mixed-resume] phase 2: new process resumes the next packed sequence"
export RUN_NAME=real_mixed_resumed TRAIN_ITERS=2 RESUME_FROM="$FIRST"
bash "${PROJECT_ROOT}/scripts/41_train_bc_full_megatron.sh"

FINAL="$(
    python "$RESOLVER" --latest-under "$OUTPUT_DIR" \
        --require-full-state --field path
)"
[[ "$(python "$RESOLVER" --checkpoint "$FINAL" --field iteration)" == "2" ]]
[[ "$(python "$RESOLVER" --checkpoint "$FINAL" --field consumed_samples)" == "4" ]]
echo "[real-mixed-resume] PASS: real packed Genshin+LLaVA stream resumed at iteration 2"
echo "[real-mixed-resume] final checkpoint: $FINAL"
