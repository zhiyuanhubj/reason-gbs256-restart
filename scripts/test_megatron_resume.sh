#!/usr/bin/env bash
# End-to-end checkpoint test: train one step, stop, then resume for one more.
# Run this inside an allocated 8xH200 node (for example through srun --overlap).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
if ! command -v megatron >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /fsx/home/zhiyuan/miniconda3/etc/profile.d/conda.sh
    conda activate megatron-sft
fi
SMOKE_ROOT="${SMOKE_ROOT:-/opt/dlami/nvme/${USER}/genshin-training/resume-smoke-${SLURM_JOB_ID:-local}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
MODEL_PATH="${MODEL_PATH:-/fsx/home/zhiyuan/nfs/models/Qwen3.5-9B}"
DATA_FILE="${SMOKE_ROOT}/train.jsonl"
OUTPUT_DIR="${SMOKE_ROOT}/output"
RESOLVER="${PROJECT_ROOT}/scripts/resolve_mcore_checkpoint.py"
START_CHECKPOINT="${START_CHECKPOINT:-}"
if [[ -n "$START_CHECKPOINT" ]]; then
    START_CHECKPOINT="$(
        python "$RESOLVER" --checkpoint "$START_CHECKPOINT" \
            --require-full-state --field path
    )"
    OUTPUT_DIR="$(dirname "$START_CHECKPOINT")"
fi

[[ -d "$MODEL_PATH" ]] || { echo "missing model: $MODEL_PATH" >&2; exit 1; }
mkdir -p "$SMOKE_ROOT"

python - "$DATA_FILE" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "w", encoding="utf-8") as handle:
    for index in range(256):
        row = {
            "messages": [
                {"role": "user", "content": f"Return the integer after {index}."},
                {"role": "assistant", "content": str(index + 1)},
            ]
        }
        handle.write(json.dumps(row) + "\n")
PY

export PROJECT_ROOT MODEL_PATH DATA_FILE OUTPUT_DIR
export TRAIN_FILE="$DATA_FILE" VAL_FILE="$DATA_FILE"
export GAME_NAME=resume_smoke
export GPUS=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 TP=4 PP=1
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29671}"
export MICRO_BS=1 GLOBAL_BS=2
export MAX_LEN=512 IMG_MAX_TOK=256
export PACKING="${PACKING_SMOKE:-false}" PACKING_LENGTH=512 PACKING_STRATEGY=binpack
export STREAMING=true DATASET_SHUFFLE=true
export TRAIN_DATALOADER_SHUFFLE=false SHUFFLE_BUFFER_SIZE=64
export ADD_GENSHIN_SPECIAL_TOKENS=false ENABLE_CHANNEL_LOSS=false
export CHANNEL_COLUMNS_JSON="" INTERLEAVE_PROB="" PASS_AWARE_MIXTURE=false
export MTP_LAYERS=0 RECOMPUTE=none
export CROSS_ENTROPY_LOSS_FUSION=true CROSS_ENTROPY_FUSION_IMPL=native
export TORCHDYNAMO_DISABLE=1
export EVAL_ITERS=0 EVAL_STEPS=0 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=2
export LOGGING_STEPS=1 NO_SAVE_OPTIM=false NO_SAVE_RNG=false
export NO_LOAD_OPTIM=false NO_LOAD_RNG=false
export SAVE_SAFETENSORS=false ADD_VERSION=false
export WANDB_MODE=disabled WANDB_PROJECT=resume-smoke

if [[ -z "$START_CHECKPOINT" ]]; then
    echo "[resume-smoke] phase 1: create a complete checkpoint at iteration 1"
    export RUN_NAME=resume_smoke_initial TRAIN_ITERS=1
    unset RESUME_FROM
    export AUTO_RESUME=false
    bash "${PROJECT_ROOT}/scripts/41_train_bc_full_megatron.sh"
    FIRST="$(
        python "$RESOLVER" --latest-under "$OUTPUT_DIR" \
            --require-full-state --field path
    )"
else
    FIRST="$START_CHECKPOINT"
    echo "[resume-smoke] reusing complete checkpoint: $FIRST"
fi
[[ "$(python "$RESOLVER" --checkpoint "$FIRST" --field iteration)" == "1" ]]
[[ "$(python "$RESOLVER" --checkpoint "$FIRST" --field consumed_samples)" == "2" ]]
echo "[resume-smoke] complete checkpoint: $FIRST"

echo "[resume-smoke] phase 2: restore optimizer/RNG/iteration and continue to iteration 2"
export RUN_NAME=resume_smoke_resumed TRAIN_ITERS=2 RESUME_FROM="$FIRST"
bash "${PROJECT_ROOT}/scripts/41_train_bc_full_megatron.sh"

FINAL="$(
    python "$RESOLVER" --latest-under "$OUTPUT_DIR" \
        --require-full-state --field path
)"
[[ "$(python "$RESOLVER" --checkpoint "$FINAL" --field iteration)" == "2" ]]
[[ "$(python "$RESOLVER" --checkpoint "$FINAL" --field consumed_samples)" == "4" ]]

echo "[resume-smoke] PASS: checkpoint-1 resumed and produced checkpoint-2"
echo "[resume-smoke] final checkpoint: $FINAL"
