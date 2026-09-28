#!/bin/bash
# Lossless 128-GPU continuation of the Cyberpunk 2077 + LLaVA-OneVision run
# from its full-state MCore checkpoint-800.  Later checkpoints are written to
# a separate run directory so this branch cannot collide with the source job.
#
# Submit from the repository root:
#   sbatch scripts/46_4_resume_2077_llava_onevision_128k_128gpu_gbs128_from400.slurm
#SBATCH --account=brics.b6db
#SBATCH --partition=workq
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=1-00:00:00
#SBATCH --job-name=2077-r800-128g-pf2
#SBATCH --output=logs/2077_q35_128k_128g_b128_s42_pf2_resume800_%j.log

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BASE_PRESET="${SUBMIT_DIR}/scripts/46_train_genshin_llava_onevision_128k_multinode.slurm"
SOURCE_RUN="/projects/b6db/opensima/outputs/2077_q35_128k_128g_b128_s42_pf2_20260901_084055"
SOURCE_CHECKPOINT="${SOURCE_RUN}/checkpoint-800"

[[ -f "$BASE_PRESET" ]] || {
    echo "[ERR] base preset not found: $BASE_PRESET" >&2
    exit 1
}
[[ -d "$SOURCE_CHECKPOINT" ]] || {
    echo "[ERR] resume checkpoint not found: $SOURCE_CHECKPOINT" >&2
    exit 1
}
[[ -f "${SOURCE_RUN}/mixture_plan.json" ]] || {
    echo "[ERR] source mixture plan not found: ${SOURCE_RUN}/mixture_plan.json" >&2
    exit 1
}

# Match the source checkpoint's topology and batch construction exactly:
# 128 GPUs / TP=4 / PP=1 / CP=1 => DP=32, with grad accumulation 4.
export NNODES=32
export GPUS_PER_NODE=4
export NPROC_PER_NODE=4
export TP=4
export PP=1
export CP=1
export MICRO_BS=1
export GLOBAL_BS=128

export MODEL_PATH="/projects/b6db/opensima/models/Qwen3.5-9B"
export MIXTURE_YAML="${SUBMIT_DIR}/data_configs/2077_llava_onevision_128K.yaml"
export MIXTURE_PLAN="${SOURCE_RUN}/mixture_plan.json"

# Full-state MCore resume: restore iteration, optimizer, RNG, and the indexed
# streaming dataloader cursor.  Do not auto-select a newer source checkpoint.
export RESUME_FROM="$SOURCE_CHECKPOINT"
export AUTO_RESUME=false
export RESUME_REQUIRE_FULL_STATE=true
export FINETUNE=false
export NO_LOAD_OPTIM=false
export NO_LOAD_RNG=false
export NO_SAVE_OPTIM=false
export NO_SAVE_RNG=false

# Keep the resumed branch isolated from the still-running source job.
export GAME_NAME="2077_llava_q35_128g_pf2_resume800"
export RUN_NAME="2077_q35_128k_128g_b128_s42_pf2_resume800_j${SLURM_JOB_ID}"
export OUTPUT_DIR="/projects/b6db/opensima/outputs/${RUN_NAME}"

# These values match the source run.  The loaded optimizer/scheduler state
# retains the already completed 100-step warmup.
export WARMUP_ITERS=100
export SEED=42
export SHUFFLE_BUFFER_SIZE=128
export PACKING_INTERVAL=1
export PACKING_NUM_PROC=1
export DATASET_NUM_PROC=1
export ASYNC_PACKING_PREFETCH_DEPTH=2
export JOB_LOCAL_TMPDIR="${JOB_LOCAL_TMPDIR:-/dev/shm/${USER:-u}_train_tmp_${SLURM_JOB_ID}}"

if (( ${#RUN_NAME} > 64 )); then
    echo "[ERR] RUN_NAME/WANDB_NAME exceeds 64 characters (${#RUN_NAME}): $RUN_NAME" >&2
    exit 1
fi

exec bash "$BASE_PRESET"
