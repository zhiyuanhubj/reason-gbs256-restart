#!/bin/bash
# Fresh Qwen3.5-9B base-model run of the Cyberpunk 2077 + LLaVA-OneVision 128K
# recipe on 128 GH200 GPUs. This is the non-resume, 128-GPU counterpart to
# 46_1; dataset, optimizer, packing, and checkpoint settings come from 46.
#
# Submit from the repository root:
#   sbatch scripts/46_2_train_2077_llava_onevision_128k_128gpu_gbs128_seed42.slurm
#SBATCH --account=brics.b6db
#SBATCH --partition=workq
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=1-00:00:00
#SBATCH --job-name=2077-q35-128k-128g-pf2
#SBATCH --output=logs/2077_q35_128k_128g_b128_s42_pf2_%j.log

set -euo pipefail

# Slurm executes a spooled copy, so resolve 46 from the repository directory
# where sbatch was invoked.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BASE_PRESET="${SUBMIT_DIR}/scripts/46_train_genshin_llava_onevision_128k_multinode.slurm"
if [[ ! -f "$BASE_PRESET" ]]; then
    echo "[ERR] base preset not found: $BASE_PRESET" >&2
    exit 1
fi

# 32 nodes x 4 GPUs, TP=4, PP=1, CP=1 -> DP=32. With micro batch 1 and
# global batch 128, each optimizer step uses four microbatches per DP rank.
export NNODES="${NNODES:-32}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-128}"

# Start from the untouched Qwen3.5-9B base model. Resume is deliberately
# disabled even if the submission environment contains resume variables.
export MODEL_PATH="/projects/b6db/opensima/models/Qwen3.5-9B"
export MIXTURE_YAML="${SUBMIT_DIR}/data_configs/2077_llava_onevision_128K.yaml"
export RESUME_FROM=""
export AUTO_RESUME="false"

# Unlike the step-700 resume preset, a fresh base-model run keeps warmup.
export WARMUP_ITERS="${WARMUP_ITERS:-100}"
export SEED="${SEED:-42}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-128}"
export PACKING_NUM_PROC="${PACKING_NUM_PROC:-1}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
# The cluster default TMPDIR can point at a missing /local/user/<uid> path.
# 43_train_inner.sh creates this job-scoped directory independently per node.
export JOB_LOCAL_TMPDIR="${JOB_LOCAL_TMPDIR:-/dev/shm/${USER:-u}_train_tmp_${SLURM_JOB_ID:-local}}"
# Phase-2 data pipeline: prepare the next two packed microbatches while the
# current optimizer step is on GPU.  Set to 0 at submission time to disable.
export ASYNC_PACKING_PREFETCH_DEPTH="${ASYNC_PACKING_PREFETCH_DEPTH:-2}"
export GAME_NAME="${GAME_NAME:-2077_llava_q35_128g_pf2}"
export RUN_NAME="${RUN_NAME:-2077_q35_128k_128g_b128_s42_pf2_j${SLURM_JOB_ID:-local}}"
if (( ${#RUN_NAME} > 64 )); then
    echo "[ERR] RUN_NAME/WANDB_NAME exceeds 64 characters (${#RUN_NAME}): $RUN_NAME" >&2
    exit 1
fi

exec bash "$BASE_PRESET"
