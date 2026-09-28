#!/bin/bash
# Genshin instruction + reasoning, five exact passes per dataset, 128K.
# 128 GH200 GPUs on 32 nodes x 4 GPUs, global batch size 64.
# Learning rate warms up to 1e-5, then cosine-decays to 1e-7.
#
# Submit:
#   sbatch scripts/49_train_gensin_instruct_reasoning_5pass_128k_128gpu_gbs32_cosine1e7_1d.slurm
#
# This preset only overrides allocation/parallelism/schedule settings and reuses
# the dataset validation and launch path from the 48 preset.
#SBATCH --account=brics.b6db
#SBATCH --partition=workq
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=1-00:00:00
#SBATCH --job-name=wh_genshin_llava700_ins_reasoning_b64_5epoch_decay
#SBATCH --output=logs/wh_genshin_llava700_ins_reasoning_b64_5epoch_decay_%j.log

set -euo pipefail

# Slurm executes a spooled copy of this file, so BASH_SOURCE points under
# /var/spool/slurmd rather than back to this repository.  Resolve the reused
# preset from the directory where sbatch was invoked instead.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BASE_PRESET="${SUBMIT_DIR}/scripts/48_train_gensin_instruct_reasoning_5pass_128k_multinode.slurm"
if [[ ! -f "${BASE_PRESET}" ]]; then
    echo "[ERR] base preset not found: ${BASE_PRESET}" >&2
    exit 1
fi

# 32 nodes x 4 GPUs, TP=4, PP=1, CP=1 -> DP=32.  With micro batch 1 and
# global batch 64, each optimizer step uses two microbatches per DP rank.
export GPUS="${GPUS:-0,1,2,3}"
export NNODES="${NNODES:-32}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-64}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-128}"
export PACKING_NUM_PROC="${PACKING_NUM_PROC:-1}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
# Avoid the missing Isambard /local/user/<uid> TMPDIR during TileLang/NVCC JIT.
# 43_train_inner.sh creates this job-scoped directory on each node.
export JOB_LOCAL_TMPDIR="${JOB_LOCAL_TMPDIR:-/dev/shm/${USER:-u}_train_tmp_${SLURM_JOB_ID:-local}}"
export ASYNC_PACKING_PREFETCH_DEPTH="${ASYNC_PACKING_PREFETCH_DEPTH:-2}"
export IMG_MAX_TOK="${IMG_MAX_TOK:-16384}"
export MTP_LAYERS="${MTP_LAYERS:-1}"

# Keep distributed-optimizer collectives synchronous.  Parameter-gather
# overlap is incompatible with this multimodal model's parameter-registration
# versus forward order, and the combined overlap configuration regressed
# throughput substantially in the 32-node run.
export OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-false}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-false}"

# Keep the original peak LRs and 30-step warmup, but enable cosine decay.
export LR="${LR:-1e-5}"
export VIT_LR="${VIT_LR:-3e-6}"
export ALIGNER_LR="${ALIGNER_LR:-1e-5}"
export WARMUP_ITERS="${WARMUP_ITERS:-30}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
export MIN_LR="${MIN_LR:-1e-7}"

# Start a fresh instruction/reasoning run from the HF export of the complete
# Genshin+LLaVA checkpoint-700 produced by job 6210204.  MODEL_PATH is used as
# initialization only: optimizer/RNG/iteration and the old data cursor are not
# restored as they would be with RESUME_FROM.
export MODEL_PATH="${MODEL_PATH:-/projects/b6db/opensima/outputs/genshin_llava019_128k_128gpu_gbs128_700step_resume_s3407_tp4_full_20260831_092432_j6210204/checkpoint-700-hf-mtp}"
export GAME_NAME="${GAME_NAME:-gi5p_llava700_resume_700_gbs64}"
export RUN_NAME="${RUN_NAME:-${GAME_NAME}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}}"

exec bash "${BASE_PRESET}"
