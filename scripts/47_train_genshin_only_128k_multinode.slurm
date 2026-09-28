#!/bin/bash
# Genshin-only 3-pass, 128K full-parameter SFT on the current Isambard
# allocation: 2 nodes x 4 GH200 GPUs (8 GPUs total).
#
# Submit a fresh allocation:
#   sbatch scripts/47_train_genshin_only_128k_multinode.slurm
#
# Attach the whole training step to an already running allocation from a login
# node (do not wrap this in another one-node srun):
#   EXISTING_JOB_ID=6158395 GLOBAL_BS=8 \
#     bash scripts/47_train_genshin_only_128k_multinode.slurm
#
# The lightweight dataset view must first exist at:
#   /projects/b6db/opensima/genshin-training/genshin-only/
# It is created by scripts/build_genshin_only_dataset.py and contains symlinks;
# the 2.6 TB source dataset is not copied to node-local storage.
#SBATCH --account=brics.b6db
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=1-00:00:00
#SBATCH --job-name=genshin-only-128k
#SBATCH --output=logs/genshin_only_128k_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ASSET_ROOT="${ASSET_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"

# When invoked from a login node, discover the complete allocation instead of
# first entering a one-node srun step.  A nested two-node srun cannot expand
# beyond that outer step and either allocates one node or fails immediately.
if [[ -n "${EXISTING_JOB_ID:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" != "$EXISTING_JOB_ID" ]]; then
        echo "[ERR] already inside Slurm job $SLURM_JOB_ID; refusing to attach job $EXISTING_JOB_ID"
        exit 1
    fi
    _JOB_INFO="$(scontrol show job "$EXISTING_JOB_ID" -o 2>/dev/null)" || {
        echo "[ERR] cannot query existing Slurm job: $EXISTING_JOB_ID"
        exit 1
    }
    _job_field() {
        local _key="$1" _field
        for _field in $_JOB_INFO; do
            if [[ "$_field" == "${_key}="* ]]; then
                printf '%s\n' "${_field#*=}"
                return 0
            fi
        done
        return 1
    }
    _JOB_STATE="$(_job_field JobState)"
    [[ "$_JOB_STATE" == "RUNNING" ]] || {
        echo "[ERR] existing Slurm job $EXISTING_JOB_ID is $_JOB_STATE, not RUNNING"
        exit 1
    }
    export SLURM_JOB_ID="$EXISTING_JOB_ID"
    export SLURM_JOB_NODELIST="$(_job_field NodeList)"
    export SLURM_NNODES="$(_job_field NumNodes)"
    export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$(_job_field CPUs/Task)}"
    export NNODES="${NNODES:-$SLURM_NNODES}"
    export ALLOC_NODELIST="${ALLOC_NODELIST:-$SLURM_JOB_NODELIST}"
    export SRUN_JOB_ID_OVERRIDE="$EXISTING_JOB_ID"
    echo "[attach] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST nnodes=$SLURM_NNODES cpus/task=$SLURM_CPUS_PER_TASK"
elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[ERR] not inside a Slurm allocation; use sbatch or set EXISTING_JOB_ID=<running-job-id>"
    exit 1
fi

# Shared model/data/output paths on this cluster.
export NFS_DIR="${NFS_DIR:-${ASSET_ROOT}}"
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-${ASSET_ROOT}/genshin-training}"
export MIXTURE_YAML="${MIXTURE_YAML:-${LOCAL_DATA_ROOT}/genshin-only/genshin-only-128k.local.yaml}"
export MODEL_PATH="${MODEL_PATH:-${ASSET_ROOT}/models/Qwen3.5-9B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ASSET_ROOT}/outputs/${GAME_NAME:-genshin3_qwen35_128k}_${SLURM_JOB_ID:-local}}"

[[ -f "$MIXTURE_YAML" ]] || {
    echo "[ERR] Genshin training view is missing: $MIXTURE_YAML"
    echo "[ERR] Run scripts/build_genshin_only_dataset.py first (see the header of this file)."
    exit 1
}
[[ -f "$MODEL_PATH/config.json" ]] || {
    echo "[ERR] Qwen3.5-9B model is missing: $MODEL_PATH"
    exit 1
}

# Isambard software stack and shared conda environment.
export CONDA_BASE="${CONDA_BASE:-/home/b6db/hongnanvideo1.b6db/.miniconda3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-megatron-sft}"
export PYTHON_BIN="${PYTHON_BIN:-/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/bin/python}"
export ENV_SETUP_MODULES="${ENV_SETUP_MODULES:-cuda/12.6 brics/nccl/2.26.6-1 brics/aws-ofi-nccl/1.8.1}"
export CUDA_HOME="${CUDA_HOME:-/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/targets/sbsa-linux}"
# TileLang JITs the Gated DeltaNet backward kernel with the system CUDA 12.6
# nvcc.  Conda puts GCC 14 first on PATH, while CUDA 12.6 supports at most
# GCC 13, so pin the host compiler explicitly on every Slurm task.
export CUDA_HOST_CC="${CUDA_HOST_CC:-/usr/bin/gcc-13}"
export CUDA_HOST_CXX="${CUDA_HOST_CXX:-/usr/bin/g++-13}"
export CC="$CUDA_HOST_CC"
export CXX="$CUDA_HOST_CXX"

# Two nodes x four GPUs. TP stays inside a node; DP spans the two nodes.
export GPUS="${GPUS:-0,1,2,3}"
export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export GLOBAL_BS="${GLOBAL_BS:-8}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-128}"
export PACKING_NUM_PROC="${PACKING_NUM_PROC:-1}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
# Avoid the missing Isambard /local/user/<uid> TMPDIR during TileLang/NVCC JIT.
# 43_train_inner.sh creates this job-scoped directory on each node.
export JOB_LOCAL_TMPDIR="${JOB_LOCAL_TMPDIR:-/dev/shm/${USER:-u}_train_tmp_${SLURM_JOB_ID:-local}}"
export ASYNC_PACKING_PREFETCH_DEPTH="${ASYNC_PACKING_PREFETCH_DEPTH:-2}"
export IMG_MAX_TOK="${IMG_MAX_TOK:-16384}"

export GAME_NAME="${GAME_NAME:-genshin3_qwen35_128k}"
export PASS_AWARE_MIXTURE="${PASS_AWARE_MIXTURE:-true}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec bash "${SCRIPT_DIR}/46_train_genshin_llava_onevision_128k_multinode.slurm"
