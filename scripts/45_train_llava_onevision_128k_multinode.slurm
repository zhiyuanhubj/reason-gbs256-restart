#!/bin/bash
# LLaVA-OneVision 128K full-parameter SFT, without the Lepton launcher.
#
# Required before submission:
#   export NFS_DIR=/path/whose/data/ contains
#     LLaVA-OneVision-1.5-Instruct-Data-qwen-format/
#     LLaVA-OneVision-1.5-Instruct-Data-qwen-format-val/
#
# Smoke test (2 nodes):
#   NFS_DIR=/path/to/shared/cache sbatch --nodes=2 --export=ALL,TRAIN_ITERS=8,\
#     SAVE_STEPS=99999,EVAL_STEPS=99999 scripts/45_train_llava_onevision_128k_multinode.slurm
#
# Production (4 nodes):
#   NFS_DIR=/path/to/shared/cache sbatch scripts/45_train_llava_onevision_128k_multinode.slurm
#
# Use 4-way TP, sequence parallel, 128K
# padding-free packing, MCore GDN, and Transformer Engine fused CE.
# With four 8-GPU nodes this gives DP=8 and GLOBAL_BS=128 (grad-accum=16).
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=llava-ov-128k
#SBATCH --output=logs/llava_ov_128k_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
: "${NFS_DIR:?Set NFS_DIR to the parent directory whose data/ contains the LLaVA-OneVision datasets}"
export NFS_DIR

export MIXTURE_YAML="${MIXTURE_YAML:-${PROJECT_ROOT}/data_configs/llava_onevision.yaml}"
export MODEL_PATH="${MODEL_PATH:-${NFS_DIR}/models/Qwen3.5-9B}"
export GAME_NAME="${GAME_NAME:-llava_onevision_qwen35_128k}"
export RUN_NAME="${RUN_NAME:-${GAME_NAME}_tp4_full_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${NFS_DIR}/outputs/${RUN_NAME}}"

# Match the data template's loader policy.  43_train_* derives DATASET_SPEC
# and VAL_FILE from MIXTURE_YAML; this script supplies the remaining runtime
# knobs that are deliberately model/job-specific rather than data-specific.
export STREAMING="${STREAMING:-true}"
export DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-128}"
export ENABLE_CHANNEL_LOSS="${ENABLE_CHANNEL_LOSS:-true}"
export CHANNEL_COLUMN="${CHANNEL_COLUMN:-data_source}"
export ADD_GENSHIN_SPECIAL_TOKENS="${ADD_GENSHIN_SPECIAL_TOKENS:-false}"
export STOPPING_STRATEGY="${STOPPING_STRATEGY:-all_exhausted}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export INTERLEAVE_PROB="${INTERLEAVE_PROB:-$("${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" "${MIXTURE_YAML}" --print-interleave-prob)}"

export TP="${TP:-4}"
export PP="${PP:-1}"
export SEQ_PARALLEL="${SEQ_PARALLEL:-true}"
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-128}"
export MAX_LEN="${MAX_LEN:-131072}"
export PACKING="${PACKING:-true}"
export PACKING_LENGTH="${PACKING_LENGTH:-131072}"
export PACKING_NUM_PROC="${PACKING_NUM_PROC:-1}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
export ASYNC_PACKING_PREFETCH_DEPTH="${ASYNC_PACKING_PREFETCH_DEPTH:-2}"
export SWIFT_USE_MCORE_GDN="${SWIFT_USE_MCORE_GDN:-1}"
export DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-false}"
export IMG_MAX_TOK="${IMG_MAX_TOK:-16384}"
export EPOCHS="${EPOCHS:-1}"
export LR="${LR:-1e-5}"
export MIN_LR="${MIN_LR:-1e-6}"
export VIT_LR="${VIT_LR:-3e-6}"
export ALIGNER_LR="${ALIGNER_LR:-1e-5}"
export WARMUP_ITERS="${WARMUP_ITERS:-100}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export RECOMPUTE="${RECOMPUTE:-full}"
export CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"
export CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-te}"
export MTP_LAYERS="${MTP_LAYERS:-1}"
export MTP_LOSS_SCALE="${MTP_LOSS_SCALE:-0.1}"
export EVAL_ITERS="${EVAL_ITERS:-163}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export EVAL_STEPS="${EVAL_STEPS:-5000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-true}"
export NO_SAVE_RNG="${NO_SAVE_RNG:-true}"

exec bash "${PROJECT_ROOT}/scripts/43_train_genshin_multinode.slurm"
