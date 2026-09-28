#!/bin/bash
# Genshin, Cyberpunk 2077 and Spider-Man 2 polished reasoning history15/sl120, three passes each.
# Base: general_game_4183 (HF-format weights; override MODEL_PATH when ready).
# New fine-tuning run: 128K, 128 GH200 GPUs, GBS 256, seed 42.
# Uses 06/07 JSONL: 77,453 training samples and 192 validation samples.
# Submit from the repository root:
#   sbatch scripts/57_train_general_game_reasoning_history15_3pass_128k_128gpu_gbs256_0924.slurm
#SBATCH --account=brics.b6db
#SBATCH --partition=workq
#SBATCH --nodes=32
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=1-00:00:00
#SBATCH --job-name=reason-hist15-0924-128g
#SBATCH --output=logs/general_reason_hist15_0924_128k_b256_s42_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# Under sbatch, BASH_SOURCE is the spooled copy. Resolve this repository from
# the directory where sbatch was invoked, as in the 49 preset.
if [[ ! -f "${SCRIPT_DIR}/../data_configs/general_game_reasoning_history15_128K_0924.yaml" \
      && -n "${SLURM_SUBMIT_DIR:-}" \
      && -f "${SLURM_SUBMIT_DIR}/data_configs/general_game_reasoning_history15_128K_0924.yaml" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/scripts" && pwd)"
fi
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# Keep the documented short Lustre alias in job configuration.
export ASSET_ROOT="${ASSET_ROOT:-/projects/b6db/opensima}"
export NFS_DIR="${NFS_DIR:-${ASSET_ROOT}}"

export MIXTURE_YAML="${PROJECT_ROOT}/data_configs/general_game_reasoning_history15_128K_0924.yaml"
# The YAML selects three reasoning training splits and 192 validation records.
# Clear inherited validation paths; the launcher reads them from the YAML.
export ENABLE_EVAL="${ENABLE_EVAL:-true}"
export VAL_FILE=""
export FULL_VAL_EVAL="${FULL_VAL_EVAL:-true}"
[[ -f "$MIXTURE_YAML" ]] || {
    echo "[ERR] mixture config not found: $MIXTURE_YAML"
    exit 1
}
# Load HF weights without resuming optimizer or step count.
# Placeholder name until the general_game_4183 HF export is available.
export MODEL_PATH="${MODEL_PATH:-${ASSET_ROOT}/models/general_game_4183}"
for _model_file in config.json model.safetensors.index.json tokenizer.json; do
    [[ -s "$MODEL_PATH/$_model_file" ]] || {
        echo "[ERR] base model HF export is missing a required file: $MODEL_PATH/$_model_file"
        exit 1
    }
done
export GAME_NAME="${GAME_NAME:-general_reason_hist15_0924}"
export RUN_NAME="${RUN_NAME:-general_reason_hist15_0924_128k_b256_s42_j${SLURM_JOB_ID:-local}}"
if (( ${#RUN_NAME} > 64 )); then
    echo "[ERR] RUN_NAME/WANDB_NAME exceeds 64 characters (${#RUN_NAME}): $RUN_NAME" >&2
    exit 1
fi
export RESUME_FROM="${RESUME_FROM:-}"
export AUTO_RESUME="${AUTO_RESUME:-false}"
# 43_train_inner.sh creates this job-scoped temporary directory on each node.
export JOB_LOCAL_TMPDIR="${JOB_LOCAL_TMPDIR:-/dev/shm/${USER:-u}_train_tmp_${SLURM_JOB_ID:-local}}"
_OUTPUT_DIR_EXPLICIT="${OUTPUT_DIR+x}"
if [[ -n "$RESUME_FROM" && "$RESUME_FROM" != "auto" && -z "$_OUTPUT_DIR_EXPLICIT" ]]; then
    export OUTPUT_DIR="$(dirname "$(readlink -f "$RESUME_FROM")")"
    export RUN_NAME="${RUN_NAME:-$(basename "$OUTPUT_DIR")}"
else
    export RUN_NAME="${RUN_NAME:-${GAME_NAME}_tp4_full_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}}"
    export OUTPUT_DIR="${OUTPUT_DIR:-${NFS_DIR}/outputs/${RUN_NAME}}"
fi
if [[ "$AUTO_RESUME" == "true" || "$RESUME_FROM" == "auto" ]]; then
    [[ -n "$_OUTPUT_DIR_EXPLICIT" ]] || {
        echo "[ERR] AUTO_RESUME requires a stable, explicitly supplied OUTPUT_DIR"
        exit 1
    }
fi
export MIXTURE_PLAN="${MIXTURE_PLAN:-${OUTPUT_DIR}/mixture_plan.json}"

# ----------------------------- data loading -----------------------------
export STREAMING="${STREAMING:-true}"
export STREAMING_SHARD_BY_DP="${STREAMING_SHARD_BY_DP:-true}"
export DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
export TRAIN_DATALOADER_SHUFFLE="${TRAIN_DATALOADER_SHUFFLE:-true}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-128}"
export STOPPING_STRATEGY="${STOPPING_STRATEGY:-all_exhausted_without_replacement}"
export STOP_AT_DATASET_END="${STOP_AT_DATASET_END:-true}"
export PASS_AWARE_MIXTURE="${PASS_AWARE_MIXTURE:-true}"
# A TP group consumes one identical packed sequence. Only TP rank zero owns
# physical readers/packing workers; collated tensors are broadcast on-device.
export TP_DATALOADER_BROADCAST="${TP_DATALOADER_BROADCAST:-true}"
export CHANNEL_COLUMNS_JSON="${CHANNEL_COLUMNS_JSON:-{\"conversations\":\"messages\",\"game\":\"channel\",\"data_source\":\"channel\"}}"
export ENABLE_CHANNEL_LOSS="${ENABLE_CHANNEL_LOSS:-true}"
export ADD_GENSHIN_SPECIAL_TOKENS="${ADD_GENSHIN_SPECIAL_TOKENS:-true}"

# Isambard software stack and shared Conda environment.
export CONDA_BASE="${CONDA_BASE:-/home/b6db/hongnanvideo1.b6db/.miniconda3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-megatron-sft}"
PYTHON_BIN="${PYTHON_BIN:-/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/bin/python}"
export PYTHON_BIN
export ENV_SETUP_MODULES="${ENV_SETUP_MODULES:-cuda/12.6 brics/nccl/2.26.6-1 brics/aws-ofi-nccl/1.8.1}"
export CUDA_HOME="${CUDA_HOME:-/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/targets/sbsa-linux}"
export CUDA_HOST_CC="${CUDA_HOST_CC:-/usr/bin/gcc-13}"
export CUDA_HOST_CXX="${CUDA_HOST_CXX:-/usr/bin/g++-13}"
export CC="$CUDA_HOST_CC"
export CXX="$CUDA_HOST_CXX"
export INTERLEAVE_PROB="${INTERLEAVE_PROB:-$("${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" "${MIXTURE_YAML}" --print-interleave-prob)}"

# -------------------------- GPU / parallelism ---------------------------
export GPUS="${GPUS:-0,1,2,3}"
export NNODES="${NNODES:-32}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export SEQ_PARALLEL="${SEQ_PARALLEL:-true}"
# NCCL 2.27.x RAS can segfault in rasMsgHandlePeersUpdate during long runs.
# Disabling RAS does not disable collectives or EFA communication.
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"

# 32 nodes x 4 GPUs, TP=4, PP=1, CP=1 -> DP=32. GBS=256 gives grad-accum=8.
# Qwen3.5-9B has four KV heads; TP=8 fails Megatron's query-group divisibility check.
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-256}"

# Match the stable 128-GPU communication configuration used by preset 49.
export OVERLAP_GRAD_REDUCE="${OVERLAP_GRAD_REDUCE:-false}"
export OVERLAP_PARAM_GATHER="${OVERLAP_PARAM_GATHER:-false}"

# ----------------------- context / multimodality ------------------------
export MAX_LEN="${MAX_LEN:-131072}"
export IMG_MAX_TOK="${IMG_MAX_TOK:-16384}"
export PACKING="${PACKING:-true}"
export PACKING_LENGTH="${PACKING_LENGTH:-131072}"
export PACKING_NUM_PROC="${PACKING_NUM_PROC:-1}"
export PACKING_INTERVAL="${PACKING_INTERVAL:-128}"
export PACKING_STRATEGY="${PACKING_STRATEGY:-binpack}"
# IterablePackingDataset uses dataset_num_proc for its own encoding processes;
# zero creates no workers and blocks forever waiting on an empty output queue.
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
export ASYNC_PACKING_PREFETCH_DEPTH="${ASYNC_PACKING_PREFETCH_DEPTH:-2}"
# Keep PyTorch DataLoader workers at zero: the packing dataset already owns the
# encoding worker above, and another worker layer costs excessive RAM.
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
if [[ "$STREAMING" == "true" && "$PACKING" == "true" && "$DATASET_NUM_PROC" -le 0 ]]; then
    echo "[ERR] streaming packing requires DATASET_NUM_PROC >= 1; zero deadlocks IterablePackingDataset"
    exit 1
fi
export SWIFT_USE_MCORE_GDN="${SWIFT_USE_MCORE_GDN:-1}"
export DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-false}"
export SEED="${SEED:-42}"

# ------------------------ schedule / optimizer --------------------------
# Mixture passes already define total exposure, so the trainer runs one epoch.
export EPOCHS="${EPOCHS:-1}"
export TRAIN_ITERS="${TRAIN_ITERS:-400}"
export LR="${LR:-2e-5}"
export MIN_LR="${MIN_LR:-5e-7}"
# Freeze the vision encoder while continuing to train the projector and LLM.
export FREEZE_VIT=true
export FREEZE_ALIGNER=false
export VIT_LR=""
export ALIGNER_LR="${ALIGNER_LR:-2e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export WARMUP_ITERS="${WARMUP_ITERS:-20}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"

# ---------------------- memory / fused kernels --------------------------
export RECOMPUTE="${RECOMPUTE:-full}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
export CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"
export CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-te}"
export DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-true}"
# MTP checkpoint TP-replica metadata is corrected by
# scripts/mcore_mtp_checkpoint_patch.py and covered by crash/resume validation.
export MTP_LAYERS="${MTP_LAYERS:-1}"
export MTP_LOSS_SCALE="${MTP_LOSS_SCALE:-0.1}"

# ----------------------- evaluation / checkpoint ------------------------
# Full validation covers all three 64-record splits and ignores EVAL_ITERS.
export EVAL_ITERS="${EVAL_ITERS:-50}"
export SWIFT_EVAL_NUM_MICROBATCHES="${SWIFT_EVAL_NUM_MICROBATCHES:-2}"
export SAVE_STEPS="${SAVE_STEPS:-20}"
export EVAL_STEPS="${EVAL_STEPS:-20}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-false}"
export ADD_VERSION="${ADD_VERSION:-false}"
export NO_SAVE_RNG="${NO_SAVE_RNG:-false}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-false}"
export NO_LOAD_OPTIM="${NO_LOAD_OPTIM:-false}"
export NO_LOAD_RNG="${NO_LOAD_RNG:-false}"

# Default TRAIN_ITERS=400 sets the optimizer-step cap and cosine schedule.
# STOP_AT_DATASET_END may finish earlier when the three-pass mixture is exhausted.
# Explicit TRAIN_ITERS=0 retains automatic estimation below.
# Iterable datasets have no __len__. With packing, raw rows / GBS is only a
# generous safety ceiling and scheduler horizon; packing can end the run before
# the cosine schedule reaches MIN_LR. STOP_AT_DATASET_END ends at the exact final
# complete packed optimizer step without cycling the mixture.
if [[ "$STREAMING" == "true" && "$TRAIN_ITERS" -le 0 ]]; then
    _MIXTURE_TOTAL_SAMPLES="$(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" \
            "${MIXTURE_YAML}" --print-total-samples
    )"
    export TRAIN_ITERS="$(( _MIXTURE_TOTAL_SAMPLES / GLOBAL_BS ))"
    echo "[mixture] streaming train_iters=$TRAIN_ITERS from $_MIXTURE_TOTAL_SAMPLES samples / GBS=$GLOBAL_BS"
fi

# -------------------------------- WandB ---------------------------------
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-opensima}"
export WANDB_ENTITY="${WANDB_ENTITY:-OpenSIMA}"

# This file is the complete experiment configuration. 43 only provides the
# Slurm rendezvous/srun layer; 41 converts these values into Megatron flags.
if [[ "${DIRECT_INNER:-false}" == "true" ]]; then
    if [[ "$PASS_AWARE_MIXTURE" == "true" ]]; then
        mkdir -p "$(dirname "$MIXTURE_PLAN")"
        export DATASET_SPEC="$(
            "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" \
                "${MIXTURE_YAML}" --seed "$SEED" \
                --write-plan "$MIXTURE_PLAN" --print-source-spec
        )"
    else
        export DATASET_SPEC="$(
            "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" \
                "${MIXTURE_YAML}" --print-spec
        )"
    fi
    _MIX_VAL="$(
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" \
            "${MIXTURE_YAML}" --print-val
    )"
    if [[ "$ENABLE_EVAL" == "true" ]]; then
        [[ -n "$_MIX_VAL" ]] || { echo "[ERR] mixture config has no validation paths"; exit 1; }
        export VAL_FILE="$_MIX_VAL"
    else
        export VAL_FILE=""
    fi
    export ENABLE_CHANNEL_LOSS=true
    exec bash "${PROJECT_ROOT}/scripts/43_train_inner.sh"
else
    exec bash "${PROJECT_ROOT}/scripts/43_train_genshin_multinode.slurm"
fi
