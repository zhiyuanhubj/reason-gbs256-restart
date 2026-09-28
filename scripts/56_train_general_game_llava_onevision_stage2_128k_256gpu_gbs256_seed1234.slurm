#!/bin/bash
# Stage 2 of preset 51: 128K context, 256 GPUs, GBS 256, seed 1234.
# Initialize from the stage-1 HF checkpoint export; optimizer/step count restart.
# Set STAGE1_MODEL_PATH (or MODEL_PATH), OUTPUT_DIR and the dataset variables below
# on the destination cluster. MODEL_PATH must contain HF weights, not a native
# Megatron checkpoint directory. Software paths/modules are inherited from env.
# TRAIN_ITERS defaults to 4000 packed optimizer steps; override if needed (>100).
# Cosine decay reaches MIN_LR at TRAIN_ITERS; data exhaustion can stop earlier.
# Activate the training Python environment before submitting from the repo root:
#   sbatch scripts/56_train_general_game_llava_onevision_stage2_128k_256gpu_gbs256_seed1234.slurm
# Override the account/partition/resource directives with sbatch for your cluster.
#SBATCH --account=brics.b6db
#SBATCH --partition=workq
#SBATCH --nodes=64
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=1-00:00:00
#SBATCH --job-name=general-game-stage2-256g
#SBATCH --output=logs/general_game_stage2_128k_256g_b256_s1234_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# Under sbatch, BASH_SOURCE is the spooled copy. Resolve this repository from
# the directory where sbatch was invoked, as in the 49 preset.
if [[ ! -f "${SCRIPT_DIR}/../data_configs/general_game_llava_onevision_stage2.yaml" \
      && -n "${SLURM_SUBMIT_DIR:-}" \
      && -f "${SLURM_SUBMIT_DIR}/data_configs/general_game_llava_onevision_stage2.yaml" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/scripts" && pwd)"
fi
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export MIXTURE_YAML="${PROJECT_ROOT}/data_configs/general_game_llava_onevision_stage2.yaml"
# Dataset paths are resolved on the training cluster, including the exact
# stage-1 guided validation files. WoW must use the new 674-Parquet release.
_REQUIRED_DATA_VARS=(
    LLAVA_DATA_ROOT GENERAL_GAME_DATA_ROOT game_caption_qa_0905
    WORLD_OF_WARCRAFT_DATA_ROOT GENSHIN_INSTRUCTION_DATA_ROOT
    CYBERPUNK2077_INSTRUCTION_DATA_ROOT GENERAL_INSTRUCTION_DATA_ROOT
    GENSHIN_REASONING_DATA_ROOT CYBERPUNK2077_REASONING_DATA_ROOT
    SPIDERMAN_REASONING_DATA_ROOT GENSHIN_GUIDED_VAL
    SPIDERMAN_GUIDED_VAL CYBERPUNK2077_GUIDED_VAL
)
for _var in "${_REQUIRED_DATA_VARS[@]}"; do
    [[ -n "${!_var:-}" ]] || { echo "[ERR] Set $_var on the training cluster" >&2; exit 1; }
    export "$_var"
done
export ENABLE_EVAL=true
export VAL_FILE=""
[[ -f "$MIXTURE_YAML" ]] || {
    echo "[ERR] mixture config not found: $MIXTURE_YAML"
    exit 1
}
export MODEL_PATH="${STAGE1_MODEL_PATH:-${MODEL_PATH:-}}"
: "${MODEL_PATH:?Set STAGE1_MODEL_PATH to the stage-1 HF checkpoint export}"
[[ -s "$MODEL_PATH/config.json" ]] || {
    echo "[ERR] Stage-1 HF checkpoint is missing config.json: $MODEL_PATH" >&2
    exit 1
}
if [[ ! -s "$MODEL_PATH/model.safetensors.index.json" && ! -s "$MODEL_PATH/model.safetensors" \
      && ! -s "$MODEL_PATH/pytorch_model.bin.index.json" && ! -s "$MODEL_PATH/pytorch_model.bin" ]]; then
    echo "[ERR] Stage-1 checkpoint must contain exported HF model weights: $MODEL_PATH" >&2
    exit 1
fi
export GAME_NAME="${GAME_NAME:-general_game_stage2}"
export RUN_NAME="${RUN_NAME:-general_game_stage2_128k_256g_b256_s1234_j${SLURM_JOB_ID:-local}}"
if (( ${#RUN_NAME} > 64 )); then
    echo "[ERR] RUN_NAME/WANDB_NAME exceeds 64 characters (${#RUN_NAME}): $RUN_NAME" >&2
    exit 1
fi
export RESUME_FROM=""
export AUTO_RESUME="false"
export FINETUNE=true
# 43_train_inner.sh creates this job-scoped temporary directory on each node.
export JOB_LOCAL_TMPDIR="${JOB_LOCAL_TMPDIR:-/dev/shm/${USER:-u}_train_tmp_${SLURM_JOB_ID:-local}}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new stage-2 output directory on shared storage}"
export OUTPUT_DIR
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

# Use the destination cluster's environment; 43_train_inner.sh activates Conda.
# Set CONDA_BASE / CONDA_ENV_NAME and, if needed, ENV_SETUP_MODULES,
# CUDA_HOME / CUDA_HOST_CC / CUDA_HOST_CXX before submission.
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-megatron-sft}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export INTERLEAVE_PROB="${INTERLEAVE_PROB:-$("${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/lib_mixture.py" "${MIXTURE_YAML}" --print-interleave-prob)}"

# -------------------------- GPU / parallelism ---------------------------
export GPUS="${GPUS:-0,1,2,3}"
export NNODES="${NNODES:-64}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export SEQ_PARALLEL="${SEQ_PARALLEL:-true}"
# NCCL 2.27.x RAS can segfault in rasMsgHandlePeersUpdate during long runs.
# Disabling RAS does not disable collectives or EFA communication.
export NCCL_RAS_ENABLE="${NCCL_RAS_ENABLE:-0}"

# 64 nodes x 4 GPUs, TP=4, PP=1, CP=1 -> DP=64. GBS=256 gives grad-accum=4.
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
export SEED="${SEED:-1234}"
# SWIFT data_seed is independent of seed and otherwise defaults to 42.
export DATA_SEED="${DATA_SEED:-${SEED}}"
echo "[stage2] seed=$SEED data_seed=$DATA_SEED"

# ------------------------ schedule / optimizer --------------------------
# Mixture passes already define total exposure, so the trainer runs one epoch.
export EPOCHS="${EPOCHS:-1}"
export TRAIN_ITERS="${TRAIN_ITERS:-4000}"
export LR="${LR:-2e-5}"
export MIN_LR="${MIN_LR:-6e-6}"
export VIT_LR="${VIT_LR:-6e-6}"
export ALIGNER_LR="${ALIGNER_LR:-2e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
export WARMUP_ITERS="${WARMUP_ITERS:-100}"
export LR_WARMUP_INIT="${LR_WARMUP_INIT:-6e-6}"
# Scale the warmup start with each group's peak LR: ViT starts at 1.8e-6.
export SCALE_GROUP_WARMUP_INIT=true
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
export FULL_VAL_EVAL="${FULL_VAL_EVAL:-true}"
# Retained for the opt-out partial-evaluation mode; ignored by full validation.
export EVAL_ITERS="${EVAL_ITERS:-1}"
export SWIFT_EVAL_NUM_MICROBATCHES="${SWIFT_EVAL_NUM_MICROBATCHES:-2}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export EVAL_STEPS="${EVAL_STEPS:-100}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-false}"
export ADD_VERSION="${ADD_VERSION:-false}"
export NO_SAVE_RNG="${NO_SAVE_RNG:-false}"
export NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-false}"
export NO_LOAD_OPTIM=true
export NO_LOAD_RNG=true

# Do not estimate the cosine horizon from raw rows / GBS: packing changes
# the number of optimizer steps. Use the configured 4000-step horizon by default.
if [[ ! "$TRAIN_ITERS" =~ ^[1-9][0-9]*$ || ! "$WARMUP_ITERS" =~ ^[0-9]+$ ]]; then
    echo "[ERR] TRAIN_ITERS must be positive; WARMUP_ITERS must be nonnegative" >&2
    exit 1
fi
if (( TRAIN_ITERS <= WARMUP_ITERS )); then
    echo "[ERR] TRAIN_ITERS must exceed WARMUP_ITERS=$WARMUP_ITERS" >&2
    exit 1
fi
echo "[stage2] base=$MODEL_PATH lr=$LR -> $MIN_LR ($LR_DECAY_STYLE), warmup=$WARMUP_ITERS, steps=$TRAIN_ITERS"

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
    [[ -n "$_MIX_VAL" ]] || { echo "[ERR] mixture config has no validation paths"; exit 1; }
    export VAL_FILE="$_MIX_VAL"
    export ENABLE_CHANNEL_LOSS=true
    exec bash "${PROJECT_ROOT}/scripts/43_train_inner.sh"
else
    exec bash "${PROJECT_ROOT}/scripts/43_train_genshin_multinode.slurm"
fi
