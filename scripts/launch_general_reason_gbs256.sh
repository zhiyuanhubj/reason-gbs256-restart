#!/usr/bin/env bash
# 4183 + 0916 mixture on existing extra-holds. Keep original GBS=256; raise
# grad-accum instead of shrinking batch (4x8 H200, TP=4, DP=8 -> accum=32).
set -euo pipefail
if [[ "$(hostname)" == ip-10-1-115-182 ]]; then
    echo "[ERR] refusing launch on login $(hostname)" >&2
    exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/fsx/home/zhiyuan/training}"
HOLD_JOBS="${HOLD_JOBS:?comma-separated hold job ids}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/opt/dlami/nvme/zhiyuan-reasoning-jsonl}"
MIXTURE_YAML="${MIXTURE_YAML:-${PROJECT_ROOT}/data_configs/general_game_reasoning_128K_0921.yaml}"
MODEL_PATH="${MODEL_PATH:-/fsx/home/zhiyuan/nfs/models/Qwen3.5-9B-General-Game/checkpoint-4183}"
NFS_DIR="${NFS_DIR:-/fsx/home/zhiyuan/nfs}"
KEY_FILE="${KEY_FILE:-/fsx/home/zhiyuan/.config/wandb/opensima_mirror.key}"
RUN_NAME="${RUN_NAME:-reason0916_gbs256_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${LOCAL_DATA_ROOT}/checkpoints/${RUN_NAME}}"
MASTER_PORT="${MASTER_PORT:-30111}"

export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d ' \n' < "$KEY_FILE")}"
test -n "$WANDB_API_KEY"
test -s "$MODEL_PATH/config.json"
test -s "$MODEL_PATH/tokenizer.json"
test -s "$MODEL_PATH/model.safetensors.index.json"
test -s "$MIXTURE_YAML"

IFS=, read -r -a jobs <<<"$HOLD_JOBS"
njobs=${#jobs[@]}
if (( njobs < 1 )); then
    echo "[ERR] HOLD_JOBS is empty" >&2
    exit 2
fi

allocations=()
nodes=()
for rank in "${!jobs[@]}"; do
    job="${jobs[$rank]}"
    line="$(squeue -h -j "$job" -o '%T|%N')"
    state="${line%%|*}"
    node="${line#*|}"
    if [[ "$state" != "RUNNING" || -z "$node" || "$node" == "(null)" ]]; then
        echo "[ERR] job $job is not a running one-node allocation: $line" >&2
        exit 2
    fi
    allocations+=("$rank:$job:$node")
    nodes+=("$node")
done
if (( $(printf '%s\n' "${nodes[@]}" | sort -u | wc -l) != njobs )); then
    echo "[ERR] allocations do not resolve to $njobs unique nodes: ${nodes[*]}" >&2
    exit 2
fi

MASTER_ADDR="${MASTER_ADDR:-${nodes[0]#ip-}}"
MASTER_ADDR="${MASTER_ADDR//-/.}"
RUN_STATE_DIR="${NFS_DIR}/run-state/${RUN_NAME}"
mkdir -p "$RUN_STATE_DIR" "$PROJECT_ROOT/logs" "$OUTPUT_DIR"

export PROJECT_ROOT HOLD_JOBS LOCAL_DATA_ROOT MIXTURE_YAML MODEL_PATH NFS_DIR
export RUN_NAME OUTPUT_DIR MASTER_ADDR MASTER_PORT
export MIXTURE_PLAN="${OUTPUT_DIR}/mixture_plan.json"
export ASSET_ROOT="$NFS_DIR"
export NNODES="$njobs" NPROC_PER_NODE=8 GPUS_PER_NODE=8 GPUS=0,1,2,3,4,5,6,7
export TP="${TP:-4}" PP=1 CP=1
export GLOBAL_BS="${GLOBAL_BS:-256}" MICRO_BS="${MICRO_BS:-1}"
export DIRECT_INNER=true PASS_AWARE_MIXTURE=true
export FINETUNE="${FINETUNE:-true}" AUTO_RESUME="${AUTO_RESUME:-false}" RESUME_FROM="${RESUME_FROM:-}"
export ALLOW_GBS_CHANGE="${ALLOW_GBS_CHANGE:-false}"
export EPOCHS=1 TRAIN_ITERS="${TRAIN_ITERS:-0}"
export MAX_LEN=131072 PACKING_LENGTH=131072 IMG_MAX_TOK=16384
export SAVE_STEPS="${SAVE_STEPS:-20}" EVAL_STEPS="${EVAL_STEPS:-20}" EVAL_ITERS="${EVAL_ITERS:-8}" SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-0}"
export STOP_AT_DATASET_END=true
export FREEZE_VIT=true FREEZE_ALIGNER=false
export LR=2e-5 MIN_LR=5e-7 WARMUP_ITERS=20 LR_DECAY_STYLE=cosine SEED=42
export CONDA_BASE=/fsx/home/zhiyuan/miniconda3
export CONDA_ENV_NAME=megatron-sft
export PYTHON_BIN=/fsx/home/zhiyuan/miniconda3/envs/megatron-sft/bin/python
export ENV_SETUP_MODULES=none
export MODEL_TYPE="${MODEL_TYPE:-qwen3_5}"
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_HOST_CC=/usr/bin/gcc
export CUDA_HOST_CXX=/usr/bin/g++
export NCCL_RAS_ENABLE=0
export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"
export WANDB_MODE=online WANDB_ENTITY="${WANDB_ENTITY:-zhiyuan-hu-bj-nus}" WANDB_PROJECT="${WANDB_PROJECT:-opensima}"
export WANDB_DIR="$OUTPUT_DIR/wandb"
export WANDB_DISABLE_CODE=true
export WANDB_CONSOLE=off
unset WANDB_RUN_ID WANDB_RESUME WANDB_RUN_GROUP WANDB_JOB_TYPE WANDB_SWEEP_ID WANDB_API_KEY_HAOYU WANDB_SILENT
export PYTHONUNBUFFERED=1
export GAME_NAME="${GAME_NAME:-general_reason_0916}"
export STEP_TIME="${STEP_TIME:-6-00:00:00}"

DP=$(( NNODES * NPROC_PER_NODE / TP / PP / CP ))
if (( GLOBAL_BS % DP != 0 )); then
    echo "[ERR] GLOBAL_BS=$GLOBAL_BS not divisible by DP=$DP (nnodes=$NNODES tp=$TP)" >&2
    exit 2
fi
ACCUM=$(( GLOBAL_BS / (MICRO_BS * DP) ))

echo "[launcher] run=$RUN_NAME jobs=${jobs[*]} nodes=${nodes[*]}"
echo "[launcher] master=$MASTER_ADDR:$MASTER_PORT data=$MIXTURE_YAML output=$OUTPUT_DIR"
echo "[launcher] gbs=$GLOBAL_BS micro=$MICRO_BS tp=$TP dp=$DP accum=$ACCUM nnodes=$NNODES"

while IFS= read -r var; do
    unset "$var"
done < <(env | awk -F= '/^SLURM_/ {print $1}')
unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER

pids=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM

for entry in "${allocations[@]}"; do
    IFS=: read -r rank job node <<<"$entry"
    log="$PROJECT_ROOT/logs/${RUN_NAME}.launcher.rank${rank}.log"
    echo "[launcher] rank=$rank job=$job node=$node log=$log"
    NODE_RANK="$rank" \
        srun --jobid="$job" --overlap --nodes=1 --ntasks=1 \
        --time="$STEP_TIME" --cpus-per-task=96 --mem=0 --gres=gpu:8 --nodelist="$node" \
        bash "$PROJECT_ROOT/scripts/55_train_general_game_reasoning_3pass_128k_128gpu_gbs256_0921.slurm" \
        >"$log" 2>&1 &
    pids+=("$!")
done

rc=0
remaining=${#pids[@]}
while (( remaining > 0 )); do
    if wait -n; then
        remaining=$((remaining - 1))
    else
        rc=$?
        echo "[launcher] a rank failed rc=$rc; terminating remaining ranks" >&2
        cleanup
        break
    fi
done
wait 2>/dev/null || true
echo "[launcher] finished rc=$rc"
exit "$rc"
