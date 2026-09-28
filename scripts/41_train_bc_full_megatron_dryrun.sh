#!/usr/bin/env bash
# Megatron 后端 dry-run：5 step，验证 full_sft + MTP + new_special_tokens 都能跑
# 用途：拷贝项目到新机器后，第一次跑这个看模型能加载、显存够、loss 下降
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/env.sh"

MODEL_PATH="${PROJECT_ROOT}/models/Qwen3.5-4B"
TRAIN_FILE="${PROJECT_ROOT}/data/bc_sft_processed/train.jsonl"
VAL_FILE="${PROJECT_ROOT}/data/bc_sft_processed/val_small.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/_megatron_dryrun_$(date +%H%M%S)"
mkdir -p "$OUTPUT_DIR"

export WANDB_MODE=offline
export FULL_VAL_EVAL="${FULL_VAL_EVAL:-true}"
EVAL_FLAGS=()
DATALOADER_NUM_WORKERS=2
if [[ "$FULL_VAL_EVAL" == "true" ]]; then
    EVAL_FLAGS+=(--external_plugins "${PROJECT_ROOT}/scripts/swift_full_val_eval.py"
        --streaming true --streaming_shard_by_dp true --val_dataset_shuffle false)
    DATALOADER_NUM_WORKERS=0
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
SWIFT_USE_MCORE_GDN=1 \
IMAGE_MAX_TOKEN_NUM=1024 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
megatron sft \
    --model "$MODEL_PATH" \
    --dataset "$TRAIN_FILE" \
    --val_dataset "$VAL_FILE" \
    --split_dataset_ratio 0.0 \
    --new_special_tokens '<|action_start|>' '<|action_end|>' '<|thought_start|>' '<|thought_end|>' \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --sequence_parallel true \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --max_length 32768 \
    --packing false \
    --train_iters 5 \
    --finetune true \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --vit_gradient_checkpointing true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --cross_entropy_loss_fusion true \
    --attention_backend flash \
    --bf16 true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --lr_decay_style cosine \
    --weight_decay 0.01 \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 1 \
    --save_steps 100000 \
    --eval_steps 100000 \
    --no_save_optim true \
    --no_save_rng true \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataset_num_proc 4 \
    --report_to wandb \
    --mtp_num_layers 1 \
    --mtp_loss_scaling_factor 0.1 \
    --seed 42 \
    "${EVAL_FLAGS[@]}"
