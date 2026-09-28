#!/usr/bin/env bash
# 推理 (单卡，stream，交互式)
#
# 用法:
#   bash scripts/31_infer.sh outputs/<run>/<ckpt>/safetensors_dir
#   GPU_ID=2 bash scripts/31_infer.sh path/to/ckpt
#
# Megatron 训练带 --save_safetensors true，ckpt 目录内会自动有 HF safetensors，
# 直接传那个目录就行。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/env.sh"

MODEL_PATH="${1:?用法: bash scripts/31_infer.sh <ckpt_dir>}"
GPU_ID="${GPU_ID:-0}"

[[ -d "$MODEL_PATH" ]] || { echo "[infer] ERR: 模型目录不存在: $MODEL_PATH"; exit 1; }

echo "[infer] model: $MODEL_PATH  (GPU $GPU_ID)"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift infer \
    --model "$MODEL_PATH" \
    --stream true \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9 \
    --infer_backend pt
