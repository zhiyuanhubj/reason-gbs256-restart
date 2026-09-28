#!/usr/bin/env bash
# Megatron mcore checkpoint → HuggingFace safetensors
#
# 注意：训练脚本里我们已经设了 --save_safetensors true，正常情况下保存的就是 HF 格式。
# 但在某些场景（PP/EP 较大）下你可能拿到 mcore 格式 checkpoint，需要离线转。
#
# 用法:
#   bash scripts/99_convert_mcore_to_hf.sh outputs/megatron_full_xxx/iter_001000

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/env.sh"

MCORE_DIR="${1:?用法: bash scripts/99_convert_mcore_to_hf.sh <mcore_ckpt_dir>}"

if [[ ! -d "$MCORE_DIR" ]]; then
    echo "[convert] 目录不存在: $MCORE_DIR"
    exit 1
fi

OUT_DIR="${MCORE_DIR%/}-hf"

echo "[convert] mcore: $MCORE_DIR"
echo "[convert] hf   : $OUT_DIR"

# ms-swift 提供的 mcore -> hf 转换
swift export \
    --mcore_model "$MCORE_DIR" \
    --to_hf true \
    --output_dir "$OUT_DIR" \
    --use_hf false

echo "[convert] 完成: $OUT_DIR"
