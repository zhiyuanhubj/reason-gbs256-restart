#!/usr/bin/env bash
# 下载 Qwen3.5-4B（或其他 model）到 models/<basename>/
#
# 用法:
#   bash scripts/01_download_model.sh                       # 默认 Qwen3.5-4B
#   MODEL_NAME=Qwen/Qwen3.5-30B-A3B  bash scripts/01_*.sh   # 改 model
#   HF_OFFICIAL=1 bash scripts/01_*.sh                      # 用 HF 官方源（境外/有梯子）
#   MS_DOWNLOAD=1 bash scripts/01_*.sh                      # 用 ModelScope
#
# 默认源：hf-mirror.com（国内 HF 镜像，~28 MB/s）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/env.sh"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
TARGET_DIR="${TARGET_DIR:-${PROJECT_ROOT}/models/$(basename "$MODEL_NAME")}"

mkdir -p "$(dirname "$TARGET_DIR")"

if [[ -d "$TARGET_DIR" && -n "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]]; then
    echo "[download] $TARGET_DIR 已存在且非空，跳过下载。如需覆盖：rm -rf $TARGET_DIR"
    exit 0
fi

# 下载源选择：
#   - 默认 hf-mirror.com（HuggingFace 中国镜像，国内最快，~28 MB/s on H200 测试机）
#   - MS_DOWNLOAD=1 改用 ModelScope（备选）
#   - HF_OFFICIAL=1 用 HuggingFace 官方（境外/有梯子时）
if [[ "${HF_OFFICIAL:-0}" == "1" ]]; then
    echo "[download] 通过 HuggingFace 官方下载 $MODEL_NAME ..."
    unset HF_ENDPOINT
    huggingface-cli download "$MODEL_NAME" --local-dir "$TARGET_DIR"
elif [[ "${MS_DOWNLOAD:-0}" == "1" ]]; then
    echo "[download] 通过 ModelScope 下载 $MODEL_NAME ..."
    TARGET_DIR="$TARGET_DIR" MODEL_NAME="$MODEL_NAME" python - <<'PY'
import os
from modelscope import snapshot_download
target = os.environ['TARGET_DIR']
src = os.environ['MODEL_NAME']
path = snapshot_download(src, local_dir=target)
print(f"[download] downloaded to: {path}")
PY
else
    echo "[download] 通过 hf-mirror.com 下载 $MODEL_NAME (HF 国内镜像) ..."
    export HF_ENDPOINT="https://hf-mirror.com"
    # hf_transfer 可以多线程加速（如未装则忽略）
    pip install -q hf_transfer 2>/dev/null && export HF_HUB_ENABLE_HF_TRANSFER=1 || true
    hf download "$MODEL_NAME" --local-dir "$TARGET_DIR" --max-workers 8
fi

echo "[download] 完成: $TARGET_DIR"
ls -lh "$TARGET_DIR" | head -20
