#!/usr/bin/env bash
# 一键安装 megatron-sft env 的所有依赖
#
# 前置：
#   - python 3.10 的 conda env 已创建：
#     conda create -n megatron-sft python=3.10 -y
#   - 机器上 nvidia-driver + CUDA 12.x 可用
#
# 用法:
#   bash scripts/00_install_env.sh           # 装基础环境（不装 apex）
#   bash scripts/00_install_env.sh apex      # 额外编译 apex (~30 min；mcore≥0.16 不强依赖)
#
# 环境变量（可覆盖）:
#   CONDA_ENV_NAME   conda env 名（默认 megatron-sft）
#   CONDA_BASE       miniconda 安装目录（默认从 `conda info --base` 自动探测）
#   CUDA_HOME        CUDA 路径（默认 /usr/local/cuda-12.8）

set -euo pipefail

# Versions used by the validated 64-GPU save/stop/resume run.
MS_SWIFT_COMMIT="${MS_SWIFT_COMMIT:-dc40c652fa9b8fa096613055ccf124dd49f8890c}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.14.1}"
MEGATRON_CORE_VERSION="${MEGATRON_CORE_VERSION:-0.16.1}"
MCORE_BRIDGE_VERSION="${MCORE_BRIDGE_VERSION:-1.6.1}"

# ----------------------------------------------------------------------
# 0. 自动探测 PROJECT_ROOT + 激活 env
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEPS_DIR="${PROJECT_ROOT}/.deps"
mkdir -p "$DEPS_DIR"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-megatron-sft}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /opt/miniconda3)}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# NVIDIA's cuda-nvcc conda activation hook reads optional variables without
# defaults, which is incompatible with nounset. Limit the relaxation to conda.
set +u
conda activate "${CONDA_ENV_NAME}"
set -u

# Environment modules on Cray systems may set CUDA_HOME as a non-exported
# shell variable. Recover it from nvcc so child pip build processes see it.
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    _NVCC_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    if [[ -d "${_NVCC_ROOT}/targets/sbsa-linux/include" ]]; then
        CUDA_HOME="${_NVCC_ROOT}/targets/sbsa-linux"
    else
        CUDA_HOME="$_NVCC_ROOT"
    fi
    export CUDA_HOME
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    # Conda CUDA keeps internal compiler tools such as cicc outside CUDA_HOME.
    export PATH="${CONDA_PREFIX}/nvvm/bin:${CONDA_PREFIX}/bin:${CUDA_HOME}/bin:${PATH}"
fi

py_ver=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[install] python=$py_ver, env=$CONDA_DEFAULT_ENV"

# ----------------------------------------------------------------------
# 1. 校验 torch 已装
# ----------------------------------------------------------------------
if ! python -c 'import torch' >/dev/null 2>&1; then
    if [[ "$(uname -m)" == "aarch64" ]]; then
        # PyTorch's cu128 index does not publish 2.8 ARM64 wheels. The cu129
        # ARM64 wheel is validated on the Isambard GH200s.
        echo "[install] torch 未装，安装 ARM64 torch 2.8.0 ..."
        pip install --upgrade 'torch==2.8.0+cu129' 'torchvision==0.23.0' \
            --index-url https://download.pytorch.org/whl/cu129
    else
        echo "[install] torch 未装，先装 torch 2.8.0+cu128 ..."
        pip install --upgrade 'torch==2.8.0' 'torchvision==0.23.0' \
            --index-url https://download.pytorch.org/whl/cu128
    fi
fi
python -c 'import torch; print("[install] torch:", torch.__version__, "cuda:", torch.version.cuda)'

# ----------------------------------------------------------------------
# 2. flash-attn (prebuilt wheel)
#    注意：torch 2.8.0+cu128 是 cxx11abi=TRUE 编译的，必须装匹配的 wheel，
#    否则会报 undefined symbol 错。pip install flash-attn 默认装的可能是 abiFALSE。
# ----------------------------------------------------------------------
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
need_install_fa=0
if ! python -c 'import flash_attn' >/dev/null 2>&1; then
    need_install_fa=1
else
    # 已装但 ABI 不对
    if ! python -c 'from flash_attn.flash_attn_interface import flash_attn_func' >/dev/null 2>&1; then
        echo "[install] flash-attn 已装但 ABI 不匹配，重装 ..."
        pip uninstall -y flash-attn || true
        need_install_fa=1
    fi
fi
if [[ "$need_install_fa" == "1" ]]; then
    if [[ "$(uname -m)" == "aarch64" ]]; then
        echo "[install] ARM64 从源码编译 flash-attn 2.8.3 ..."
        PATH="${CONDA_PREFIX}/nvvm/bin:${CUDA_HOME}/bin:${PATH}" \
            FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-90}" \
            MAX_JOBS="${MAX_JOBS:-32}" \
            pip install --no-build-isolation 'flash-attn==2.8.3'
    else
        echo "[install] 装 flash-attn 2.8.3 (cxx11abiTRUE) ..."
        pip install --no-build-isolation "$FA_WHEEL"
    fi
fi
python -c 'import flash_attn; from flash_attn.flash_attn_interface import flash_attn_func; print("[install] flash_attn:", flash_attn.__version__, "ok")'

# ----------------------------------------------------------------------
# 3. transformers（固定为完成验证的版本）
# ----------------------------------------------------------------------
echo "[install] 装 transformers ${TRANSFORMERS_VERSION} ..."
pip install -U "transformers==${TRANSFORMERS_VERSION}"
python -c 'import transformers; print("[install] transformers:", transformers.__version__)'

# ----------------------------------------------------------------------
# 4. ms-swift（固定 commit + 本仓库可复现 patch）
# ----------------------------------------------------------------------
echo "[install] 装 ms-swift ${MS_SWIFT_COMMIT} ..."
if [[ ! -d "${DEPS_DIR}/ms-swift" ]]; then
    git clone https://github.com/modelscope/ms-swift.git "${DEPS_DIR}/ms-swift"
fi
[[ -d "${DEPS_DIR}/ms-swift/.git" ]] || {
    echo "[ERR] ${DEPS_DIR}/ms-swift exists but is not a git clone"
    exit 1
}
if [[ "$(git -C "${DEPS_DIR}/ms-swift" rev-parse HEAD)" != "$MS_SWIFT_COMMIT" ]]; then
    if ! git -C "${DEPS_DIR}/ms-swift" diff --quiet ||
       ! git -C "${DEPS_DIR}/ms-swift" diff --cached --quiet; then
        echo "[ERR] existing ms-swift checkout has unrelated changes; use a clean DEPS_DIR"
        exit 1
    fi
    git -C "${DEPS_DIR}/ms-swift" fetch origin "$MS_SWIFT_COMMIT"
    git -C "${DEPS_DIR}/ms-swift" checkout --detach "$MS_SWIFT_COMMIT"
fi
# Verified against the actual training checkout; the existing cumulative
# patch includes finite-stream, TP broadcast, and packed dataloader resume.
# See docs/EXPLAN0909_ENVIRONMENT.md for validation details.
SWIFT_PATCHES=(
    "${PROJECT_ROOT}/patches/ms-swift-dc40c652-stateful-streaming-tp.patch"
)
for SWIFT_PATCH in "${SWIFT_PATCHES[@]}"; do
    if git -C "${DEPS_DIR}/ms-swift" apply --check "$SWIFT_PATCH"; then
        git -C "${DEPS_DIR}/ms-swift" apply "$SWIFT_PATCH"
    elif git -C "${DEPS_DIR}/ms-swift" apply --reverse --check "$SWIFT_PATCH"; then
        echo "[install] ms-swift patch already applied: $(basename "$SWIFT_PATCH")"
    else
        echo "[ERR] ms-swift patch does not apply cleanly: $SWIFT_PATCH"
        exit 1
    fi
done
pip install --no-deps -e "${DEPS_DIR}/ms-swift"
python -c 'import swift; print("[install] swift:", swift.__version__)'

# The editable install above deliberately uses --no-deps so it cannot replace
# the versions pinned for this training stack.  Install SWIFT's CLI/runtime
# dependencies explicitly; json_repair is imported even by `megatron sft
# --help`.
pip install -U \
    addict attrdict binpacking cpm_kernels dacite \
    'fastapi' 'gradio>=3.40,<6.0' json_repair matplotlib nltk openai oss2 \
    rouge scipy sortedcontainers transformers_stream_generator uvicorn zstandard

# ----------------------------------------------------------------------
# 5. 训练相关基础
# ----------------------------------------------------------------------
_MEDIA_DEPS=(qwen_vl_utils)
if [[ "$(uname -m)" != "aarch64" ]]; then
    _MEDIA_DEPS+=(decord)
else
    echo "[install] ARM64 跳过 decord（无官方 wheel；纯图片训练不使用视频解码）"
fi
pip install -U \
    'accelerate==1.14.0' \
    'peft==0.20.0' \
    'trl==1.9.2' \
    'datasets==5.0.1' \
    'pyarrow==25.0.0' \
    'pycryptodomex==3.23.0' \
    'wandb==0.28.1' \
    'pyyaml' \
    'deepspeed>=0.15.0' \
    'modelscope' \
    'einops' 'sentencepiece' 'tiktoken' \
    'tensorboard' 'swanlab' \
    'rich' 'tqdm' 'ipython' \
    "${_MEDIA_DEPS[@]}" 'huggingface_hub'   # qwen3.5 多模态 / 下载工具

# ----------------------------------------------------------------------
# 6. flash-linear-attention (GatedDeltaNet 内核) + tilelang
#    tilelang 必装：Triton >= 3.4 + Hopper GPU (H100/H200) 上 fla 的
#    gated chunk_bwd_dqkwg 会产生错误结果（fla bug #640），fla 检测到此场景
#    会 raise 错误，必须用 tilelang 后端替代。
# ----------------------------------------------------------------------
echo "[install] 装 flash-linear-attention + tilelang ..."
pip install -U 'flash-linear-attention==0.5.2' 'tilelang==0.1.13'
python -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule; import tilelang; print("[install] fla + tilelang ok")'

# ----------------------------------------------------------------------
# 7. Megatron 后端依赖
#    - megatron-core / mcore-bridge: pip wheel，秒装
#    - transformer_engine (TE): 需要 nccl.h 才能编译 frontend；编译约 1-2 min
#    - apex: 源码编译 30+ min，且 mcore>=0.16 不强依赖（缺失会 fallback 到 Torch Norm
#            略慢但功能完整）。所以默认不装。想装：bash scripts/00_install_env.sh apex
# ----------------------------------------------------------------------
echo "[install] 装 megatron-core + mcore-bridge ..."
pip install --no-deps -U \
    "megatron-core==${MEGATRON_CORE_VERSION}" \
    "mcore-bridge==${MCORE_BRIDGE_VERSION}"

# transformer_engine
if ! python -c 'import transformer_engine.pytorch' >/dev/null 2>&1; then
    echo "[install] 装 transformer_engine[pytorch,core_cu12] ... (~1 min 编译)"
    # The PyTorch CUDA wheel contains cuDNN runtime libraries but not headers.
    # TE's PyTorch frontend needs matching headers when built on ARM64.
    if [[ "$(uname -m)" == "aarch64" ]] && \
       [[ ! -f "${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/include/cudnn.h" ]]; then
        pip install --no-deps 'nvidia-cudnn-cu12==9.10.2.21'
    fi
    # 设置 NCCL 头路径，否则编译会找不到 nccl.h
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
    export PATH="$CUDA_HOME/bin:$PATH"
    NCCL_HOME="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nccl"
    CUDNN_HOME="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn"
    export CPATH="${CUDNN_HOME}/include:${NCCL_HOME}/include:${CPATH:-}"
    export LIBRARY_PATH="${NCCL_HOME}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
    if [[ "$(uname -m)" == "aarch64" ]]; then
        _HPC_SDK_ROOT="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11"
        _CUDA_MATH_ROOT="${_HPC_SDK_ROOT}/math_libs/12.6/targets/sbsa-linux"
        _CUDA_TOOLKIT_ROOT="${_HPC_SDK_ROOT}/cuda/12.6/targets/sbsa-linux"
        _NCCL_ROOT="${_HPC_SDK_ROOT}/comm_libs/12.6/nccl"
        export CPATH="${_CUDA_MATH_ROOT}/include:${_CUDA_TOOLKIT_ROOT}/include:${_NCCL_ROOT}/include:${CPATH:-}"
        export LIBRARY_PATH="${CUDNN_HOME}/lib:${_CUDA_MATH_ROOT}/lib:${_CUDA_TOOLKIT_ROOT}/lib:${_NCCL_ROOT}/lib:${LIBRARY_PATH:-}"
    fi

    pip install --no-build-isolation 'transformer-engine[pytorch,core_cu12]==2.14.1'
fi
python -c 'import transformer_engine; import transformer_engine.pytorch as te; \
print("[install] TE:", transformer_engine.__version__, "te.Linear:", hasattr(te, "Linear"))'

# apex（仅当 `apex` 参数显式传入时才编译）
if [[ "${1:-}" == "apex" ]]; then
    if ! python -c 'import apex' >/dev/null 2>&1; then
        echo "[install] === 编译 apex 源码（30+ min，需要 GPU 空闲）==="
        if [[ ! -d "${DEPS_DIR}/apex" ]]; then
            git clone https://github.com/NVIDIA/apex.git "${DEPS_DIR}/apex"
        fi
        (
            cd "${DEPS_DIR}/apex"
            if [[ "$(uname -m)" == "aarch64" ]]; then
                # Conda's GCC 14 headers are not compatible with nvcc 12.9 on
                # this GH200 image.  The system GCC 12 toolchain is supported.
                export CC="${APEX_CC:-/usr/bin/gcc-12}"
                export CXX="${APEX_CXX:-/usr/bin/g++-12}"
                export CUDAHOSTCXX="$CXX"
                export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
            fi
            pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
                --config-settings "--build-option=--cpp_ext" \
                --config-settings "--build-option=--cuda_ext" \
                ./
        )
    fi
fi

# ----------------------------------------------------------------------
# 8. 检验 (轻量)
# ----------------------------------------------------------------------
echo ""
echo "============================================================"
echo "[install] 环境装完。运行 'python tools/verify_env.py' 校验。"
echo "[install] 如果想装 apex（30+ min 编译，性能略提升）："
echo "[install]   bash scripts/00_install_env.sh apex"
echo "============================================================"
