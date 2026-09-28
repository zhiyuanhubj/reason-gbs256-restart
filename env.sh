# 用法：source env.sh
# 激活 conda env + 设置训练相关环境变量
#
# 可移植性：PROJECT_ROOT 自动 = 本文件所在目录（脚本可以 cp 到任何机器）
# 可被覆盖的环境变量（在 source 前 export）：
#   CONDA_ENV_NAME      conda env 名（默认 megatron-sft）
#   CONDA_BASE          miniconda 安装目录（默认从 `which conda` 自动探测）
#   CUDA_HOME           CUDA 安装路径（默认 /usr/local/cuda-12.8，下面会探测）
#   WANDB_PROJECT       wandb 项目名（默认 bc-sft-qwen35）
#   WANDB_ENTITY        wandb team (默认 OpenSIMA；改成你的 entity 或 unset)

# --- 项目根（自动探测）---
# bash 兼容写法：用 BASH_SOURCE 取本文件路径，再 readlink -f 取绝对路径
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# --- conda env ---
CONDA_ENV_NAME="${CONDA_ENV_NAME:-megatron-sft}"
if [[ -z "${CONDA_PREFIX:-}" || "$(basename "${CONDA_PREFIX:-}")" != "${CONDA_ENV_NAME}" ]]; then
    set +u   # conda activate 内部用了未绑定变量，临时关掉 nounset
    if [[ -z "${CONDA_BASE:-}" ]]; then
        CONDA_BASE="$(conda info --base 2>/dev/null || echo /opt/miniconda3)"
    fi
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    set -u 2>/dev/null || true
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- 模型 / 数据缓存路径 ---
export MODELSCOPE_CACHE="${PROJECT_ROOT}/models/.modelscope_cache"
export HF_HOME="${PROJECT_ROOT}/models/.hf_cache"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/data/.hf_datasets_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
mkdir -p "$MODELSCOPE_CACHE" "$HF_HOME" "$HF_DATASETS_CACHE"

# --- CUDA / NCCL（编译 transformer_engine 等依赖时用）---
if [[ -z "${CUDA_HOME:-}" ]]; then
    if [[ -d /usr/local/cuda-12.8 ]]; then
        export CUDA_HOME=/usr/local/cuda-12.8
    elif [[ -d /usr/local/cuda ]]; then
        export CUDA_HOME=/usr/local/cuda
    fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi
# Conda's ARM64 CUDA toolkit keeps the internal device compiler here. nvcc
# needs it for flash-attn/TileLang JIT builds on GH200 nodes.
if [[ -d "${CONDA_PREFIX}/nvvm/bin" ]]; then
    # Keep Conda's matching ptxas ahead of any older CUDA module in PATH.
    export PATH="${CONDA_PREFIX}/nvvm/bin:${CONDA_PREFIX}/bin:${PATH}"
fi
_NCCL_HOME="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nccl"
if [[ -d "$_NCCL_HOME" ]]; then
    export CPATH="${_NCCL_HOME}/include:${CPATH:-}"
    export LIBRARY_PATH="${_NCCL_HOME}/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${_NCCL_HOME}/lib:${CUDA_HOME:-}/lib64:${LD_LIBRARY_PATH:-}"
fi

# pip 安装的 CUDA 运行库(cublas/cusparse/...)的 .so 在 site-packages/nvidia/*/lib，
# 作为兜底加进 LD_LIBRARY_PATH。
_SP_NVIDIA="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia"
if [[ -d "$_SP_NVIDIA" ]]; then
    _NV_LIBS=""
    for _d in "$_SP_NVIDIA"/*/lib; do
        [[ -d "$_d" ]] && _NV_LIBS="${_NV_LIBS:+$_NV_LIBS:}$_d"
    done
    [[ -n "$_NV_LIBS" ]] && export LD_LIBRARY_PATH="${_NV_LIBS}:${LD_LIBRARY_PATH:-}"
fi

# ★ cuDNN 一致性修复（关键）★
# transformer_engine 是用 ${CUDA_HOME}(=/usr/local/cuda-12.8) 编译的，其 RPATH 指向
# 系统 cuda-12.8/lib，import TE 时会加载【系统】的 libcudnn 主库；
# 若 cudnn 子库(libcudnn_engines_*.so 等)却从 pip 的 nvidia/cudnn/lib 解析，
# 主库/子库虽同为 9.10.2 但属不同构建 → 视觉塔 conv 报:
#   CUDNN_BACKEND_TENSOR_DESCRIPTOR ... CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
# 系统 ${CUDA_HOME}/lib 里有【完整】的 cudnn 9.10.2(主库+全部子库)，把它放最前，
# 让主库和子库都来自同一处，与 TE 一致，彻底消除错配。
if [[ -d "${CUDA_HOME:-}/lib" ]]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

# --- 训练相关 ---
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

# Triton / Inductor JIT 内核缓存必须放【节点本地磁盘】，不能放共享 Lustre。
# 否则多节点多卡(几十个进程)并发读写/重命名 Lustre 上的 ~/.triton/cache 会触发竞态:
#   OSError [Errno 14] Bad address: .../causal_conv1d_fwd_kernel.ttir
#   FileNotFoundError: .../l2norm_fwd_kernel.source (atomic rename 失败)
# GatedDeltaNet 的 causal_conv1d / l2norm 等都是 fla 的 Triton 内核，必须用本地缓存。
# 注意：不能用 /tmp(在根分区 /，某些节点会 100% 满 → No space left on device)；
# 优先用大本地 NVMe (/opt/dlami/nvme, 28T)，再退到 /dev/shm(1T tmpfs)，最后才 /tmp。
_JIT_BASE=""
for _cand in /opt/dlami/nvme /dev/shm /tmp; do
    if [[ -d "$_cand" && -w "$_cand" ]]; then _JIT_BASE="$_cand"; break; fi
done
_JIT_CACHE="${_JIT_BASE:-/tmp}/${USER:-u}_jit${SLURM_JOB_ID:+_$SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${_JIT_CACHE}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${_JIT_CACHE}/inductor}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${_JIT_CACHE}/tilelang}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TILELANG_CACHE_DIR" 2>/dev/null || true

# Qwen3.5 关键开关：用 megatron-core 实现的 GatedDeltaNet
# (transformers 实现不支持 packing 且 TP 不可用)
export SWIFT_USE_MCORE_GDN=1

# Qwen3.5 多模态相关（即便纯文本数据训也安全留默认）
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-128}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-12}"

# --- 实验追踪 (wandb) ---
# 上线到你的 wandb: 设 WANDB_API_KEY(或先 `wandb login`)即可 online；否则自动 offline。
# 推荐把 key 放进 ~/.wandb_key 文件(下面会自动读取)，不用每次 export。
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.wandb_key" ]]; then
    export WANDB_API_KEY="$(tr -d ' \n' < "${HOME}/.wandb_key")"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-offline}"   # 没 key 就离线写本地，可后续 wandb sync
fi
export WANDB_PROJECT="${WANDB_PROJECT:-opensima}"
export WANDB_ENTITY="${WANDB_ENTITY:-OpenSIMA}"   # 团队 entity（wandb.ai/OpenSIMA/opensima）
export WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-false}"   # 不上传 ckpt 到 wandb
export WANDB_WATCH="${WANDB_WATCH:-false}"           # 不开 watch（VLM 多图慢）

echo "[env.sh] activated: $CONDA_DEFAULT_ENV"
echo "[env.sh] PROJECT_ROOT=$PROJECT_ROOT"
echo "[env.sh] SWIFT_USE_MCORE_GDN=$SWIFT_USE_MCORE_GDN"
