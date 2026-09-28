#!/bin/bash
# 由 43_train_genshin_multinode.slurm 的 srun 在【每个节点】上调用。
# 职责: 激活 conda env + 配置跨节点 NCCL/EFA + 设置本节点分布式 rank, 再调用 41 训练脚本。
# 训练配置(MODEL_PATH/TP/GLOBAL_BS/MASTER_ADDR/...)由外层 sbatch export, srun(--export=ALL 默认)透传。
set -euo pipefail

# Optional per-job scratch override.  The benchmark launcher uses this because
# the cluster-provided TMPDIR can point at a missing /local/user/<uid> path.
# Create it independently on every allocated node before nvcc/TileLang starts.
if [[ -n "${JOB_LOCAL_TMPDIR:-}" ]]; then
    mkdir -p "$JOB_LOCAL_TMPDIR"
    export TMPDIR="$JOB_LOCAL_TMPDIR"
    export TMP="$JOB_LOCAL_TMPDIR"
    export TEMP="$JOB_LOCAL_TMPDIR"
fi

# Load the destination cluster's CUDA/NCCL stack if the outer launcher asks
# for it. This is required on Cray/Isambard and is a no-op on the old cluster.
if [[ -n "${ENV_SETUP_MODULES:-}" && "${ENV_SETUP_MODULES}" != "none" ]]; then
    # shellcheck disable=SC1090
    source /etc/profile.d/modules.sh 2>/dev/null || true
    read -r -a _ENV_MODULES <<<"$ENV_SETUP_MODULES"
    module load "${_ENV_MODULES[@]}"
fi

CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || true)}"
[[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || {
    echo "[ERR] Cannot find conda initialization; set CONDA_BASE"
    exit 1
}
# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
set +u
conda activate "${CONDA_ENV_NAME:-megatron-sft}"
set -u

# Conda compiler packages overwrite CC/CXX from their activate.d hooks.  Allow
# cluster launchers to repin a CUDA-compatible host compiler after activation
# (needed by CUDA 12.6 + TileLang on the Isambard GH200 image).
if [[ -n "${CUDA_HOST_CC:-}" || -n "${CUDA_HOST_CXX:-}" ]]; then
    export CC="${CUDA_HOST_CC:-${CC:-cc}}"
    export CXX="${CUDA_HOST_CXX:-${CXX:-c++}}"
    [[ -x "$CC" ]] || { echo "[ERR] CUDA host C compiler is not executable: $CC"; exit 1; }
    [[ -x "$CXX" ]] || { echo "[ERR] CUDA host C++ compiler is not executable: $CXX"; exit 1; }
    # The Conda nvcc wrapper also reads this variable; keep it consistent with
    # TileLang's explicit -ccbin argument.
    export NVCC_PREPEND_FLAGS="-ccbin=${CXX}"
fi

# ---------- 跨节点 NCCL / EFA (p5en 有 EFA + aws-ofi-nccl 插件) ----------
# 把 EFA libfabric + ofi-nccl 网络插件加进 LD_LIBRARY_PATH, 让 NCCL 走 EFA RDMA。
# 若插件加载失败 NCCL 会回退到 socket(走下面探测到的网卡), 慢但不影响正确性。
if [[ -d /opt/amazon/efa ]]; then
    export LD_LIBRARY_PATH="/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}"
    export FI_PROVIDER="${FI_PROVIDER:-efa}"
    export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
fi
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# bootstrap/GLOO 用的主网卡(取默认路由网卡)
IFACE="$(awk '$2=="00000000"{print $1; exit}' /proc/net/route 2>/dev/null)"
if [[ -z "$IFACE" ]]; then
    for n in /sys/class/net/*; do [[ -e "$n/device" ]] && { IFACE="$(basename "$n")"; break; }; done
fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$IFACE}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$IFACE}"

# ---------- 分布式 rank ----------
# Explicit values allow one coordinated run to span several existing one-node
# allocations. A normal multi-node sbatch still falls back to Slurm's values.
export NNODES="${NNODES:-${SLURM_NNODES}}"
export NODE_RANK="${NODE_RANK:-${SLURM_NODEID}}"
export IMG_MAX_TOK="${IMG_MAX_TOK:-16384}"

echo "[inner] host=$(hostname) NODE_RANK=${NODE_RANK}/${NNODES} MASTER=${MASTER_ADDR}:${MASTER_PORT} iface=${NCCL_SOCKET_IFNAME} env=${CONDA_DEFAULT_ENV} CXX=${CXX:-unset}"

exec bash "${PROJECT_ROOT}/scripts/41_train_bc_full_megatron.sh"
