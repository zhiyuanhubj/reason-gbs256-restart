#!/usr/bin/env bash
# Run mcore -> HuggingFace conversion on an already allocated Slurm node.
#
# From a login node:
#   EXISTING_JOB_ID=<job-id> \
#     bash scripts/99_convert_mcore_to_hf_existing_node.sh <mcore_ckpt_dir>
#
# From a shell/step that is already running on the compute node:
#   bash scripts/99_convert_mcore_to_hf_existing_node.sh <mcore_ckpt_dir>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CONVERT_SCRIPT="${SCRIPT_DIR}/99_convert_mcore_to_hf.sh"
MCORE_DIR="${1:?用法: EXISTING_JOB_ID=<job-id> bash scripts/99_convert_mcore_to_hf_existing_node.sh <mcore_ckpt_dir>}"

# A non-interactive srun does not necessarily inherit a shell with conda on
# PATH.  These are the validated Isambard locations and remain overridable.
export CONDA_BASE="${CONDA_BASE:-/home/b6db/hongnanvideo1.b6db/.miniconda3}"
export CUDA_HOME="${CUDA_HOME:-/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/targets/sbsa-linux}"

if [[ ! -x "${CONVERT_SCRIPT}" ]]; then
    echo "[ERR] 转换脚本不存在或不可执行: ${CONVERT_SCRIPT}" >&2
    exit 1
fi

# SLURMD_NODENAME is set for commands that are already executing on a compute
# node.  The private flag prevents recursion after this launcher starts srun.
if [[ -n "${SLURMD_NODENAME:-}" || "${_MCORE_CONVERT_IN_STEP:-0}" == "1" ]]; then
    echo "[convert-launcher] node=${SLURMD_NODENAME:-$(hostname)} job=${SLURM_JOB_ID:-unknown}"
    exec bash "${CONVERT_SCRIPT}" "${MCORE_DIR}"
fi

JOB_ID="${EXISTING_JOB_ID:-${SLURM_JOB_ID:-}}"
if [[ -z "${JOB_ID}" ]]; then
    echo "[ERR] 当前不在计算节点上，请设置 EXISTING_JOB_ID=<running-job-id>" >&2
    exit 1
fi

JOB_INFO="$(scontrol show job "${JOB_ID}" -o 2>/dev/null)" || {
    echo "[ERR] 无法查询 Slurm 作业: ${JOB_ID}" >&2
    exit 1
}

job_field() {
    local key="$1" field
    for field in ${JOB_INFO}; do
        if [[ "${field}" == "${key}="* ]]; then
            printf '%s\n' "${field#*=}"
            return 0
        fi
    done
    return 1
}

JOB_STATE="$(job_field JobState)"
if [[ "${JOB_STATE}" != "RUNNING" ]]; then
    echo "[ERR] Slurm 作业 ${JOB_ID} 状态为 ${JOB_STATE}，不是 RUNNING" >&2
    exit 1
fi

NODE_LIST="$(job_field NodeList)"
CPUS_PER_TASK="${CPUS_PER_TASK:-$(job_field CPUs/Task)}"
if [[ -z "${CPUS_PER_TASK}" || "${CPUS_PER_TASK}" == "N/A" ]]; then
    CPUS_PER_TASK=1
fi

echo "[convert-launcher] attach job=${JOB_ID} node=${NODE_LIST} cpus=${CPUS_PER_TASK} gpus=4"
exec srun \
    --jobid="${JOB_ID}" \
    --overlap \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gres=gpu:4 \
    --export="ALL,_MCORE_CONVERT_IN_STEP=1" \
    bash "${BASH_SOURCE[0]}" "${MCORE_DIR}"
