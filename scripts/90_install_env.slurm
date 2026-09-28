#!/bin/bash
# 在一个 GPU 节点上创建 megatron-sft conda 环境并安装所有依赖。
# 安装脚本(00_install_env.sh)末尾会 import fla/tilelang/transformer_engine 做校验，
# 这些需要 GPU + 驱动，所以必须在 GPU 节点上跑（登录节点无 GPU）。
# conda env 装在 /fsx/home(共享 Lustre)，装一次所有节点都能用。
# 注: 用 low-pri 账号 + ml.p5en.48xlarge-low 分区。
#   - xgen-mm 账号有 26 节点组上限且已被现有任务占满；
#   - low-pri 账号只被 *-low / *-ultra-low 分区允许(标准分区 DenyAccounts=low-pri)；
#   - -low 分区: tier 10, 无节点上限, 4 天上限。
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-low
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=megsft-install
#SBATCH --output=/fsx/home/zhiyuan/game/extracted/training-main/logs/install_%j.log

set -uo pipefail

PROJECT_ROOT=/fsx/home/zhiyuan/game/extracted/training-main
CONDA_ENV_NAME=megatron-sft

echo "=========================================================="
echo " megatron-sft 环境安装  job=$SLURM_JOB_ID  node=$(hostname)"
echo " 时间: $(date)"
echo "=========================================================="

source /fsx/home/zhiyuan/miniconda3/etc/profile.d/conda.sh

# 1) 创建 conda env（若不存在）
if ! conda env list | grep -qE "/${CONDA_ENV_NAME}$"; then
    echo "[install] 创建 conda env: ${CONDA_ENV_NAME} (python 3.10)"
    conda create -n "${CONDA_ENV_NAME}" python=3.10 -y
else
    echo "[install] conda env ${CONDA_ENV_NAME} 已存在，跳过创建"
fi

conda activate "${CONDA_ENV_NAME}"
echo "[install] active env: $CONDA_DEFAULT_ENV  python: $(which python)"

# 2) CUDA（编译 transformer_engine 用，节点上 nvcc 在 12.8）
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
nvidia-smi || true
nvcc --version || true

# 3) 跑项目安装脚本（torch/flash-attn/transformers/ms-swift/megatron-core/TE/fla/tilelang）
cd "$PROJECT_ROOT"
bash scripts/00_install_env.sh
INSTALL_RC=$?

echo "=========================================================="
echo "[install] 00_install_env.sh 返回码: ${INSTALL_RC}"

# 4) 校验环境
echo "[install] 运行 tools/verify_env.py ..."
python tools/verify_env.py || true

echo "=========================================================="
if [[ "${INSTALL_RC}" -eq 0 ]]; then
    echo "MEGSFT_INSTALL_DONE_OK"
else
    echo "MEGSFT_INSTALL_DONE_FAIL rc=${INSTALL_RC}"
fi
echo " 时间: $(date)"
echo "=========================================================="
