#!/bin/bash
# 多节点 Megatron-SWIFT 全参 SFT：Qwen3.5-9B VLM，原神 BC 数据。
#
# 默认 4 节点 × 8 H200 = 32 GPU，TP=4 PP=1 DP=8（TP 不跨节点走 NVLink；DP 跨节点走 EFA）。
# 提交:
#   sbatch scripts/43_train_genshin_multinode.slurm                       # 4 节点正式训练
#   sbatch --nodes=2 --export=ALL,TRAIN_ITERS=8,SAVE_STEPS=99999,EVAL_STEPS=99999 \
#          scripts/43_train_genshin_multinode.slurm                       # 2 节点跨节点短跑验证
#
# 账号/分区：用户已释放 xgen-mm 配额(上限26)，改用标准分区——更高优先级、不被抢占、无 2 节点上限。
#   (备选 low-pri: -low 有 max2nodes 上限只够 2 节点; -ultra-low 无上限但 tier5 最低优先级会被抢占)
#SBATCH --account=xgen-mm
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --job-name=sft
#SBATCH --output=logs/train_%j.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# ====================== 训练配置（可被 sbatch --export 覆盖）======================
# 数据 / 模型
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-9B}"
export TRAIN_FILE="${TRAIN_FILE:-/fsx/sfr/zhiyuan/yuanshen_processed_filtered/_merged/train.jsonl}"
export VAL_FILE="${VAL_FILE:-/fsx/sfr/zhiyuan/yuanshen_processed_filtered/_merged/val.jsonl}"
export GAME_NAME="${GAME_NAME:-yuanshen}"

# ----------------- 多游戏混训(可选) -----------------
# 设了 MIXTURE_YAML 就用它生成多游戏采样规格(DATASET_SPEC)并覆盖 TRAIN_FILE/VAL_FILE，
# 同时自动开启按游戏分 loss。单游戏训练留空即可(走 TRAIN_FILE)。
# 模板见 ${PROJECT_ROOT}/data_configs/mixture.example.yaml。
MIXTURE_YAML="${MIXTURE_YAML:-}"
export ENABLE_CHANNEL_LOSS="${ENABLE_CHANNEL_LOSS:-false}"
if [[ -n "$MIXTURE_YAML" ]]; then
    [[ -f "$MIXTURE_YAML" ]] || { echo "[ERR] MIXTURE_YAML 不存在: $MIXTURE_YAML"; exit 1; }
    _PY="${PYTHON_BIN:-python}"
    if [[ "${PASS_AWARE_MIXTURE:-false}" == "true" ]]; then
        [[ -n "${MIXTURE_PLAN:-}" ]] || {
            echo "[ERR] PASS_AWARE_MIXTURE=true requires MIXTURE_PLAN"
            exit 1
        }
        mkdir -p "$(dirname "$MIXTURE_PLAN")"
        export DATASET_SPEC="$("$_PY" "${PROJECT_ROOT}/scripts/lib_mixture.py" \
            "$MIXTURE_YAML" --seed "${SEED:-42}" \
            --write-plan "$MIXTURE_PLAN" --print-source-spec)"
    else
        export DATASET_SPEC="$("$_PY" "${PROJECT_ROOT}/scripts/lib_mixture.py" "$MIXTURE_YAML" --print-spec)"
    fi
    _MIX_VAL="$("$_PY" "${PROJECT_ROOT}/scripts/lib_mixture.py" "$MIXTURE_YAML" --print-val)"
    if [[ "${ENABLE_EVAL:-true}" == "true" ]]; then
        [[ -n "$_MIX_VAL" ]] && export VAL_FILE="$_MIX_VAL"
    else
        export VAL_FILE=""
    fi
    # Mixture training keeps per-channel metrics enabled by default. Diagnostics
    # can explicitly disable them without changing normal training behavior.
    export ENABLE_CHANNEL_LOSS="${MIXTURE_ENABLE_CHANNEL_LOSS:-true}"
    echo "[mixture] yaml=$MIXTURE_YAML"
    echo "[mixture] DATASET_SPEC=$DATASET_SPEC"
    echo "[mixture] VAL_FILE=$VAL_FILE"
fi

# 并行：9B(KV heads=4) → TP 上限 4；每节点 8 卡全用
export GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TP="${TP:-4}"
export PP="${PP:-1}"

# batch：DP = NNODES*8/(TP*PP)。GLOBAL_BS 必须能被 DP 整除。
# 显存利用：上次 micro=1+recompute=selective 只用了一半显存。recompute=none 已设(见下)，
# 真正吃满显存的杠杆是 MICRO_BS。先用安全的 1（自动提交不赌 OOM），dryrun 验证 micro=2
# 在 max_len=40960 下能否放下；放得下再设 MICRO_BS=2 重挂。
# DP=NNODES*8/(TP*PP)=8（4节点）。micro=1,DP=8,global=64 -> grad_accum=8。
export MICRO_BS="${MICRO_BS:-1}"
export GLOBAL_BS="${GLOBAL_BS:-64}"

# 序列 / 视觉：数据每条 30 帧（务必让 MAX_LEN >= 1700 + 30×(IMG_MAX_TOK+40)）
export IMG_MAX_TOK="${IMG_MAX_TOK:-16384}"
export MAX_LEN="${MAX_LEN:-40960}"

# 超参
if [[ -n "$MIXTURE_YAML" ]]; then
    export EPOCHS="${EPOCHS:-1}"  # passes 已经是整个训练任务的总倍率
else
    export EPOCHS="${EPOCHS:-3}"
fi
export LR="${LR:-1e-5}"
export TRAIN_ITERS="${TRAIN_ITERS:-0}"   # >0 时短跑(dryrun)
export RECOMPUTE="${RECOMPUTE:-none}"    # H200 显存有富余时优先关重算提速；OOM 再退 selective/full
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"

# ckpt / log
export SAVE_STEPS="${SAVE_STEPS:-100}"
export EVAL_STEPS="${EVAL_STEPS:-100}"

# 共享 run_name（所有节点必须一致，否则各写各的输出目录）
export RUN_NAME="${RUN_NAME:-sft_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID}}"

# wandb：未登录 → offline（本地存 wandb/，可后续 `wandb sync`）。登录后可传 WANDB_MODE=online。
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-opensima}"
export WANDB_ENTITY="${WANDB_ENTITY:-OpenSIMA}"

# ====================== 分布式 rendezvous ======================
ALLOC_NODELIST="${ALLOC_NODELIST:-${SLURM_JOB_NODELIST}}"
JOB_NNODES="${NNODES:-${SLURM_NNODES}}"
nodes=( $(scontrol show hostnames "$ALLOC_NODELIST") )
if (( ${#nodes[@]} < JOB_NNODES )); then
    echo "[ERR] requested NNODES=$JOB_NNODES but allocation only exposes: ${nodes[*]}"
    exit 1
fi
head=${nodes[0]}
# 解析 head 节点【IPv4】——不要用嵌套 srun(在某些节点会 socket 超时挂住)。
# 注意必须 IPv4: getent hosts 可能先返回 IPv6 link-local(fe80::)，跨节点无法路由会挂。
# 用 ahostsv4 强制 IPv4；兜底从 AWS 节点名(ip-10-0-140-50 → 10.0.140.50)解析。
MASTER_ADDR="$(getent ahostsv4 "$head" 2>/dev/null | awk '{print $1; exit}')"
if [[ -z "$MASTER_ADDR" || "$MASTER_ADDR" == fe80* ]]; then
    MASTER_ADDR="$(echo "$head" | sed -E 's/^ip-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+).*/\1.\2.\3.\4/')"
fi
export MASTER_ADDR
export MASTER_PORT="$(( 20000 + SLURM_JOB_ID % 20000 ))"
TOTAL_GPUS=$(( JOB_NNODES * NPROC_PER_NODE ))
DP=$(( TOTAL_GPUS / (TP * PP) ))

echo "============================================================"
echo " MULTI-NODE SFT   job=$SLURM_JOB_ID"
echo "   nodes(${JOB_NNODES})=${nodes[*]}"
echo "   head=$head  MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "   model=$MODEL_PATH"
echo "   data=$TRAIN_FILE"
echo "   parallel: TP=$TP PP=$PP DP=$DP  (WORLD_SIZE=$TOTAL_GPUS)"
echo "   global_bs=$GLOBAL_BS micro=$MICRO_BS  max_len=$MAX_LEN img_tok=$IMG_MAX_TOK"
echo "   epochs=$EPOCHS train_iters=$TRAIN_ITERS  run=$RUN_NAME"
echo "   wandb_mode=$WANDB_MODE"
echo "============================================================"

if (( GLOBAL_BS % DP != 0 )); then
    echo "[WARN] GLOBAL_BS=$GLOBAL_BS 不能被 DP=$DP 整除，megatron 可能报错/调整。"
fi

if [[ "${PREFLIGHT_ONLY:-false}" == "true" ]]; then
    echo "[preflight] topology, model, and dataset configuration validated; training not started"
    exit 0
fi

# ====================== ckpt 上传 watcher (同 job, 跑在 head 节点空闲 CPU, 不占额外节点) ======================
WPID=""
if [[ -n "${CKPT_HF_REPO:-}" ]]; then
    mkdir -p "$OUTPUT_DIR"   # 41内层脚本稍后也会建, 这里先建好供watcher日志重定向
    echo "[ckpt-watcher] 在 head 节点后台启动: $OUTPUT_DIR -> HF $CKPT_HF_REPO"
    RUN_DIR="$OUTPUT_DIR" HF_REPO="$CKPT_HF_REPO" POLL_SEC="${CKPT_POLL_SEC:-300}" \
        bash "${PROJECT_ROOT}/scripts/ckpt_upload_watcher.sh" >> "$OUTPUT_DIR/ckpt_watcher.out" 2>&1 &
    WPID=$!
fi

# ====================== 启动（每节点 1 task）======================
# The original AWS nodes have eight GPUs; current Isambard nodes have four.
GPUS_PER_NODE="${GPUS_PER_NODE:-${NPROC_PER_NODE}}"
CPUS_PER_NODE_TASK="${CPUS_PER_NODE_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
_SRUN_ALLOCATION_ARGS=()
# This override lets a launcher running on a login node attach directly to an
# existing allocation. A normal sbatch launch leaves it unset.
if [[ -n "${SRUN_JOB_ID_OVERRIDE:-}" ]]; then
    _SRUN_ALLOCATION_ARGS+=(--jobid="${SRUN_JOB_ID_OVERRIDE}" --overlap)
fi
srun "${_SRUN_ALLOCATION_ARGS[@]}" \
     --nodes="${JOB_NNODES}" --ntasks="${JOB_NNODES}" --ntasks-per-node=1 \
     --cpus-per-task="${CPUS_PER_NODE_TASK}" \
     --gres="gpu:${GPUS_PER_NODE}" \
     bash "${PROJECT_ROOT}/scripts/43_train_inner.sh"
TRAIN_RC=$?

# 训练结束: 让 watcher 再扫一轮上传最后的 ckpt, 然后收尾
if [[ -n "$WPID" ]]; then
    echo "[ckpt-watcher] 训练结束(rc=$TRAIN_RC), 最后扫一轮上传剩余 ckpt"
    sleep 60
    kill "$WPID" 2>/dev/null || true
fi
exit $TRAIN_RC
