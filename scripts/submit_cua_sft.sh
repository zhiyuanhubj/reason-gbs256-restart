#!/bin/bash
# 提交单独的 CUA (VideoCUA GUI) SFT。复用 genshin 同一套 Megatron-SWIFT 框架，仅换数据：
#   1) 数据 -> /fsx/sfr/zhiyuan/videocua/videocua_processed/_merged (9484 train / 128 val)
#      流式多轮 GUI 轨迹，动作 = pyautogui DSL(0-1000 整数)，system 内嵌任务指令
#   2) 1 节点: TP=4 PP=1 -> DP=2, GLOBAL_BS=64(grad_accum=32)。数据轻(均 7 帧)，1 节点足够，排队也快
#   3) 3 epoch(~445 步), lr=1e-5 constant, max_len=40960, img_tok=1024, recompute=none, MTP1
#   4) MICRO_BS=1(沿用 genshin OOM 教训), EVAL_STEPS=50, ckpt 自动传 HF
#   5) 输出到 /fsx/sfr, wandb online
# 用法:
#   bash scripts/submit_cua_sft.sh
#   NODES=2 bash scripts/submit_cua_sft.sh          # 想用 2 节点(DP4)就传 NODES=2
#   EXCLUDE_NODES=ip-10-0-155-72 bash scripts/submit_cua_sft.sh
set -uo pipefail
PROJECT_ROOT=/fsx/home/zhiyuan/game/extracted/training-main
DATA="${DATA:-/fsx/sfr/zhiyuan/videocua/videocua_processed/_merged}"
NODES="${NODES:-1}"
DEP_JOB="${DEP_JOB:-}"

cd "$PROJECT_ROOT"
ts="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="cua_sft_${NODES}n_dp$((NODES*2))_${ts}"
TRAIN_OUT=/fsx/sfr/zhiyuan/train_outputs/$RUN_NAME

dep_args=()
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
[[ -n "$EXCLUDE_NODES" ]] && { dep_args+=(--exclude="$EXCLUDE_NODES"); echo "[submit] 排除脏节点: $EXCLUDE_NODES"; }
if [[ -n "$DEP_JOB" ]]; then
    dep_args+=(--dependency="afterok:$DEP_JOB")
    echo "[submit] 将在 job $DEP_JOB 成功后自动开跑"
else
    if [[ ! -f "$DATA/train.jsonl" || ! -f "$DATA/val.jsonl" ]]; then
        echo "[ERR] 数据未就绪: $DATA/{train,val}.jsonl 不存在。"
        exit 1
    fi
fi

TRAIN_JOB=$(sbatch --parsable "${dep_args[@]}" \
  --nodes="$NODES" --time=4-00:00:00 --job-name=cua_sft \
  --export=ALL,TRAIN_FILE="$DATA/train.jsonl",VAL_FILE="$DATA/val.jsonl",GAME_NAME=cua,TP=4,PP=1,MICRO_BS=1,GLOBAL_BS=64,RECOMPUTE=none,MAX_LEN=40960,IMG_MAX_TOK=1024,EPOCHS=3,LR=1e-5,LR_DECAY_STYLE=constant,SAVE_STEPS=100,EVAL_STEPS=50,SAVE_TOTAL_LIMIT=10,NO_SAVE_OPTIM=true,NO_SAVE_RNG=true,CKPT_HF_REPO="${CKPT_HF_REPO:-WeihaoTan/cua-sft}",OUTPUT_DIR="$TRAIN_OUT",RUN_NAME="$RUN_NAME",WANDB_MODE=online,WANDB_PROJECT=opensima,WANDB_ENTITY=OpenSIMA \
  scripts/43_train_genshin_multinode.slurm)
echo "[submit] 已排队 CUA SFT: job=$TRAIN_JOB run=$RUN_NAME nodes=$NODES"
echo "[submit] 数据=$DATA  输出=$TRAIN_OUT"
echo "[submit] HF ckpt 仓库: ${CKPT_HF_REPO:-WeihaoTan/cua-sft}"
echo "[submit] wandb: https://wandb.ai/OpenSIMA/opensima  (run name=$RUN_NAME)"
echo "$TRAIN_JOB train=$RUN_NAME data=$DATA out=$TRAIN_OUT" > /fsx/sfr/zhiyuan/videocua/.train_jobid
