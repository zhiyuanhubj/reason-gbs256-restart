#!/bin/bash
# 提交 2 节点【私服】训练。复刻 noprivate 的超参, 仅:
#   1) 数据 -> yuanshen_private_processed(只私服 + 去钟德智, 同样过滤参数)
#   2) 2 节点: TP=4 PP=1 -> DP=4, GLOBAL_BS=64(grad_accum=8), 其余同 noprivate
#   3) 3 epoch, lr=1e-5 constant, max_len=40960, img_tok=1024, recompute=none, MTP1
#   4) 输出到 /fsx/sfr(绕开 HOME 配额), wandb online
# 用法:
#   bash scripts/submit_gs_2n_private.sh              # 立即提交(数据须已就绪)
#   DEP_JOB=16986 bash scripts/submit_gs_2n_private.sh # 等数据处理 job afterok 后自动开跑
set -uo pipefail
PROJECT_ROOT=/fsx/home/zhiyuan/game/extracted/training-main
DATA="${DATA:-/fsx/sfr/zhiyuan/yuanshen_private_processed/_merged}"
DEP_JOB="${DEP_JOB:-}"

cd "$PROJECT_ROOT"
ts="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="bcft_gs_2n_dp4_private_${ts}"
TRAIN_OUT=/fsx/sfr/zhiyuan/train_outputs/$RUN_NAME

dep_args=()
# 脏节点(残留 orphan vLLM 容器占 ~128GiB/GPU)规避: EXCLUDE_NODES=ip-10-0-155-72[,...]
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
[[ -n "$EXCLUDE_NODES" ]] && { dep_args+=(--exclude="$EXCLUDE_NODES"); echo "[submit] 排除脏节点: $EXCLUDE_NODES"; }
if [[ -n "$DEP_JOB" ]]; then
    dep_args+=(--dependency="afterok:$DEP_JOB")
    echo "[submit] 将在数据处理 job $DEP_JOB 成功后自动开跑"
else
    if [[ ! -f "$DATA/train.jsonl" || ! -f "$DATA/val.jsonl" ]]; then
        echo "[ERR] 数据还没就绪: $DATA/{train,val}.jsonl 不存在; 等处理(merge)跑完或用 DEP_JOB=<jobid>。"
        exit 1
    fi
fi

TRAIN_JOB=$(sbatch --parsable "${dep_args[@]}" \
  --nodes=2 --time=4-00:00:00 --job-name=bcft_gs_2n_priv \
  --export=ALL,TRAIN_FILE="$DATA/train.jsonl",VAL_FILE="$DATA/val.jsonl",GAME_NAME=gs,TP=4,PP=1,MICRO_BS=1,GLOBAL_BS=64,RECOMPUTE=none,MAX_LEN=40960,IMG_MAX_TOK=1024,EPOCHS=3,LR=1e-5,LR_DECAY_STYLE=constant,SAVE_STEPS=500,EVAL_STEPS=50,SAVE_TOTAL_LIMIT=10,NO_SAVE_OPTIM=true,NO_SAVE_RNG=true,CKPT_HF_REPO="${CKPT_HF_REPO:-WeihaoTan/genshi-private}",OUTPUT_DIR="$TRAIN_OUT",RUN_NAME="$RUN_NAME",WANDB_MODE=online,WANDB_PROJECT=opensima,WANDB_ENTITY=OpenSIMA \
  scripts/43_train_genshin_multinode.slurm)
echo "[submit] 已排队 2 节点私服训练: job=$TRAIN_JOB run=$RUN_NAME"
echo "[submit] 数据=$DATA  输出=$TRAIN_OUT"
echo "[submit] wandb: https://wandb.ai/OpenSIMA/opensima  (run name=$RUN_NAME)"
echo "$TRAIN_JOB train=$RUN_NAME data=$DATA out=$TRAIN_OUT" > /fsx/sfr/zhiyuan/yuanshen_private_processed/.train_jobid
