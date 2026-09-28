#!/bin/bash
# 提交 8 节点 noprivate 训练 + ckpt 上传HF watcher。
# 复刻当前 8n 配置(args.json): TP=4 PP=1 -> DP=16, GLOBAL_BS=128, MICRO_BS=2(grad_accum=4),
#   3 epoch, lr=1e-5 constant, max_len=40960, img_tok=1024, recompute=none, MTP1。
# 本次改动:
#   1) 数据 -> yuanshen_noprivate_processed(无私服 + 去钟德智 + 保留大鼠标位移 + 修复损坏session)
#   2) 可无损重启: NO_SAVE_OPTIM/RNG=false(存优化器+RNG)
#   3) ckpt 存完整 safetensors, 起 watcher 传到 HF(WeihaoTan)并在确认传完后删本地
#   4) save/eval 频率调疏(500), save_total_limit 调大(交给 watcher 管理删除)
# 用法: bash scripts/submit_gs_8n_noprivate.sh
set -uo pipefail
PROJECT_ROOT=/fsx/home/zhiyuan/game/extracted/training-main
NEW_DATA="${NEW_DATA:-/fsx/sfr/zhiyuan/yuanshen_noprivate_processed/_merged}"
HF_REPO="${HF_REPO:-WeihaoTan/yuanshen-bc-9b-noprivate}"

if [[ ! -f "$NEW_DATA/train.jsonl" || ! -f "$NEW_DATA/val.jsonl" ]]; then
    echo "[ERR] 数据还没准备好: $NEW_DATA/{train,val}.jsonl 不存在; 等数据处理(merge)跑完再来。"
    exit 1
fi

cd "$PROJECT_ROOT"
ts="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="bcft_gs_8n_dp16_noprivate_${ts}"

TRAIN_JOB=$(sbatch --parsable --nodes=8 --time=4-00:00:00 --job-name=bcft_gs_8n_nopriv \
  --export=ALL,TRAIN_FILE="$NEW_DATA/train.jsonl",VAL_FILE="$NEW_DATA/val.jsonl",GAME_NAME=gs,TP=4,PP=1,MICRO_BS=2,GLOBAL_BS=128,RECOMPUTE=none,MAX_LEN=40960,IMG_MAX_TOK=1024,EPOCHS=3,LR=1e-5,LR_DECAY_STYLE=constant,SAVE_STEPS=500,EVAL_STEPS=500,SAVE_TOTAL_LIMIT=100,NO_SAVE_OPTIM=false,NO_SAVE_RNG=false,RUN_NAME="$RUN_NAME",WANDB_MODE=online,WANDB_PROJECT=opensima,WANDB_ENTITY=OpenSIMA \
  scripts/43_train_genshin_multinode.slurm)
echo "[submit] 已排队 8 节点训练: job=$TRAIN_JOB run=$RUN_NAME (可重启ckpt: 存优化器+RNG)"

WJOB=$(sbatch --parsable \
  --export=ALL,RUN_DIR="$PROJECT_ROOT/outputs/$RUN_NAME",HF_REPO="$HF_REPO",TRAIN_JOB="$TRAIN_JOB" \
  scripts/watcher_job.sbatch)
echo "[submit] ckpt 上传 watcher: job=$WJOB  RUN_DIR=$PROJECT_ROOT/outputs/$RUN_NAME -> HF $HF_REPO"
echo "$TRAIN_JOB train=$RUN_NAME watcher=$WJOB repo=$HF_REPO" > /fsx/sfr/zhiyuan/data/_regen/train_jobids.txt
echo "[submit] data=$NEW_DATA"
echo "[submit] 训练凑齐8节点后自动开跑; watcher 自动传ckpt并传完删本地。"
