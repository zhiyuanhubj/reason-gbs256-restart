#!/usr/bin/env bash
# Resume 0919 GBS=256 from complete checkpoint-80 after FSx ENOSPC kill at 100.
# Load weights from FSx ckpt-80; save new ckpts on local NVMe (SAVE_TOTAL_LIMIT=0).
# Does not scancel holds. Does not touch 0916 (port 30256).
set -euo pipefail
if [[ "$(hostname)" == ip-10-1-115-182 ]]; then
    echo "[ERR] refusing on login $(hostname)" >&2
    exit 2
fi
LOCK=/fsx/home/zhiyuan/logs/resume_reason_0919_from_80.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[ERR] resume already running"; exit 0; }

specs=(
  2165:ip-10-1-25-253
  2167:ip-10-1-59-206
  2220:ip-10-1-69-108
  2216:ip-10-1-31-190
)
CKPT=/fsx/home/zhiyuan/nfs/outputs/reason0919_gbs256_20260925T083029Z/checkpoint-80
test -f "$CKPT/latest_checkpointed_iteration.txt"
test -d "$CKPT/iter_0000080"

ROOT=/opt/dlami/nvme/zhiyuan-reasoning-jsonl
RUN_NAME="reason0919_gbs256_resume80_$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/checkpoints/$RUN_NAME"
LAUNCH=/fsx/home/zhiyuan/training/scripts/launch_general_reason_gbs256.sh
SIDECAR=/fsx/home/zhiyuan/training/scripts/wandb_jsonl_sidecar.sh
LAUNCH_LOG=/fsx/home/zhiyuan/logs/${RUN_NAME}.launch.log

echo "[resume0919] $(date -u +%FT%TZ) host=$(hostname) run=$RUN_NAME"
echo "[resume0919] load=$CKPT save=$OUT GBS=256 SAVE_TOTAL_LIMIT=0"

for spec in "${specs[@]}"; do
    job=${spec%%:*}; node=${spec##*:}
    echo "[resume0919] mkdir $job $node"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=0 \
        --time=00:01:00 --nodelist="$node" bash -lc "mkdir -p '$OUT' && df -h /opt/dlami/nvme | tail -1"
done

while IFS= read -r var; do
    unset "$var"
done < <(env | awk -F= '/^SLURM_/ {print $1}')
unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER
: >"$LAUNCH_LOG"

setsid nohup env \
    HOLD_JOBS=2165,2167,2220,2216 \
    LOCAL_DATA_ROOT="$ROOT" \
    MIXTURE_YAML=/fsx/home/zhiyuan/training/data_configs/general_game_reasoning_0919_128K.yaml \
    OUTPUT_DIR="$OUT" \
    RUN_NAME="$RUN_NAME" \
    MASTER_PORT=30119 \
    GLOBAL_BS=256 \
    MICRO_BS=1 \
    TP=4 \
    MODEL_TYPE=qwen3_5 \
    FI_EFA_FORK_SAFE=1 \
    EVAL_ITERS=8 \
    SAVE_STEPS=20 \
    EVAL_STEPS=20 \
    SAVE_TOTAL_LIMIT=0 \
    STEP_TIME=6-00:00:00 \
    TRAIN_ITERS=907 \
    WANDB_ENTITY=zhiyuan-hu-bj-nus \
    WANDB_PROJECT=opensima \
    GAME_NAME=general_reason_0919 \
    FINETUNE=false \
    AUTO_RESUME=false \
    RESUME_FROM="$CKPT" \
    bash "$LAUNCH" >"$LAUNCH_LOG" 2>&1 < /dev/null &
echo $! > /fsx/home/zhiyuan/logs/reason0919_gbs256.launcher.pid
sleep 10
if kill -0 "$(cat /fsx/home/zhiyuan/logs/reason0919_gbs256.launcher.pid)" 2>/dev/null; then
    echo "[resume0919] launcher alive pid=$(cat /fsx/home/zhiyuan/logs/reason0919_gbs256.launcher.pid)"
    tail -40 "$LAUNCH_LOG" || true
else
    echo "[resume0919] FATAL launcher died"
    tail -80 "$LAUNCH_LOG" || true
    exit 1
fi

echo "$RUN_NAME" > /fsx/home/zhiyuan/logs/reason0919_gbs256_resume80.run_name
SIDECAR_LOG=/fsx/home/zhiyuan/logs/${RUN_NAME}.wandb_sidecar.log
SIDECAR_DIR=/fsx/home/zhiyuan/logs/wandb_sidecar_${RUN_NAME}_zhiyuan-hu-bj-nus
: >"$SIDECAR_LOG"
srun --jobid=2216 --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=0 \
    --time=00:02:00 --nodelist=ip-10-1-31-190 \
    bash -lc "export RUN_NAME='$RUN_NAME' JSONL='$OUT/logging.jsonl' SIDECAR_DIR='$SIDECAR_DIR' WANDB_ENTITY=zhiyuan-hu-bj-nus WANDB_PROJECT=opensima; setsid nohup bash '$SIDECAR' >>'$SIDECAR_LOG' 2>&1 < /dev/null & echo PLANTED=\$!; sleep 2; if kill -0 \$! 2>/dev/null; then echo ALIVE host=\$(hostname) pid=\$!; else echo DEAD; exit 1; fi"
echo "[resume0919] $(date -u +%FT%TZ) planted run=$RUN_NAME"
