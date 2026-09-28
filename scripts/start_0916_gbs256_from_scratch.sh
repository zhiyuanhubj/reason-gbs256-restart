#!/usr/bin/env bash
# 0916 GBS=256 from HF 4183 on 2166/2170/2168/2169. Fresh run, do not resume 920.
# Checkpoints on local NVMe (same path on each node). SAVE_TOTAL_LIMIT=0.
# Does not scancel holds. Does not touch 0919 (port 30119).
set -euo pipefail
if [[ "$(hostname)" == ip-10-1-115-182 ]]; then
    echo "[ERR] refusing on login $(hostname)" >&2
    exit 2
fi
LOCK=/fsx/home/zhiyuan/logs/start_0916_gbs256_scratch.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[ERR] scratch launch already running"; exit 0; }

specs=(
  2166:ip-10-1-30-86
  2170:ip-10-1-48-227
  2168:ip-10-1-13-41
  2169:ip-10-1-24-61
)
RUN_NAME="reason0916_gbs256_scratch_$(date -u +%Y%m%dT%H%M%SZ)"
ROOT=/opt/dlami/nvme/zhiyuan-reasoning-jsonl
OUT="$ROOT/checkpoints/$RUN_NAME"
LAUNCH=/fsx/home/zhiyuan/training/scripts/launch_general_reason_0916_4n.sh
SIDECAR=/fsx/home/zhiyuan/training/scripts/wandb_jsonl_sidecar.sh
LAUNCH_LOG=/fsx/home/zhiyuan/logs/${RUN_NAME}.launch.log
STOP=/fsx/home/zhiyuan/training/scripts/stop_0916_port30135.sh

echo "[scratch0916] $(date -u +%FT%TZ) host=$(hostname) run=$RUN_NAME"
echo "[scratch0916] output=$OUT GBS=256 SAVE_TOTAL_LIMIT=0 from 4183"

for spec in "${specs[@]}"; do
    job=${spec%%:*}; node=${spec##*:}
    echo "[scratch0916] stop leftover 30135 on $job $node"
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=0 \
        --time=00:02:00 --nodelist="$node" bash "$STOP" || true
done

for spec in "${specs[@]}"; do
    job=${spec%%:*}; node=${spec##*:}
    srun --jobid="$job" --overlap --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=0 \
        --time=00:01:00 --nodelist="$node" bash -lc "mkdir -p '$OUT' && df -h /opt/dlami/nvme | tail -1"
done

while IFS= read -r var; do
    unset "$var"
done < <(env | awk -F= '/^SLURM_/ {print $1}')
unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER
: >"$LAUNCH_LOG"

setsid nohup env \
    HOLD_JOBS=2166,2170,2168,2169 \
    LOCAL_DATA_ROOT="$ROOT" \
    MIXTURE_YAML=/fsx/home/zhiyuan/training/data_configs/general_game_reasoning_128K_0921.yaml \
    OUTPUT_DIR="$OUT" \
    RUN_NAME="$RUN_NAME" \
    MASTER_PORT=30256 \
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
    TRAIN_ITERS=0 \
    WANDB_ENTITY=zhiyuan-hu-bj-nus \
    WANDB_PROJECT=opensima \
    GAME_NAME=general_reason_0916 \
    FINETUNE=true \
    AUTO_RESUME=false \
    RESUME_FROM="" \
    bash "$LAUNCH" >"$LAUNCH_LOG" 2>&1 < /dev/null &
echo $! > /fsx/home/zhiyuan/logs/reason0916_gbs256_scratch.launcher.pid
sleep 10
if kill -0 "$(cat /fsx/home/zhiyuan/logs/reason0916_gbs256_scratch.launcher.pid)" 2>/dev/null; then
    echo "[scratch0916] launcher alive pid=$(cat /fsx/home/zhiyuan/logs/reason0916_gbs256_scratch.launcher.pid)"
    tail -30 "$LAUNCH_LOG" || true
else
    echo "[scratch0916] FATAL launcher died"
    tail -80 "$LAUNCH_LOG" || true
    exit 1
fi

setsid nohup env \
    RUN_NAME="$RUN_NAME" \
    JSONL="$OUT/logging.jsonl" \
    SIDECAR_DIR="/fsx/home/zhiyuan/logs/wandb_sidecar_${RUN_NAME}_zhiyuan-hu-bj-nus" \
    bash "$SIDECAR" >/fsx/home/zhiyuan/logs/${RUN_NAME}.wandb_sidecar.log 2>&1 < /dev/null &
echo $! > /fsx/home/zhiyuan/logs/reason0916_gbs256_scratch.sidecar.pid
echo "$RUN_NAME" > /fsx/home/zhiyuan/logs/reason0916_gbs256_scratch.run_name
echo "[scratch0916] sidecar pid=$(cat /fsx/home/zhiyuan/logs/reason0916_gbs256_scratch.sidecar.pid)"
echo "[scratch0916] $(date -u +%FT%TZ) planted run=$RUN_NAME"
