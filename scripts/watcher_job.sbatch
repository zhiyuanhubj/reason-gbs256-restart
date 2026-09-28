#!/usr/bin/env bash
# ckpt 上传 watcher 作为轻量作业跑(low-pri 1核)。ultra-low 上限1天,
# 到点前若训练作业还在/还有未传ckpt, 自动 resubmit 自己续命。
#SBATCH --account=low-pri
#SBATCH --partition=ml.p5en.48xlarge-ultra-low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=23:30:00
#SBATCH --job-name=ckpt-upload
#SBATCH --output=/fsx/sfr/zhiyuan/data/_regen/logs/watcher_%j.log

set -uo pipefail
: "${RUN_DIR:?}"; : "${HF_REPO:?}"; : "${TRAIN_JOB:?}"
SCRIPTS=/fsx/home/zhiyuan/game/extracted/training-main/scripts

# 后台跑 watcher
RUN_DIR="$RUN_DIR" HF_REPO="$HF_REPO" POLL_SEC="${POLL_SEC:-300}" \
    bash "$SCRIPTS/ckpt_upload_watcher.sh" &
WPID=$!

# 到 ~23h 时收尾: 若训练作业仍在 -> resubmit 续命
END=$(( $(date +%s) + 23*3600 + 1500 ))
while kill -0 "$WPID" 2>/dev/null; do
    sleep 300
    [ "$(date +%s)" -ge "$END" ] && break
    # 训练作业已结束 且 无未传 ckpt -> 让 watcher 再扫一轮后退出
    if ! squeue -j "$TRAIN_JOB" -h >/dev/null 2>&1 || [ -z "$(squeue -j "$TRAIN_JOB" -h 2>/dev/null)" ]; then
        echo "[watcher_job] 训练 $TRAIN_JOB 已结束, 最后扫一轮后退出"
        sleep 600; kill "$WPID" 2>/dev/null; break
    fi
done

# 还在训练 -> 续命
if squeue -j "$TRAIN_JOB" -h 2>/dev/null | grep -q .; then
    kill "$WPID" 2>/dev/null
    echo "[watcher_job] 训练仍在, resubmit watcher 续命"
    RUN_DIR="$RUN_DIR" HF_REPO="$HF_REPO" TRAIN_JOB="$TRAIN_JOB" \
        sbatch --export=ALL,RUN_DIR="$RUN_DIR",HF_REPO="$HF_REPO",TRAIN_JOB="$TRAIN_JOB" "$SCRIPTS/watcher_job.sbatch"
fi
echo "[watcher_job] done $(date)"
