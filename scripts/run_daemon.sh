#!/usr/bin/env bash
# 把任意训练脚本变成真 daemon（setsid + nohup + 重定向 stdio）
# 不依赖 controlling terminal，SSH 断 / cursor 关 / 父 shell 死都不影响
#
# 用法:
#   bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh
#   bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh GPUS=2,3 NPROC_PER_NODE=2
#
# 之后:
#   tail -f logs/<最新>.log     # 看进度
#   pkill -f "megatron sft"     # 停训练（或 kill -TERM <pid>）
#   pgrep -af "megatron sft"    # 看是否在跑

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <script.sh> [VAR=val ...]"; exit 1
fi
SCRIPT="$1"; shift

[[ -f "$SCRIPT" ]] || { echo "[daemon] script not found: $SCRIPT"; exit 1; }

mkdir -p "${PROJECT_ROOT}/logs"
DAEMON_LOG="${PROJECT_ROOT}/logs/daemon_$(date +%Y%m%d_%H%M%S).log"

echo "[daemon] launching:  $SCRIPT  $*"
echo "[daemon] daemon_log: $DAEMON_LOG  (脚本内部的 tee 还会另存一份)"

# setsid 让进程脱离当前 session（成为新 session leader），完全 detach controlling tty
# nohup 忽略 SIGHUP（双保险）
# stdin → /dev/null  (避免被 EOF 杀)
# stdout/stderr → daemon log (脚本里的 tee 是另一份训练详细日志)
# & + disown 从 shell job table 移除
setsid nohup env "$@" bash "$SCRIPT" </dev/null >>"$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!
disown $DAEMON_PID 2>/dev/null || true

# 等 1 秒让进程稳定
sleep 1

# 验证 reparent: 新进程的 PPID 应该是 1 (init) 而不是当前 shell
PPID_OF=$(ps -o ppid= -p "$DAEMON_PID" 2>/dev/null | tr -d ' ' || echo "?")
echo "[daemon] launched PID=$DAEMON_PID  PPID=$PPID_OF  (PPID=1 表示成功 reparent 到 init)"
echo "[daemon] tail -f $DAEMON_LOG"
echo "[daemon] tail -f ${PROJECT_ROOT}/logs/<run_name>.log    # 训练详细日志"
echo "[daemon] pkill -f 'megatron sft'                         # 停训练"
