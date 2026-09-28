#!/usr/bin/env bash
# Concat three per-game val.jsonl into val_mixed.jsonl on THIS node's NVMe.
# Does not touch train.jsonl. Atomic replace of val_mixed.jsonl only.
set -euo pipefail
if [[ "$(hostname)" == ip-10-1-115-182 ]]; then
    echo "[ERR] refusing concat on login $(hostname)" >&2
    exit 2
fi
ROOT="${ROOT:-/opt/dlami/nvme/zhiyuan-reasoning-jsonl}"
OUT="${OUT:-/fsx/home/zhiyuan/logs/val_0916_concat.txt}"
G="$ROOT/genshin_polish_reasoning_0916/_merged/val.jsonl"
C="$ROOT/cyberpunk2077_polish_reasoning_0916/_merged/val.jsonl"
S="$ROOT/spiderman2_polish_reasoning_0916/_merged/val.jsonl"
MIX="$ROOT/val_mixed.jsonl"
TMP="$ROOT/val_mixed.jsonl.tmp.$$"
{
  echo "==== $(date -u '+%F %T UTC') host=$(hostname) job=${SLURM_JOB_ID:-none} ===="
  busy=0
  if pgrep -af 'torch.distributed.run|_megatron/sft.py' | grep -v pgrep >/tmp/val_0916_pgrep.$$ 2>/dev/null; then
    if [[ -s /tmp/val_0916_pgrep.$$ ]]; then
      busy=1
      echo TRAINING_BUSY=1
      head -3 /tmp/val_0916_pgrep.$$
    fi
  fi
  rm -f /tmp/val_0916_pgrep.$$
  echo TRAINING_BUSY=$busy
  for f in "$G" "$C" "$S"; do
    if [[ ! -s "$f" ]]; then
      echo "MISSING_OR_EMPTY $f"
      echo CONCAT_SKIP
      exit 3
    fi
    echo "SRC $(wc -l < "$f") $f"
  done
  cat "$G" "$C" "$S" > "$TMP"
  n=$(wc -l < "$TMP")
  echo "TMP_LINES $n $TMP"
  if [[ "$n" -ne 192 ]]; then
    echo "ERR expected 192 lines got $n"
    rm -f "$TMP"
    exit 4
  fi
  mv -f "$TMP" "$MIX"
  echo "WROTE $(wc -l < "$MIX") bytes=$(stat -c%s "$MIX") $MIX"
  echo CONCAT_OK
} >"$OUT" 2>&1
echo CONCAT_WRITTEN "$OUT"
