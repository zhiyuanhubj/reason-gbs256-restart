#!/usr/bin/env bash
# Convert 0919 history15 reasoning parquet -> jsonl+images, then 07 merge/val.
# Output lives next to 0916 polish jsonl but in *_reasoning_history15_0919 dirs.
set -euo pipefail
if [[ "$(hostname)" == ip-10-1-115-182 ]]; then
    echo "[ERR] refusing jsonl convert on login $(hostname)" >&2
    exit 2
fi
PYTHON="${PYTHON:-/fsx/home/zhiyuan/miniconda3/envs/megatron-sft/bin/python}"
GAME="${GAME:?genshin|cyberpunk2077|spiderman2}"
INPUT="${INPUT:?parquet directory}"
OUT_ROOT="${OUT_ROOT:-/opt/dlami/nvme/zhiyuan-reasoning-jsonl}"
WORKERS="${WORKERS:-8}"

test -d "$INPUT"
mkdir -p "$OUT_ROOT"
game_dir="$OUT_ROOT/${GAME}_reasoning_history15_0919"
pack="$OUT_ROOT/.pack_${GAME}_0919"
rm -rf "$pack"
mkdir -p "$pack/$GAME"

echo "[convert0919] host=$(hostname) game=$GAME input=$INPUT time=$(date -u +%FT%TZ)"
df -h /opt/dlami/nvme | tail -1
echo "[convert0919] parquet=$(ls "$INPUT"/*.parquet 2>/dev/null | wc -l)"

"$PYTHON" /fsx/home/zhiyuan/training/scripts/06_prepare_bc_dataset.py \
    --input_dir "$INPUT" \
    --output_dir "$pack/$GAME" \
    --val_chunks 0 \
    --num_workers "$WORKERS" \
    --abs_paths \
    --game "$GAME"

"$PYTHON" /fsx/home/zhiyuan/training/scripts/07_merge_shuffle_genshin.py \
    --input_dir "$pack" \
    --output_dir "$game_dir/_merged" \
    --val_size 64 \
    --seed 42 \
    --check_images

if [[ -d "$pack/$GAME/images" && ! -e "$game_dir/images" ]]; then
    mv "$pack/$GAME/images" "$game_dir/images"
fi
if [[ -d "$game_dir/images" ]]; then
    old="$pack/$GAME/images"
    new="$game_dir/images"
    "$PYTHON" - "$old" "$new" "$game_dir/_merged" <<'PY'
import sys
from pathlib import Path
old, new, merged = sys.argv[1], sys.argv[2], Path(sys.argv[3])
for name in ("train.jsonl", "val.jsonl"):
    path = merged / name
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"rewrote {path} {old} -> {new}", flush=True)
PY
fi
rm -rf "$pack"
echo "[convert0919] COMPLETE game=$GAME out=$game_dir/_merged"
wc -l "$game_dir/_merged/train.jsonl" "$game_dir/_merged/val.jsonl"
df -h /opt/dlami/nvme | tail -1
