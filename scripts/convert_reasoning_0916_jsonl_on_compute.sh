#!/usr/bin/env bash
# Convert one 0916 reasoning parquet dir to jsonl+images, then 07 merge/val.
# Must run on a compute node.
set -euo pipefail
if [[ "$(hostname)" == ip-10-1-115-182 ]]; then
    echo "[ERR] refusing jsonl convert on login $(hostname)" >&2
    exit 2
fi
PYTHON="${PYTHON:-/fsx/home/zhiyuan/miniconda3/envs/megatron-sft/bin/python}"
GAME="${GAME:?genshin|cyberpunk2077|spiderman2}"
INPUT="${INPUT:?parquet directory}"
OUT_ROOT="${OUT_ROOT:-/opt/dlami/nvme/zhiyuan/reasoning-jsonl}"
WORKERS="${WORKERS:-16}"

test -d "$INPUT"
mkdir -p "$OUT_ROOT"
game_dir="$OUT_ROOT/${GAME}_polish_reasoning_0916"
pack="$OUT_ROOT/.pack_${GAME}"
rm -rf "$pack"
mkdir -p "$pack/$GAME"

echo "[convert] host=$(hostname) game=$GAME input=$INPUT time=$(date -u +%FT%TZ)"
df -h /opt/dlami/nvme | tail -1

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

# Keep images next to merged jsonl; 06 wrote them under pack/$GAME/images
if [[ -d "$pack/$GAME/images" && ! -e "$game_dir/images" ]]; then
    mv "$pack/$GAME/images" "$game_dir/images"
fi
# jsonl already has absolute paths into pack/$GAME/images — rewrite to new location
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
echo "[convert] COMPLETE game=$GAME out=$game_dir/_merged"
wc -l "$game_dir/_merged/train.jsonl" "$game_dir/_merged/val.jsonl"
df -h /opt/dlami/nvme | tail -1
