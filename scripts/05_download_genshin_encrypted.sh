#!/usr/bin/env bash
# Download the encrypted Genshin split archive to node-local storage.
set -euo pipefail

REPO_ID="${REPO_ID:-thomaslee1818/yuanshen-bc-formatted-sl135-encrypted}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:?Set LOCAL_DATA_ROOT to node-local NVMe}"
DESTINATION="${DESTINATION:-${LOCAL_DATA_ROOT}/genshin-encrypted}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-16}"

command -v hf >/dev/null 2>&1 || {
    echo "[ERR] Hugging Face CLI is missing: pip install -U huggingface_hub"
    exit 1
}
mkdir -p "$DESTINATION"

export HF_HOME="${HF_HOME:-${LOCAL_DATA_ROOT}/hf-home}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

echo "[download] repo=$REPO_ID"
echo "[download] destination=$DESTINATION"
hf download "$REPO_ID" \
    --repo-type dataset \
    --local-dir "$DESTINATION" \
    --max-workers "$HF_MAX_WORKERS"

for required in archive_manifest.json SHA256SUMS; do
    [[ -s "$DESTINATION/$required" ]] || {
        echo "[ERR] downloaded archive is missing $required"
        exit 1
    }
done
echo "[download] complete: $DESTINATION"
