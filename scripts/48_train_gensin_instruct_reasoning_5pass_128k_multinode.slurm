#!/bin/bash
# Genshin instruction + reasoning, five exact passes per dataset, 128K
# full-parameter SFT on 2 nodes x 4 GH200 GPUs (8 GPUs total).  The default
# initialization is the merged HF export of Genshin+LLaVA checkpoint-700.
#
# Fresh allocation:
#   sbatch scripts/48_train_gensin_instruct_reasoning_5pass_128k_multinode.slurm
#
# Existing allocation from a login node:
#   EXISTING_JOB_ID=6158395 GLOBAL_BS=8 \
#     bash scripts/48_train_gensin_instruct_reasoning_5pass_128k_multinode.slurm
#
# Source datasets (train/val JSONL plus extracted images) default to:
#   /projects/b6db/opensima/gensin-instruct-tars
#   /projects/b6db/opensima/gi_reasoning_0718-tars
# The source files are never modified.  At launch, this script creates a
# small absolute-image-path JSONL view under genshin-training/ so both sources
# can coexist even though each uses relative images/chunk_xxx paths.
#SBATCH --account=brics.b6db
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --time=1-00:00:00
#SBATCH --job-name=gi-ins-reason-5p
#SBATCH --output=logs/gi_ins_reason_5pass_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# sbatch runs a spooled copy under /var/spool/slurmd, so BASH_SOURCE no longer
# points at this repository.  Fall back to the directory from which sbatch was
# invoked when the sibling launcher is not present beside the executing copy.
if [[ ! -f "${SCRIPT_DIR}/46_train_genshin_llava_onevision_128k_multinode.slurm" \
      && -n "${SLURM_SUBMIT_DIR:-}" \
      && -f "${SLURM_SUBMIT_DIR}/scripts/46_train_genshin_llava_onevision_128k_multinode.slurm" ]]; then
    SCRIPT_DIR="$(cd "${SLURM_SUBMIT_DIR}/scripts" && pwd)"
fi
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export ASSET_ROOT="${ASSET_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"

# Attach directly to the complete existing allocation.  Entering a one-node
# srun first would prevent the launcher from spanning both nodes.
if [[ -n "${EXISTING_JOB_ID:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" && "$SLURM_JOB_ID" != "$EXISTING_JOB_ID" ]]; then
        echo "[ERR] already inside Slurm job $SLURM_JOB_ID; refusing to attach job $EXISTING_JOB_ID"
        exit 1
    fi
    _JOB_INFO="$(scontrol show job "$EXISTING_JOB_ID" -o 2>/dev/null)" || {
        echo "[ERR] cannot query existing Slurm job: $EXISTING_JOB_ID"
        exit 1
    }
    _job_field() {
        local _key="$1" _field
        for _field in $_JOB_INFO; do
            if [[ "$_field" == "${_key}="* ]]; then
                printf '%s\n' "${_field#*=}"
                return 0
            fi
        done
        return 1
    }
    _JOB_STATE="$(_job_field JobState)"
    [[ "$_JOB_STATE" == "RUNNING" ]] || {
        echo "[ERR] existing Slurm job $EXISTING_JOB_ID is $_JOB_STATE, not RUNNING"
        exit 1
    }
    export SLURM_JOB_ID="$EXISTING_JOB_ID"
    export SLURM_JOB_NODELIST="$(_job_field NodeList)"
    export SLURM_NNODES="$(_job_field NumNodes)"
    export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-$(_job_field CPUs/Task)}"
    export NNODES="${NNODES:-$SLURM_NNODES}"
    export ALLOC_NODELIST="${ALLOC_NODELIST:-$SLURM_JOB_NODELIST}"
    export SRUN_JOB_ID_OVERRIDE="$EXISTING_JOB_ID"
    echo "[attach] job=$SLURM_JOB_ID nodes=$SLURM_JOB_NODELIST nnodes=$SLURM_NNODES cpus/task=$SLURM_CPUS_PER_TASK"
elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[ERR] not inside a Slurm allocation; use sbatch or set EXISTING_JOB_ID=<running-job-id>"
    exit 1
fi

export NFS_DIR="${NFS_DIR:-${ASSET_ROOT}}"
export INSTRUCT_ROOT="${INSTRUCT_ROOT:-${ASSET_ROOT}/gensin-instruct-tars}"
export REASONING_ROOT="${REASONING_ROOT:-${ASSET_ROOT}/gi_reasoning_0718-tars}"
export JSONL_VIEW_ROOT="${JSONL_VIEW_ROOT:-${ASSET_ROOT}/genshin-training/gensin-instruct-reasoning-jsonl}"
export MIXTURE_YAML="${MIXTURE_YAML:-${PROJECT_ROOT}/data_configs/gensin_instruct_reasoning_5pass.yaml}"
export MODEL_PATH="${MODEL_PATH:-${ASSET_ROOT}/model/genshin_llava_qwen35_9b_step700}"

for _root in "$INSTRUCT_ROOT" "$REASONING_ROOT"; do
    [[ -f "$_root/train.jsonl" && -f "$_root/val.jsonl" && -d "$_root/images" ]] || {
        echo "[ERR] dataset is incomplete (need train.jsonl, val.jsonl, images/): $_root"
        exit 1
    }
done
[[ -f "$MIXTURE_YAML" ]] || { echo "[ERR] mixture config is missing: $MIXTURE_YAML"; exit 1; }
[[ -f "$MODEL_PATH/config.json" && -f "$MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "[ERR] Genshin+LLaVA step-700 HF model is incomplete: $MODEL_PATH"
    exit 1
}

# Isambard software stack and shared conda environment.
export CONDA_BASE="${CONDA_BASE:-/home/b6db/hongnanvideo1.b6db/.miniconda3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-megatron-sft}"
export PYTHON_BIN="${PYTHON_BIN:-/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/bin/python}"
export ENV_SETUP_MODULES="${ENV_SETUP_MODULES:-cuda/12.6 brics/nccl/2.26.6-1 brics/aws-ofi-nccl/1.8.1}"
export CUDA_HOME="${CUDA_HOME:-/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/targets/sbsa-linux}"
export CUDA_HOST_CC="${CUDA_HOST_CC:-/usr/bin/gcc-13}"
export CUDA_HOST_CXX="${CUDA_HOST_CXX:-/usr/bin/g++-13}"
export CC="$CUDA_HOST_CC"
export CXX="$CUDA_HOST_CXX"

# Build an atomic metadata-only view.  The two source validation files remain
# unchanged; val.jsonl in the view contains all 128 + 128 validation rows.
mkdir -p "$JSONL_VIEW_ROOT"
"$PYTHON_BIN" - "$INSTRUCT_ROOT" "$REASONING_ROOT" "$JSONL_VIEW_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

instruct_root, reasoning_root, output_root = map(Path, sys.argv[1:])
sources = (
    ("gensin-instruct", "gensin_instruct", instruct_root),
    ("gi-reasoning", "genshin_reasoning", reasoning_root),
)
output_root.mkdir(parents=True, exist_ok=True)

def rewrite(inputs, destination):
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    rows = images = 0
    try:
        with temporary.open("w", encoding="utf-8") as fout:
            for source_path, source_root, channel in inputs:
                with source_path.open(encoding="utf-8") as fin:
                    for line_number, line in enumerate(fin, 1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        paths = record.get("images") or []
                        record["images"] = [
                            image if os.path.isabs(image) else os.path.normpath(os.path.join(source_root, image))
                            for image in paths
                        ]
                        # Keep validation metrics separated by the YAML source
                        # identity. Both upstream val files currently carry
                        # game=genshin_instruction, including reasoning rows.
                        record["game"] = channel
                        record["channel"] = channel
                        if "messages" not in record or not paths:
                            raise ValueError(f"invalid row: {source_path}:{line_number}")
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        rows += 1
                        images += len(paths)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return rows, images

stats = {}
for name, channel, root in sources:
    stats[name] = rewrite([(root / "train.jsonl", root, channel)], output_root / f"{name}-train.jsonl")
stats["validation"] = rewrite(
    [(root / "val.jsonl", root, channel) for _name, channel, root in sources],
    output_root / "val.jsonl",
)
(output_root / "manifest.json").write_text(
    json.dumps({name: {"rows": value[0], "images": value[1]} for name, value in stats.items()},
               ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("[jsonl-view] " + " ".join(f"{name}={value[0]}" for name, value in stats.items()))
PY

# Two nodes x four GPUs. TP remains within each node; DP spans the nodes.
export GPUS="${GPUS:-0,1,2,3}"
export NNODES="${NNODES:-2}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export TP="${TP:-4}"
export PP="${PP:-1}"
export CP="${CP:-1}"
export GLOBAL_BS="${GLOBAL_BS:-16}"
export SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-128}"
export PACKING_NUM_PROC="${PACKING_NUM_PROC:-1}"
export DATASET_NUM_PROC="${DATASET_NUM_PROC:-1}"
# Avoid the missing Isambard /local/user/<uid> TMPDIR during TileLang/NVCC JIT.
# 43_train_inner.sh creates this job-scoped directory on each node.
export JOB_LOCAL_TMPDIR="${JOB_LOCAL_TMPDIR:-/dev/shm/${USER:-u}_train_tmp_${SLURM_JOB_ID:-local}}"
export ASYNC_PACKING_PREFETCH_DEPTH="${ASYNC_PACKING_PREFETCH_DEPTH:-2}"
export IMG_MAX_TOK="${IMG_MAX_TOK:-16384}"

# Schedule overrides for this five-pass run.
export WARMUP_ITERS="${WARMUP_ITERS:-30}"
export EVAL_STEPS="${EVAL_STEPS:-50}"
# Full validation is enabled by the common launcher and ignores EVAL_ITERS.
# The following count applies only when FULL_VAL_EVAL=false:
# validation has 256 rows. With DP=32, each rank owns eight rows; the finite
# stream is flushed at dataset end even though packing_interval=128. With two
# microbatches per iteration, four eval iterations consume those eight rows
# (four instruction + four reasoning) exactly once.
export EVAL_ITERS="${EVAL_ITERS:-4}"
export SAVE_STEPS="${SAVE_STEPS:-50}"

export GAME_NAME="${GAME_NAME:-genshin_instruct_reasoning_5pass_gbs16_qwen35_128k}"
export OUTPUT_DIR="${OUTPUT_DIR:-${ASSET_ROOT}/outputs/${GAME_NAME}_${SLURM_JOB_ID:-local}}"
export PASS_AWARE_MIXTURE="${PASS_AWARE_MIXTURE:-true}"
export WANDB_MODE="${WANDB_MODE:-online}"

exec bash "${SCRIPT_DIR}/46_train_genshin_llava_onevision_128k_multinode.slurm"
