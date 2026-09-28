#!/usr/bin/env bash
# ★ BC SFT 正式训练（Megatron-SWIFT 后端 + 全参 + MTP=1）
#
# 任务  : Qwen3.5-4B 多模态全参 SFT (GatedDeltaNet hybrid 架构)
# 后端  : Megatron-SWIFT (mcore-bridge ≥1.2.1, megatron-core ≥0.16.1)
# 数据  : data/bc_sft_processed/{train,val_small}.jsonl
#
# 加速点（Megatron 相对 HF + ZeRO-3）:
#   1. TP=4 + Sequence Parallel: 模型权重和激活按 hidden 切片
#   2. cross_entropy_loss_fusion: CE loss kernel 融合 (TE)
#   3. attention_backend=flash: TE flash attention
#   4. MTP num_layers=1: 多 token 预测 (~等效 +30% 数据)
#   5. recompute full uniform: 重算全部 transformer block 1 次，省显存
#
# 默认 4 卡 TP=4 PP=1 DP=1，单卡 ~31GB；切大模型时见 docs/scaling.md (新机器请新建)
#
# 用法:
#   bash scripts/41_train_bc_full_megatron.sh                           # 默认全配置
#   GAME_NAME=pubg TRAIN_FILE=... VAL_FILE=... bash scripts/41_*.sh     # 切游戏/数据
#   MTP_LAYERS=0 bash scripts/41_*.sh                                   # 关 MTP
#   GPUS=2,3 NPROC_PER_NODE=2 TP=2 bash scripts/41_*.sh                 # 只用 2 卡（需调小）
#
# 实际生产请用：
#   bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh
# 这会把训练真正 detach 成 daemon，SSH 断不影响。

set -euo pipefail

# 自动定位 PROJECT_ROOT (本脚本所在目录的上一级)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/env.sh"

# ----------------- 数据 / 模型 -----------------
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Qwen3.5-4B}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/bc_sft_processed}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.jsonl}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val_small.jsonl}"   # 32 条小 val (~2 min/eval)；用 val.jsonl 是全 397 条 (~30 min)

GAME_NAME="${GAME_NAME:-delta_force}"
RUN_NAME="${RUN_NAME:-${GAME_NAME}_megatron_full_$(date +%Y%m%d_%H%M%S)}"
_OUTPUT_DIR_EXPLICIT="${OUTPUT_DIR+x}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${RUN_NAME}}"

# ----------------- 多游戏混训 -----------------
# DATASET_SPEC: 多个数据集(空格分隔, 每个可带 #N 控制采样条数, 超采自动重复)，
#   例如 "a/genshin.jsonl#80000 b/honkai.jsonl#40000"。设了它就覆盖单一 TRAIN_FILE。
#   一般由 43_*.slurm 读 MIXTURE_YAML 自动生成。
DATASET_SPEC="${DATASET_SPEC:-}"
# 按游戏分别追踪 loss：每条数据需有 channel 字段(=游戏名)，开启后日志出现 loss_<游戏>
ENABLE_CHANNEL_LOSS="${ENABLE_CHANNEL_LOSS:-false}"
CHANNEL_COLUMN="${CHANNEL_COLUMN:-}"
CHANNEL_COLUMNS_JSON="${CHANNEL_COLUMNS_JSON:-}"
# All presets inherit full validation; export the flag for the Python plugin.
export FULL_VAL_EVAL="${FULL_VAL_EVAL:-true}"
STREAMING="${STREAMING:-$FULL_VAL_EVAL}"
STREAMING_SHARD_BY_DP="${STREAMING_SHARD_BY_DP:-$FULL_VAL_EVAL}"
DATASET_SHUFFLE="${DATASET_SHUFFLE:-true}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-10000}"
ASYNC_PACKING_PREFETCH_DEPTH="${ASYNC_PACKING_PREFETCH_DEPTH:-0}"
INTERLEAVE_PROB="${INTERLEAVE_PROB:-}"
STOPPING_STRATEGY="${STOPPING_STRATEGY:-first_exhausted}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
if [[ "$FULL_VAL_EVAL" == "true" ]]; then
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
else
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
fi
ADD_GENSHIN_SPECIAL_TOKENS="${ADD_GENSHIN_SPECIAL_TOKENS:-true}"
# Strict pass ordering is opt-in.  Its manifest owns sample order, therefore
# neither HF streaming shuffle nor Megatron's dataloader shuffle may run.
PASS_AWARE_MIXTURE="${PASS_AWARE_MIXTURE:-false}"
MIXTURE_PLAN="${MIXTURE_PLAN:-}"
TRAIN_DATALOADER_SHUFFLE="${TRAIN_DATALOADER_SHUFFLE:-true}"
STOP_AT_DATASET_END="${STOP_AT_DATASET_END:-false}"

# ----------------- GPU / 并行 -----------------
GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
TP="${TP:-4}"
PP="${PP:-1}"
SEQ_PARALLEL="${SEQ_PARALLEL:-true}"
# Context parallel. Default 1 keeps current 128K runs unchanged. Set CP=2/4
# only for memory-saving speed trials (e.g. recompute=none).
CP="${CP:-1}"
# Split packed images across TP ranks instead of replicating the HF ViT on
# every rank. Default on in this copy (training-vittp); production training/
# does not load the plugin.
VIT_TP="${VIT_TP:-true}"
# Experimental until the packing/GDN production checklist is complete.
# Keep the default false; short-run validation must opt in explicitly.
PACKING="${PACKING:-false}"

# ----------------- 多节点（分布式）-----------------
# 单机默认: NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1。
# 多节点由 SLURM 启动脚本(43_*.slurm)在每个节点上 export 这些变量后再调用本脚本。
# 全局进程数 WORLD_SIZE = NNODES × NPROC_PER_NODE = TP × PP × DP。
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ----------------- batch / 训练超参 -----------------
MICRO_BS="${MICRO_BS:-1}"
GLOBAL_BS="${GLOBAL_BS:-8}"          # NPROC=4,TP=4 → DP=1，global_bs = micro * 1 * grad_acc(=8)
# Qwen3.5 一张 720P 图 ≈ 900+ token；32K 大约能放 ~30 张图 + system(~1.7k) + 文本。
# ★ 改数据每条 trajectory 的帧数时，务必同步调这里：MAX_LEN >= 1700 + 帧数×(每图token + ~40)
MAX_LEN="${MAX_LEN:-32768}"
PACKING_LENGTH="${PACKING_LENGTH:-$MAX_LEN}"
PACKING_NUM_PROC="${PACKING_NUM_PROC:-4}"
PACKING_INTERVAL="${PACKING_INTERVAL:-128}"
PACKING_STRATEGY="${PACKING_STRATEGY:-binpack}"
# Reject obviously pathological raw rows before the multimodal tokenizer
# allocates/processes token lists far beyond MAX_LEN. Set 0 to disable.
SWIFT_PACKING_MAX_RAW_TEXT_CHARS="${SWIFT_PACKING_MAX_RAW_TEXT_CHARS:-$(( MAX_LEN * 2 ))}"
SWIFT_USE_MCORE_GDN="${SWIFT_USE_MCORE_GDN:-1}"
DETERMINISTIC_MODE="${DETERMINISTIC_MODE:-false}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-3}"
# TRAIN_ITERS>0 时按固定步数训练(用于 dryrun 短跑验证)，否则按 EPOCHS 跑满。
TRAIN_ITERS="${TRAIN_ITERS:-0}"
LR="${LR:-1e-5}"
MIN_LR="${MIN_LR:-1e-6}"
# warmup: WARMUP_FRAC 是相对【整个训练(全部 epoch)的总步数】的比例(0.05=总步数5%)，不是一个 epoch。
# 想精确控制就用 WARMUP_ITERS 直接指定 warmup 步数(设了它则忽略 WARMUP_FRAC)。
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
WARMUP_ITERS="${WARMUP_ITERS:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
# LR 调度: cosine(默认,带衰减) / constant(只 warmup 不衰减) / linear / WSD ...
# 要"warmup 但不 decay"就设 LR_DECAY_STYLE=constant（warmup 后恒定在 LR，min_lr 被忽略）。
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"
# ViT(视觉塔)/aligner(merger) 单独学习率。留空 = 与 LLM 的 LR 相同(全模型同 lr co-train)。
# VLM SFT 常用更小的 ViT lr(如 LLM 的 1/5~1/10)以稳住预训练视觉编码器。
VIT_LR="${VIT_LR:-}"
ALIGNER_LR="${ALIGNER_LR:-}"

# ----------------- MTP -----------------
MTP_LAYERS="${MTP_LAYERS:-1}"          # 1 = 开 MTP（多 token 预测），0 = 关
MTP_LOSS_SCALE="${MTP_LOSS_SCALE:-0.1}"

# ----------------- 激活重算 / 显存 -----------------
# full(默认: 重算全部 block, 最省显存但慢~30%) / selective(重算指定模块) /
# none(不重算, 最快最吃显存——显存有富余时用它"拉满显存"提速)。
RECOMPUTE="${RECOMPUTE:-full}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
# Space-separated Megatron selective modules.  "mlp" recomputes only the
# MLP/SwiGLU path and leaves layernorm, fused core attention, and the complete
# GDN block outside activation checkpointing.
RECOMPUTE_MODULES="${RECOMPUTE_MODULES:-core_attn}"
CROSS_ENTROPY_LOSS_FUSION="${CROSS_ENTROPY_LOSS_FUSION:-true}"
CROSS_ENTROPY_FUSION_IMPL="${CROSS_ENTROPY_FUSION_IMPL:-native}"
# TE 2.14.1's op fuser currently crashes inside TorchDynamo while tracing its
# packaging.version check. Eager TE ops are correct and avoid that compiler bug.
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-true}"

# ----------------- ckpt / log -----------------
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
# MCore is the resumable training checkpoint. HF safetensors are an export
# artifact and duplicate the model weights; keep them off during training.
SAVE_SAFETENSORS="${SAVE_SAFETENSORS:-false}"
# RUN_NAME already gives each new run a unique directory. Disabling SWIFT's
# extra vN-* nesting keeps OUTPUT_DIR stable across process restarts.
ADD_VERSION="${ADD_VERSION:-false}"
# 默认保存优化器和 RNG；否则 checkpoint 只有权重，不能无损续训。
NO_SAVE_OPTIM="${NO_SAVE_OPTIM:-false}"
NO_SAVE_RNG="${NO_SAVE_RNG:-false}"
# RESUME_FROM 指向 checkpoint-N 目录；设为 auto 或 AUTO_RESUME=true 时从
# OUTPUT_DIR 选择最新的完整 checkpoint。续训默认恢复 optimizer/RNG/iteration。
RESUME_FROM="${RESUME_FROM:-}"
AUTO_RESUME="${AUTO_RESUME:-false}"
RESUME_REQUIRE_FULL_STATE="${RESUME_REQUIRE_FULL_STATE:-true}"
NO_LOAD_OPTIM="${NO_LOAD_OPTIM:-false}"
NO_LOAD_RNG="${NO_LOAD_RNG:-false}"
FINETUNE="${FINETUNE:-}"
SAVE_LIMIT_ARGS=()
if [[ "$SAVE_TOTAL_LIMIT" =~ ^[0-9]+$ && "$SAVE_TOTAL_LIMIT" -gt 0 ]]; then
    SAVE_LIMIT_ARGS=(--save_total_limit "$SAVE_TOTAL_LIMIT")
elif [[ "$SAVE_TOTAL_LIMIT" != "0" ]]; then
    echo "[ERR] SAVE_TOTAL_LIMIT must be 0 (unlimited) or an integer >=2"
    exit 1
fi

# ----------------- 视觉 token -----------------
# 每图视觉 token 上限。Qwen3.5 一张 720P 实际 ≈ 900+ token。
# 想在固定 MAX_LEN 下塞更多帧，就调小它（会降清晰度）；反之调大。
IMG_MAX_TOK="${IMG_MAX_TOK:-1024}"

# ----------------- 检查 -----------------
[[ -d "$MODEL_PATH" ]] || { echo "[ERR] model not found: $MODEL_PATH"; exit 1; }
case "$AUTO_RESUME" in
    true|false) ;;
    *) echo "[ERR] AUTO_RESUME must be true or false (got $AUTO_RESUME)"; exit 1 ;;
esac
case "$RESUME_REQUIRE_FULL_STATE" in
    true|false) ;;
    *) echo "[ERR] RESUME_REQUIRE_FULL_STATE must be true or false (got $RESUME_REQUIRE_FULL_STATE)"; exit 1 ;;
esac
case "$SAVE_SAFETENSORS" in
    true|false) ;;
    *) echo "[ERR] SAVE_SAFETENSORS must be true or false (got $SAVE_SAFETENSORS)"; exit 1 ;;
esac
case "$ADD_VERSION" in
    true|false) ;;
    *) echo "[ERR] ADD_VERSION must be true or false (got $ADD_VERSION)"; exit 1 ;;
esac
if [[ "$RESUME_FROM" == "auto" ]]; then
    AUTO_RESUME=true
    RESUME_FROM=""
fi

_CKPT_RESOLVER="${PROJECT_ROOT}/scripts/resolve_mcore_checkpoint.py"
_REQUIRE_FULL_ARGS=()
[[ "$RESUME_REQUIRE_FULL_STATE" == "true" ]] && _REQUIRE_FULL_ARGS+=(--require-full-state)
if [[ "$AUTO_RESUME" == "true" && -z "$RESUME_FROM" && -d "$OUTPUT_DIR" ]]; then
    if _latest_ckpt="$(
        python "$_CKPT_RESOLVER" --latest-under "$OUTPUT_DIR" \
            "${_REQUIRE_FULL_ARGS[@]}" --field path 2>/dev/null
    )"; then
        RESUME_FROM="$_latest_ckpt"
        echo "[resume] auto-selected $RESUME_FROM"
    else
        echo "[resume] no complete checkpoint under $OUTPUT_DIR; starting a new run"
    fi
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
    RESUME_FROM="$(
        python "$_CKPT_RESOLVER" --checkpoint "$RESUME_FROM" \
            "${_REQUIRE_FULL_ARGS[@]}" --field path
    )"
    if [[ -z "$_OUTPUT_DIR_EXPLICIT" ]]; then
        OUTPUT_DIR="$(dirname "$RESUME_FROM")"
        RUN_NAME="$(basename "$OUTPUT_DIR")"
    fi
    FINETUNE="${FINETUNE:-false}"
    [[ "$FINETUNE" == "false" ]] || {
        echo "[ERR] RESUME_FROM requires FINETUNE=false for iteration/optimizer/RNG restore"
        exit 1
    }

    _CKPT_GLOBAL_BS="$(
        python "$_CKPT_RESOLVER" --checkpoint "$RESUME_FROM" --field global_batch_size
    )"
    ALLOW_GBS_CHANGE="${ALLOW_GBS_CHANGE:-false}"
    export ALLOW_GBS_CHANGE
    if [[ "$_CKPT_GLOBAL_BS" -ne "$GLOBAL_BS" ]]; then
        if [[ "$ALLOW_GBS_CHANGE" == "true" ]]; then
            echo "[resume] ALLOW_GBS_CHANGE: checkpoint GBS=$_CKPT_GLOBAL_BS -> current GBS=$GLOBAL_BS (consumed samples kept from checkpoint)"
        else
            echo "[ERR] GLOBAL_BS changed across resume: checkpoint=$_CKPT_GLOBAL_BS current=$GLOBAL_BS"
            exit 1
        fi
    fi
    _RESUME_CONSUMED_SAMPLES="$(
        python "$_CKPT_RESOLVER" --checkpoint "$RESUME_FROM" --field consumed_samples
    )"
    RESUME_ARGS=(
        --mcore_model "$RESUME_FROM"
        --finetune false
        --no_load_optim "$NO_LOAD_OPTIM"
        --no_load_rng "$NO_LOAD_RNG"
    )
    if [[ "$STREAMING" == "true" && "$_RESUME_CONSUMED_SAMPLES" -gt 0 ]]; then
        echo "[resume] trainer will restore the indexed stream cursor at consumed=$_RESUME_CONSUMED_SAMPLES (legacy checkpoints fall back to replay)"
    fi
else
    FINETUNE="${FINETUNE:-true}"
    RESUME_ARGS=(
        --finetune "$FINETUNE"
        --no_load_optim "$NO_LOAD_OPTIM"
        --no_load_rng "$NO_LOAD_RNG"
    )
fi

if [[ -n "$DATASET_SPEC" ]]; then
    # DATASET_SPEC may be a local directory (Parquet) or a whitespace-separated
    # list of paths with optional #sample suffixes.
    # shellcheck disable=SC2206
    _DATASET_CHECKS=($DATASET_SPEC)
    for _dataset in "${_DATASET_CHECKS[@]}"; do
        _dataset="${_dataset%%#*}"
        [[ -e "$_dataset" ]] || { echo "[ERR] train data not found: $_dataset"; exit 1; }
    done
else
    [[ -f "$TRAIN_FILE" ]] || { echo "[ERR] train data not found: $TRAIN_FILE"; exit 1; }
fi
# Training-only presets omit validation entirely and disable evaluation.
VAL_DATASET_ARGS=()
if [[ "${ENABLE_EVAL:-true}" == "true" ]]; then
    # lib_mixture.py --print-val emits one validation path per line.
    mapfile -t _VAL_FILES <<< "$VAL_FILE"
    for _val_file in "${_VAL_FILES[@]}"; do
        [[ -e "$_val_file" ]] || { echo "[ERR] val data not found: $_val_file"; exit 1; }
    done
    VAL_DATASET_ARGS=(--val_dataset "${_VAL_FILES[@]}")
else
    VAL_FILE=""
    EVAL_ITERS=0
fi
case "$PACKING" in
    true|false) ;;
    *) echo "[ERR] PACKING must be true or false (got $PACKING)"; exit 1 ;;
esac
case "$CROSS_ENTROPY_LOSS_FUSION" in
    true|false) ;;
    *) echo "[ERR] CROSS_ENTROPY_LOSS_FUSION must be true or false (got $CROSS_ENTROPY_LOSS_FUSION)"; exit 1 ;;
esac
case "$CROSS_ENTROPY_FUSION_IMPL" in
    native|te) ;;
    *) echo "[ERR] CROSS_ENTROPY_FUSION_IMPL must be native or te (got $CROSS_ENTROPY_FUSION_IMPL)"; exit 1 ;;
esac
case "$DISABLE_TORCH_COMPILE" in
    true|false) ;;
    *) echo "[ERR] DISABLE_TORCH_COMPILE must be true or false (got $DISABLE_TORCH_COMPILE)"; exit 1 ;;
esac
if [[ "$DISABLE_TORCH_COMPILE" == "true" ]]; then
    export TORCHDYNAMO_DISABLE=1
fi
[[ "$PACKING_LENGTH" =~ ^[0-9]+$ && "$PACKING_LENGTH" -gt 0 ]] || {
    echo "[ERR] PACKING_LENGTH must be a positive integer (got $PACKING_LENGTH)"
    exit 1
}
[[ "$PACKING_NUM_PROC" =~ ^[0-9]+$ && "$PACKING_NUM_PROC" -gt 0 ]] || {
    echo "[ERR] PACKING_NUM_PROC must be a positive integer (got $PACKING_NUM_PROC)"
    exit 1
}
[[ "$SWIFT_PACKING_MAX_RAW_TEXT_CHARS" =~ ^[0-9]+$ ]] || {
    echo "[ERR] SWIFT_PACKING_MAX_RAW_TEXT_CHARS must be a non-negative integer (got $SWIFT_PACKING_MAX_RAW_TEXT_CHARS)"
    exit 1
}
if [[ -n "$DATASET_SPEC" && "$TRAIN_ITERS" -le 0 && "$EPOCHS" != "1" ]]; then
    echo "[ERR] 多数据集 passes 已表示整个任务的总曝光量，必须设置 EPOCHS=1（当前 EPOCHS=$EPOCHS）"
    exit 1
fi
case "$PASS_AWARE_MIXTURE" in
    true|false) ;;
    *) echo "[ERR] PASS_AWARE_MIXTURE must be true or false (got $PASS_AWARE_MIXTURE)"; exit 1 ;;
esac
case "$STOP_AT_DATASET_END" in
    true|false) ;;
    *) echo "[ERR] STOP_AT_DATASET_END must be true or false (got $STOP_AT_DATASET_END)"; exit 1 ;;
esac
case "$STREAMING_SHARD_BY_DP" in
    true|false) ;;
    *) echo "[ERR] STREAMING_SHARD_BY_DP must be true or false (got $STREAMING_SHARD_BY_DP)"; exit 1 ;;
esac
if [[ "$STOP_AT_DATASET_END" == "true" && "$STREAMING" != "true" ]]; then
    echo "[ERR] STOP_AT_DATASET_END=true requires STREAMING=true"
    exit 1
fi
if [[ "$PASS_AWARE_MIXTURE" == "true" ]]; then
    [[ -n "$MIXTURE_PLAN" && -f "$MIXTURE_PLAN" ]] || {
        echo "[ERR] PASS_AWARE_MIXTURE=true requires an existing MIXTURE_PLAN"; exit 1;
    }
    [[ "$EPOCHS" == "1" ]] || {
        echo "[ERR] PASS_AWARE_MIXTURE=true requires EPOCHS=1"; exit 1;
    }
    DATASET_SHUFFLE=false
    TRAIN_DATALOADER_SHUFFLE=false
    # The pass-aware generator owns exact DP sharding. Its plugin disables the
    # pipeline's duplicate filter while this flag keeps per-rank loading
    # parallel instead of serializing all DP batches through a dispatcher.
    STREAMING_SHARD_BY_DP=true
    [[ "$DATALOADER_NUM_WORKERS" == "0" ]] || {
        echo "[ERR] PASS_AWARE_MIXTURE=true currently requires DATALOADER_NUM_WORKERS=0"; exit 1;
    }
fi

mkdir -p "$OUTPUT_DIR" "${PROJECT_ROOT}/logs"

# Keep values needed by worker processes in their environment.
export PROJECT_ROOT MODEL_PATH DATASET_SPEC TRAIN_FILE MIXTURE_YAML
export CHANNEL_COLUMNS_JSON STREAMING DATASET_SHUFFLE SHUFFLE_BUFFER_SIZE
export STREAMING_SHARD_BY_DP
export INTERLEAVE_PROB STOPPING_STRATEGY PACKING PACKING_LENGTH PACKING_STRATEGY
export PACKING_INTERVAL DATASET_NUM_PROC SEED ADD_GENSHIN_SPECIAL_TOKENS
export SWIFT_PACKING_MAX_RAW_TEXT_CHARS
export STOP_AT_DATASET_END
export NNODES NPROC_PER_NODE TP PP CP MICRO_BS GLOBAL_BS SEQ_PARALLEL MAX_LEN IMG_MAX_TOK

# ----------------- Wandb -----------------
TAGS_BASE="${GAME_NAME},full_sft,vlm,megatron"
[[ "$MTP_LAYERS" -gt 0 ]] && TAGS_BASE="${TAGS_BASE},mtp${MTP_LAYERS}"
export WANDB_TAGS="${WANDB_TAGS:-${TAGS_BASE}}"
export WANDB_NAME="${RUN_NAME}"
export WANDB_NOTES="${WANDB_NOTES:-Megatron full SFT, MTP=${MTP_LAYERS}, TP=${TP}}"
unset WANDB_RUN_ID WANDB_RESUME WANDB_RUN_GROUP WANDB_SWEEP_ID

# ----------------- 条件参数 -----------------
EXTRA_FLAGS=()
if [[ -n "${DATA_SEED:-}" ]]; then
    EXTRA_FLAGS+=(--data_seed "$DATA_SEED")
fi
EXTERNAL_PLUGINS=()
[[ -n "${MODEL_TYPE:-}" ]] && EXTRA_FLAGS+=(--model_type "$MODEL_TYPE")
PACKING_ARGS=(--packing false)
if [[ "$PACKING" == "true" ]]; then
    [[ "$MICRO_BS" == "1" ]] || {
        echo "[ERR] Qwen3.5 packing requires MICRO_BS=1 (got $MICRO_BS)"
        exit 1
    }
    (( PACKING_LENGTH <= MAX_LEN )) || {
        echo "[ERR] PACKING_LENGTH ($PACKING_LENGTH) cannot exceed MAX_LEN ($MAX_LEN)"
        exit 1
    }
    [[ "$SWIFT_USE_MCORE_GDN" == "1" ]] || {
        echo "[ERR] packing=true requires SWIFT_USE_MCORE_GDN=1"
        exit 1
    }
    [[ "$DETERMINISTIC_MODE" == "false" ]] || {
        echo "[ERR] Qwen3.5 packed GDN requires DETERMINISTIC_MODE=false"
        exit 1
    }
    python -c 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule' || {
        echo "[ERR] packing=true but the FLA GDN kernel is unavailable"
        exit 1
    }
    python -c '
import inspect
import mcore_bridge.model.modules.gated_delta_net as gdn
source = inspect.getsource(gdn)
required = ("packed_seq_params", "cu_seqlens_q", "cu_seqlens=cu_seqlens")
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit(f"mcore-bridge packed GDN path is missing: {missing}")
' || {
        echo "[ERR] packing=true but mcore-bridge lacks the verified cu_seqlens GDN path"
        exit 1
    }
    PACKING_ARGS=(
        --packing true
        --packing_length "$PACKING_LENGTH"
        --packing_num_proc "$PACKING_NUM_PROC"
        --packing_strategy "$PACKING_STRATEGY"
        --padding_free true
    )
    # packing_interval is a local ms-swift patch; stock 4.5.2 rejects it.
    if grep -q 'packing_interval' "$(python -c 'import swift,os; print(os.path.dirname(swift.__file__))')/arguments/base_args/base_args.py" 2>/dev/null; then
        PACKING_ARGS+=(--packing_interval "$PACKING_INTERVAL")
    else
        echo "[bc-megatron] skip --packing_interval (not in this swift)"
    fi
fi
if [[ "$MTP_LAYERS" -gt 0 ]]; then
    EXTRA_FLAGS+=(--mtp_num_layers "$MTP_LAYERS" --mtp_loss_scaling_factor "$MTP_LOSS_SCALE")
    EXTERNAL_PLUGINS+=("${PROJECT_ROOT}/scripts/mcore_mtp_checkpoint_patch.py")
fi
if [[ "$VIT_TP" == "true" ]]; then
    EXTERNAL_PLUGINS+=("${PROJECT_ROOT}/scripts/mcore_vit_tp_patch.py")
fi
if [[ "${SCALE_GROUP_WARMUP_INIT:-false}" == "true" ]]; then
    EXTERNAL_PLUGINS+=("${PROJECT_ROOT}/scripts/mcore_group_warmup_patch.py")
fi
# 训练时长：固定步数(dryrun) 或 跑满 epoch
if [[ "$TRAIN_ITERS" -gt 0 ]]; then
    EXTRA_FLAGS+=(--train_iters "$TRAIN_ITERS")
    echo "[bc-megatron] DRYRUN: train_iters=$TRAIN_ITERS (忽略 EPOCHS)"
else
    EXTRA_FLAGS+=(--num_train_epochs "$EPOCHS")
fi

# ViT / aligner 单独学习率（仅在显式设置时添加；否则用全局 LR）
[[ -n "$VIT_LR" ]] && EXTRA_FLAGS+=(--vit_lr "$VIT_LR")
[[ -n "$ALIGNER_LR" ]] && EXTRA_FLAGS+=(--aligner_lr "$ALIGNER_LR")

# 按游戏分别追踪 loss
if [[ "$ENABLE_CHANNEL_LOSS" == "true" ]]; then
    EXTRA_FLAGS+=(
        --enable_channel_loss true
        --callbacks channel_token_counter
    )
    EXTERNAL_PLUGINS+=("${PROJECT_ROOT}/scripts/swift_channel_token_counter.py")
fi
if [[ -n "$CHANNEL_COLUMNS_JSON" ]]; then
    EXTRA_FLAGS+=(--columns "$CHANNEL_COLUMNS_JSON")
elif [[ -n "$CHANNEL_COLUMN" ]]; then
    EXTRA_FLAGS+=(--columns "{\"$CHANNEL_COLUMN\":\"channel\"}")
fi
if [[ "$STREAMING" == "true" ]]; then
    EXTRA_FLAGS+=(
        --streaming true
        --dataset_shuffle "$DATASET_SHUFFLE"
        --shuffle_buffer_size "$SHUFFLE_BUFFER_SIZE"
    )
    if grep -q 'streaming_shard_by_dp' "$(python -c 'import swift,os; print(os.path.dirname(swift.__file__))')/arguments/base_args/data_args.py" 2>/dev/null; then
        EXTRA_FLAGS+=(--streaming_shard_by_dp "$STREAMING_SHARD_BY_DP")
    else
        echo "[bc-megatron] skip --streaming_shard_by_dp (not in this swift)"
    fi
    EXTERNAL_PLUGINS+=("${PROJECT_ROOT}/scripts/swift_streaming_schema_patch.py")
fi
if [[ "$ASYNC_PACKING_PREFETCH_DEPTH" != "0" ]]; then
    [[ "$ASYNC_PACKING_PREFETCH_DEPTH" =~ ^[1-9][0-9]*$ ]] || {
        echo "[ERR] ASYNC_PACKING_PREFETCH_DEPTH must be 0 or a positive integer"; exit 1;
    }
    [[ "$STREAMING" == "true" ]] || {
        echo "[ERR] ASYNC_PACKING_PREFETCH_DEPTH requires STREAMING=true"; exit 1;
    }
    export SWIFT_ASYNC_PACKING_PREFETCH_DEPTH="$ASYNC_PACKING_PREFETCH_DEPTH"
    EXTERNAL_PLUGINS+=("${PROJECT_ROOT}/scripts/swift_async_packing_prefetch.py")
fi
if grep -q 'stop_at_dataset_end' "$(python -c 'import swift,os; print(os.path.dirname(swift.__file__))')/megatron/arguments/megatron_args.py" 2>/dev/null; then
    EXTRA_FLAGS+=(--stop_at_dataset_end "$STOP_AT_DATASET_END")
else
    echo "[bc-megatron] skip --stop_at_dataset_end (not in this swift)"
fi
if [[ "$PASS_AWARE_MIXTURE" == "true" ]]; then
    EXTERNAL_PLUGINS+=("${PROJECT_ROOT}/scripts/swift_pass_aware_mixture.py")
    [[ -n "$DATASET_SPEC" ]] || {
        echo "[ERR] PASS_AWARE_MIXTURE=true requires DATASET_SPEC"; exit 1;
    }
    # Trigger SWIFT's interleave hook once. The plugin consumes every source
    # exactly once and owns all quota and phase decisions.
    PASS_AWARE_DATASET_SPEC="$DATASET_SPEC"
    _pass_aware_count=($PASS_AWARE_DATASET_SPEC)
    INTERLEAVE_PROB=""
    for _pass_aware_dataset in "${_pass_aware_count[@]}"; do
        INTERLEAVE_PROB+="1 "
    done
    INTERLEAVE_PROB="${INTERLEAVE_PROB% }"
    STOPPING_STRATEGY=first_exhausted
fi
if [[ "$FULL_VAL_EVAL" == "true" && "${ENABLE_EVAL:-true}" == "true" ]]; then
    EXTERNAL_PLUGINS+=("${PROJECT_ROOT}/scripts/swift_full_val_eval.py")
    EXTRA_FLAGS+=(--val_dataset_shuffle false)
fi
if [[ -n "${EXTRA_EXTERNAL_PLUGINS:-}" ]]; then
    read -r -a _EXTRA_PLUGIN_PATHS <<<"$EXTRA_EXTERNAL_PLUGINS"
    EXTERNAL_PLUGINS+=("${_EXTRA_PLUGIN_PATHS[@]}")
fi
if (( ${#EXTERNAL_PLUGINS[@]} > 0 )); then
    EXTRA_FLAGS+=(--external_plugins "${EXTERNAL_PLUGINS[@]}")
fi
if [[ -n "$INTERLEAVE_PROB" ]]; then
    # shellcheck disable=SC2206
    _INTERLEAVE_PROBS=($INTERLEAVE_PROB)
    # shellcheck disable=SC2206
    _INTERLEAVE_DATASETS=(${PASS_AWARE_DATASET_SPEC:-$DATASET_SPEC})
    [[ "${#_INTERLEAVE_PROBS[@]}" -eq "${#_INTERLEAVE_DATASETS[@]}" ]] || {
        echo "[ERR] interleave probability count (${#_INTERLEAVE_PROBS[@]}) does not match dataset count (${#_INTERLEAVE_DATASETS[@]})"
        exit 1
    }
    EXTRA_FLAGS+=(--interleave_prob "${_INTERLEAVE_PROBS[@]}" --stopping_strategy "$STOPPING_STRATEGY")
fi
[[ -n "${EVAL_ITERS:-}" ]] && EXTRA_FLAGS+=(--eval_iters "$EVAL_ITERS")

SPECIAL_TOKEN_ARGS=()
if [[ "$ADD_GENSHIN_SPECIAL_TOKENS" == "true" ]]; then
    SPECIAL_TOKEN_ARGS=(
        --new_special_tokens
        '<|action_start|>' '<|action_end|>' '<|action_sep|>'
        '<|thought_start|>' '<|thought_end|>'
    )
fi

# 数据集参数：DATASET_SPEC(多游戏混训, 带#N采样) 优先，否则单一 TRAIN_FILE
DATASET_ARGS=(--dataset)
if [[ -n "$DATASET_SPEC" ]]; then
    # shellcheck disable=SC2206
    if [[ "$PASS_AWARE_MIXTURE" == "true" ]]; then
        DATASET_ARGS+=($PASS_AWARE_DATASET_SPEC)
    else
        DATASET_ARGS+=($DATASET_SPEC)
    fi
else
    DATASET_ARGS+=("$TRAIN_FILE")
fi

# warmup: 绝对步数(WARMUP_ITERS) 优先，否则用占总步数的比例(WARMUP_FRAC)
if [[ -n "${LR_WARMUP_INIT:-}" ]]; then
    EXTRA_FLAGS+=(--lr_warmup_init "$LR_WARMUP_INIT")
fi
if [[ -n "$WARMUP_ITERS" ]]; then
    EXTRA_FLAGS+=(--lr_warmup_iters "$WARMUP_ITERS")
else
    EXTRA_FLAGS+=(--lr_warmup_fraction "$WARMUP_FRAC")
fi

# 激活重算策略
case "$RECOMPUTE" in
    full)
        RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
        [[ "$RECOMPUTE_METHOD" == "uniform" || "$RECOMPUTE_METHOD" == "block" ]] || {
            echo "[ERR] full recompute requires RECOMPUTE_METHOD=uniform or block (got $RECOMPUTE_METHOD)"
            exit 1
        }
        EXTRA_FLAGS+=(--recompute_granularity full --recompute_method "$RECOMPUTE_METHOD" --recompute_num_layers "$RECOMPUTE_NUM_LAYERS")
        ;;
    selective)
        read -r -a RECOMPUTE_MODULES_ARRAY <<< "$RECOMPUTE_MODULES"
        ((${#RECOMPUTE_MODULES_ARRAY[@]} > 0)) || { echo "[ERR] selective requires non-empty RECOMPUTE_MODULES"; exit 1; }
        EXTRA_FLAGS+=(--recompute_granularity selective --recompute_modules "${RECOMPUTE_MODULES_ARRAY[@]}")
        ;;
    none)      EXTRA_FLAGS+=(--recompute_granularity none) ;;
    *) echo "[ERR] 未知 RECOMPUTE=$RECOMPUTE (应为 full/selective/none)"; exit 1 ;;
esac
[[ "$CP" =~ ^[0-9]+$ && "$CP" -ge 1 ]] || {
    echo "[ERR] CP must be a positive integer (got $CP)"; exit 1;
}
if [[ "$CP" -gt 1 ]]; then
    EXTRA_FLAGS+=(--context_parallel_size "$CP")
fi
if [[ "${OVERLAP_GRAD_REDUCE:-false}" == "true" ]]; then
    EXTRA_FLAGS+=(--overlap_grad_reduce true)
fi
if [[ "${OVERLAP_PARAM_GATHER:-false}" == "true" ]]; then
    EXTRA_FLAGS+=(--overlap_param_gather true)
fi
if [[ "${OPTIMIZER_CPU_OFFLOAD:-false}" == "true" ]]; then
    EXTRA_FLAGS+=(--optimizer_cpu_offload true)
fi
if [[ "${TP_COMM_OVERLAP:-false}" == "true" ]]; then
    EXTRA_FLAGS+=(--tp_comm_overlap true)
fi

# ----------------- 信息 -----------------
echo "============================================================"
echo "[bc-megatron] model     : $MODEL_PATH"
echo "[bc-megatron] train     : $TRAIN_FILE"
echo "[bc-megatron] val       : $VAL_FILE"
echo "[bc-megatron] game      : $GAME_NAME"
echo "[bc-megatron] run_name  : $RUN_NAME"
echo "[bc-megatron] output    : $OUTPUT_DIR"
echo "[bc-megatron] gpus      : $GPUS  (NPROC=$NPROC_PER_NODE TP=$TP PP=$PP CP=$CP SP=$SEQ_PARALLEL)"
echo "[bc-megatron] dist      : NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT  WORLD_SIZE=$((NNODES * NPROC_PER_NODE))  DP=$((NNODES * NPROC_PER_NODE / (TP * PP * CP)))"
echo "[bc-megatron] global_bs : $GLOBAL_BS  (micro=$MICRO_BS)"
echo "[bc-megatron] max_len   : $MAX_LEN  img_max_tok=$IMG_MAX_TOK"
echo "[bc-megatron] packing   : $PACKING  length=$PACKING_LENGTH num_proc=$PACKING_NUM_PROC deterministic=$DETERMINISTIC_MODE"
echo "[bc-megatron] epochs    : $EPOCHS  lr=$LR (warmup=$WARMUP_FRAC, decay=$LR_DECAY_STYLE -> $MIN_LR)"
echo "[bc-megatron] vit_lr    : ${VIT_LR:-=LLM($LR)}  aligner_lr: ${ALIGNER_LR:-=LLM($LR)}"
echo "[bc-megatron] recompute : $RECOMPUTE"
[[ "$RECOMPUTE" != "selective" ]] || echo "[bc-megatron] modules   : $RECOMPUTE_MODULES"
echo "[bc-megatron] vit_tp    : $VIT_TP"
echo "[bc-megatron] CE fusion : $CROSS_ENTROPY_LOSS_FUSION  impl=$CROSS_ENTROPY_FUSION_IMPL"
echo "[bc-megatron] torch.compile: $([[ "$DISABLE_TORCH_COMPILE" == "true" ]] && echo disabled || echo enabled)"
echo "[bc-megatron] MTP       : layers=$MTP_LAYERS  scale=$MTP_LOSS_SCALE"
echo "[bc-megatron] channel_loss: $ENABLE_CHANNEL_LOSS  (日志: trained_tokens_<游戏>; WandB: trained_tokens/<游戏>)"
echo "[bc-megatron] data mode : streaming=$STREAMING shuffle=$DATASET_SHUFFLE buffer=$SHUFFLE_BUFFER_SIZE channel_columns=${CHANNEL_COLUMNS_JSON:-${CHANNEL_COLUMN:-none}}"
echo "[bc-megatron] pass-aware: $PASS_AWARE_MIXTURE plan=${MIXTURE_PLAN:-none} dataloader_shuffle=$TRAIN_DATALOADER_SHUFFLE"
echo "[bc-megatron] checkpoint: save_optim=$([[ "$NO_SAVE_OPTIM" == "false" ]] && echo yes || echo no) save_rng=$([[ "$NO_SAVE_RNG" == "false" ]] && echo yes || echo no)"
echo "[bc-megatron] artifacts : mcore=yes safetensors=$SAVE_SAFETENSORS add_version=$ADD_VERSION"
if [[ -n "$RESUME_FROM" ]]; then
    echo "[bc-megatron] resume    : $RESUME_FROM (consumed=$_RESUME_CONSUMED_SAMPLES, load_optim=$([[ "$NO_LOAD_OPTIM" == "false" ]] && echo yes || echo no), load_rng=$([[ "$NO_LOAD_RNG" == "false" ]] && echo yes || echo no))"
else
    echo "[bc-megatron] resume    : new run (finetune=$FINETUNE)"
fi
[[ -n "$INTERLEAVE_PROB" ]] && echo "[bc-megatron] interleave: datasets=${#_INTERLEAVE_PROBS[@]} stopping=$STOPPING_STRATEGY"
[[ -n "$DATASET_SPEC" ]] && echo "[bc-megatron] dataset_spec: $DATASET_SPEC"
echo "[bc-megatron] save/eval : every $SAVE_STEPS / $EVAL_STEPS  (keep $SAVE_TOTAL_LIMIT)"
echo "[bc-megatron] wandb     : project=$WANDB_PROJECT entity=${WANDB_ENTITY:-default}  tags=$WANDB_TAGS"
echo "============================================================"

# ----------------- 启动 -----------------
# 多节点时每个节点写各自的日志(避免并发写同一文件)；单机仍是 ${RUN_NAME}.log
if [[ "$NNODES" -gt 1 ]]; then
    LOG_FILE="${PROJECT_ROOT}/logs/${RUN_NAME}.node${NODE_RANK}.log"
else
    LOG_FILE="${PROJECT_ROOT}/logs/${RUN_NAME}.log"
fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
SWIFT_USE_MCORE_GDN="$SWIFT_USE_MCORE_GDN" \
IMAGE_MAX_TOKEN_NUM="$IMG_MAX_TOK" \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
NNODES="$NNODES" \
NODE_RANK="$NODE_RANK" \
MASTER_ADDR="$MASTER_ADDR" \
MASTER_PORT="$MASTER_PORT" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
CUDA_VISIBLE_DEVICES="$GPUS" \
megatron sft \
    --model "$MODEL_PATH" \
    "${DATASET_ARGS[@]}" \
    "${VAL_DATASET_ARGS[@]}" \
    --split_dataset_ratio 0.0 \
    --train_dataloader_shuffle "$TRAIN_DATALOADER_SHUFFLE" \
    --data_sharding false \
    "${SPECIAL_TOKEN_ARGS[@]}" \
    --add_non_thinking_prefix false \
    --tensor_model_parallel_size "$TP" \
    --pipeline_model_parallel_size "$PP" \
    --sequence_parallel "$SEQ_PARALLEL" \
    --micro_batch_size "$MICRO_BS" \
    --global_batch_size "$GLOBAL_BS" \
    --max_length "$MAX_LEN" \
    "${PACKING_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    --freeze_llm false \
    --freeze_vit "${FREEZE_VIT:-false}" \
    --freeze_aligner "${FREEZE_ALIGNER:-false}" \
    --vit_gradient_checkpointing "${VIT_GRADIENT_CHECKPOINTING:-true}" \
    --cross_entropy_loss_fusion "$CROSS_ENTROPY_LOSS_FUSION" \
    --cross_entropy_fusion_impl "$CROSS_ENTROPY_FUSION_IMPL" \
    --attention_backend flash \
    --bf16 true \
    --lr "$LR" \
    --min_lr "$MIN_LR" \
    --lr_decay_style "$LR_DECAY_STYLE" \
    --weight_decay "$WEIGHT_DECAY" \
    --output_dir "$OUTPUT_DIR" \
    --add_version "$ADD_VERSION" \
    --save_safetensors "$SAVE_SAFETENSORS" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    "${SAVE_LIMIT_ARGS[@]}" \
    --no_save_optim "$NO_SAVE_OPTIM" \
    --no_save_rng "$NO_SAVE_RNG" \
    --logging_steps "$LOGGING_STEPS" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --report_to wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_exp_name "$RUN_NAME" \
    --seed "$SEED" \
    "${EXTRA_FLAGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
