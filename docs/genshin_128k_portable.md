# Genshin-only 128K training runbook

This runbook reproduces the validated 8-node Qwen3.5-9B configuration without
mixing LLaVA-OneVision data.

## Data and model sources

| Asset | Source | Size / role |
|---|---|---|
| Genshin SL135 | [thomaslee1818/yuanshen-bc-formatted-sl135-encrypted](https://huggingface.co/datasets/thomaslee1818/yuanshen-bc-formatted-sl135-encrypted) | 15 encrypted split volumes, about 2.82 TB; 560 trainable Parquet shards / 89,503 trajectories after extraction |
| Qwen3.5-9B | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | base VLM |
| LLaVA-OneVision (mixed-run reference only) | [ackermans26/LLaVA-OneVision-1.5-Instruct-Data-qwen-format](https://huggingface.co/datasets/ackermans26/LLaVA-OneVision-1.5-Instruct-Data-qwen-format) | about 6.24 TB; **not downloaded or used for this Genshin-only run** |

The Genshin repository is already the final training representation: sequence
length 135, 128K-class trajectories, private-server data removed, and the
key-release issue fixed. The local pipeline decrypts and validates it; it does
not regenerate actions or images.

Do not put the archive password, Hugging Face token, or WandB key in this
repository. Supply them through `GENSHIN_ARCHIVE_PASSWORD`, `HF_TOKEN`, and
`WANDB_API_KEY` (or the corresponding CLI login stores).

For reference, the original mixed-data preprocessing path was:

```text
Genshin HF archive
  -> extract_genshin_split_archive.py
  -> genshin-parquet/

LLaVA-OneVision HF repository
  -> build_llava_fraction_view.py --fraction 0.162
  -> deterministic 16.2% canonical subset

both roots
  -> build_local_genshin_llava_mixture.py
  -> genshin-llava-128k.local.yaml
  -> 46_train_genshin_llava_onevision_128k_multinode.slurm
```

`build_llava_fraction_view.py` hard-links complete selected shards and rewrites
only each subset's boundary shard, so its physical rows exactly match the
requested fraction. The pure-Genshin workflow below skips all LLaVA steps.

## 1. Install the validated environment

For the Isambard ARM64/GH200 cluster, including Conda terms acceptance, the
CUDA 12.9 compiler toolchain, source-built FlashAttention and Apex, follow
[`isambard_gh200_environment.md`](isambard_gh200_environment.md). The commands
below describe the original x86_64 CUDA 12.8 environment.

CUDA 12.8 and Python 3.10 were used for the validated run:

```bash
conda create -n megatron-sft python=3.10 -y
CONDA_ENV_NAME=megatron-sft CUDA_HOME=/usr/local/cuda-12.8 \
  bash scripts/00_install_env.sh
python tools/verify_env.py
```

`00_install_env.sh` checks out ms-swift commit
`dc40c652fa9b8fa096613055ccf124dd49f8890c` and applies
`patches/ms-swift-dc40c652-stateful-streaming-tp.patch`. That patch is required
for DP-local indexed streaming, finite-dataset EOF synchronization, exact
packing/cursor state recovery, TP-local batch broadcast, and deterministic
legacy replay fallback. Running against an arbitrary ms-swift `main` does not
reproduce the tested behavior.

Core validated versions:

```text
torch 2.8.0+cu128          transformers 5.14.1
ms-swift 4.5.0.dev0        megatron-core 0.16.1
mcore-bridge 1.6.1         transformer-engine 2.14.1
datasets 5.0.1             pyarrow 25.0.0
```

## 2. Download and extract on compute-node NVMe

The encrypted and extracted copies coexist during extraction, so reserve at
least 5.7 TB. Run these commands on a compute node, not a login node:

```bash
export LOCAL_DATA_ROOT=/local/nvme/$USER/genshin-training
export HF_TOKEN=...                         # only if the repository requires it
bash scripts/05_download_genshin_encrypted.sh

export GENSHIN_ARCHIVE_PASSWORD=...         # obtain from the dataset owner
python scripts/extract_genshin_split_archive.py \
  --source "$LOCAL_DATA_ROOT/genshin-encrypted" \
  --destination "$LOCAL_DATA_ROOT/genshin-parquet" \
  --workers 32 \
  --verify-volumes
```

The extractor:

1. reads all split volumes as one seekable stream without creating a 2.82 TB
   concatenated archive;
2. validates each volume against `SHA256SUMS`;
3. decrypts WinZip AES-256 members in parallel;
4. validates every member HMAC and refuses truncated output.

Download/extract once, then copy `genshin-parquet/` and
`genshin-encrypted/SHA256SUMS` to the same `LOCAL_DATA_ROOT` path on every
training node. Node-local copies avoid 64 ranks repeatedly decoding images from
the shared filesystem.

## 3. Build the Genshin-only training view on every node

```bash
python scripts/build_genshin_only_dataset.py \
  --genshin-root "$LOCAL_DATA_ROOT/genshin-parquet" \
  --output-root "$LOCAL_DATA_ROOT/genshin-only" \
  --passes 3 \
  --val-rows 32 \
  --workers 32
```

This is mostly metadata and symlinks; it does not duplicate the 2.82 TB
dataset. It validates every Parquet footer, automatically reserves the minimum
number of complete shards containing at least 32 rows, excludes those shards
from training, and writes:

```text
$LOCAL_DATA_ROOT/genshin-only/
  parquet/genshin/*.parquet        # links to training-only shards
  parquet/canonical_index.json
  parquet/source_manifest.json
  validation/genshin-val.parquet   # 32 non-overlapping trajectories
  genshin-only-128k.local.yaml
```

The held-out trajectories are not present in the training view. With the
current archive, automatic holdout reserves two small shards (64 source rows),
leaving 89,439 trajectories. Three logical Genshin passes expose 268,317 raw
trajectories. SWIFT itself runs one expanded epoch because the YAML `passes`
field already defines total exposure.

To run nine logical Genshin passes, rebuild into a fresh output root with
`--passes 9`.

## 4. Launch the run

The training contract was originally validated on 8 nodes / 64 H200s. The
current `47` preset defaults to the Isambard allocation used on 2026-08-28:
2 nodes, 4 GH200s per node, and global batch size 8. For that machine, use the
exact launch and troubleshooting instructions in
[`isambard_gh200_environment.md`](isambard_gh200_environment.md#8-在-slurm-训练中使用).

Place the model on a shared path such as
`$NFS_DIR/models/Qwen3.5-9B`:

```bash
MODEL_NAME=Qwen/Qwen3.5-9B \
TARGET_DIR=/shared/project/models/Qwen3.5-9B \
HF_OFFICIAL=1 \
  bash scripts/01_download_model.sh
```

Then submit with the account and partition used by the destination cluster:

```bash
sbatch -A <account> -p <partition> \
  --export=ALL,NFS_DIR=/shared/project,\
LOCAL_DATA_ROOT=/local/nvme/$USER/genshin-training,\
CONDA_BASE=/shared/miniconda3,\
WANDB_PROJECT=opensima,WANDB_ENTITY=OpenSIMA \
  scripts/47_train_genshin_only_128k_multinode.slurm
```

The original 64-GPU validation contract is:

| Setting | Value |
|---|---:|
| Nodes / GPUs | 8 / 64 H200 |
| TP / PP / DP | 4 / 1 / 16 |
| Micro / global batch | 1 / 128 |
| Gradient accumulation | 8 |
| `MAX_LEN` / image max tokens | 131,072 / 832 |
| Packing | enabled, binpack to 131,072 |
| MTP | 1 layer, loss scale 0.1 |
| Recompute | full |
| LLM / ViT / aligner LR | 1e-5 / 3e-6 / 1e-5 |
| LR schedule | 100-step warmup, then constant |
| Checkpoint | every 100 steps, optimizer + scheduler + RNG included |
| Validation | 32 held-out rows, 2 eval iterations every 5 steps |
| Logging | every step to WandB |

Qwen3.5-9B has four KV heads, so `TP=8` is invalid; keep `TP=4`.
`recompute=none` was tested at 128K and ran out of H200 memory. MTP checkpoint
metadata is repaired by `scripts/mcore_mtp_checkpoint_patch.py`.

## 5. Checkpoint storage and resume

The launcher defaults to `CHECKPOINT_STORAGE=local`. Each node writes only its
MCore shards to:

```text
$LOCAL_DATA_ROOT/checkpoints/$RUN_NAME/checkpoint-N
```

This avoids accumulating 122+ GiB checkpoints on shared storage. It also means
the same node-local volumes must survive and be present for restart. MCore may
redistribute shard reads across DP ranks during load, so **do not point all
ranks directly at their partial local directory**.

Inside an allocation containing those NVMe volumes, stage only the selected
checkpoint temporarily:

```bash
export LOCAL_CHECKPOINT="$LOCAL_DATA_ROOT/checkpoints/$RUN_NAME/checkpoint-100"
export STAGING_ROOT="/shared/resume-staging/$RUN_NAME"
bash scripts/assemble_local_mcore_checkpoint.sh
```

Then restart with a stable run name and output path:

```bash
sbatch -A <account> -p <partition> \
  --export=ALL,NFS_DIR=/shared/project,\
LOCAL_DATA_ROOT=/local/nvme/$USER/genshin-training,\
RUN_NAME="$RUN_NAME",\
OUTPUT_DIR="$LOCAL_DATA_ROOT/checkpoints/$RUN_NAME",\
RESUME_FROM="/shared/resume-staging/$RUN_NAME/checkpoint-100" \
  scripts/47_train_genshin_only_128k_multinode.slurm
```

After every rank reports `Successfully loaded Megatron model weights`, the
temporary staged checkpoint can be deleted. Keep it until at least one resumed
optimizer step has finite loss and gradient norm.

If the destination scheduler can replace nodes or erase local NVMe, use
`CHECKPOINT_STORAGE=shared` instead. That supports `AUTO_RESUME=true` directly
but requires enough shared capacity.

## Relevant code

```text
scripts/05_download_genshin_encrypted.sh       Hugging Face download
scripts/extract_genshin_split_archive.py       split AES archive verification/extraction
scripts/build_genshin_only_dataset.py          footer checks, train/val split, YAML
scripts/47_train_genshin_only_128k_multinode.slurm  final pure-Genshin preset
scripts/46_train_genshin_llava_onevision_128k_multinode.slurm shared 128K contract
scripts/43_train_genshin_multinode.slurm       Slurm rendezvous
scripts/43_train_inner.sh                      conda/NCCL/node-rank setup
scripts/41_train_bc_full_megatron.sh           Megatron-SWIFT command
scripts/assemble_local_mcore_checkpoint.sh     safe local-checkpoint staging
scripts/resolve_mcore_checkpoint.py            completeness/full-state validation
scripts/mcore_mtp_checkpoint_patch.py          MTP distributed-checkpoint fix
patches/ms-swift-dc40c652-stateful-streaming-tp.patch exact framework changes
```
