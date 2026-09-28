# 重启 0916 / 0919 reasoning（GBS=256）

不要从 `checkpoint-4183` 重头训。权重和 jsonl 都不在这个 git 仓库里。能直接下载的是下面这些 Hugging Face 地址。

## 先下载权重

这两份都是 Hugging Face 格式（safetensors），不是 Megatron DCP。下载后设 `MODEL_PATH`，`FINETUNE=true`，`RESUME_FROM` 留空。优化器状态和 step 计数会重新开始，但权重不是 4183。

**0919，用 step 60。这是网上能下到的最远一份。**

https://huggingface.co/zhiyuanhucs/Qwen3.5-9B-General-Game-reason0919/tree/checkpoint-60

```bash
huggingface-cli download zhiyuanhucs/Qwen3.5-9B-General-Game-reason0919 \
  --revision checkpoint-60 \
  --local-dir /path/to/reason0919-checkpoint-60
```

同仓库还有更早的 `checkpoint-40`。`main` 不要拿来续这次训练。step 80 的 Megatron checkpoint 只在 AWS FSx，没有上传：

`/fsx/home/zhiyuan/nfs/outputs/reason0919_gbs256_20260925T083029Z/checkpoint-80`

那一份才能 `FINETUNE=false`、`RESUME_FROM` 接着 optimizer 训。另一台机器访问不到 FSx，就用上面的 `checkpoint-60`。step 100/120/140 的分片不完整，下不下来。

**0916，用 step 920。这次从 4183 新开、训到 step 125 的 checkpoint 没有上传，下不下来。**

https://huggingface.co/zhiyuanhucs/Qwen3.5-9B-General-Game-reason0916/tree/checkpoint-920

```bash
huggingface-cli download zhiyuanhucs/Qwen3.5-9B-General-Game-reason0916 \
  --revision checkpoint-920 \
  --local-dir /path/to/reason0916-checkpoint-920
```

同仓库还有 `checkpoint-880`、`checkpoint-600`，都比 920 更早。

## 数据不用重新下载，val 也不要重切

`/projects/b6db` 上切好的 jsonl 已经在配置里，每个游戏 64 条 val。文件在就直接训练：

- 0916：`data_configs/general_game_reasoning_128K_0921.yaml`
- 0919 history15：`data_configs/general_game_reasoning_history15_128K_0924.yaml`

这些 jsonl 不在 Hugging Face 上。只有原始 parquet 在 Hugging Face，而且是加密包，下下来还要再跑 `06`（`--val_chunks 0`）和 `07`（`--val_size 64 --seed 42`）。jsonl 已经在磁盘上时不要走这条。

| 游戏 | 0916 polish parquet | 0919 history15 parquet |
|---|---|---|
| Genshin | https://huggingface.co/datasets/opensima12/genshin_polish_reasoning_0916_encrypted | https://huggingface.co/datasets/opensima12/genshin_reasoning_history15_sl120_0919_encrypted |
| Cyberpunk 2077 | https://huggingface.co/datasets/opensima12/cyberpunk2077_polish_reasoning_0916_encrypted | https://huggingface.co/datasets/opensima12/cyberpunk2077_reasoning_history15_sl120_0920_encrypted |
| Spider-Man 2 | https://huggingface.co/datasets/opensima12/spiderman2_polish_reasoning_0916_encrypted | https://huggingface.co/datasets/opensima12/spiderman2_reasoning_history15_sl120_0920_encrypted |

Cyberpunk / Spider-Man 的 0919 仓库名带 `0920`，就是这套 0919 数据。

这份说明给另一台机器上的 coding agent。目标是把两次 **GBS=256** 的三游戏 mixed-reasoning SFT 接着训，而不是从 4183 重开。

在 `/projects/b6db` 这台机器上，数据和 val 切分已经在仓库里，不要重切：

| 这次 AWS 上的 run | 这边直接用的配置 | train 样本（×3 passes 之前） | 全集 step（GBS=256） |
|---|---|---|---|
| 0916 polish，从 4183 重开 | `data_configs/general_game_reasoning_128K_0921.yaml` | 38286 + 27807 + 9642 = 75735 | 227205/256 = **887** |
| 0919 history15 | `data_configs/general_game_reasoning_history15_128K_0924.yaml` | 39003 + 28556 + 9894 = 77453 | 232359/256 = **907** |

两个 YAML 的 `val` 都是三个游戏各 64 条，一共 192。这就是训练 eval 集。`scripts/lib_mixture.py --print-val` 一行一个路径。`scripts/41_train_bc_full_megatron.sh` 用 `--val_dataset` 和 `--split_dataset_ratio 0.0`，日志指标是 `eval_loss`。

对应脚本：

- 0916：`scripts/55_train_general_game_reasoning_3pass_128k_128gpu_gbs256_0921.slurm`
- 0919 history15：`scripts/57_train_general_game_reasoning_history15_3pass_128k_128gpu_gbs256_0924.slurm`

这两个脚本现在都认环境变量 `RESUME_FROM`。不设的时候是新开（`FINETUNE` 默认为 true）。设了路径时 `41_train_bc_full_megatron.sh` 要求 `FINETUNE=false`。脚本里的 `TRAIN_ITERS` 默认是 300（55）和 400（57），那是步数上限，不是全集。要跑完一个 epoch，启动前设 `TRAIN_ITERS=887` 或 `907`，或者设 `TRAIN_ITERS=0` 让脚本按样本数重算。`SAVE_TOTAL_LIMIT` 建议设为 2。

0916 这次在 AWS 上从 4183 训到 step 125 的 checkpoint 没有上传。要接着训就下载上一节的 `checkpoint-920`，设成 `MODEL_PATH`，`FINETUNE=true`。不要改回 `checkpoint-4183`。

0919 网上最远是上一节的 `checkpoint-60`，同样 `FINETUNE=true`。只有把 AWS 上的 Megatron `checkpoint-80` 拷过来时，才设 `FINETUNE=false` 和 `RESUME_FROM`。step 100/120/140 的分片不完整，不能当 resume。

下面是 AWS 原集群上的路径和当时的 4×8 H200 合同，用来对照，不是这边的启动路径。

不要直接执行 `scripts/start_0916_gbs256_from_scratch.sh` 或 `scripts/resume_reason_0919_from_80.sh`。这两个脚本把已经过期的 Slurm job id 写死了。用 `scripts/launch_general_reason_gbs256.sh`，把 `HOLD_JOBS` 换成当前 4 个正在跑的单节点 8 卡 hold。

登录节点是 `ip-10-1-115-182`。启动脚本会拒绝在这台机器上跑。协调进程用 `setsid nohup` 放在计算节点上，`srun` 不要超过 2 分钟。不要 `scancel` 别人的作业，也不要取消 eval / vLLM 的 hold。

## 共同合同

| 项 | 值 |
|---|---|
| 节点 | 4 × 8 H200 |
| 并行 | TP=4，PP=1，CP=1，DP=8，micro-batch=1，GBS=256，accum=32 |
| 长度 | `MAX_LEN=131072`，packing 开，`STOP_AT_DATASET_END=true` |
| 优化 | lr `2e-5`，min lr `5e-7`，warmup 20，cosine，seed 42 |
| 冻结 | `FREEZE_VIT=true`，`FREEZE_ALIGNER=false` |
| 模型类型 | `MODEL_TYPE=qwen3_5` |
| 保存 / eval | `SAVE_STEPS=20`，`EVAL_STEPS=20`，`EVAL_ITERS=8` |
| eval | YAML 里的 `val` 就是训练的 eval 集。`scripts/41_train_bc_full_megatron.sh` 传 `--val_dataset` 和 `--split_dataset_ratio 0.0`。日志里的指标名是 `eval_loss`。没有第二套官方 eval split |
| wandb | entity `zhiyuan-hu-bj-nus`，project `opensima`。key 在本机 `~/.config/wandb/opensima_mirror.key`，不在 git 里。`logging.jsonl` 只写在最后一个 rank，sidecar 是 `scripts/wandb_jsonl_sidecar.sh` |

4 节点时 `GBS = MICRO_BS * DP * accum = 1 * 8 * 32 = 256`。不要把 GBS 降到 64。

`TRAIN_ITERS=0` 时，`scripts/55_train_general_game_reasoning_3pass_128k_128gpu_gbs256_0921.slurm` 用 `scripts/lib_mixture.py --print-total-samples` 除以 GBS。已经算过的结果：0916 是 227205/256 = **887**，0919 是 232359/256 = **907**。

Checkpoint 是 Megatron DCP。一份完整 checkpoint 约 120G，64 个 `*.distcp`，外加 rank 0 才有的 `iter_*/.metadata`、`metadata.json`、`common.pt`、`args.json`、`latest_checkpointed_iteration.txt`。4 台机器各写 16 个 shard 到本地 NVMe。节点被回收后，单台机器上的分片不能 resume。以后保存到共享盘，并设 `SAVE_TOTAL_LIMIT=2`。原集群共享盘上次只剩约 1.8T（98%），不限份数会再次写满。

环境：`/fsx/home/zhiyuan/miniconda3/envs/megatron-sft`。`ENV_SETUP_MODULES=none`。

## Val 是怎么切的

两次都一样，没有事先存在的 eval parquet。

1. `scripts/06_prepare_bc_dataset.py --val_chunks 0`：全部 parquet 进 train，不按 chunk 留 val。
2. `scripts/07_merge_shuffle_genshin.py --val_size 64 --seed 42`：打乱后每个游戏取前 64 条做 val，其余做 train。
3. 三个游戏的 `val.jsonl` 按 genshin、cyberpunk2077、spiderman2 的顺序拼成 192 行。

0916 转换脚本：`scripts/convert_reasoning_0916_jsonl_on_compute.sh`。  
0919 转换脚本：`scripts/convert_reasoning_0919_jsonl_on_compute.sh`。  
必须在计算节点上跑。每个游戏各跑一次，`GAME` 取 `genshin`、`cyberpunk2077`、`spiderman2`。

拼 0916 val：

```bash
# 在持有 jsonl 的计算节点上
ROOT=/opt/dlami/nvme/zhiyuan-reasoning-jsonl
cat \
  "$ROOT/genshin_polish_reasoning_0916/_merged/val.jsonl" \
  "$ROOT/cyberpunk2077_polish_reasoning_0916/_merged/val.jsonl" \
  "$ROOT/spiderman2_polish_reasoning_0916/_merged/val.jsonl" \
  > "$ROOT/val_mixed.jsonl"
# 必须是 192 行
```

拼 0919 val 时换成 `*_reasoning_history15_0919/_merged/val.jsonl`，输出 `$ROOT/val_mixed_0919.jsonl`，同样必须是 192 行。

## 0916：从 4183 重开，不要 resume 这次丢失的 shard

这次从 4183 新开的 run 是 `reason0916_gbs256_scratch_20260926T052632Z`。停在 2026-09-27 17:06 UTC，最后训练步 **125/887**，loss 0.190。最后一次完整 eval 在 step 120，eval_loss 0.209。`checkpoint-20` 到 `checkpoint-120` 只在四台 NVMe 上，没有拷到 FSx，也没有上传 Hugging Face。四台机器都已被别人占用，git 里没有这些权重。

Hugging Face `zhiyuanhucs/Qwen3.5-9B-General-Game-reason0916` 上的 `checkpoint-600`、`checkpoint-880`、`checkpoint-920`、`main` 是更早那次 0916，不是这次从 4183 重开的 GBS=256。不要拿它们当这次的 resume。

重启：

- `FINETUNE=true`
- `AUTO_RESUME=false`
- `RESUME_FROM` 为空
- `MODEL_PATH=/fsx/home/zhiyuan/nfs/models/Qwen3.5-9B-General-Game/checkpoint-4183`  
  同一份起点也在 Hugging Face `ltzheng/Qwen3.5-9B-General-Game` 的 `checkpoint-4183`
- `MIXTURE_YAML=data_configs/general_game_reasoning_128K_0921.yaml`
- `GAME_NAME=general_reason_0916`
- `TRAIN_ITERS=887`（或留 0，让脚本按样本数重算）
- 输出目录放到新机器的共享存储，`SAVE_TOTAL_LIMIT=2`

数据（原集群 NVMe，换机器需要把 jsonl 和 images 一起拷走）：

```text
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/val_mixed.jsonl
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/genshin_polish_reasoning_0916/_merged/train.jsonl
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/cyberpunk2077_polish_reasoning_0916/_merged/train.jsonl
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/spiderman2_polish_reasoning_0916/_merged/train.jsonl
```

每个游戏 `passes: 3.0`。YAML 里的路径是绝对路径，换盘之后要改 `data_configs/general_game_reasoning_128K_0921.yaml`。

如果别人把四台机器上的 `checkpoint-120` 都拷回来了，才可以改成从 step 120 resume。四份缺一不可，尤其是 rank 0 的 metadata。原路径：

```text
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/checkpoints/reason0916_gbs256_scratch_20260926T052632Z/checkpoint-120
```

| rank | 原节点 | 原作业 | shard |
|---|---|---|---|
| 0 | ip-10-1-30-86 | 2166 | `__0`–`__7`，以及 `.metadata` / `common.pt` |
| 1 | ip-10-1-48-227 | 2170 | `__8`–`__15` |
| 2 | ip-10-1-13-41 | 2168 | `__16`–`__23` |
| 3 | ip-10-1-24-61 | 2169 | `__24`–`__31` |

约定的接收目录：`/fsx/home/zhiyuan/nfs/outputs/reason0916_nvme_shards/<主机名>/checkpoint-120`。2026-09-28 09:00 UTC 时这个目录还是空的。

## 0919：从 FSx checkpoint-80 续，不要从残缺的 140 续

能直接加载的完整 checkpoint 只有：

```text
/fsx/home/zhiyuan/nfs/outputs/reason0919_gbs256_20260925T083029Z/checkpoint-80
```

64 个 shard 齐全，目录内 `latest_checkpointed_iteration.txt` 内容是 `80`，约 120G。这是 run `reason0919_gbs256_20260925T083029Z`。后来的 resume run `reason0919_gbs256_resume80_20260926T154446Z` 停在 2026-09-27 16:00 UTC，step **160/907** 的 eval 做到 2/8。NVMe 上的 `checkpoint-100`、`120`、`140` 不完整。本机只保存了 rank 3 的分片：

```text
/fsx/home/zhiyuan/nfs/outputs/reason0919_nvme_shards/ip-10-1-31-190/
```

约 92G，含 100/120/140，只有 `__24`–`__31`。没有另外三台的分片就不能从 140 续。

Hugging Face `zhiyuanhucs/Qwen3.5-9B-General-Game-reason0919` 上有 `checkpoint-40`、`checkpoint-60`、`main`，都比 80 更早。80 没有上传。

从 80 续：

- `FINETUNE=false`（`RESUME_FROM` 非空时必须是 false）
- `AUTO_RESUME=false`
- `ALLOW_GBS_CHANGE=false`（GBS 仍然是 256，和 checkpoint 一致）
- `RESUME_FROM` 指向上面的 `checkpoint-80`
- `OUTPUT_DIR` 用新的共享目录，不要写回旧的 NVMe run 目录
- `MIXTURE_YAML=data_configs/general_game_reasoning_0919_128K.yaml`
- `GAME_NAME=general_reason_0919`
- `TRAIN_ITERS=907`
- `SAVE_TOTAL_LIMIT=2`

数据：

```text
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/val_mixed_0919.jsonl
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/genshin_reasoning_history15_0919/_merged/train.jsonl
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/cyberpunk2077_reasoning_history15_0919/_merged/train.jsonl
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/spiderman2_reasoning_history15_0919/_merged/train.jsonl
```

原神 parquet 来自 `opensima12/genshin_reasoning_history15_sl120_0919_encrypted`。Cyberpunk 和 Spider-Man 的 Hugging Face 仓库名是 `*_0920_encrypted`，那就是这套 0919 数据，不是第三套。每个游戏 `passes: 3.0`。换盘之后改 YAML 里的绝对路径。

step 140 若要拼回去，还缺这三份，每份约 31G，用 `rsync -a` 拷整个 `checkpoint-140`（里面有隐藏文件 `.metadata`）：

| rank | 原节点 | 当时占用者 | shard |
|---|---|---|---|
| 0 | ip-10-1-25-253 | weikai.huang 作业 2172 | `__0`–`__7`，以及 metadata |
| 1 | ip-10-1-59-206 | dominick.reilly 作业 2262 | `__8`–`__15` |
| 2 | ip-10-1-69-108 | yifan.zhang 作业 2273 | `__16`–`__23` |
| 3 | ip-10-1-31-190 | 已拷到 FSx | `__24`–`__31` |

源目录：

```text
/opt/dlami/nvme/zhiyuan-reasoning-jsonl/checkpoints/reason0919_gbs256_resume80_20260926T154446Z/checkpoint-140
```

接收目录：`/fsx/home/zhiyuan/nfs/outputs/reason0919_nvme_shards/<主机名>/`。四份 merge 到同一个 `checkpoint-140` 之后，再把 `FINETUNE=false`、`RESUME_FROM` 指到合并结果。在那之前用 checkpoint-80。

## 在新的 4 个 hold 上启动

把 `HOLD_JOBS` 换成 4 个正在 `RUNNING` 的单节点作业，四个节点必须不同。下面是 0919 从 checkpoint-80 续的形状；0916 重开时把 `FINETUNE` 改成 `true`，`RESUME_FROM` 置空，并换 YAML、`GAME_NAME`、`TRAIN_ITERS`、`RUN_NAME`。

```bash
# 在计算节点上，不要在登录节点上跑
export HOLD_JOBS=job0,job1,job2,job3
export LOCAL_DATA_ROOT=/opt/dlami/nvme/zhiyuan-reasoning-jsonl
export MIXTURE_YAML=/fsx/home/zhiyuan/training/data_configs/general_game_reasoning_0919_128K.yaml
export MODEL_PATH=/fsx/home/zhiyuan/nfs/models/Qwen3.5-9B-General-Game/checkpoint-4183
export OUTPUT_DIR=/fsx/home/zhiyuan/nfs/outputs/reason0919_gbs256_restart_<UTC>
export RUN_NAME=reason0919_gbs256_restart_<UTC>
export GLOBAL_BS=256 MICRO_BS=1 TP=4
export FINETUNE=false AUTO_RESUME=false
export RESUME_FROM=/fsx/home/zhiyuan/nfs/outputs/reason0919_gbs256_20260925T083029Z/checkpoint-80
export TRAIN_ITERS=907 SAVE_STEPS=20 EVAL_STEPS=20 EVAL_ITERS=8 SAVE_TOTAL_LIMIT=2
export MASTER_PORT=30119
export WANDB_ENTITY=zhiyuan-hu-bj-nus WANDB_PROJECT=opensima
export GAME_NAME=general_reason_0919
export MODEL_TYPE=qwen3_5 FI_EFA_FORK_SAFE=1
bash /fsx/home/zhiyuan/training/scripts/launch_general_reason_gbs256.sh
```

启动前确认：

- `RESUME_FROM/latest_checkpointed_iteration.txt` 存在，`iter_0000080` 里有 64 个 `*.distcp`、`.metadata`、`metadata.json`、`common.pt`
- 四个节点都能读到同一份 YAML、同一份 jsonl/images，以及同一个 `OUTPUT_DIR`
- `OUTPUT_DIR` 所在文件系统至少还能再放两份 checkpoint（约 240G），并且 `SAVE_TOTAL_LIMIT=2`
- 四个 hold 的剩余时间长于这一轮还要跑的时间。0916 全集约 887 step，0919 从 80 续到 907；原集群一步大约 1000 秒量级，eval 一轮大约 1 小时

这次死掉时的 wandb（仅供对照，不要 resume 这两个 run id）：

- 0916 scratch：https://wandb.ai/zhiyuan-hu-bj-nus/opensima/runs/ocb3idh8
- 0919 resume-from-80：https://wandb.ai/zhiyuan-hu-bj-nus/opensima/runs/mrfkgc6a
