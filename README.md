# megatron-sft-qwen35

0916 / 0919 三游戏 reasoning、GBS=256、4×8 H200 的续训合同（起点权重、val 切分、checkpoint、启动命令）见 [`RESTART_REASON_GBS256.md`](RESTART_REASON_GBS256.md)。

游戏行为克隆（Behavior Cloning）VLM **SFT 训练流水线**，基于 [ms-swift](https://github.com/modelscope/ms-swift) 的 **Megatron 后端** + Qwen3.5-VL 系列模型。

> 8 节点、128K、纯原神数据的可移植下载/处理/训练/恢复说明见
> [`docs/genshin_128k_portable.md`](docs/genshin_128k_portable.md)。该文档和
> `scripts/47_train_genshin_only_128k_multinode.slurm` 包含经过 64 GPU
> save-stop-resume 验证的训练合同；当前脚本默认资源已适配 Isambard 的
> 2 节点 × 4 GH200、GBS=8。
>
> Isambard ARM64/GH200 的 Conda、CUDA 12.9、FlashAttention、Transformer
> Engine 和 Apex 完整重装流程见
> [`docs/isambard_gh200_environment.md`](docs/isambard_gh200_environment.md)。

设计目标：
- 多模态多图（VLM）+ 全参数微调，开箱即用
- 单机多卡 H100/H200，支持 4B / 30B / 70B 切换（参数化的并行策略）
- 一键起 daemon、wandb 实时监控、ckpt 自动校验输出格式
- 把项目目录拷到任何机器都能跑（路径全部自动探测）

## 当前最快的 128K 四节点配置

同一 Genshin speed subset、GBS=128、micro-batch=1、MTP=1 的实测最快配置是：

- 4 节点 × 8 H200
- TP=4、PP=1、CP=1、sequence parallel
- ViT-TP 开启
- full recompute（uniform，每次 1 层）
- 128K binpack packing

排除首步后平均约 **296.5 s/it**，最后两步约 **290 s/it**，记录显存约
**67.35 GiB/GPU**。启动时传入四个已经运行的单节点 allocation：

```bash
HOLD_JOBS=1001,1002,1003,1004 \
NFS_DIR=/path/to/shared/cache \
LOCAL_DATA_ROOT=/opt/dlami/nvme/$USER/genshin-training \
RECOMPUTE=full CP=1 VIT_TP=true TRAIN_ITERS=5 \
  bash scripts/launch_speed_trial_4nodes.sh
```

四台机器需要提前具有相同的 node-local dataset 路径。launcher 会从 Slurm
allocation 自动解析节点和 rendezvous 地址，不包含固定 job ID 或主机名。

---

## 0. 快速开始（5 步）

```bash
# 1) 装环境（1 次，~10 min）
conda create -n megatron-sft python=3.10 -y
bash scripts/00_install_env.sh
python tools/verify_env.py                                  # 全绿即 OK

# 2) 下载基础模型 (~9 GB / 6 min on hf-mirror)
bash scripts/01_download_model.sh                           # 默认 Qwen3.5-4B
# 大模型: MODEL_NAME=Qwen/Qwen3.5-30B-A3B bash scripts/01_*.sh

# 3) 下载并预处理数据集
HF_TOKEN=hf_xxxxx python scripts/05_download_dataset.py     # 18GB / ~10 min
python scripts/06_prepare_bc_dataset.py --num_workers 6     # parquet -> jpg + jsonl, ~20s

# 4) wandb 登录（1 次）
python -c "import wandb; wandb.login(key='YOUR_KEY')"

# 5) 起训练（**真 daemon**，SSH 断不影响）
bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh

# 训完: 抽样验证 ckpt 是否真学会了输出格式
BEST_CKPT=$(ls -td outputs/delta_force_megatron_full_*/iter_* | head -1)
python scripts/42_check_format.py "$BEST_CKPT"
```

---

## 1. 文件结构

```
megatron-sft-qwen35/
├── README.md                              # 本文件
├── env.sh                                 # 环境激活（PROJECT_ROOT 自动探测）
├── requirements.txt                       # 依赖参考清单
├── scripts/
│   ├── 00_install_env.sh                 # 装环境（含 flash-attn / TE / fla / tilelang / megatron-core）
│   ├── 01_download_model.sh              # 下基础模型（默认 hf-mirror）
│   ├── 05_download_dataset.py            # 下 BC SFT 数据（HF dataset）
│   ├── 06_prepare_bc_dataset.py          # parquet → jsonl + jpg；切 train/val
│   ├── 41_train_bc_full_megatron.sh      # ★ 正式训练（Megatron 全参 + MTP=1 + TP=4）
│   ├── 41_train_bc_full_megatron_dryrun.sh   # 5 step dryrun，验证机器/参数
│   ├── 42_check_format.py                # ★ 训完用 val 抽样 + 7 项格式指标
│   ├── 43_train_genshin_multinode.slurm  # 多节点 Megatron-SWIFT 入口
│   ├── 45_train_llava_onevision_128k_multinode.slurm # LLaVA-OneVision 128K 预设
│   ├── lib_mixture.py                    # passes 配额、Parquet mixture 与 manifest 工具
│   ├── 31_infer.sh                       # 单卡 stream 推理
│   ├── 99_convert_mcore_to_hf.sh         # 万一拿到 mcore-format ckpt 转回 HF
│   └── run_daemon.sh                     # 把任意训练脚本变成真 daemon (setsid + nohup)
├── tools/
│   └── verify_env.py                     # 装完环境后跑一遍验证
├── docs/
│   ├── faq.md                            # 排错手册
│   └── scaling_to_large_models.md        # ★ 切到 30B/70B 的并行/显存策略
├── data/                                 # 训练数据（gitignore）
│   ├── delta-force-bc-sft-shuffled/     # 原始 parquet
│   └── bc_sft_processed/                # 解码后的 jsonl + jpg images
├── models/                               # 基础模型权重（gitignore）
├── outputs/                              # 训练 ckpt（gitignore）
└── logs/                                 # 训练日志（gitignore）
```

---

## 1.1 多数据集 mixture 与 LLaVA-OneVision 128K

`data_configs/mixture.example.yaml` 是 JSONL 多游戏混训模板；每个数据集的 `passes`
表示本次任务总共看到该数据的次数。`data_configs/llava_onevision.yaml` 是纯
LLaVA-OneVision 的完整 Parquet 子数据集模板；
`data_configs/genshin_llava_onevision_128K.yaml` 则用于原神与 JSONL 版
LLaVA-OneVision 的 128K 混训。

LLaVA-OneVision 数据应放在 `$NFS_DIR/data/` 下，至少包含：

```text
$NFS_DIR/data/LLaVA-OneVision-1.5-Instruct-Data-qwen-format/
  canonical_index.json
$NFS_DIR/data/LLaVA-OneVision-1.5-Instruct-Data-qwen-format-val/
```

先做两节点短跑，确认数据、NCCL 和 128K 显存配置：

```bash
NFS_DIR=/path/to/shared/cache \
sbatch --nodes=2 --export=ALL,TRAIN_ITERS=8,SAVE_STEPS=99999,EVAL_STEPS=99999 \
  scripts/45_train_llava_onevision_128k_multinode.slurm
```

正式训练使用四节点：

```bash
NFS_DIR=/path/to/shared/cache \
sbatch scripts/45_train_llava_onevision_128k_multinode.slurm
```

该脚本当前默认使用 Qwen3.5-9B、`MAX_LEN=131072`、128K packing、
`IMG_MAX_TOK=1024`、`TP=8`、`MICRO_BS=1`、四节点 `GLOBAL_BS=128`、
sequence parallel、MCore GDN、full recompute、TE fused CE 和 MTP=1。所有值均可经
`sbatch --export=ALL,KEY=value` 覆盖。它不依赖 Lepton。
若提交节点的默认 `python` 没有 PyYAML，额外传入
`PYTHON_BIN=/path/to/megatron-sft/bin/python`。

### 原神 3 遍 + LLaVA-OneVision 0.162 遍

混训配置按 `llava_onevision.yaml` 的形式列出全部 134 个 LLaVA 子集：原神
`passes=3.0`，每个 LLaVA 子集 `passes=0.162`。passes 是样本曝光倍率，不是 token
比例；实际 interleave 概率按各子集的 `样本数 × passes` 计算。

默认数据布局：

```text
$NFS_DIR/data/genshin_bc_sl135/_merged/train.jsonl
$NFS_DIR/data/LLaVA-OneVision-1.5-Instruct-Data-qwen-format/
  CLEVR/*.jsonl                  # 也支持 CLEVR.jsonl
  CLEVR-Math/*.jsonl
  ...                            # 共 134 个命名子集
$NFS_DIR/data/LLaVA-OneVision-1.5-Instruct-Data-qwen-format-val/
  _merged/val.jsonl
```

四节点正式提交：

```bash
NFS_DIR=/path/to/shared/cache \
sbatch scripts/46_train_genshin_llava_onevision_128k_multinode.slurm
```

路径不符合上述布局时，可分别覆盖：

```bash
sbatch --export=ALL,NFS_DIR=/cache,GENSHIN_TRAIN_FILE=/data/genshin.jsonl,\
LLAVA_DATA_ROOT=/data/llava_subsets,LLAVA_VAL_FILE=/data/llava_val.jsonl \
  scripts/46_train_genshin_llava_onevision_128k_multinode.slurm
```

`46` 是完整独立的实验配置，不继承 `45`。默认关键参数为：

```text
8 nodes × 8 GPUs  TP=4  PP=1  DP=16
MICRO_BS=1  GLOBAL_BS=128  grad-accum=8
MAX_LEN=131072  PACKING=true  PACKING_LENGTH=131072
LR=1e-5  VIT_LR=3e-6  WARMUP_ITERS=100  RECOMPUTE=full
CROSS_ENTROPY_FUSION_IMPL=te  EVAL_STEPS=5  SAVE_STEPS=100
MTP_LAYERS=1  IMG_MAX_TOK=832
```

WandB 会记录累计总 token `trained_tokens/total`，并通过 `game → channel` 和
`data_source → channel` 分别统计原神及各 LLaVA 子集的 loss/token。当前小验证集
每次运行 `EVAL_ITERS=2`；checkpoint 默认完整保存 optimizer、scheduler、RNG 和
iteration，`SAVE_TOTAL_LIMIT=0` 表示不自动删除。

---

## 2. 数据格式

数据集是 ShareGPT 风格的 parquet，每条 trajectory = **N 帧 720p 图 + N 个 action 文本**（N = 转换时的 `--sequence_length`）。

> **token 预算（务必和 MAX_LEN 对齐）**：Qwen3.5 一张 720P 图 ≈ **900+ token**，所以 32K(`MAX_LEN=32768`) 大约能放 **~30 帧** + system(~1.7k) + 文本。
> 改帧数时同步改 `41_train` 的 `MAX_LEN` / `IMG_MAX_TOK`：`MAX_LEN >= 1700 + N×(每图token + ~40)`。转换脚本 `create_bc_jsonl_genshin.py` 跑完会打印这条提醒。

```python
# 1 行 parquet:
{
    "images": [jpeg_bytes_1, ..., jpeg_bytes_N],           # N 张 720p JPEG
    "conversations": [
        {"from": "system", "value": "你是一名经验丰富的游戏玩家..."},
        {"from": "human",  "value": "<image>"},                   # 第 1 帧
        {"from": "gpt",    "value": "<|action_start|>dx dy wheel<|action_sep|> W Shift<|action_sep|> ...<|action_end|>"},
        # ... N 轮
    ]
}
```

`scripts/06_prepare_bc_dataset.py` 把 parquet 转成 ms-swift 标准 messages 格式：

```jsonl
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "<image>"},
    {"role": "assistant", "content": "<|action_start|>...<|action_end|>"},
    ...
  ],
  "images": ["images/chunk_000/row_0000/00.jpg", ...],   # 相对路径，整目录可整体搬走
  "game": "delta_force"          # 标记游戏，给 wandb tag 用
}
```

> 原神（Genshin）的数据可以用 data 仓库里的 `data_process/process_sft_data/create_bc_jsonl_genshin.py`
> 直接从加密的中间 parquet 一步转成这里的 jsonl + jpg（自带相对路径），不用先落地中间 BC parquet。

**Action 文本格式**：`<|action_start|>dx dy wheel<|action_sep|> k1 k2<|action_sep|> k3<|action_sep|> k4 k5<|action_sep|> k6<|action_sep|> k7<|action_sep|> k8<|action_end|>`
- `dx dy wheel` = 累计鼠标位移 + 滚轮（dx>0 右、dy>0 下、wheel>0 上滚），按 5px 量化
- 6 段按键（200ms 内 6 个 33ms 帧），**用 special token `<|action_sep|>` 分隔**（分号 `;` 现在是合法可输出字符，不能再当分隔符）；字母键大写 `W`/`A`/`D`，鼠标 `LB`/`RB`/`MB`，修饰键首字母大写 `Shift`/`Ctrl`，F 键 `F1`/`F2`...

**5 个 special tokens 必须注册**（脚本里 `--new_special_tokens` 自动做）：
- `<|action_start|>` / `<|action_end|>` / `<|action_sep|>` / `<|thought_start|>` / `<|thought_end|>`
- 没注册的话每个 token 会被切碎成多个 sub-token，模型学不到原子边界

---

## 3. 训练脚本配置

`scripts/41_train_bc_full_megatron.sh` 所有参数都通过环境变量可覆盖。**完整列表**：

| 类别 | 变量 | 默认值 | 含义 |
|---|---|---|---|
| **数据/模型** | `MODEL_PATH` | `models/Qwen3.5-4B` | 基础模型路径 |
| | `TRAIN_FILE` | `data/bc_sft_processed/train.jsonl` | 训练 jsonl |
| | `VAL_FILE` | `data/bc_sft_processed/val_small.jsonl` | val（默认 32 条小集） |
| | `GAME_NAME` | `delta_force` | wandb tag + run_name 前缀 |
| | `RUN_NAME` | `${GAME_NAME}_megatron_full_<ts>` | 自定义 run name |
| **GPU/并行** | `GPUS` | `0,1,2,3` | CUDA_VISIBLE_DEVICES |
| | `NPROC_PER_NODE` | `4` | 进程数 = NPROC = TP × PP × DP |
| | `TP` | `4` | tensor parallel |
| | `PP` | `1` | pipeline parallel |
| | `SEQ_PARALLEL` | `true` | 序列并行（仅 TP>1 有效）|
| **batch** | `MICRO_BS` | `1` | 单次 forward 样本数（多图必须 1） |
| | `GLOBAL_BS` | `8` | 一次梯度更新看的样本数 |
| **训练超参** | `EPOCHS` | `3` | epoch 数 |
| | `LR` | `1e-5` | 全参 SFT 经验值 |
| | `MIN_LR` | `1e-6` | cosine 衰减下限 |
| | `WARMUP_FRAC` | `0.05` | warmup 占总步数比例 |
| | `WEIGHT_DECAY` | `0.01` | AdamW |
| | `MAX_LEN` | `32768` | 序列长度上限（容纳 20×1024 视觉 + 文本） |
| | `IMG_MAX_TOK` | `1024` | 单图视觉 token 数 |
| **MTP** | `MTP_LAYERS` | `1` | 多 token 预测头数；0 关 |
| | `MTP_LOSS_SCALE` | `0.1` | MTP loss 权重 |
| **ckpt/log** | `SAVE_STEPS` | `100` | 每 N step 存 ckpt |
| | `EVAL_STEPS` | `100` | 每 N step 在 val 上 eval |
| | `SAVE_TOTAL_LIMIT` | `3` | 只留最近 3 个 ckpt |
| | `LOGGING_STEPS` | `1` | 每 step 都 log（wandb 曲线密） |

**常见调参**：
```bash
# 更长训练（5 epoch）
EPOCHS=5 bash scripts/41_train_bc_full_megatron.sh

# 切游戏（PUBG）
GAME_NAME=pubg \
  TRAIN_FILE=data/pubg_processed/train.jsonl \
  VAL_FILE=data/pubg_processed/val.jsonl \
  bash scripts/41_train_bc_full_megatron.sh

# 关 MTP（如果 30B 显存吃紧）
MTP_LAYERS=0 bash scripts/41_train_bc_full_megatron.sh

# 只用 GPU 2,3
GPUS=2,3 NPROC_PER_NODE=2 TP=2 \
  IMG_MAX_TOK=512 MAX_LEN=24576 \
  bash scripts/41_train_bc_full_megatron.sh
```

---

## 4. Daemon 模式（强烈推荐）

直接 `bash scripts/41_*.sh` 起的训练，是当前 shell 的子进程，**SSH 断 / cursor 关 / 父 shell 死会被 SIGHUP 杀掉**。生产环境务必用 daemon：

```bash
bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh
# 或者带参数:
bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh GAME_NAME=pubg EPOCHS=5
```

`run_daemon.sh` 用 `setsid + nohup + disown` 把训练 reparent 到 init (PPID=1)，stdin → /dev/null，stdout/stderr → `logs/daemon_<ts>.log`。这样 SSH 断完全不影响。

**管理 daemon**：
```bash
pgrep -af "megatron sft"                                     # 查是否在跑
tail -f logs/delta_force_megatron_full_*.log                 # 看进度
pkill -SIGTERM -f "megatron sft"                             # 优雅停（保存最后 ckpt）
pkill -SIGKILL -f "megatron sft"                             # 强杀（不存 ckpt）
```

---

## 5. Wandb 实时监控

`env.sh` 默认 `WANDB_PROJECT=bc-sft-qwen35  WANDB_ENTITY=OpenSIMA`。每个 run：
- **run name** = `<GAME_NAME>_megatron_full_<timestamp>`
- **tags** = `<game>, full_sft, vlm, qwen3_5_4b, megatron, mtp1`
- **每 1 step** 上报：`loss / mtp_1_loss / lr / grad_norm / token_acc / epoch / num_input_tokens_seen / memory(GiB)`
- **每 100 step** 上报：`eval/loss / eval/token_acc`
- **每 ~30 s** 系统级 GPU 显存

切换 entity / project：
```bash
WANDB_PROJECT=my-proj WANDB_ENTITY=my-team bash scripts/run_daemon.sh scripts/41_*.sh
```

**多游戏对比**：每个游戏一个 run（按 `GAME_NAME` 自动区分），dashboard 上按 `tags` 过滤就能并排比较 loss 曲线 / 收敛速度。

---

## 6. 训完验证：ckpt 是否真学会了输出格式

`scripts/42_check_format.py` 抽样 val 数据让模型推理，统计 7 项指标 + edit distance：

```bash
BEST_CKPT=$(ls -td outputs/delta_force_megatron_full_*/iter_* | head -1)
python scripts/42_check_format.py "$BEST_CKPT" --num_samples 50 --gpu 0
```

输出示例：
```
=================== FORMAT CHECK SUMMARY ===================
  ckpt              : outputs/.../iter_001000
  total inferred    : 50 (errors: 0)
  contains_start    :   50/50  (100.0%)
  contains_end      :   50/50  (100.0%)
  well_formed       :   50/50  (100.0%)
  parses_xyz        :   48/50  ( 96.0%)
  has_action_sep    :   50/50  (100.0%)
  exact_match       :   12/50  ( 24.0%)
  avg edit distance : 18.4
=============================================================
```

每条详情写到 `outputs/<run>/format_check_<step>.jsonl` 让你抽查 (gt, pred)。

---

## 7. 切换更大模型（30B / 70B / MoE）

**详见** [`docs/scaling_to_large_models.md`](docs/scaling_to_large_models.md)。要点速查：

| 模型 | 推荐配置 | 单卡显存 (H200 141GB) |
|---|---|---|
| **Qwen3.5-4B** (现在) | 4 卡 TP=4 PP=1, micro=1, MTP=1 | ~31 GB |
| **Qwen3.5-30B-A3B** (MoE) | 8 卡 TP=4 EP=2 PP=1, micro=1, MTP=0 | ~70 GB |
| **Qwen3.5-72B** (dense) | 8 卡 TP=8 PP=1 + offload optimizer, MTP=0 | ~110 GB |
| 跨节点 | TP=8 PP=2 + sequence_parallel | 看 layer 切分 |

通用扩展原则：
1. 显存不够先开 `recompute_granularity full` + `recompute_method uniform` + `recompute_num_layers 1`
2. 还不够 → `--use_distributed_optimizer` + `--optimizer_cpu_offload`
3. 70B+ 上 PP=2/4，Megatron 自动做 1F1B pipeline schedule
4. MoE 模型加 `--expert_model_parallel_size`
5. 极大模型先关 MTP（额外 1 层 head 的显存压力）

---

## 8. 推理 / 部署

```bash
# 单卡交互式推理
bash scripts/31_infer.sh outputs/delta_force_megatron_full_*/iter_001000 GPU_ID=0

# 如果训练存的是 mcore 格式（开了 PP 时可能），先转 HF
bash scripts/99_convert_mcore_to_hf.sh outputs/.../iter_001000
```

部署到 vLLM 推理（生产环境）：训练默认 `--save_safetensors true`，存出来直接是 HF 格式可被 vLLM 加载。

---

## 9. 关于 MTP（Multi-Token Prediction）

**MTP 是 DeepSeek-V3 推广的训练技巧**：标准 LM 只预测 t+1，MTP 加 N 个**辅助 head** 同时预测 t+2, t+3, ..., t+1+N。
- 主 head 推理时正常用，辅助 head 训完丢
- 论文报告 N=1 大约等效于 +30% 数据量的提升
- ms-swift Megatron 后端通过 `--mtp_num_layers N` 开（HF 后端不支持）

**当前默认 N=1**。在 wandb 里能看到 `mtp_1_loss` 字段，应该比主 `loss` 略高（因为预测更远）。极大模型显存吃紧时关掉 (`MTP_LAYERS=0`)。

---

## 10. 排错

详见 [`docs/faq.md`](docs/faq.md)。常见：

| 现象 | 解决 |
|---|---|
| OOM at start | 加 `recompute_num_layers`、降 `IMG_MAX_TOK`/`MAX_LEN`、加 TP |
| OOM after step 50 | 激活值积累，开 `--optimizer_cpu_offload` 或加 `recompute_method uniform` |
| 卡 0 被别人占 | `GPUS=1,2,3,...` + 相应 NPROC，但 TP 必须能整除 hidden_size |
| `<\|action_start\|>` 输出被切碎 | 检查 ckpt 里 tokenizer.json 的 added_token 是否含全部 5 个 special token (含 `<\|action_sep\|>`) |
| eval_loss 比 train_loss 高很多 | 过拟合，降 `EPOCHS` 或加 dropout |
| wandb 看不见 | check `OpenSIMA` team 权限；或 `WANDB_ENTITY=` unset 用个人账号 |
| 训练慢但 GPU 利用率低 | dataloader 卡，加 `--dataloader_num_workers` 或减小 `--max_length` |
| Hopper GPU 报 fla `chunk_bwd_dqkwg` | 装 `tilelang`（见 faq.md） |
