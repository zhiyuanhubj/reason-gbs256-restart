# 原神 BC SFT —— 全量正式实验参数
> 8 节点 × 8×H200 = 64 GPU，对 Qwen3.5-9B VLM 做全参 SFT（行为克隆）。


---

## 1. 模型

| 项 | 值 |
|---|---|
| 模型 | Qwen3.5-9B VLM（`Qwen3_5ForConditionalGeneration`，GatedDeltaNet 混合架构） |
| 路径 | `/fsx/home/zhiyuan/game/extracted/training-main/models/Qwen3.5-9B` |
| 结构 | hidden=4096，layers=32（线性注意力 + 每 4 层 1 个 full attention），**KV heads=4**，vocab=248320 |
| 并行约束 | **KV heads=4 ⇒ TP 上限 = 4** |

---

## 2. 数据（全量）

| 项 | 值 |
|---|---|
| 原始 | `/fsx/home/zhiyuan/yuanshen`（3.9TB，1.75TB parquet） |
| 处理脚本 | `data-main/data_process/process_sft_data/create_bc_jsonl_genshin.py`（已用 `<|action_sep|>`，分号为普通字符） |
| 处理输出 | `/fsx/home/zhiyuan/yuanshen_processed/<玩家>/{train.jsonl, images/...}` |
| 训练数据 | `/fsx/home/zhiyuan/yuanshen_processed/_merged/{train,val}.jsonl`（合并 + 完全 shuffle + 绝对图片路径，已验证） |
| 规模 | **20 个玩家，190,062 条轨迹**（train **189,934** / val 128） |
| 每条轨迹 | 30 帧 1280×720 JPEG + 30 个 action（`<|action_start|>...<|action_end|>`，6 段按键用 `<|action_sep|>` 分隔） |
| val | **128 条**（从全量随机抽，shuffle 后切出），仅用于 eval_loss，不占训练 |
| 跳过 | `杨远`（无中间 parquet，只有原始 video，需先在 Windows 后处理生成 parquet 才能纳入） |

---

## 3. 并行 / 资源

| 参数 | 值 | 说明 |
|---|---|---|
| 节点 / GPU | 8 节点 × 8 H200 = **64 GPU** | |
| `TP` | **4** | 模型上限；TP 不跨节点（NVLink） |
| `PP` | 1 | 9B 单卡放得下 |
| **DP** | **16** | = 64 / (TP4 × PP1)，跨节点走 EFA |
| `NPROC_PER_NODE` | 8 | |

---

## 4. 完整训练超参

### 4.1 batch / 序列
| 参数 | 值 | 说明 |
|---|---|---|
| `micro_batch_size` | 1 | 多图 VLM 限制，不可增 |
| `global_batch_size` | **64** | 须被 DP=16 整除；grad_accum = 64/16 = 4 |
| `max_length` | **40960** | 30 帧放得下，无截断 |
| `IMAGE_MAX_TOKEN_NUM` | 1024 | 单图视觉 token 上限 |
| `packing` | false | |

### 4.2 学习率（warmup，但**不衰减**）
| 参数 | 值 | 说明 |
|---|---|---|
| `lr`（LLM） | **1e-5** | 9B 全参 SFT 经验值 |
| `vit_lr`（视觉塔） | **3e-6（LLM 的 1/3）** | `VIT_LR=3e-6`，保护预训练视觉编码器 |
| `aligner_lr`（merger） | = LLM lr (1e-5) | 未单独设（如需可加 `ALIGNER_LR`） |
| `lr_decay_style` | **constant** | warmup 后恒定，**不 decay** |
| warmup | **`WARMUP_ITERS=148` 步** | = 一个 epoch(≈2968 步)的 5%（不要太多）；用绝对步数精确控制 |
| `weight_decay` | 0.01 | |
| `clip_grad` | 1.0 | 梯度裁剪 |
| optimizer | Adam（β1=0.9, β2=0.95, eps=1e-8） | |

### 4.3 微调范围（全参）
| 参数 | 值 |
|---|---|
| `finetune` | true |
| `freeze_llm` / `freeze_vit` / `freeze_aligner` | 全 false（**全参微调**） |
| `vit_gradient_checkpointing` | true |

### 4.4 显存 / 性能（拉满显存）
| 参数 | 值 | 说明 |
|---|---|---|
| `RECOMPUTE` | **none** | 不重算激活，**最快最吃显存**（起跑前 dryrun 确认不 OOM；若 OOM 退 `selective`） |
| `cross_entropy_loss_fusion` | true | CE kernel 融合（TE） |
| `attention_backend` | flash | |
| `bf16` | true | |
| `sequence_parallel` | true | 仅 TP>1 有效 |

### 4.5 MTP（开启）
| 参数 | 值 |
|---|---|
| `mtp_num_layers` | **1** |
| `mtp_loss_scaling_factor` | 0.1 |

### 4.6 训练时长
| 参数 | 值 |
|---|---|
| `num_train_epochs` | **3** |

### 4.7 数据加载 / 杂项
| 参数 | 值 |
|---|---|
| `dataloader_num_workers` | 4 |
| `dataset_num_proc` | 8 |
| `seed` | 42 |
| `split_dataset_ratio` | 0.0（用独立 val_dataset） |
| `new_special_tokens` | `<|action_start|> <|action_end|> <|action_sep|> <|thought_start|> <|thought_end|>` |
| 关键环境变量 | `SWIFT_USE_MCORE_GDN=1`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、JIT 缓存→`/opt/dlami/nvme`、EFA NCCL |

---

## 5. CKPT（仅 safetensors）

| 项 | 值 | 说明 |
|---|---|---|
| 格式 | **仅 safetensors（HF 格式）** | `save_safetensors=true` |
| 优化器状态 | **不存** | `no_save_optim=true`（不存 fsdp/优化器分片） |
| RNG | 不存 | `no_save_rng=true` |
| 单个 ckpt 大小 | **18GB**（实测，4 分片 safetensors，无优化器文件） | |
| `save_steps` | **600** | 约 8700 步 → **保存约 15 次**，全部保留（多存无妨） |
| `eval_steps` | 600 | val 上算 eval_loss |
| `logging_steps` | 1 | 每步记录 |

---

## 6. 时间（8 节点，全量 189,934 训练轨迹，3 epoch）—— 实测

- **实测稳态 ~14.1 s/step**（RECOMPUTE=none, DP=16, global_bs=64, 显存 ~70GB/卡）
- 总步数 **8902**（189934/64 ≈ 2968 步/epoch × 3）
- **3 epoch ≈ 35 小时（~1.5 天）**
- warmup 148 步 = 1 epoch 的 5%
- 即 **约 1.2 ～ 1.7 天**，`--time` 设 2 天，ckpt 可续训兜底。
- `SAVE_STEPS=600` ⇒ ~8900/600 ≈ **15 次保存**。

---

## 7. Loss 曲线 / 日志（详细记载）
- **`outputs/<run>/v0-*/logging.jsonl`**：每步 `loss / mtp_1_loss / grad_norm / token_acc / lr / memory(GiB) / consumed_samples / epoch`
- **wandb**（project=bc-sft-qwen35）：offline 写本地，可 `wandb sync`；或 `WANDB_MODE=online`+登录看实时曲线
- 训完用 `scripts/42_check_format.py` 抽样验证输出格式

---

## 8. 存储估算（充足）
| 项 | 大小 |
|---|---|
| /fsx/home 剩余 | ~11 TB |
| 全量处理数据 | ~1.1-1.3 TB |
| ckpt（~15 个 × 18GB，全部保留） | ~270 GB |
| 模型本体 | 18 GB |
| **结论** | **充足** |

---

## 9. 启动命令（核对后用）

```bash
cd /fsx/home/zhiyuan/game/extracted/training-main
sbatch --nodes=8 --time=2-00:00:00 \
  --export=ALL,TRAIN_FILE=/fsx/home/zhiyuan/yuanshen_processed/_merged/train.jsonl,VAL_FILE=/fsx/home/zhiyuan/yuanshen_processed/_merged/val.jsonl,TP=4,GLOBAL_BS=64,RECOMPUTE=none,LR=1e-5,VIT_LR=3e-6,LR_DECAY_STYLE=constant,WARMUP_ITERS=148,EPOCHS=3,MTP_LAYERS=1,SAVE_STEPS=600,EVAL_STEPS=600,SAVE_TOTAL_LIMIT=50 \
  scripts/43_train_genshin_multinode.slurm
```

> 起正式跑前先 2 节点短 dryrun 确认 `RECOMPUTE=none` 不 OOM（`TRAIN_ITERS=8`、`SAVE_STEPS/EVAL_STEPS=99999`）。
