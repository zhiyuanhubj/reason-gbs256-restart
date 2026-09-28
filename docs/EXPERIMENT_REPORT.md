# 原神行为克隆 SFT —— 实验总结报告

> Job 11772 · 全量数据 · 8 节点 × 8×H200 · 已训练完成（3 epoch / 8902 步）

---

## 1. 结论速览

- ✅ **训练成功完成**：Qwen3.5-9B VLM 全参 SFT，3 epoch / 8902 步，耗时约 **1 天 11.5 小时**（8 节点 64×H200）。
- ✅ **格式学习完美**：贪心解码下动作格式正确率 **100%**（action_start/sep/end、可解析 dx dy wheel）。
- ✅ **train/eval loss 收敛且无过拟合**：train 4.879 → 0.232（最低 0.185），eval 稳定在 **0.230**，全程贴合。
- ⚠️ **已定位一个推理侧问题**：贪心解码会把鼠标/视角移动**幅度压小**（pred/GT≈0.25-0.32），改用采样可缓解（见 §5）。这是**解码方式问题，不是数据/训练 bug**。
- ✅ **15 个 ckpt 全部上传** HuggingFace [`zhiyuanhucs/sft`](https://huggingface.co/zhiyuanhucs/sft)，`main` = 最终 checkpoint-8902。

---

## 2. 数据

| 项 | 值 |
|---|---|
| 原始数据 | `/fsx/home/zhiyuan/yuanshen`（3.9TB，20 玩家，1.75TB parquet） |
| 处理 | `create_bc_jsonl_genshin.py`：parquet→jsonl+jpg，30 帧/轨迹，5px 量化，**95% 空闲帧过滤** |
| 动作格式 | `<\|action_start\|>dx dy wheel<\|action_sep\|> k1..<\|action_end\|>`（6 段按键用 `<\|action_sep\|>` 分隔；分号为普通字符） |
| 合并+shuffle | 20 玩家合并、完全 shuffle、图片转绝对路径 |
| 训练集 | **train 189,934 / val 128** 条轨迹（每条 30 帧 1280×720 + 30 动作） |
| 跳过 | `杨远`（仅原始 video 无 parquet） |
| 数据特征 | dx=dy=0 占 37%（其中 94.6% 是"按键但不转视角"的有效动作，纯空闲仅 2.2%）；鼠标位移中位 20px、长尾 |

---

## 3. 训练配置

| 类别 | 参数 | 值 |
|---|---|---|
| 模型 | Qwen3.5-9B VLM | hidden=4096, 32层(GatedDeltaNet混合), KV heads=4, vocab=248320 |
| 并行 | TP / PP / DP | **4 / 1 / 16**（64 GPU；TP 走 NVLink，DP 跨节点走 EFA） |
| batch | micro / global | 1 / 64（grad_accum=4） |
| 序列 | max_length / img_tok | 40960 / 1024 |
| 学习率 | LLM / ViT / aligner | **1e-5 / 3e-6(1/3) / 1e-5** |
| LR 调度 | warmup / decay | **148 步(=1 epoch 的 5%) / constant(不衰减)** |
| 微调 | 范围 | 全参（LLM+ViT+aligner 都不冻结） |
| MTP | num_layers / scale | 1 / 0.1 |
| 显存优化 | recompute | none（拉满显存，实测 ~70GB/卡） |
| 精度 | bf16 + flash-attn + CE fusion | |
| special tokens | 5 个全注册 | action_start/end/sep, thought_start/end |
| 时长 | epoch / 步数 / 用时 | 3 / 8902 / ~35.5h |

---

## 4. 训练曲线与数值

完整曲线：`outputs/loss_curve.png`（train/eval loss、log 尺度、MTP loss、grad norm 四联图）

**train loss**：4.879 起步 → 前 ~40 步断崖式下降 → step 100 收敛到 ~0.25 → 末 0.232（最低 0.185）。
**grad norm**：早期峰值 ~85（special token embedding 初始化调整）→ step 40 后稳定个位数。
**MTP head loss**：与主 loss 同步收敛到 ~0.23。

| step | train_loss | eval_loss |
|---:|---:|---:|
| 600 | 0.236 | 0.2446 |
| 1200 | 0.255 | 0.2390 |
| 1800 | 0.238 | 0.2365 |
| 2400 | 0.259 | 0.2339 |
| 3000 | 0.216 | 0.2330 |
| 3600 | 0.220 | 0.2322 |
| 4200 | 0.219 | 0.2316 |
| 4800 | 0.237 | 0.2302 |
| 5400 | 0.237 | 0.2301 |
| 6000 | 0.206 | 0.2305 |
| 6600 | 0.213 | 0.2303 |
| 7200 | 0.221 | 0.2305 |
| 7800 | 0.226 | 0.2309 |
| 8400 | 0.219 | 0.2306 |
| 8902 | 0.232 | 0.2301 |

> eval_loss 从 0.245 平滑降到 ~0.230 后**走平**（约 step 4800 起），train≈eval 全程贴合 → 收敛良好、无过拟合。3 个 epoch 充分；继续训练边际收益已很小。
> 完整数值：`outputs/loss_table.txt`；逐步原始记录：`outputs/.../logging.jsonl`（每步 loss/mtp/grad_norm/lr/显存）。

---

## 5. Val 评测（动作质量）

在完整 val（128 轨迹 × 每条 30 turn，8 卡并行）上推理，量化预测动作 vs 人类 GT：

### 最终模型 checkpoint-8902
| 指标 | 贪心(temp=0) | 采样(temp=1) |
|---|---|---|
| 格式正确率 | **100%** | 81.6% |
| pred 平均幅度 | 21.8 | 65.9 |
| GT 平均幅度 | 69.1 | 80.7 |
| **pred/GT 幅度比** | **0.316** | **0.816** |
| pred 完全不动占比 | 61.2% | 42.8% |

### 中途 checkpoint-5400（对照）
| 指标 | 贪心(temp=0) | 采样(temp=1) |
|---|---|---|
| 格式正确率 | 100% | 100% |
| pred/GT 幅度比 | 0.253 | **0.766** |
| pred 完全不动 | 58.5% | 45.5% |

逐条结果（含完整输入+GT+预测）：`outputs/val_full_checkpoint-{5400,8902}/T{0.0,1.0}_all.jsonl`，已打包 `/fsx/home/zhiyuan/sft_val_eval_ckpt5400.zip`。

---

## 6. 已知问题：动作幅度偏小

**现象**：测试时模型输出的鼠标/视角移动幅度明显小于人类、常"不动"。

**原因**：**解码（推理）问题，非数据/训练 bug**。同一 ckpt 同一批样本：贪心解码幅度只有真值的 ~25-32%、56-61% 输出 0；改用采样幅度恢复到 77-82%；两种方式格式都能正常解析。
根本机理：鼠标位移在动作串里仅占 3-5 个 token，是长尾高熵数字，加上数据 37% 的步鼠标本就为 0，模型学到的概率分布**众数落在小值/0**，贪心每步取 argmax 即系统性塌缩幅度。

**解决办法**：
1. **推理侧（立即可用）**：用采样解码而非贪心。注意最终 ckpt 在 temp=1 时格式正确率掉到 81.6%（分布更尖锐，采样易采出畸形），建议用**中等温度 + top_p**（如 temp≈0.7, top_p≈0.9）在"幅度"和"格式"间取平衡，或对鼠标段单独提温。
2. **训练侧（治本，可选）**：① 对 dx/dy 数字 token 做 loss 加权；② 把鼠标位移**分桶成离散动作 token**（类 VPT），将数字回归转为分类，让大幅度在 argmax 下也能稳定出现。

---

## 7. 产物清单

| 产物 | 位置 |
|---|---|
| 全部 15 个 ckpt | HF [`zhiyuanhucs/sft`](https://huggingface.co/zhiyuanhucs/sft)（分支 checkpoint-600~8902，main=8902） |
| 训练 loss 曲线 | `outputs/loss_curve.png` / 前100步 `outputs/loss_first100.png` |
| loss 数值表 | `outputs/loss_table.txt` / 逐步 `outputs/.../logging.jsonl` |
| val 评测逐条 | `outputs/val_full_checkpoint-{5400,8902}/T{0,1}_all.jsonl` |
| 训练脚本 | `scripts/41_train_bc_full_megatron.sh` + `43_train_genshin_multinode.slurm` |
| 数据处理 | `data-main/.../create_bc_jsonl_genshin.py` + `07_merge_shuffle_genshin.py` |
| 评测脚本 | `scripts/42b_eval_actions.py` + `42c_summarize.py` |
| 实验参数 | `docs/EXPERIMENT_PLAN.md` |

---

## 8. 训练/工程过程中解决的关键问题

1. **环境**：全新建 `megatron-sft` conda 环境（torch2.8/cu128 + ms-swift + megatron-core + TE + fla/tilelang）。
2. **cuDNN 主库/子库错配**：TE 用系统 cuda-12.8 编译，import 时加载系统 cuDNN 主库，子库却从 pip 解析 → 视觉塔卷积崩。修复：env.sh 把系统 `cuda-12.8/lib` 完整 cuDNN 置于 LD_LIBRARY_PATH 最前。
3. **Triton/TileLang JIT 缓存竞态**：多节点并发写 Lustre 上 `~/.triton` 触发 Bad address。修复：缓存改到节点本地 `/opt/dlami/nvme`。
4. **SLURM 多节点**：坏节点规避、`getent ahostsv4` 解析 MASTER IPv4（避免 IPv6 link-local）、EFA NCCL 配置、账号/分区适配（xgen-mm 标准分区）。
