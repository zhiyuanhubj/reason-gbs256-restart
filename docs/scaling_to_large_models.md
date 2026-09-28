# 切到更大模型的并行/显存策略

本文档解释如何把当前 Qwen3.5-4B 的训练流水线扩展到 30B / 70B / MoE 大模型。

---

## 0. TL;DR 速查表

| 模型 | 参数量 | 推荐并行 | 单卡显存 (H200 141GB) | 备注 |
|---|---|---|---|---|
| **Qwen3.5-4B** (现在) | 4B dense + GDN | TP=4 PP=1 DP=1 | ~31 GB | MTP=1 ok |
| **Qwen3.5-8B** | 8B dense | TP=4 PP=1, recompute full | ~50 GB | MTP=1 ok |
| **Qwen3.5-30B-A3B** | 30B MoE (3B activated) | TP=4 EP=2 PP=1, micro=1 | ~70 GB | MTP=0 推荐 |
| **Qwen3.5-72B** | 72B dense | TP=8 PP=1 + dist optim + cpu offload | ~110 GB | MTP=0 |
| **跨节点 (2×8卡)** | 70B+ | TP=8 PP=2 SP=true | 看 layer 切分 | TP 不跨节点 |

> 数字是经验值，要做 dry-run 确认。一定 OOM 时按本文 § 4 顺序往下 fallback。

---

## 1. 修改训练脚本的步骤

`scripts/41_train_bc_full_megatron.sh` 所有关键参数都参数化了。改大模型只需改 4 个地方：

```bash
# 1) 模型路径（先用 scripts/01_download_model.sh 下载）
MODEL_NAME=Qwen/Qwen3.5-30B-A3B bash scripts/01_download_model.sh

# 2) 起训练，覆盖并行参数 + 一些显存优化
MODEL_PATH=models/Qwen3.5-30B-A3B \
  GPUS=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  TP=4 PP=1 \
  MICRO_BS=1 GLOBAL_BS=8 \
  IMG_MAX_TOK=512 MAX_LEN=24576 \
  MTP_LAYERS=0 \
  bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh
```

**额外参数**（要进 `41_*.sh` 修改的）：MoE 时加 `--expert_model_parallel_size 2`，offload 时加 `--use_distributed_optimizer --optimizer_cpu_offload --optimizer_offload_fraction 1.0`。下面 § 5 给出 30B/70B 完整脚本片段。

---

## 2. 三种并行的取舍

| 类型 | 原理 | 显存压力 | 通信 | 限制 |
|---|---|---|---|---|
| **TP** (Tensor Parallel) | hidden 维度切 N 份 | 模型权重 1/N、激活 1/N (开 SP) | TP 内部高频 all-reduce / all-gather | hidden_size 必须能整除 TP；TP 跨节点慢 |
| **PP** (Pipeline Parallel) | layer 切 N 段 | 模型权重 1/N | 1F1B 调度，pipeline bubble 浪费 | 需 micro_batch ≥ PP 才能流水线满载 |
| **DP** (Data Parallel) | 不切模型，每卡完整 | **无减少**；优化器 + 梯度可用 ZeRO 切 | 反向后 grad all-reduce | 模型必须能塞单卡 |
| **EP** (Expert Parallel) | MoE 专家分散到不同卡 | expert 参数 1/N | alltoall expert routing | 仅 MoE |
| **SP** (Sequence Parallel) | 序列长度切 N（必须配 TP）| 激活值再切 1/N | TP-region 内的额外 all-gather | 仅在 TP>1 有意义 |

**经验法则**：
1. 模型小（≤8B 单卡能塞）：DP=N（4 卡时 DP=4，TP=1），简单粗暴
2. 模型中等（4B–30B）：TP=4 + DP=N/4，**当前 4B 用 TP=4 是为了显存留余量**
3. 模型大（30B+）：TP=8（一个 NUMA / 一个 NVLink island 内）+ PP=N/8 跨节点
4. MoE 模型：TP × EP = 总卡数，dense 部分 TP=4，专家 EP=2 是常见组合

> **TP 不要跨节点**（NVLink 才扛得住高频通信，跨节点 IB 会让训练慢 5-10 倍）。

---

## 3. 显存预算公式（粗算）

对 **bf16 训练 + AdamW**：

```
单卡显存 ≈ 模型参数 (BF16, 2 byte/param) / TP
        + AdamW 状态 (FP32 m + v, 8 byte/param) / TP / DP        ← ZeRO-1/dist optim 时才/DP
        + 梯度 (BF16, 2 byte/param) / TP
        + 激活值 (与 batch、seq_len、hidden 相关) / TP / SP
        + ViT 部分 (Qwen3.5-VL ViT 通常 ~2 B 参数, 不切 TP)
```

**Qwen3.5-30B-A3B (MoE) 实算**（8 卡 TP=4 EP=2）：
- LLM 参数 30B × 2 byte = 60 GB → /TP=4 = 15 GB
- Adam 30B × 8 = 240 GB → /TP=4/EP=2 = 30 GB （但稀疏激活，实际可能更低）
- 激活 (max_len 24K, bs 1, hidden 2048) ≈ 25 GB → /TP/SP = 6 GB
- ViT ~2B × 2 = 4 GB
- **合计 ~55 GB** ← 留 50% 余量到 H200 的 141GB

**Qwen3.5-72B (dense) 实算**（8 卡 TP=8）：
- LLM 72B × 2 / 8 = 18 GB
- Adam 72B × 8 / 8 = 72 GB → 必须开 `optimizer_cpu_offload` 才行
- 激活 ~ 30 GB / 8 / 2 (SP) = 2 GB
- 合计 ~ 90 GB（含 offload 后的 buffer）

---

## 4. OOM 时的 fallback 顺序

按这个顺序加，每加一个跑 5-step dryrun 看是否能起：

```
1. 加 recompute (默认 selective core_attn → full uniform)
       --recompute_granularity full --recompute_method uniform --recompute_num_layers 1
2. 降 IMG_MAX_TOK (1024 → 512 → 256)
       IMG_MAX_TOK=512 MAX_LEN=24576
3. 降 MAX_LEN
       MAX_LEN=16384
4. 加 distributed optimizer (类 ZeRO-1, 优化器状态切 DP)
       --use_distributed_optimizer
       (注意：TP=8 DP=1 时这个无效)
5. 加 CPU offload (代价：DRAM 占 30B*8=240 GB)
       --optimizer_cpu_offload --optimizer_offload_fraction 1.0
6. 加 TP（必要时跨 NVLink island，但仍单节点内）
       TP=8
7. 加 PP（1F1B pipeline，跨节点也行）
       PP=2  (NPROC=TP*PP*DP)
8. 关 MTP（省 1 层 head 的参数 + 激活）
       MTP_LAYERS=0
9. 关 ViT 全参（半冻结）
       --freeze_vit true   (但用户可能不接受，因为要全参)
10. 上更多卡 / 跨节点
```

---

## 5. 完整脚本片段示例

### 5.1 Qwen3.5-30B-A3B (MoE, 8 卡 H200)

```bash
# 在 41_train_bc_full_megatron.sh 末尾的 megatron sft 命令里加几行：
#   --expert_model_parallel_size 2 \
#   --moe_grouped_gemm true \              # 已默认 true
#   --use_distributed_optimizer \

MODEL_PATH=models/Qwen3.5-30B-A3B \
GAME_NAME=delta_force_30b \
GPUS=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
TP=4 PP=1 \
MICRO_BS=1 GLOBAL_BS=16 \
IMG_MAX_TOK=512 MAX_LEN=24576 \
MTP_LAYERS=0 \
LR=5e-6 \
SAVE_STEPS=50 EVAL_STEPS=50 \
bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh
```

> 30B 模型 lr 通常用 5e-6，比 4B 的 1e-5 小一半（大模型 lr 经验：dense 1e-5 → MoE 5e-6 → 70B 2e-6）。

### 5.2 Qwen3.5-72B (dense, 8 卡 H200 + offload)

```bash
# 41_*.sh 里加：
#   --use_distributed_optimizer \
#   --optimizer_cpu_offload --optimizer_offload_fraction 1.0 \

MODEL_PATH=models/Qwen3.5-72B \
GAME_NAME=delta_force_72b \
GPUS=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
TP=8 PP=1 \
MICRO_BS=1 GLOBAL_BS=16 \
IMG_MAX_TOK=256 MAX_LEN=16384 \
MTP_LAYERS=0 \
LR=2e-6 \
SAVE_STEPS=50 \
bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh
```

### 5.3 跨节点（2 节点 × 8 卡 = 16 卡，72B + PP=2）

跨节点需要 `--master_addr` 和 `--node_rank`，目前 41_*.sh 默认单机。要扩展，参考 ms-swift `examples/megatron/multi-node/`。

---

## 6. 验证扩展是否成功

切大模型后**务必**先跑 dry-run：

```bash
# 复制一份 dryrun 脚本，改成你的大模型
cp scripts/41_train_bc_full_megatron_dryrun.sh /tmp/dryrun_30b.sh
# 把里面的 model / TP / EP / batch 都改了
bash /tmp/dryrun_30b.sh                                # 5 step 验证

# 看 step 5 的 memory(GiB)，应该 < 单卡显存的 80%
# 看是否出现 mtp_1_loss（如果 MTP 开着）
# 看 train_speed(s/it) 估算总耗时
```

5 step 通过 → 用 daemon 起正式训练。

---

## 7. 已知坑

1. **TP 必须整除 hidden_size**：Qwen3.5-4B hidden_size=2560，TP 可选 1/2/4/5/8（但 8 切 320 太碎，性能差）
2. **MoE EP 必须整除 num_experts**：Qwen3.5-30B-A3B 有 128 experts，EP 可选 1/2/4/8/16
3. **跨 NUMA 的 NVLink 效率**：H200 单机 8 卡 NVLink full mesh，TP=8 全在一个 island 没问题；某些定制机器只有 4-4 NVLink island，TP=8 会跨 island 变慢
4. **CPU offload 吃 DRAM**：72B × 8 (FP32 AdamW state) = 576 GB DRAM，机器内存不够会 swap 到磁盘极慢
5. **mcore_bridge 不一定支持所有 model**：截至 1.2.1 支持 qwen2/qwen3/qwen3_5/qwen3_omni 等，新模型出来要等更新或 fallback HF 后端
