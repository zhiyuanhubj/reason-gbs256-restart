# FAQ / 排错

## 1. 模型 / Tokenizer

### 1.1 Qwen3.5-4B 是 LLM 还是 VLM？
VLM。Qwen3.5 整个系列都是 image-text-to-text 多模态 dense / MoE 模型，架构上把
GatedDeltaNet 线性注意力和全注意力混合在一起。如果你只想训纯文本：数据 jsonl 里没
`images` 字段就不会触发视觉路径；可加 `--freeze_vit true --freeze_aligner true` 把视觉
部分冻结。

### 1.2 transformers 报 `Unrecognized model_type 'qwen3_5'`
升级 transformers 主分支：
```bash
pip install -U 'git+https://github.com/huggingface/transformers.git@main'
```
（已写入 `scripts/00_install_env.sh`）

### 1.3 训练后 `<|action_start|>` 等 token 输出被切碎
检查 ckpt 目录下的 `tokenizer.json`，应包含全部 5 个 added special token
（`<|action_start|>` / `<|action_end|>` / `<|action_sep|>` / `<|thought_start|>` / `<|thought_end|>`，
再加 padding 对齐）。如果数量不对说明 `--new_special_tokens` 没生效，
检查训练命令里这一行参数是否被覆盖。注意：分号 `;` 现在是合法可输出字符，chunk 分隔统一用 `<|action_sep|>`。

---

## 2. 环境 / 依赖

### 2.1 Hopper GPU 报 `Triton >= 3.4.0 ... gated chunk_bwd_dqkwg`
完整错误：
```
RuntimeError: Triton >= 3.4.0 on Hopper GPUs produces incorrect results for
gated chunk_bwd_dqkwg (see #640). Please install tilelang
```

H100/H200 上 fla 的 GatedDeltaNet backward kernel 在新版 Triton 下结果错误，fla 主动
raise 让换 tilelang：
```bash
pip install tilelang
```
（已写入 `scripts/00_install_env.sh`）

### 2.2 Megatron 后端训练 GatedDeltaNet 极慢（每 step 数分钟）
参考 [ms-swift issue #8389](https://github.com/modelscope/ms-swift/issues/8389)：
```bash
export SWIFT_USE_MCORE_GDN=1
```
（已在 `env.sh` 默认开启）

### 2.3 flash-attn 报 `undefined symbol: _ZN3c105ErrorC2...`
ABI 不匹配。torch 2.8.0+cu128 是 cxx11abi=TRUE 编译的，但 `pip install flash-attn`
默认装的可能是 cxx11abi=FALSE。装匹配的 prebuilt wheel：
```bash
pip uninstall -y flash-attn
pip install --no-build-isolation \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```
（已在 `00_install_env.sh` 处理）

### 2.4 transformer_engine 编译报 `nccl.h: No such file or directory`
TE pytorch frontend 编译时找不到 NCCL 头文件。`env.sh` 已经把 conda env 里的
`nvidia/nccl/include` 加到 `CPATH`。如果是新装环境第一次编 TE，确保已经 source 过
`env.sh` 再 `pip install transformer-engine`。

### 2.5 apex 编译失败 / 装不上
mcore≥0.16 已经不强依赖 apex（缺失会 fallback 到 Torch Norm，性能略差但功能完整），
默认我们不装。想装：`bash scripts/00_install_env.sh apex`（30+ min 编译）。
Isambard ARM64/GH200 的 GCC 12、CUDA 12.9 `cicc/ptxas` 路径和验证步骤见
[`isambard_gh200_environment.md`](isambard_gh200_environment.md)。

### 2.6 Isambard 第一轮 backward 才报 CUDA 12.6 不支持 GCC 14

Conda 的 compiler activate hook 会覆盖进入环境前设置的 `CC/CXX`。因此必须在
`conda activate` 之后把 TileLang host compiler 重新设为 `/usr/bin/g++-13`。
`scripts/47_train_genshin_only_128k_multinode.slurm` 通过 `CUDA_HOST_CC/CXX`
传递选择，`scripts/43_train_inner.sh` 在激活后执行固定。完整错误和检查方法见
[`isambard_gh200_environment.md`](isambard_gh200_environment.md#tilelang-在第一步-backward-报-unsupported-gnu-version)。

---

## 3. 训练时 OOM

按以下顺序逐项加，每加一个跑 5-step dryrun：

```
1. recompute 升级
       --recompute_granularity full --recompute_method uniform --recompute_num_layers 1
2. 降视觉 token
       IMG_MAX_TOK=512   (1024 → 512 → 256)
3. 降序列长度
       MAX_LEN=24576
4. distributed optimizer (类 ZeRO-1)
       --use_distributed_optimizer
5. CPU offload (代价：DRAM, 30B 模型 ~240GB DRAM)
       --optimizer_cpu_offload --optimizer_offload_fraction 1.0
6. 加 TP（必须能整除 hidden_size）
       TP=8
7. 加 PP（layer 维度切分）
       PP=2
8. 关 MTP
       MTP_LAYERS=0
9. 上更多卡
```

详见 [`scaling_to_large_models.md`](scaling_to_large_models.md)。

---

## 4. 训练流程

### 4.0 128K streaming packing 在 0% 不动

`DATASET_NUM_PROC` 是 `IterablePackingDataset` 自己的编码进程数，不能为 0；
`DATALOADER_NUM_WORKERS` 才是外层 PyTorch DataLoader worker。推荐组合是前者 4、
后者 0。前者为 0 会导致所有 rank 等待空队列，表现为模型已初始化、GPU 占用显存
但利用率为 0，而且不会立即 traceback。

### 4.1 batch size 怎么算
```
global_batch_size = micro_batch_size × DP × gradient_accumulation_steps
DP                = world_size / (TP × PP)
```
例：4 GPU、TP=4、PP=1 → DP=1，micro=1，要 global_bs=8 → grad_acc=8。
ms-swift 会自动算 grad_acc，**不要手设**，只设 `--global_batch_size`。

### 4.2 多少 step 一个 epoch
```
step_per_epoch = num_train_samples / global_batch_size
total_step     = step_per_epoch × num_train_epochs
```
当前 3400 train + global_bs 8 + 3 epoch = 1275 step。

### 4.3 ckpt 多大？
- bf16 safetensors ≈ 2 byte × 模型参数。4B → 8 GB，30B → 60 GB，72B → 144 GB
- `save_total_limit=3` 保留最近 3 个，旧的自动删
- `--no_save_optim true` 不存优化器状态（恢复训练用不到，省 4 倍空间）

### 4.4 Megatron 的 mcore checkpoint 怎么推理
当前 `--save_safetensors true` 训练就直接存 HF 格式，可以直接 `vllm serve`。
万一需要把 mcore-format ckpt 转回 HF：
```bash
bash scripts/99_convert_mcore_to_hf.sh outputs/<run>/iter_xxxx
# 输出在 outputs/<run>/iter_xxxx-hf
```

---

## 5. Daemon / 后台

### 5.1 SSH 一断训练就死了
你直接 `bash scripts/41_*.sh` 起的训练是当前 shell 的子进程，会被 SIGHUP 杀。用：
```bash
bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh
```
这个脚本用 `setsid + nohup + disown` 把训练 reparent 到 init (PPID=1)。

### 5.2 怎么验证 daemon 真起来了
```bash
PID=$(pgrep -f "bash.*41_train_bc_full_megatron")
ps -o pid,ppid,sid,cmd -p $PID
# PPID 应该 = 1，SID 应该 = 自己的 PID
```

### 5.3 怎么停 daemon
```bash
pkill -SIGTERM -f "megatron sft"     # 优雅，会保存最后一个 ckpt
pkill -SIGKILL -f "megatron sft"     # 强杀，不存
```

---

## 6. Wandb

### 6.1 wandb 上看不见 run
- run 在 `WANDB_ENTITY` 设置的 team 下（默认 `OpenSIMA`）。如果你不在这个 team，
  改 `WANDB_ENTITY=` 为空（用个人账号）或换成你自己的 team
- 第一次用要 `python -c "import wandb; wandb.login(key='...')"`
- 验证：log 里搜 `wandb: 🚀 View run at https://...`，点这个 URL

### 6.2 想离线 / 不上报
```bash
WANDB_MODE=offline bash scripts/41_*.sh           # 完全本地
WANDB_MODE=disabled bash scripts/41_*.sh          # 完全关
```

### 6.3 多个游戏怎么对比
每个游戏起一个 run（`GAME_NAME=xxx` 区分），dashboard 上按 tag 过滤就能并排比较。

---

## 7. GPU 资源

### 7.1 卡被别人占着
```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# 选你能用的卡：
GPUS=2,3 NPROC_PER_NODE=2 TP=2 IMG_MAX_TOK=512 MAX_LEN=24576 \
  bash scripts/run_daemon.sh scripts/41_train_bc_full_megatron.sh
```

### 7.2 NPROC 必须能整除什么？
- `NPROC = TP × PP × DP × CP`（CP=context parallel，默认 1）
- TP 必须整除 hidden_size（Qwen3.5-4B hidden=2560，可 TP=1/2/4/5/8）

### 7.3 GPU 0 1 也能用但显存少
- 如果只有部分卡空：`GPUS=1,2,3 NPROC_PER_NODE=3` 但 TP=3 通常不行（hidden 不整除），改用 `TP=1 DP=3`，但单卡显存压力大
- 大模型先 `TP=4` 在 4 卡上跑，留 GPU 0 给别人

---

## 8. 调试小技巧

### 8.1 看训练在做什么
```bash
tail -f logs/<run_name>.log                                      # 训练详细日志
tail -f logs/daemon_*.log                                         # daemon 包装日志
watch -n 5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'
```

### 8.2 验证 ckpt 是否能加载 + 输出格式正确
```bash
python scripts/42_check_format.py outputs/<run>/iter_xxx --num_samples 30
```

### 8.3 训练突然崩了怎么找原因
```bash
grep -B5 -A30 "Traceback\|RuntimeError\|OutOfMemory" logs/<run_name>.log | tail -60
```
