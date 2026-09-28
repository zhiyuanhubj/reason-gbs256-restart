# Isambard GH200 环境安装手册

本文记录 2026-08-28 在 Isambard ARM64/GH200 节点上实际安装并验证通过的
`megatron-sft` 环境。它适用于本仓库的 Qwen3.5-9B、Megatron-SWIFT、128K
原神行为克隆训练。

当前验证配置：

| 项目 | 配置 |
|---|---|
| CPU 架构 | `aarch64` |
| GPU | NVIDIA GH200 120GB，每节点 4 卡 |
| 节点 | 2 节点，共 8 卡 |
| 驱动 | 565.57.01 |
| 系统 CUDA module | CUDA 12.6 |
| Conda CUDA 编译工具 | CUDA 12.9 |
| Python | 3.10.21 |
| PyTorch | 2.8.0+cu129 |
| Conda 环境 | `/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft` |

系统 CUDA 12.6 module 主要提供集群运行库，Torch 和本地编译扩展使用 Conda
CUDA 12.9。当前 NVIDIA 驱动可以运行这套 CUDA 12.9 wheel。不要让 12.6 的
`ptxas` 和 12.9 的 `nvcc/cicc` 混用。

有一个重要例外：正式训练加载 `cuda/12.6` module 后，TileLang 的运行时 JIT 会
调用系统 CUDA 12.6 `nvcc`。因此训练入口会在 Conda 激活完成后，把 TileLang 的
host compiler 固定为系统 GCC/G++ 13；否则 Conda 的 GCC 14 会在第一个 backward
才报错。Apex 的安装仍按第 5 节使用 Conda CUDA 12.9 与 GCC 12。

## 1. 设置路径并加载集群模块

安装和编译应在分配到的计算节点上进行，不要在登录节点上编译 CUDA 扩展。

```bash
export PROJECT_ROOT=/projects/b6db/opensima/code/training-best-cp1-full-vittp
export CONDA_BASE=/home/b6db/hongnanvideo1.b6db/.miniconda3
export CONDA_ENV_PATH=/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft

module purge
module load cuda/12.6
module load brics/nccl/2.26.6-1
module load brics/aws-ofi-nccl/1.8.1
source "$CONDA_BASE/etc/profile.d/conda.sh"
```

如果使用新的账号或目录，只需修改上面三个路径。模型、数据集和输出目录由
`scripts/47_train_genshin_only_128k_multinode.slurm` 单独配置。

## 2. 接受 Conda 条款

Anaconda defaults channel 第一次使用时会要求接受条款。只需执行一次：

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

如果命令提示已经接受，可以直接继续。不要通过克隆旧环境绕过条款；本手册使用
全新 Conda 环境。

## 3. 创建 Python 环境和 CUDA 12.9 编译工具

先确认环境目录的父目录在 Conda 的 `envs_dirs` 中：

```bash
conda config --show envs_dirs
# 如果输出中没有 /home/b6db/hongnanvideo1.b6db/.conda-envs，再执行：
conda config --add envs_dirs /home/b6db/hongnanvideo1.b6db/.conda-envs
```

这一步保证安装脚本和训练脚本以后都能用名称 `megatron-sft` 激活该 prefix。

```bash
conda create -p "$CONDA_ENV_PATH" python=3.10 pip -y

set +u
conda activate "$CONDA_ENV_PATH"
set -u

conda install -p "$CONDA_ENV_PATH" \
  -c nvidia/label/cuda-12.9.1 \
  cuda-nvcc=12.9 -y
```

`cuda-nvcc` 会同时安装 ARM64 的 `nvcc`、`cicc`、`ptxas`、CUDA headers 和
Conda 编译工具链。环境的 CUDA 根目录不是普通的 `$CONDA_PREFIX`，而是：

```bash
export CUDA_HOME="$CONDA_ENV_PATH/targets/sbsa-linux"
export PATH="$CONDA_ENV_PATH/nvvm/bin:$CONDA_ENV_PATH/bin:$CUDA_HOME/bin:$PATH"

nvcc --version
ptxas --version
```

两者都应显示 CUDA 12.9。PATH 中 `$CONDA_ENV_PATH/bin` 必须排在系统 CUDA
module 前面，否则可能出现 PTX 版本不匹配。

## 4. 安装训练依赖

```bash
cd "$PROJECT_ROOT"
export CONDA_BASE
export CONDA_ENV_NAME=megatron-sft
export CUDA_HOME="$CONDA_ENV_PATH/targets/sbsa-linux"
export MAX_JOBS=32

bash scripts/00_install_env.sh
```

脚本会安装和配置：

- ARM64 PyTorch 2.8.0+cu129 和 torchvision 0.23.0；
- 从源码为 GH200/SM90 编译 FlashAttention 2.8.3；
- transformers 5.14.1；
- 固定 commit `dc40c652fa9b8fa096613055ccf124dd49f8890c` 的 ms-swift，
  加上仓库内的恢复训练 patch；
- accelerate、PEFT、TRL、datasets、DeepSpeed 等训练依赖；
- flash-linear-attention 0.5.2 和 TileLang 0.1.13；
- Megatron Core 0.16.1、MCore Bridge 1.6.1；
- Transformer Engine 2.14.1 及 ARM64 所需的 cuDNN headers。

ms-swift 以 `--no-deps` editable 模式安装，随后再显式安装 CLI 运行依赖。这是为了
保留本仓库验证过的 PEFT、TRL 和 datasets 版本。pip 可能报告 ms-swift metadata
的版本冲突警告；只要下面的验证通过，不要根据该警告擅自降级这些包。

ARM64 没有合适的 `decord` wheel，因此脚本会跳过它。当前数据是图片帧 Parquet，
不需要视频解码，所以不影响训练。

## 5. 编译安装 Apex

基础环境装好后执行：

```bash
cd "$PROJECT_ROOT"
export CONDA_BASE
export CONDA_ENV_NAME=megatron-sft
export CUDA_HOME="$CONDA_ENV_PATH/targets/sbsa-linux"
export MAX_JOBS=32

bash scripts/00_install_env.sh apex
```

GH200 是 ARM64，Apex 没有适用的预编译 wheel。脚本会从 NVIDIA Apex 源码编译
C++/CUDA 扩展，并自动：

- 使用 `/usr/bin/gcc-12` 和 `/usr/bin/g++-12`，避开 Conda GCC 14 与 nvcc
  12.9 的头文件冲突；
- 设置 `TORCH_CUDA_ARCH_LIST=9.0`，只编译 GH200/Hopper 内核；
- 把 Conda CUDA 12.9 的 `cicc` 和 `ptxas` 放到正确的 PATH 顺序。

该步骤可能需要 10–30 分钟。当前验证的 Apex 源码 revision 为
`77a4b7a824c9`，安装包显示版本 `0.1`。

## 6. 验证环境

先激活仓库运行环境：

```bash
module purge
module load cuda/12.6 brics/nccl/2.26.6-1 brics/aws-ofi-nccl/1.8.1
export CONDA_BASE=/home/b6db/hongnanvideo1.b6db/.miniconda3
export CONDA_ENV_NAME=megatron-sft
export CUDA_HOME=/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/targets/sbsa-linux
source "$PROJECT_ROOT/env.sh"
```

运行完整导入检查：

```bash
python tools/verify_env.py
megatron sft --help >/dev/null
```

再验证 Apex CUDA 内核，而不只是 Python import：

```bash
python - <<'PY'
import torch
import apex
import amp_C
import fused_layer_norm_cuda
from apex.normalization import FusedLayerNorm
from apex.optimizers import FusedAdam

x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
layer = FusedLayerNorm(128).cuda().to(torch.bfloat16)
y = layer(x)
y.float().square().mean().backward()
optimizer = FusedAdam(layer.parameters(), lr=1e-3)
optimizer.step()
torch.cuda.synchronize()
print("APEX_GPU_OK", apex.__file__, y.dtype, float(x.grad.float().norm()))
PY
```

成功时最后一行以 `APEX_GPU_OK` 开头。

Python 3.10 下 TileLang 会提示推荐 Python 3.11，并把 `torch.compile` 设为 identity。
本仓库的训练配置默认 `DISABLE_TORCH_COMPILE=true`，这是已知提示，不是安装失败。

## 7. 准备当前原神数据视图

当前源数据集是：

```text
/projects/b6db/opensima/yuanshen_bc_formatted_sl135
```

训练不直接改动源目录。先生成仅包含元数据、验证集和 Parquet 符号链接的轻量视图：

```bash
export PROJECT_ROOT=/projects/b6db/opensima/code/training-best-cp1-full-vittp
export PYTHON=/home/b6db/hongnanvideo1.b6db/.conda-envs/megatron-sft/bin/python

"$PYTHON" "$PROJECT_ROOT/scripts/build_genshin_only_dataset.py" \
  --genshin-root /projects/b6db/opensima/yuanshen_bc_formatted_sl135 \
  --output-root /projects/b6db/opensima/genshin-training/genshin-only \
  --passes 3 \
  --val-rows 32 \
  --workers 32
```

该操作不会复制 2.6 TB 图片数据。正式入口检查的配置文件是：

```text
/projects/b6db/opensima/genshin-training/genshin-only/genshin-only-128k.local.yaml
```

## 8. 在 Slurm 训练中使用

训练入口已经配置好 module、Conda 和 CUDA 路径：

```bash
sbatch scripts/47_train_genshin_only_128k_multinode.slurm
```

该入口在 Isambard 上默认使用：

| 参数 | 值 |
|---|---:|
| 节点 / GPU | 2 × 4 = 8 GPU |
| TP / PP / CP / DP | 4 / 1 / 1 / 2 |
| micro / global batch | 1 / 8 |
| gradient accumulation | 4 |
| sequence / packing length | 131072 |
| dataset encoding processes | 每个数据 rank 4 |
| PyTorch DataLoader workers | 0 |

从登录节点附着到一个已经处于 `RUNNING` 的两节点 allocation 时，直接执行：

```bash
cd /projects/b6db/opensima/code/training-best-cp1-full-vittp
EXISTING_JOB_ID=<job-id> GLOBAL_BS=8 \
  bash scripts/47_train_genshin_only_128k_multinode.slurm
```

脚本会通过 `scontrol show job` 自动读取 `NodeList`、`NumNodes` 和 `CPUs/Task`，
然后用 `srun --jobid=<job-id> --overlap` 在完整 allocation 上启动每节点一个 task。
不要先用一个单节点 `srun` 包住脚本；这样内层两节点 step 无法扩展到外层 step 之外。
同一个 allocation 上也不要同时启动两个训练 step，否则 `--overlap` 会让它们争用 GPU。

只检查模型、数据和并行配置，不启动训练：

```bash
EXISTING_JOB_ID=<job-id> PREFLIGHT_ONLY=true \
  OUTPUT_DIR=/tmp/genshin-preflight-<job-id> \
  bash "$PROJECT_ROOT/scripts/47_train_genshin_only_128k_multinode.slurm"
```

建议为正式运行显式给稳定输出目录：

```bash
EXISTING_JOB_ID=<job-id> GLOBAL_BS=8 \
OUTPUT_DIR=/projects/b6db/opensima/outputs/genshin3_qwen35_128k_gbs8_<job-id> \
  bash scripts/47_train_genshin_only_128k_multinode.slurm
```

确认真正跑起来不能只看 `Train: 0%`。应等待 `logging.jsonl` 出现 iteration 1：

```bash
tail -f "$OUTPUT_DIR/logging.jsonl"
```

2026-08-28 的 8 卡实跑首步记录为：loss `6.0541`、首步显存约
`69.45 GiB/GPU`、首步 `133.99 s`；到 iteration 10 显存稳定在约
`80.02 GiB/GPU`，平均速度下降到约 `78–81 s/it`。首步包含 TileLang JIT，不能用
它准确外推总时长。

## 9. 常见错误

### `CondaToSNonInteractiveError` 或要求接受条款

执行第 2 节的两个 `conda tos accept` 命令，然后重新创建环境。

### Apex 编译出现 `0.0bf16`、`__is_array` 等 GCC 错误

NVCC 使用了 Conda GCC 14。重新运行最新版 `scripts/00_install_env.sh apex`；
脚本在 ARM64 上会固定系统 GCC 12。

### TileLang 在第一步 backward 报 `unsupported GNU version`

典型日志结尾包含：

```text
host_config.h: error: #error -- unsupported GNU version!
Command: .../cuda/12.6/bin/nvcc
  -ccbin=.../aarch64-conda-linux-gnu-c++
```

这不是 Apex 安装问题。Conda 激活脚本把 `CC/CXX` 改成了 GCC 14，而 CUDA 12.6
最多支持 GCC 13。最新版 `47` 设置 `CUDA_HOST_CC=/usr/bin/gcc-13`、
`CUDA_HOST_CXX=/usr/bin/g++-13`，`43_train_inner.sh` 会在 `conda activate` 之后
再次固定它们。启动日志必须出现：

```text
[inner] ... CXX=/usr/bin/g++-13
```

随后 8 个 rank 应出现 `TileLang completes to compile kernel`。

### 模型初始化完成后一直是 `0/33539`，GPU 利用率为 0

如果 `DATASET_NUM_PROC=0`，streaming 的 `IterablePackingDataset` 不会创建任何
编码进程，却会一直等待空的输出队列。这里的参数不是 PyTorch DataLoader worker。
当前默认和安全配置是：

```text
DATASET_NUM_PROC=4
DATALOADER_NUM_WORKERS=0
```

最新版 `46` 会拒绝 streaming packing 下的 `DATASET_NUM_PROC <= 0`，避免静默死锁。

### `srun` 只启动一台节点或提示节点不属于当前 step

原因通常是在一个单节点 `srun` 里面再次调用两节点 `srun`。退出这个外层 step，
从登录节点使用第 8 节的 `EXISTING_JOB_ID=<job-id>` 启动。底层 launcher 显式设置
`--nodes`、`--ntasks`、`--ntasks-per-node=1`，每个节点只启动一个 torchrun。

### pass-aware 模式提示 mixture plan 不存在

严格三遍采样需要 `MIXTURE_PLAN`，不能只生成普通的 dataset spec。最新版外层
launcher 会在启动 `srun` 前调用 `lib_mixture.py --write-plan`，并让所有 rank 读取
同一个共享 plan。不要手工删除正在运行任务输出目录中的 `mixture_plan.json`。

### `decord`、Python 3.10、FlashAttention v3 等 warning

当前原神数据使用图片帧，不读取视频，所以缺少 `decord` 不阻塞训练。Python 3.10
下 `torch.compile requires Python >= 3.11` 也只是提示，因为正式配置已关闭
torch.compile。FlashAttention v3 的安装建议和 Qwen vision input-embedding warning
同样不是启动失败判据；以 iteration 日志、有限 loss/grad norm 和 GPU 利用率为准。

### `sh: cicc: command not found`

Conda CUDA 的内部编译器不在默认 CUDA_HOME 下。确认：

```bash
export PATH="$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
```

### `ptxas fatal: Unsupported .version 8.8; current version is 8.5`

CUDA 12.9 的 `cicc` 错误调用了系统 CUDA 12.6 的 `ptxas`。确保
`$CONDA_PREFIX/bin` 位于系统 CUDA module 前面，并验证 `ptxas --version` 是
12.9。最新版 `env.sh` 和安装脚本已处理这个 PATH 顺序。

### Transformer Engine 找不到 `cudnn.h`、`cusparse.h` 或 `nccl.h`

不要单独反复安装 TE。使用 `scripts/00_install_env.sh`，它会安装匹配的 cuDNN
headers，并加入 Isambard HPC SDK 的 CUDA math、toolkit 和 NCCL headers。

### `Apex is not installed. Falling back to Torch Norm`

说明 Apex 尚未成功导入。执行第 5 节，然后运行第 6 节的 GPU 内核测试。Apex
缺失不会改变训练正确性，但会失去对应 fused 优化。

## 10. 重装原则

- `scripts/00_install_env.sh` 可以重复运行，已安装组件会被检查或复用。
- 不要从另一架构或另一 CUDA 版本克隆 Conda 环境；ARM64 CUDA 扩展需要本机编译。
- 不要在仓库里保存 Conda 密码、Hugging Face token 或 WandB key。
- 环境验证完成后再提交正式 128K 训练，先用 `PREFLIGHT_ONLY=true` 检查路径和拓扑。
