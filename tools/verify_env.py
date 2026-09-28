#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 megatron-sft env 完整性。"""
from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[OK ]{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def err(msg: str) -> None:
    print(f"{RED}[ERR ]{RESET} {msg}")


def title(msg: str) -> None:
    print(f"\n{BOLD}== {msg} =={RESET}")


def check_module(name: str, *, version_attr: str = "__version__", required: bool = True) -> bool:
    try:
        m = importlib.import_module(name)
    except ImportError as e:
        if required:
            err(f"{name}: ImportError ({e})")
        else:
            warn(f"{name}: ImportError ({e})  [可选]")
        return False
    ver = getattr(m, version_attr, "?")
    ok(f"{name} {ver}")
    return True


def main() -> int:
    title("Python")
    ok(f"python {platform.python_version()}  ({sys.executable})")

    title("CUDA / GPU")
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
                text=True,
            )
            for line in out.strip().splitlines():
                ok(f"GPU: {line}")
        except subprocess.CalledProcessError as e:
            err(f"nvidia-smi failed: {e}")
    else:
        warn("nvidia-smi 不在 PATH")

    title("Torch")
    try:
        import torch  # type: ignore

        ok(f"torch {torch.__version__} (cuda={torch.version.cuda})")
        ok(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            ok(f"device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                ok(f"  device[{i}]: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        err(f"torch import/check 失败: {e}")
        return 2

    title("核心训练框架")
    check_module("transformers")
    check_module("accelerate")
    check_module("peft")
    check_module("trl")
    check_module("datasets")
    check_module("deepspeed")
    check_module("flash_attn")

    title("ms-swift")
    check_module("swift")

    title("Qwen3.5 必备")
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa
        ok("flash-linear-attention.chunk_gated_delta_rule")
    except ImportError as e:
        err(f"fla.chunk_gated_delta_rule 缺失: {e}")

    title("Megatron 后端")
    check_module("megatron", required=True)
    check_module("megatron.core", version_attr="__version__", required=True)
    check_module("transformer_engine", required=True)
    check_module("mcore_bridge", required=True)
    check_module("apex", required=False)        # mcore≥0.16 可选
    check_module("tilelang", required=True)     # Hopper GPU 的 fla 必须

    title("实验追踪 / 工具")
    check_module("wandb", required=False)
    check_module("decord", required=False)      # qwen3.5-vl 多模态
    check_module("qwen_vl_utils", required=False)

    title("环境变量")
    for k in [
        "PROJECT_ROOT",
        "SWIFT_USE_MCORE_GDN",
        "PYTORCH_CUDA_ALLOC_CONF",
        "MODELSCOPE_CACHE",
        "HF_HOME",
    ]:
        v = os.environ.get(k)
        (ok if v else warn)(f"{k}={v}")

    title("本地模型探测")
    try:
        from pathlib import Path
        from transformers import AutoConfig  # type: ignore

        proj_root = Path(os.environ.get("PROJECT_ROOT", "")) if os.environ.get("PROJECT_ROOT") \
            else Path(__file__).resolve().parent.parent
        models_dir = proj_root / "models"
        if not models_dir.is_dir():
            warn(f"未找到 {models_dir}（先跑 scripts/01_download_model.sh）")
        else:
            found = 0
            for d in sorted(models_dir.iterdir()):
                if d.is_dir() and not d.name.startswith(".") and (d / "config.json").exists():
                    try:
                        cfg = AutoConfig.from_pretrained(str(d), trust_remote_code=True)
                        # 多模态模型的 hidden/layers 在 text_config / language_config 子节点
                        text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "language_config", None) or cfg
                        h = getattr(text_cfg, "hidden_size", "?")
                        n = getattr(text_cfg, "num_hidden_layers", "?")
                        ok(f"{d.name:24s}  type={cfg.model_type}  hidden={h}  layers={n}")
                        found += 1
                    except Exception as e:
                        warn(f"{d.name}: 配置加载失败 ({e})")
            if found == 0:
                warn(f"{models_dir} 下没有可识别模型")
    except Exception as e:
        warn(f"模型探测失败: {e}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
