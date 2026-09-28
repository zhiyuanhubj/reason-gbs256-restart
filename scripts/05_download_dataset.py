#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 BC SFT 训练数据集

用法:
    python scripts/05_download_dataset.py                    # 默认 delta-force-bc-sft-shuffled
    REPO_ID=user/other-game-bc-sft  python scripts/05_*.py   # 切换数据集
    OUTPUT_DIR=/path/to/dir         python scripts/05_*.py
    HF_TOKEN=hf_xxx                 python scripts/05_*.py   # 私有 repo 必填

环境变量:
    HF_TOKEN     HuggingFace API token（私有 repo 必填；公开 repo 可省略）
    REPO_ID      数据集 repo（默认 zhiyuanhucs/delta-force-bc-sft-shuffled）
    OUTPUT_DIR   输出目录，默认是 ${PROJECT_ROOT}/data/<basename(repo_id)>
    HF_ENDPOINT  HF 端点（默认 hf-mirror.com 国内镜像；改 https://huggingface.co 用官方）

获取 HF token: https://huggingface.co/settings/tokens
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, login

# 默认走 hf-mirror（国内 ~50× 速度），覆盖：HF_ENDPOINT=https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)
else:
    print("[download] WARN: HF_TOKEN 未设置；如果 repo 是私有的会失败。"
          "公开 repo 可忽略此警告。")

REPO_ID = os.environ.get("REPO_ID", "zhiyuanhucs/delta-force-bc-sft-shuffled")

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / REPO_ID.split("/")[-1]
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[download] {REPO_ID}  ->  {OUTPUT_DIR}")
print(f"[download] HF_ENDPOINT = {os.environ.get('HF_ENDPOINT', '<official>')}")

local_dir = snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(OUTPUT_DIR),
    allow_patterns=["*.parquet", "*.json", "*.jsonl", "*.md", "*.yaml", "*.yml", "*.txt"],
    max_workers=8,
)

print(f"[download] done. files at: {local_dir}")
print(f"[download] verify with: ls -lh {local_dir}")

# 简要清单
files = sorted(Path(local_dir).rglob("*"))
print(f"\n[download] file listing ({len([f for f in files if f.is_file()])} files):")
for f in files:
    if f.is_file():
        sz = f.stat().st_size
        sz_h = (
            f"{sz/1024/1024/1024:.2f} GB" if sz > 1024**3
            else f"{sz/1024/1024:.2f} MB" if sz > 1024**2
            else f"{sz/1024:.1f} KB"
        )
        print(f"  {sz_h:>10}  {f.relative_to(local_dir)}")
