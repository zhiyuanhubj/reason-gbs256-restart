#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把训练产出的 HF safetensors ckpt 上传到 HuggingFace。

布局：单 repo(zhiyuanhucs/sft)，每个 ckpt 一个分支(checkpoint-XXX)，
最新 step 的 ckpt 同时同步到 main。

- 只上传 HF 格式文件(safetensors + config/tokenizer/...)，自动排除 mcore 残留(iter_*/、*.pt、latest_checkpointed_iteration.txt)。
- 可重复运行：已存在的分支会跳过(除非 --overwrite)，方便后续补传新 ckpt。

用法：
  HF_TOKEN=hf_xxx python scripts/09_upload_ckpts_hf.py
  HF_TOKEN=hf_xxx python scripts/09_upload_ckpts_hf.py --only 600 1200   # 只传指定 step
  HF_TOKEN=hf_xxx python scripts/09_upload_ckpts_hf.py --overwrite       # 覆盖已存在分支
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from huggingface_hub import HfApi

DEFAULT_RUN_GLOB = "/fsx/home/zhiyuan/game/extracted/training-main/outputs/sft_*/v0-*"
IGNORE = ["iter_*", "*/iter_*", "*.pt", "latest_checkpointed_iteration.txt", "**/*.pt"]


def find_ckpts(run_glob: str) -> list[tuple[int, str]]:
    runs = sorted(glob.glob(run_glob))
    if not runs:
        return []
    run = runs[-1]  # 最新 run
    out = []
    for d in glob.glob(os.path.join(run, "checkpoint-*")):
        m = re.search(r"checkpoint-(\d+)$", d)
        if m and os.path.isdir(d):
            # 必须有 safetensors 才算有效 HF ckpt
            if glob.glob(os.path.join(d, "*.safetensors")):
                out.append((int(m.group(1)), d))
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="zhiyuanhucs/sft")
    ap.add_argument("--run_glob", default=DEFAULT_RUN_GLOB)
    ap.add_argument("--only", nargs="*", type=int, default=None, help="只传这些 step")
    ap.add_argument("--overwrite", action="store_true", help="分支已存在也重新上传")
    ap.add_argument("--no_main", action="store_true", help="不把最新 ckpt 同步到 main")
    ap.add_argument("--private", action="store_true", help="repo 设为私有")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERR] 需要 HF_TOKEN 环境变量（对 zhiyuanhucs 有写权限的 token）")
        return 1

    ckpts = find_ckpts(args.run_glob)
    if args.only:
        ckpts = [(s, d) for s, d in ckpts if s in args.only]
    if not ckpts:
        print(f"[ERR] 没找到 ckpt: {args.run_glob}/checkpoint-*")
        return 1

    api = HfApi(token=token)
    print(f"[upload] repo: {args.repo}")
    print(f"[upload] 待传 ckpt: {[s for s, _ in ckpts]}")

    # 建 repo（已存在则忽略）
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    existing_branches = {r.name for r in api.list_repo_refs(args.repo).branches}
    latest_step = ckpts[-1][0]

    for step, path in ckpts:
        branch = f"checkpoint-{step}"
        if branch in existing_branches and not args.overwrite:
            print(f"[skip] 分支 {branch} 已存在（--overwrite 可覆盖）")
        else:
            print(f"[upload] {branch}  <-  {path}")
            api.create_branch(args.repo, branch=branch, exist_ok=True)
            api.upload_folder(
                repo_id=args.repo,
                folder_path=path,
                revision=branch,
                ignore_patterns=IGNORE,
                commit_message=f"Add {branch}",
            )
        # 最新 ckpt 同步到 main
        if step == latest_step and not args.no_main:
            print(f"[upload] 同步最新 {branch} -> main")
            api.upload_folder(
                repo_id=args.repo,
                folder_path=path,
                revision="main",
                ignore_patterns=IGNORE,
                commit_message=f"Update main to {branch}",
            )

    print("[upload] 完成。")
    print(f"  https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
