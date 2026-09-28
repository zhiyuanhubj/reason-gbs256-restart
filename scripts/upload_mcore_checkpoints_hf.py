#!/usr/bin/env python3
"""Upload complete MCore checkpoints to separate Hugging Face branches."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi


def complete_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in run_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match or not path.is_dir():
            continue
        step = int(match.group(1))
        iteration_dir = path / f"iter_{step:07d}"
        marker = path / "latest_checkpointed_iteration.txt"
        distcp_files = list(iteration_dir.glob("*.distcp"))
        if not marker.is_file() or not (iteration_dir / ".metadata").is_file() or not distcp_files:
            print(f"[skip] incomplete checkpoint: {path.name}", flush=True)
            continue
        if marker.read_text(encoding="utf-8").strip() != str(step):
            raise RuntimeError(f"iteration marker mismatch: {path}")
        checkpoints.append((step, path))
    return sorted(checkpoints)


def local_files(path: Path) -> set[str]:
    return {
        file.relative_to(path).as_posix()
        for file in path.rglob("*")
        if file.is_file() and ".cache" not in file.relative_to(path).parts
    }


def verify_branch(api: HfApi, repo_id: str, revision: str, path: Path) -> None:
    expected = local_files(path)
    remote = set(api.list_repo_files(repo_id, repo_type="model", revision=revision))
    missing = expected - remote
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"{revision}: {len(missing)} remote files missing ({preview})")
    print(f"[verify] {revision}: {len(expected)} files present", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    checkpoints = complete_checkpoints(args.run_dir)
    if not checkpoints:
        raise RuntimeError(f"no complete checkpoints under {args.run_dir}")

    api = HfApi()
    who = api.whoami()["name"]
    if args.repo_id.split("/", 1)[0] != who:
        print(f"[warning] authenticated as {who}; uploading to {args.repo_id}", flush=True)
    api.create_repo(args.repo_id, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(args.repo_id, repo_type="model", private=False, gated="manual")

    manifest = {
        "format": "Megatron Core distributed checkpoint",
        "source_run": args.run_dir.name,
        "checkpoints": [step for step, _ in checkpoints],
        "latest_complete_checkpoint": checkpoints[-1][0],
        "excluded_incomplete_checkpoints": [567],
    }
    manifest_path = args.run_dir / "hf_checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=str(manifest_path),
        path_in_repo="checkpoint_manifest.json",
        commit_message="Add checkpoint manifest",
    )

    for step, path in checkpoints:
        branch = f"checkpoint-{step}"
        print(f"[upload] {branch} <- {path}", flush=True)
        api.create_branch(args.repo_id, repo_type="model", branch=branch, exist_ok=True)
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="model",
            revision=branch,
            folder_path=path,
            ignore_patterns=[".cache/**"],
            num_workers=args.workers,
            print_report=True,
            print_report_every=60,
        )
        verify_branch(api, args.repo_id, branch, path)

    latest_step, latest_path = checkpoints[-1]
    print(f"[upload] main <- checkpoint-{latest_step}", flush=True)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="model",
        revision="main",
        folder_path=latest_path,
        ignore_patterns=[".cache/**"],
        num_workers=args.workers,
        print_report=True,
        print_report_every=60,
    )
    verify_branch(api, args.repo_id, "main", latest_path)
    print(f"[done] https://huggingface.co/{args.repo_id}", flush=True)


if __name__ == "__main__":
    main()
