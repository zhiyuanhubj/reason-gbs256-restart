#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 _merged/val.jsonl(绝对路径) 打成一个自包含、可移植的 val 包：
   val_pkg/
     ├── val.jsonl          # 图片路径改为相对(images/sample_XXX/ZZ.jpg)
     └── images/sample_XXX/ZZ.jpg
然后 zip 成一个文件，方便下载确认。
"""
import argparse, json, os, shutil, sys, zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="/fsx/home/zhiyuan/yuanshen_processed/_merged/val.jsonl")
    ap.add_argument("--pkg_dir", default="/fsx/home/zhiyuan/yuanshen_val_128_pkg")
    ap.add_argument("--zip", default="/fsx/home/zhiyuan/yuanshen_val_128.zip")
    args = ap.parse_args()

    pkg = Path(args.pkg_dir)
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "images").mkdir(parents=True)

    lines = [l for l in open(args.val, encoding="utf-8").read().splitlines() if l.strip()]
    print(f"[pack] val 记录数: {len(lines)}")
    out_lines = []
    n_img = 0
    for i, line in enumerate(lines):
        rec = json.loads(line)
        sample_dir = pkg / "images" / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        new_imgs = []
        for k, src in enumerate(rec["images"]):
            dst = sample_dir / f"{k:02d}.jpg"
            shutil.copyfile(src, dst)
            new_imgs.append(os.path.relpath(dst, pkg))
            n_img += 1
        rec["images"] = new_imgs
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    (pkg / "val.jsonl").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[pack] 复制图片: {n_img} 张 -> {pkg}")

    # 打 zip
    print(f"[pack] 压缩到 {args.zip} ...")
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(pkg):
            for f in files:
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(pkg.parent))
    size_gb = os.path.getsize(args.zip) / 1024**3
    print(f"[pack] done. zip = {args.zip}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    sys.exit(main())
