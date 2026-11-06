"""把 CrossPan 转成 nnU-Net v2 的 raw 格式。

产出:
    nnUNet/nnUNet_raw/Dataset310_PancreasT1DG/
        imagesTr/<case>_0000.nii.gz
        labelsTr/<case>.nii.gz
        dataset.json

数据集编号 310 与 CrossPan 官方仓库的 Task310_PancreasT1DG 一致，
方便以后对照他们的训练脚本。

CrossPan 的文件名本来就符合 nnU-Net 规范（图像 <case>_0000.nii.gz，
标签 <case>.nii.gz），所以只需要挑出带标注的病例并复制过去。

用法:
    python nnunet_prepare.py                 # 默认 T1/train
    python nnunet_prepare.py --seq T1 --splits train test
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from config import ROOT, WORK

NNUNET_BASE = WORK / "nnUNet"
RAW = NNUNET_BASE / "nnUNet_raw"


def collect(seq, splits):
    out = []
    for split in splits:
        img_dir = ROOT / seq / split / "images"
        lab_dir = ROOT / seq / split / "labels"
        if not img_dir.is_dir():
            continue
        imgs = {f.replace("_0000.nii.gz", ""): f for f in os.listdir(img_dir) if f.endswith(".nii.gz")}
        labs = {f.replace(".nii.gz", ""): f for f in os.listdir(lab_dir) if f.endswith(".nii.gz")}
        for stem in sorted(set(imgs) & set(labs)):
            out.append((stem, img_dir / imgs[stem], lab_dir / labs[stem]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="T1")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--dataset", default="Dataset310_PancreasT1DG")
    ap.add_argument("--channel-name", default="T1")
    args = ap.parse_args()

    ds = RAW / args.dataset
    images_tr = ds / "imagesTr"
    labels_tr = ds / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    cases = collect(args.seq, args.splits)
    print(f"{args.seq}/{args.splits}: 找到 {len(cases)} 例带标注")
    if not cases:
        raise SystemExit("没有找到任何图-标配对")

    n_new = 0
    for stem, img, lab in cases:
        dst_i = images_tr / f"{stem}_0000.nii.gz"
        dst_l = labels_tr / f"{stem}.nii.gz"
        if not dst_i.exists():
            shutil.copy2(img, dst_i)
            n_new += 1
        if not dst_l.exists():
            shutil.copy2(lab, dst_l)
    print(f"复制完成（新复制 {n_new} 例，其余已存在）")
    print(f"imagesTr: {len(list(images_tr.glob('*.nii.gz')))}  "
          f"labelsTr: {len(list(labels_tr.glob('*.nii.gz')))}")

    meta = {
        "channel_names": {"0": args.channel_name},
        "labels": {"background": 0, "pancreas": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
        "name": args.dataset,
        "description": f"CrossPan {args.seq} pancreas, splits={args.splits}",
    }
    (ds / "dataset.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\ndataset.json 写入: {ds / 'dataset.json'}")
    print(f"\n下一步（在同一个 PowerShell 窗口里先设好环境变量）:")
    print(f'  $env:nnUNet_raw="{NNUNET_BASE / "nnUNet_raw"}"')
    print(f'  $env:nnUNet_preprocessed="{NNUNET_BASE / "nnUNet_preprocessed"}"')
    print(f'  $env:nnUNet_results="{NNUNET_BASE / "nnUNet_results"}"')
    print("  nnUNetv2_plan_and_preprocess -d 310 -np 3")


if __name__ == "__main__":
    main()
