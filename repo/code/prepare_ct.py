"""把 MSD Task07（CT 胰腺）转成和 CrossPan 一样的缓存格式，并注册为新的目标域 CT。

为什么单独一个脚本：CrossPan 和 MSD 的目录结构、标签编码、强度量纲都不同，
硬塞进 prepare.py 会让那个脚本变得难懂。两者产出的缓存格式一致，
所以下游的 train_eval / ensemble_eval / run_grid 完全不用改。

MSD Task07 的标签：1 = 胰腺，2 = 肿瘤。肿瘤算背景，只学胰腺。

产出：
    cache/CT_train_<case>.npz        与 CrossPan 病例同格式
    splits.json                      追加 targets["CT"] = {pool: [...], eval: [...]}
    cache/index.json                 追加 CT 病例的元数据

用法：
    python prepare_ct.py
    python prepare_ct.py --force          # 忽略已有缓存，全部重做
    python prepare_ct.py --norm percentile  # 不用 HU 窗，改用和 MRI 一样的百分位缩放
"""

import argparse
import json
import os

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from config import (
    CT_ROOT, CACHE, SPLITS, SIZE, TARGET_SPACING, CT_HU_WINDOW, CT_EVAL_FRAC,
)
from prepare import fit_to_size

FG = 1          # MSD：1 = 胰腺
TUMOR = 2       # 2 = 肿瘤，算背景


def list_cases():
    img_dir = CT_ROOT / "imagesTr"
    lab_dir = CT_ROOT / "labelsTr"

    def stem(f):
        # MSD 原始格式是 pancreas_001.nii.gz（无 _0000 通道后缀），
        # 但转成 nnU-Net 格式后会带。两种都接受。
        s = f[:-len(".nii.gz")]
        return s[:-5] if s.endswith("_0000") else s

    imgs = {stem(f): f for f in os.listdir(img_dir) if f.endswith(".nii.gz")}
    labs = {stem(f): f for f in os.listdir(lab_dir) if f.endswith(".nii.gz")}
    out = []
    for stem in sorted(set(imgs) & set(labs)):
        out.append(dict(
            key=f"CT_train_{stem}",
            stem=stem,
            center="MSD",
            image=str(img_dir / imgs[stem]),
            label=str(lab_dir / labs[stem]),
        ))
    return out


def process_case(c, norm="hu"):
    img = nib.as_closest_canonical(nib.load(c["image"]))
    lab = nib.as_closest_canonical(nib.load(c["label"]))
    x = np.asarray(img.dataobj, dtype=np.float32)
    y = np.asarray(lab.dataobj)
    if x.ndim == 4:
        x = x[..., 0]
    if y.ndim == 4:
        y = y[..., 0]
    # MSD 的部分病例经过 canonical 重定向后是负步长视图，torch 不接受
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    if x.shape != y.shape:
        raise ValueError(f"image{x.shape} 与 label{y.shape} 形状不一致")

    zooms = img.header.get_zooms()
    d0, d1 = float(zooms[0]), float(zooms[1])

    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(1)
    # 只学胰腺：肿瘤一并置为背景
    yt = torch.from_numpy(((y == FG) & (y != TUMOR)).astype(np.float32)).permute(2, 0, 1).unsqueeze(1)

    new_h = max(1, int(round(x.shape[0] * d0 / TARGET_SPACING)))
    new_w = max(1, int(round(x.shape[1] * d1 / TARGET_SPACING)))
    xt = F.interpolate(xt, size=(new_h, new_w), mode="bilinear", align_corners=False)
    yt = F.interpolate(yt, size=(new_h, new_w), mode="nearest")

    xt = fit_to_size(xt)
    yt = fit_to_size(yt)

    xn = xt[:, 0].numpy()
    if norm == "hu":
        # CT 用固定软组织窗。用百分位会把 -1000HU 的空气和 +1000HU 的骨都算进去，
        # 软组织的对比度被压到很窄一段，不利于学习。
        lo, hi = CT_HU_WINDOW
    elif norm == "global":
        # 固定全局窗：不做任何逐体数据自适应，保留 HU 的绝对语义。
        # 用来检验"是不是自适应归一化在起关键作用"。
        lo, hi = -1000.0, 1000.0
    else:
        lo, hi = np.percentile(xn, 0.5), np.percentile(xn, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(xn.min()), float(xn.max()) + 1e-6
    xi = np.clip((xn - lo) / (hi - lo), 0.0, 1.0)
    xi = (xi * 255.0).round().astype(np.uint8)
    yi = (yt[:, 0].numpy() > 0.5).astype(np.uint8)

    spacing_xyz = (d0, d1, float(zooms[2]))
    spacing_zyx = (float(zooms[2]), float(TARGET_SPACING), float(TARGET_SPACING))
    return xi, yi, spacing_xyz, spacing_zyx


def cached_spacing_zyx(c):
    img = nib.as_closest_canonical(nib.load(c["image"]))
    zooms = img.header.get_zooms()
    return [round(float(zooms[2]), 4), round(TARGET_SPACING, 4), round(TARGET_SPACING, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略已有缓存，全部重做")
    ap.add_argument("--norm", default="hu", choices=["hu", "percentile", "global"],
                    help="hu=软组织窗[-100,240]（默认）；"
                         "percentile=逐体数据百分位（与 MRI 同款，自适应）；"
                         "global=固定全局窗[-1000,1000]（不自适应）")
    ap.add_argument("--seed", type=int, default=0, help="切 pool/eval 的随机种子")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    cases = list_cases()
    print(f"MSD Task07：{len(cases)} 例带标注的 CT")
    if not cases:
        raise SystemExit(f"没找到数据，检查 CT_ROOT = {CT_ROOT}")

    index = {}
    if (CACHE / "index.json").exists():
        index = json.loads((CACHE / "index.json").read_text(encoding="utf-8"))

    done = 0
    for i, c in enumerate(cases, 1):
        out_path = CACHE / f"{c['key']}.npz"
        if out_path.exists() and not args.force:
            with np.load(out_path) as d:
                spacing = (d["spacing_zyx"].tolist() if "spacing_zyx" in d.files
                           else cached_spacing_zyx(c))
                z_slices = int(d["img"].shape[0])
                fg_pct = float(d["lab"].mean() * 100)
            index[c["key"]] = dict(seq="CT", split="train", stem=c["stem"], center="MSD",
                                   z=z_slices,
                                   fg_pct=fg_pct,
                                   spacing_zyx=spacing)
            continue
        try:
            xi, yi, sp, spacing_zyx = process_case(c, norm=args.norm)
        except Exception as e:
            print(f"  !! {c['key']} 失败: {type(e).__name__}: {e}")
            continue
        np.savez(out_path, img=xi, lab=yi, spacing_zyx=np.asarray(spacing_zyx, dtype=np.float32))
        index[c["key"]] = dict(seq="CT", split="train", stem=c["stem"], center="MSD",
                               z=int(xi.shape[0]), fg_pct=float(yi.mean() * 100),
                               spacing=[round(s, 4) for s in sp],
                               spacing_zyx=[round(s, 4) for s in spacing_zyx])
        done += 1
        if i % 25 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}")

    # 只保留本次成功落到磁盘的病例
    keys = [c["key"] for c in cases if (CACHE / f"{c['key']}.npz").exists()]
    if not keys:
        raise SystemExit("没有任何 CT 病例成功缓存")

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(keys))
    n_eval = max(1, int(len(keys) * CT_EVAL_FRAC))
    eval_keys = sorted(keys[i] for i in idx[:n_eval])
    pool_keys = sorted(keys[i] for i in idx[n_eval:])
    overlap = set(pool_keys) & set(eval_keys)
    if overlap:
        raise ValueError(f"pool 与 eval 有重叠: {sorted(overlap)[:5]}")

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    splits["targets"]["CT"] = dict(pool=pool_keys, eval=eval_keys)
    SPLITS.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    (CACHE / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    fg = np.mean([index[k]["fg_pct"] for k in keys])
    print(f"\n新缓存 {done} 例（共 {len(keys)} 例可用）")
    print(f"目标域 CT: pool={len(pool_keys)}  eval={len(eval_keys)}  平均前景={fg:.3f}%")
    print(f"归一化方式: {args.norm}")
    print(f"splits.json 已更新: {SPLITS}")
    print("\n下一步:")
    print("  python ensemble_eval.py --source T1W --target CT --seeds 0,1,2      # 先看零样本")
    print("  python run_grid.py --targets CT --budgets 0,0.01,0.05,0.1,0.2,1 --seeds 0")


if __name__ == "__main__":
    main()
