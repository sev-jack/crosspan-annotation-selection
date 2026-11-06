"""把 CrossPan 预处理成缓存，并写死一份 splits.json。

用法:
    python prepare.py

产出:
    cache/<key>.npz      每例一个文件，img 为 uint8 [Z,256,256]，lab 为 uint8 [Z,256,256]
    splits.json          源域 train/val、各目标域 pool/eval 的病例列表

为什么要有 splits.json: 让训练、评估、报表三个环节用的是同一份划分，
不会出现"我以为是同一批病例"的事故。
"""

import argparse
import json
import os
import re
import time

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from config import (
    ROOT,
    CACHE,
    WORK,
    SPLITS,
    ALL_SPECS,
    SOURCES,
    TARGETS,
    SIZE,
    TARGET_SPACING,
    FG_LABEL,
    BODY_CROP,
    SRC_VAL_FRAC,
    SPLIT_SEED,
)
from protocol import patient_id_from_key


def center_of(stem: str) -> str:
    m = re.match(r"([A-Za-z]+)", stem)
    return m.group(1).upper() if m else "UNK"


def list_spec(seq, split):
    img_dir = ROOT / seq / split / "images"
    lab_dir = ROOT / seq / split / "labels"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        return []
    imgs = {f.replace("_0000.nii.gz", ""): f for f in os.listdir(img_dir) if f.endswith(".nii.gz")}
    labs = {f.replace(".nii.gz", ""): f for f in os.listdir(lab_dir) if f.endswith(".nii.gz")}
    out = []
    for stem in sorted(set(imgs) & set(labs)):
        out.append(
            dict(
                key=f"{seq}_{split}_{stem}",
                seq=seq,
                split=split,
                stem=stem,
                center=center_of(stem),
                image=str(img_dir / imgs[stem]),
                label=str(lab_dir / labs[stem]),
            )
        )
    return out


def fit_to_size(t):
    """t: [Z,1,h,w] -> [Z,1,SIZE,SIZE]，居中填充或裁切。"""
    _, _, h, w = t.shape
    ph, pw = max(0, SIZE - h), max(0, SIZE - w)
    if ph or pw:
        t = F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
    h, w = t.shape[-2:]
    h0, w0 = (h - SIZE) // 2, (w - SIZE) // 2
    return t[:, :, h0:h0 + SIZE, w0:w0 + SIZE]


def process_case(c):
    img = nib.as_closest_canonical(nib.load(c["image"]))
    lab = nib.as_closest_canonical(nib.load(c["label"]))
    x = np.asarray(img.dataobj, dtype=np.float32)
    y = np.asarray(lab.dataobj)
    if x.ndim == 4:
        x = x[..., 0]
    if y.ndim == 4:
        y = y[..., 0]
    # 部分中心（MCF/NYU）的 NIfTI 经过 canonical 重定向后是负步长视图，
    # torch.from_numpy 不接受，必须先转成连续数组。
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"预期 3D 体数据，实际 image{x.ndim}D label{y.ndim}D")
    if x.shape != y.shape:
        raise ValueError(f"image{x.shape} 与 label{y.shape} 形状不一致")

    zooms = img.header.get_zooms()
    d0, d1 = float(zooms[0]), float(zooms[1])

    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(1)          # Z,1,H,W
    yt = torch.from_numpy((y == FG_LABEL).astype(np.float32)).permute(2, 0, 1).unsqueeze(1)

    # x 是 (H, W, Z)，轴 0 的物理间隔是 zooms[0]
    new_h = max(1, int(round(x.shape[0] * d0 / TARGET_SPACING)))
    new_w = max(1, int(round(x.shape[1] * d1 / TARGET_SPACING)))
    xt = F.interpolate(xt, size=(new_h, new_w), mode="bilinear", align_corners=False)
    yt = F.interpolate(yt, size=(new_h, new_w), mode="nearest")

    if BODY_CROP:
        # 按身体轮廓裁掉空气，再统一缩放到 SIZE。
        # 各中心物理视场差很多（实测 EMC 390mm、MCF 300mm），固定 1.5mm 视场会让
        # 同一器官在不同中心的画面里大小差近一倍，等于凭空多加一层域移。
        xn0 = xt[:, 0].numpy()
        proj = (xn0 > np.percentile(xn0, 60)).any(axis=0)
        if proj.sum() < 500:
            proj = (xn0 > np.percentile(xn0, 30)).any(axis=0)
        rows = np.where(proj.any(axis=1))[0]
        cols = np.where(proj.any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        pr = max(4, int((r1 - r0) * 0.10))
        pc = max(4, int((c1 - c0) * 0.10))
        r0, r1 = max(0, r0 - pr), min(xt.shape[-2], r1 + pr)
        c0, c1 = max(0, c0 - pc), min(xt.shape[-1], c1 + pc)
        xt = xt[:, :, r0:r1, c0:c1]
        yt = yt[:, :, r0:r1, c0:c1]
        crop_h, crop_w = xt.shape[-2:]
        xt = F.interpolate(xt, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
        yt = F.interpolate(yt, size=(SIZE, SIZE), mode="nearest")
        spacing_y = TARGET_SPACING * crop_h / SIZE
        spacing_x = TARGET_SPACING * crop_w / SIZE
    else:
        xt = fit_to_size(xt)
        yt = fit_to_size(yt)
        spacing_y = spacing_x = TARGET_SPACING

    # 注意: 要取 [:, 0]（全部切片）而不是 [0]（第一张切片）
    xn = xt[:, 0].numpy()
    lo, hi = np.percentile(xn, 0.5), np.percentile(xn, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(xn.min()), float(xn.max()) + 1e-6
    xi = np.clip((xn - lo) / (hi - lo), 0.0, 1.0)
    xi = (xi * 255.0).round().astype(np.uint8)
    yi = (yt[:, 0].numpy() > 0.5).astype(np.uint8)

    spacing_xyz = (d0, d1, float(zooms[2]))
    spacing_zyx = (float(zooms[2]), float(spacing_y), float(spacing_x))
    return xi, yi, spacing_xyz, spacing_zyx


def cached_spacing_zyx(c):
    """Physical voxel spacing for an existing cache without spacing metadata."""
    if BODY_CROP:
        return None
    img = nib.as_closest_canonical(nib.load(c["image"]))
    zooms = img.header.get_zooms()
    return [round(float(zooms[2]), 4), round(TARGET_SPACING, 4), round(TARGET_SPACING, 4)]


def build_splits(all_cases):
    rng = np.random.RandomState(SPLIT_SEED)

    src = SOURCES["T1W"]
    src_excl = set(src.get("exclude_centers", []))
    src_cases = [c for c in all_cases
                 if (c["seq"], c["split"]) in src["specs"] and c["center"] not in src_excl]
    source_groups = {}
    for c in src_cases:
        source_groups.setdefault(patient_id_from_key(c["key"]), []).append(c["key"])
    patient_order = list(source_groups)
    idx = rng.permutation(len(patient_order))
    n_val = max(1, int(len(patient_order) * SRC_VAL_FRAC))
    val_patients = {patient_order[i] for i in idx[:n_val]}
    val_keys = sorted(k for pid in val_patients for k in source_groups[pid])
    train_keys = sorted(k for pid in patient_order if pid not in val_patients
                        for k in source_groups[pid])

    targets = {}
    for name, spec in TARGETS.items():
        keep = set(spec.get("centers", []) or [])
        pool = [c for c in all_cases if (c["seq"], c["split"]) in spec["pool"]]
        if keep:
            pool = [c for c in pool if c["center"] in keep]
        pool = sorted(pool, key=lambda c: c["key"])
        if spec["eval_frac"] is None:
            ev = [c for c in all_cases if (c["seq"], c["split"]) in spec["eval"]]
            if keep:
                ev = [c for c in ev if c["center"] in keep]
            ev = sorted(ev, key=lambda c: c["key"])
            pool_keys = [c["key"] for c in pool]
            eval_keys = [c["key"] for c in ev]
        else:
            idx = rng.permutation(len(pool))
            n_ev = max(1, int(len(pool) * spec["eval_frac"]))
            eval_keys = sorted(pool[i]["key"] for i in idx[:n_ev])
            pool_keys = sorted(pool[i]["key"] for i in idx[n_ev:])

        overlap = set(pool_keys) & set(eval_keys)
        if overlap:
            raise ValueError(f"目标域 {name}: pool 与 eval 有重叠 {sorted(overlap)[:5]}")

        target_split = dict(pool=pool_keys, eval=eval_keys)
        if name == "OOP":
            # OOP 的原始 41 例保留为同患者跨序列分析；主评估另提供严格
            # patient-disjoint 子集，排除源训练和源验证中出现过的患者。
            source_patients = {
                patient_id_from_key(k) for k in train_keys + val_keys
            }
            target_split["eval_patient_disjoint"] = [
                k for k in eval_keys
                if patient_id_from_key(k) not in source_patients
            ]
        targets[name] = target_split

    return dict(
        source=dict(name="T1W", train=train_keys, val=val_keys),
        targets=targets,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="忽略已有缓存，全部重新处理（改了预处理开关时必须加）")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    all_cases = []
    for seq, split in ALL_SPECS:
        got = list_spec(seq, split)
        print(f"{seq}/{split}: {len(got)} 例有标注")
        all_cases.extend(got)
    print(f"合计 {len(all_cases)} 例\n")

    index = {}
    old_index = {}
    if (CACHE / "index.json").exists():
        old_index = json.loads((CACHE / "index.json").read_text(encoding="utf-8"))
    t0 = time.time()
    for i, c in enumerate(all_cases, 1):
        out_path = CACHE / f"{c['key']}.npz"
        if out_path.exists() and not args.force:
            with np.load(out_path) as d:
                spacing = (d["spacing_zyx"].tolist() if "spacing_zyx" in d.files
                           else cached_spacing_zyx(c))
                index[c["key"]] = dict(
                    seq=c["seq"], split=c["split"], stem=c["stem"], center=c["center"],
                    z=int(d["img"].shape[0]),
                    fg_pct=float(d["lab"].mean() * 100),
                    **({"spacing_zyx": spacing} if spacing is not None else {}),
                )
            continue
        try:
            xi, yi, spacing, spacing_zyx = process_case(c)
        except Exception as e:
            print(f"  !! {c['key']} 失败: {e}")
            continue
        np.savez(out_path, img=xi, lab=yi, spacing_zyx=np.asarray(spacing_zyx, dtype=np.float32))
        index[c["key"]] = dict(
            seq=c["seq"], split=c["split"], stem=c["stem"], center=c["center"],
            z=int(xi.shape[0]), fg_pct=float(yi.mean() * 100),
            spacing=[round(s, 4) for s in spacing],
            spacing_zyx=[round(s, 4) for s in spacing_zyx],
        )
        if i % 25 == 0 or i == len(all_cases):
            print(f"  {i}/{len(all_cases)}  {time.time() - t0:.0f}s  {c['key']}")

    splits = build_splits([c for c in all_cases if c["key"] in index])
    # 保留不由本脚本管理的目标域（例如 prepare_ct.py 加的 CT），
    # 否则重跑 CrossPan 预处理会把它们从 splits.json 里抹掉。
    if SPLITS.exists():
        old = json.loads(SPLITS.read_text(encoding="utf-8"))
        for name, spec in old.get("targets", {}).items():
            if name not in splits["targets"]:
                splits["targets"][name] = spec
                print(f"  保留已有目标域: {name} (pool={len(spec['pool'])} eval={len(spec['eval'])})")
    # CT 等由其它预处理脚本维护的病例元数据也保留，避免只重跑 MRI
    # 预处理时 cache/index.json 丢失 CT 物理间距。
    for key, meta in old_index.items():
        if key not in index and (CACHE / f"{key}.npz").exists():
            index[key] = meta
    SPLITS.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(CACHE / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"源域 {splits['source']['name']}: train={len(splits['source']['train'])} val={len(splits['source']['val'])}")
    for name, t in splits["targets"].items():
        print(f"目标域 {name}: pool={len(t['pool'])} eval={len(t['eval'])}")
    print(f"\nsplits 写入: {SPLITS}")
    print(f"缓存目录: {CACHE}  ({len(index)} 例)")
    print("\n下一步: python train_eval.py --source T1W --target OOP --budget 0 --seed 0")


if __name__ == "__main__":
    main()
