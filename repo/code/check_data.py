"""CrossPan 数据验收脚本。

用法:
    python check_data.py --root "<DATA_ROOT>\\CrossPan"

只依赖 numpy + nibabel，不需要 torch。
输出: 每个序列/划分的图-标对应情况、中心分布、标签取值、前景占比。
标签取值这一项是必须看的——标错一次整张结果表作废。
"""

import argparse
import os
import re
from collections import defaultdict

import numpy as np
import nibabel as nib

SPLITS = ["train", "val", "test"]


def stem_of_image(name: str) -> str:
    return name.replace("_0000.nii.gz", "")


def center_of(name: str) -> str:
    m = re.match(r"([A-Za-z]+)", name)
    return m.group(1).upper() if m else "UNK"


def scan(root: str):
    print(f"root = {root}\n")
    seqs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    print(f"sequences = {seqs}\n")

    pairs = []
    for seq in seqs:
        for split in SPLITS:
            img_dir = os.path.join(root, seq, split, "images")
            lab_dir = os.path.join(root, seq, split, "labels")
            if not os.path.isdir(img_dir):
                continue

            imgs = {stem_of_image(f): f for f in os.listdir(img_dir) if f.endswith(".nii.gz")}
            labs = {}
            if os.path.isdir(lab_dir):
                labs = {f.replace(".nii.gz", ""): f for f in os.listdir(lab_dir) if f.endswith(".nii.gz")}

            matched = sorted(set(imgs) & set(labs))
            unlabeled = sorted(set(imgs) - set(labs))
            orphan = sorted(set(labs) - set(imgs))

            print(
                f"[{seq}/{split}] images={len(imgs)} labels={len(labs)} "
                f"matched={len(matched)} unlabeled={len(unlabeled)} label_without_image={len(orphan)}"
            )

            centers = defaultdict(int)
            for s in imgs:
                centers[center_of(s)] += 1
            print(f"    centers: {dict(sorted(centers.items()))}")

            if orphan:
                print(f"    !! 有 {len(orphan)} 个标签找不到对应图像，先查: {orphan[:5]}")

            for s in matched:
                pairs.append((seq, split, s, os.path.join(img_dir, imgs[s]), os.path.join(lab_dir, labs[s])))

    return seqs, pairs


def probe(pairs, n_cases=6):
    """读若干例，报告形状/spacing/标签取值/前景占比。"""
    if not pairs:
        print("没有找到任何图-标配对，先解决对应问题。")
        return

    print("\n" + "=" * 70)
    print("标签取值普查（决定 foreground label 取几）")
    print("=" * 70)

    label_values = defaultdict(int)
    step = max(1, len(pairs) // 40)  # 最多抽查 40 例
    sampled = pairs[::step][:40]

    for seq, split, stem, ip, lp in sampled:
        try:
            y = np.asarray(nib.load(lp).dataobj)
        except Exception as e:
            print(f"  {seq}/{stem}: 读取失败 {e}")
            continue
        vals, counts = np.unique(y, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            label_values[int(v)] += c

    total = sum(label_values.values()) or 1
    for v in sorted(label_values):
        print(f"  label={v:<4} 体素占比 {label_values[v] / total * 100:8.4f}%  (共 {label_values[v]} 体素)")
    print("\n  注意: 占比 90%+ 的那个值就是背景；胰腺应该是那个占比很小但非零的值。")
    print("  如果出现 2 个非零值，先确认是不是把肿瘤/胰管也算进去了。")

    print("\n" + "=" * 70)
    print("逐例详情")
    print("=" * 70)
    for seq, split, stem, ip, lp in pairs[:n_cases]:
        img = nib.as_closest_canonical(nib.load(ip))
        lab = nib.as_closest_canonical(nib.load(lp))
        x = np.asarray(img.dataobj)
        y = np.asarray(lab.dataobj)
        if x.ndim == 4:
            x = x[..., 0]
        if y.ndim == 4:
            y = y[..., 0]

        fg = float((y > 0).mean() * 100)
        zs = np.where((y > 0).any(axis=(0, 1)))[0]
        zrange = f"{zs.min()}-{zs.max()}" if len(zs) else "无前景"

        print(f"\n  {seq}/{stem}  center={center_of(stem)}")
        print(f"    image shape={x.shape}  spacing={tuple(round(float(s), 3) for s in img.header.get_zooms()[:3])}")
        print(f"    image range=[{x.min():.1f}, {x.max():.1f}]   dtype={x.dtype}")
        print(f"    label unique={np.unique(y).tolist()}  前景占比={fg:.3f}%")
        print(f"    含前景的轴位切片 z={zrange}  (共 {len(zs)} 层)")

        if fg > 25:
            print("    !! 前景占比异常高，标签可能是反的或者不只是胰腺")
        if fg == 0:
            print("    !! 这一例没有任何前景体素，查标签文件")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"<DATA_ROOT>\CrossPan")
    ap.add_argument("--cases", type=int, default=6, help="打印多少例的逐例详情")
    args = ap.parse_args()

    seqs, pairs = scan(args.root)
    probe(pairs, n_cases=args.cases)

    print("\n" + "=" * 70)
    print("验收清单")
    print("=" * 70)
    print("  1. 每个划分的 matched 与官方元数据一致?  (T1 train 186 / OOP train 95 / OOP val 14 / OOP test 27)")
    print("  2. 标签取值只有 {0, 1}?  若还有别的值，先搞清楚是什么。")
    print("  3. 前景占比是否在 1%-5% 量级?  胰腺的正常范围。")
    print("  4. center 缩写是否可用作域划分?  (协议要求按中心/序列切，禁止随机切)")


if __name__ == "__main__":
    main()
