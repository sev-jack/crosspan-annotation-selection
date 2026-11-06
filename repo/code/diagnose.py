"""诊断源模型：在训练病例上多少分、验证病例上多少分、按中心拆开各多少分。

用法:
    python diagnose.py --ckpt runs/source_T1W_s0.pt

用途：区分"模型没学会"（训练集也低）和"泛化差"（训练集高、验证集低）。
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import config
import train_eval as TE


def center_of(key):
    stem = key.split("_", 2)[-1]
    m = re.match(r"([A-Za-z]+)", stem)
    return m.group(1).upper() if m else "UNK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(config.RUNS / "source_T1W_s0.pt"))
    ap.add_argument("--n-train", type=int, default=20)
    args = ap.parse_args()

    splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = TE.build_model()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(dev)
    print(f"载入 {args.ckpt}\n")

    rng = np.random.RandomState(0)
    tr = splits["source"]["train"]
    va = splits["source"]["val"]
    tr_s = [tr[i] for i in rng.choice(len(tr), size=min(args.n_train, len(tr)), replace=False)]

    for tag, keys in [(f"训练病例(抽样{len(tr_s)})", tr_s), (f"验证病例({len(va)})", va)]:
        d, per = TE.eval_volumes(model, keys, dev)
        by_center = defaultdict(list)
        for k, v in per.items():
            by_center[center_of(k)].append(v)
        print(f"{tag}: 平均体 Dice = {d:.4f}")
        for c in sorted(by_center):
            v = by_center[c]
            print(f"   {c:>5}: n={len(v):3d}  Dice={np.mean(v):.4f}")
        vals = sorted(per.values())
        print(f"   分位: p10={vals[len(vals) // 10]:.3f}  "
              f"中位={vals[len(vals) // 2]:.3f}  "
              f"p90={vals[len(vals) * 9 // 10]:.3f}\n")

    print("读法：训练集高、验证集低 = 泛化问题，加增广/减容量；")
    print("      两边都低 = 欠拟合，加 epoch/加容量。")


if __name__ == "__main__":
    main()
