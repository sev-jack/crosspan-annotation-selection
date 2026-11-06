"""消融 A4：不用 MRI 预训练，直接在目标域上从零训练。

用来回答「MRI 预训练到底值多少」：同一批目标域标注、同样的训练轮数，
一个从 MRI 权重出发微调，一个从随机初始化开始，差多少。

注意对比时的口径：
  transfer  = MRI 预训练(300ep) + 目标域微调(60ep, lr=1e-4)
  scratch   = 随机初始化 + 目标域训练(本脚本 --epochs, lr=1e-3)
两者轮数不完全对齐，但都是各自合适的训练配置；报告里要写清这一点。

用法：
    python scratch_eval.py --target CT --budget 1.0 --epochs 150
    python scratch_eval.py --target CT --budget 0.1 --epochs 150
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import config
import train_eval as TE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="CT")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default=None, help="结果写入的 json（追加）")
    args = ap.parse_args()

    splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
    pool = splits["targets"][args.target]["pool"]
    ev = splits["targets"][args.target]["eval"]
    n = max(1, int(round(args.budget * len(pool)))) if args.budget > 0 else 0
    if n == 0:
        raise SystemExit("从零训练至少要 1 例标注（budget > 0）")
    rng = np.random.RandomState(args.seed)
    picked = sorted(rng.choice(len(pool), size=n, replace=False).tolist())
    keys = [pool[i] for i in picked]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[scratch] {args.target} budget={args.budget:.0%} -> {n}/{len(pool)} 例，"
          f"{args.epochs} epoch 从随机初始化训练，device={dev}")

    t0 = time.time()
    model = TE.train(keys, args.epochs, config.SOURCE_LR, dev,
                     init_ckpt=None, tag=f"scratch_{args.target}")
    dice, per_case = TE.eval_volumes(model, ev, dev)
    mins = (time.time() - t0) / 60
    print(f"[scratch] {args.target} {args.budget:.0%} 无预训练 Dice = {dice:.4f}"
          f"  （{len(ev)} 例评估，用时 {mins:.0f} 分钟）")

    row = dict(mode="scratch", target=args.target, budget=args.budget,
               n_label=n, pool_size=len(pool), seed=args.seed,
               epochs=args.epochs, dice=round(dice, 6), minutes=round(mins, 1))
    out = Path(args.out) if args.out else (config.WORK / "scratch_results.json")
    rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    rows.append(row)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已追加到 {out}")


if __name__ == "__main__":
    main()
