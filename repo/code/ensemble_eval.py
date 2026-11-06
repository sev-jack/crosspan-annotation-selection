"""TTA（测试时增强）+ 多种子集成评估。

用法:
    python ensemble_eval.py --source T1W --target OOP --seeds 0,1,2
    python ensemble_eval.py --source T1W --target OOP --seeds 0      # 只看 TTA 单独效果

输出每个种子「无 TTA / 有 TTA」的 Dice，以及多模型集成后的 Dice。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import config
import train_eval as TE


def load_model(ckpt, dev):
    m = TE.build_model()
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(dev).eval()


@torch.no_grad()
def forward_prob(models, xt, dev, tta):
    """每个模型先做 TTA（或不），转成概率后跨模型平均。"""
    acc = None
    for m in models:
        logits = TE.forward_tta(m, xt, dev) if tta else TE.forward_plain(m, xt, dev)
        p = torch.softmax(logits.float(), dim=1)
        acc = p if acc is None else acc + p
    return acc / len(models)


@torch.no_grad()
def eval_with(models, keys, dev, tta, batch=16):
    per = {}
    for k in keys:
        img, lab = TE.load_case(k)
        vol = TE.volume_to_25d(img)
        Z = img.shape[0]
        pred = np.zeros_like(lab)
        for z0 in range(0, Z, batch):
            z1 = min(Z, z0 + batch)
            xt = torch.from_numpy(TE.norm(vol[z0:z1])).to(dev)
            pred[z0:z1] = forward_prob(models, xt, dev, tta).argmax(1).cpu().numpy().astype(np.uint8)
        per[k] = TE.dice3d(pred, lab)
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="T1W")
    ap.add_argument("--target", default="OOP")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default=None, help="权重文件名后缀，默认取 config.RUN_TAG")
    args = ap.parse_args()

    splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
    if args.target.upper() == "SRC":
        ev = splits["source"]["val"]
        label = "源域验证集"
    else:
        ev = splits["targets"][args.target]["eval"]
        label = f"目标域 {args.target}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    tag = config.RUN_TAG if args.tag is None else args.tag
    ckpts = []
    for s in seeds:
        p = config.RUNS / f"source_{args.source}_s{s}{tag}.pt"
        if p.exists():
            ckpts.append((s, p))
        else:
            print(f"[跳过] {p.name} 不存在")
    if not ckpts:
        raise SystemExit("没有任何可用的源模型权重，先跑 train_eval.py 训练")

    print(f"{label}  评估集 {len(ev)} 例  device={dev}\n")

    for s, p in ckpts:
        m = load_model(p, dev)
        d0, _ = eval_with([m], ev, dev, tta=False)
        d1, _ = eval_with([m], ev, dev, tta=True)
        print(f"seed {s}:   无TTA {d0:.4f}    TTA {d1:.4f}    ({d1 - d0:+.4f})")
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    models = [load_model(p, dev) for _, p in ckpts]
    de0, _ = eval_with(models, ev, dev, tta=False)
    de1, _ = eval_with(models, ev, dev, tta=True)
    print(f"\n{len(models)} 模型集成:   无TTA {de0:.4f}    TTA {de1:.4f}    ({de1 - de0:+.4f})")
    print(f"\n最终结果（TTA + {len(models)} 种子集成）= {de1:.4f}")


if __name__ == "__main__":
    main()
