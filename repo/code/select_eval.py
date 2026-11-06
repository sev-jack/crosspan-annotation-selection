"""域内选样消融：同样的小预算，换不同选样策略，看目标域 Dice 差多少。

动机：预算曲线在 10% 之后就平了（CT 前 22 例 0.0045 Dice/例，后 198 例只有
0.00036 Dice/例）。既然价值都集中在最前面那一小撮，那么「标哪些」就比
「标多少」更关键。这个脚本就是量化「标哪些」值多少。

策略：
  random     随机（基线）
  entropy    源模型预测熵最高（不确定度）
  diversity  16^3 降采样体数据的 k-center 贪心（多样性/coreset）
  contrast   强度直方图离源域（MRI）最远（对比度偏移）
  hard       用目标标签挑源模型 Dice 最低的（**oracle 上界，实盘不可用**）

用法：
    python select_eval.py --target CT --budget 0.1 --seeds 0,1 --strategies all
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
import ensemble_eval as EE
from protocol import patient_id_from_key, seed_everything, select_nested_keys


# ------------------------------------------------------------------ 特征与打分

def _volume_stats(key):
    """返回 (16^3 降采样体数据, 32 维强度直方图)，都基于缓存里的 uint8 图。"""
    img, lab = TE.load_case(key)
    x = img.astype(np.float32) / 255.0
    z = x.shape[0]
    idx = np.linspace(0, z - 1, 16).round().astype(int)
    small = torch.from_numpy(x[idx])[:, None]               # 16,1,256,256
    small = torch.nn.functional.interpolate(small, size=(16, 16), mode="area")[:, 0]
    feat = small.numpy().reshape(-1)                        # 4096
    hist = np.histogram(x, bins=32, range=(0.0, 1.0), density=False)[0].astype(np.float64)
    hist = hist / max(hist.sum(), 1)
    return feat, hist


@torch.no_grad()
def _mean_entropy(model, key, dev):
    """源模型在该病例上的平均预测熵（只统计预测为前景的体素，避免背景淹没）。"""
    img, _ = TE.load_case(key)
    vol = TE.volume_to_25d(img)
    ents = []
    for z0 in range(0, img.shape[0], 16):
        z1 = min(img.shape[0], z0 + 16)
        xt = torch.from_numpy(TE.norm(vol[z0:z1])).to(dev)
        p = torch.softmax(TE.forward_plain(model, xt, dev).float(), dim=1)
        ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1)     # B,H,W
        fg = p.argmax(1) == 1
        if fg.any():
            ents.append(ent[fg].mean().item())
    return float(np.mean(ents)) if ents else 0.0


@torch.no_grad()
def _source_dice(model, key, dev):
    img, lab = TE.load_case(key)
    vol = TE.volume_to_25d(img)
    pred = np.zeros_like(lab)
    for z0 in range(0, img.shape[0], 16):
        z1 = min(img.shape[0], z0 + 16)
        xt = torch.from_numpy(TE.norm(vol[z0:z1])).to(dev)
        pred[z0:z1] = TE.forward_plain(model, xt, dev).argmax(1).cpu().numpy().astype(np.uint8)
    return TE.dice3d(pred, lab)


def kcenter_greedy(feats, k):
    """k-center 贪心：每次选离已选集合最远的点。"""
    n = len(feats)
    chosen = [int(np.argmax(feats.mean(1)))]        # 用最"亮"的那个当起点，确定性
    d = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(feats - feats[nxt], axis=1))
    return chosen


def pick(strategy, pool, model, dev, k, seed, rand_seed=None):
    """返回被选中的 k 个 key（长度 min(k, len(pool))）。"""
    k = min(k, len(pool))

    if strategy == "random":
        # Match the random subset used by run_grid for a paired ablation baseline.
        # rand_seed 用于选样重复性实验：同一种子下多次独立随机抽样。
        return select_nested_keys(pool, k, seed if rand_seed is None else rand_seed)

    if strategy == "diversity":
        feats = np.stack([_volume_stats(kk)[0] for kk in pool])
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        return sorted(pool[i] for i in kcenter_greedy(feats, k))

    if strategy == "contrast":
        # 源域（MRI）参考直方图：用源域验证集算
        splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
        ref = np.mean([_volume_stats(kk)[1] for kk in splits["source"]["val"]], axis=0)
        dist = [float(np.abs(_volume_stats(kk)[1] - ref).sum()) for kk in pool]
        order = np.argsort(dist)[::-1][:k]
        return sorted(pool[i] for i in order)

    if strategy == "entropy":
        sc = [( _mean_entropy(model, kk, dev), kk) for kk in pool]
        sc.sort(key=lambda t: -t[0])
        return sorted(kk for _, kk in sc[:k])

    if strategy == "hard":      # oracle：需要目标标签，只作上界参考
        sc = [( _source_dice(model, kk, dev), kk) for kk in pool]
        sc.sort(key=lambda t: t[0])
        return sorted(kk for _, kk in sc[:k])

    raise ValueError(f"未知策略: {strategy}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="T1W")
    ap.add_argument("--target", default="CT")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--k-abs", type=int, default=0,
                    help="按绝对病例数选样（如 5、10）；设置后忽略 --budget")
    ap.add_argument("--rep", type=int, default=0,
                    help="随机选样的重复序号；>0 时对 random 策略使用独立的随机抽样")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--strategies", default="all")
    ap.add_argument("--eval-key", default="eval",
                    help="目标域评估列表；OOP 主分析请用 eval_patient_disjoint")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_strategies = ["random", "entropy", "diversity", "contrast", "hard"]
    strategies = all_strategies if args.strategies == "all" \
        else [s.strip() for s in args.strategies.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
    target_split = splits["targets"][args.target]
    pool = target_split["pool"]
    if args.eval_key not in target_split:
        raise KeyError(f"目标域 {args.target} 没有评估列表 {args.eval_key!r}")
    ev = target_split[args.eval_key]
    pool_patients = {patient_id_from_key(k) for k in pool}
    eval_patients = {patient_id_from_key(k) for k in ev}
    overlap = pool_patients & eval_patients
    if overlap:
        raise ValueError(f"pool/eval 患者重叠，拒绝选样实验：{sorted(overlap)[:8]}")
    if args.eval_key == "eval_patient_disjoint":
        source_patients = {
            patient_id_from_key(k)
            for k in splits["source"]["train"] + splits["source"]["val"]
        }
        leaked = eval_patients & source_patients
        if leaked:
            raise ValueError(f"patient-disjoint eval 与 source train/val 重叠：{sorted(leaked)[:8]}")
        # 选样实验的标注池同样必须患者独立，否则选出来的"最严重的病例"可能是
        # 模型已经见过其另一序列的病人，严重度分数会被假信号污染。
        pool_leaked = pool_patients & source_patients
        if pool_leaked:
            raise ValueError(
                "标注池与 source train/val 存在患者重叠，patient-disjoint 口径不成立："
                f"{sorted(pool_leaked)[:8]}（共 {len(pool_leaked)} 位）\n"
                "请改用患者独立标注池，例如 EXP_SPLITS=splits_oop_pd.json"
            )
    if args.k_abs:
        k = max(1, min(int(args.k_abs), len(pool)))
        budget_used = k / len(pool)
    else:
        budget_used = args.budget
        k = max(1, int(round(args.budget * len(pool))))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"目标域 {args.target}  预算 {budget_used:.2%} = {k}/{len(pool)} 例  "
          f"评估={args.eval_key} ({len(ev)} 例)  device={dev}\n")

    out = Path(args.out) if args.out else (config.WORK / "outputs" / "seeded_v2" / "selection_1pct.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []

    for seed in seeds:
        source_ckpt = config.RUNS / f"source_{args.source}_s{seed}{config.RUN_TAG}.pt"
        if not source_ckpt.exists():
            raise FileNotFoundError(f"缺少匹配当前 RUNS/RUN_TAG 的源模型：{source_ckpt}")
        model = EE.load_model(source_ckpt, dev)
        for strat in strategies:
            seed_everything(seed)
            t0 = time.time()
            keys = pick(strat, pool, model, dev, k, seed,
                        rand_seed=(seed * 1000 + args.rep) if args.rep else None)
            t_pick = time.time() - t0
            print(f"[seed {seed}] {strat:10s} 选出 {len(keys)} 例（挑选耗时 {t_pick:.0f}s）", flush=True)

            t1 = time.time()
            tuned = TE.train(keys, config.FINETUNE_EPOCHS, config.FINETUNE_LR, dev,
                             init_ckpt=source_ckpt,
                             tag=f"select_{strat}")
            dice, per_case = TE.eval_volumes(tuned, ev, dev)
            mins = (time.time() - t1) / 60
            print(f"[seed {seed}] {strat:10s} Dice = {dice:.4f}  （微调 {mins:.1f} 分钟）", flush=True)

            row = dict(target=args.target, budget=budget_used, n_label=len(keys),
                       pool_size=len(pool), strategy=strat, seed=seed,
                       rep=args.rep,
                       dice=round(dice, 6), pick_seconds=round(t_pick, 1),
                       finetune_minutes=round(mins, 1), eval_key=args.eval_key,
                       seed_protocol=TE.SEED_PROTOCOL, selected_keys=keys,
                       source_checkpoint=str(source_ckpt),
                       per_case={case: round(float(value), 6) for case, value in per_case.items()})
            identity = (args.target, args.eval_key, budget_used, strat, seed, args.rep)
            rows = [r for r in rows if (
                r.get("target"), r.get("eval_key", "eval"), r.get("budget"),
                r.get("strategy"), r.get("seed"), r.get("rep", 0)
            ) != identity]
            rows.append(row)
            out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            del tuned
            if dev == "cuda":
                torch.cuda.empty_cache()
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
