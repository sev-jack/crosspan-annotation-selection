"""逐例标注效用 U(x) 的多域 / 多种子汇总。

回答两件事（都只用已有产物，不需要再训练）：
  1. U(x) 的分布：均值 / SD / 范围 / 正值比例（每例先跨种子平均）
  2. U(x) 与"无标签难度分数"的秩相关，以及"按难度挑最难的 k 例"的平均效用

数据来源：
  outputs/upstream/severity_scores.csv        每例的无标签分数 + 零样本 Dice（真值）
  outputs/upstream/individual_utility.csv     OOP：68 例 × 3 种子
  outputs/upstream/individual_utility_nyu.csv NYU：17 例 × 3 种子
  outputs/upstream/individual_utility_ct.csv  CT ：220 例 × 3 种子

口径与 22 号文档一致（已用 OOP 的数字逐项反推核对过）：
  - 分布统计按"每例跨种子平均"后的 1 个数/例
  - "最难" = 该例自身的零样本 Dice（dice_mean）最低

    python code/report_utility_multiseed.py            # 打印
    python code/report_utility_multiseed.py --write    # 另写 summary json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

import config  # noqa: E402

OUT = config.WORK / "outputs" / "upstream"

DOMAINS = {
    "OOP": dict(csv="individual_utility.csv", split="pool", label="OOP 跨序列"),
    "NYU": dict(csv="individual_utility_nyu.csv", split="pool", label="NYU 跨中心"),
    "CT":  dict(csv="individual_utility_ct.csv",  split="pool", label="CT 跨模态"),
}

# 与 U(x) 对比的分数（名字 -> 表头），顺序即打印顺序
SCORES = [
    ("熵", "s_entropy"),
    ("能量", "s_energy"),
    ("多种子分歧(s_ens)", "s_ens"),
    ("特征kNN距离", "s_knn5"),
    ("逐例MMD相似度", "s_mmd_sim"),
    ("零样本Dice", "dice_mean"),
    ("前景占比", "fg_fraction"),
    ("零样本Dice的种子SD", "dice_sd"),
]

# 缺失值口径：22 号文档那张表用的是 NaN -> 0（用 OOP 的 −0.053 / +0.133 逐项反推过）。
# 同一个域里只要有一例的前景为空，S_entropy / S_energy 就会是 NaN；这里统一填 0，
# 保证新旧两列可比。CT 与 NYU 的池内没有缺失值，填与不填结果相同。
def _num(v: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if x != x else x


def load_scores() -> dict:
    path = OUT / "severity_scores.csv"
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        key = (r["domain"], r["split"], r["case_key"])
        out[key] = {k: _num(r[k])
                    for k in ("s_entropy", "s_energy", "s_ens", "s_knn5",
                              "s_mmd_sim", "dice_mean", "dice_sd", "fg_fraction")}
    return out


def missing_counts(domain: str, split: str) -> dict:
    with (OUT / "severity_scores.csv").open(encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["domain"] == domain and r["split"] == split]
    return {k: sum(1 for r in rows if r[k].strip() in ("", "nan", "NaN"))
            for k in ("s_entropy", "s_energy", "s_ens", "s_knn5", "s_mmd_sim")}


def load_utility(name: str) -> dict:
    path = OUT / name
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    per: dict[str, dict[int, float]] = {}
    for r in rows:
        per.setdefault(r["case_key"], {})[int(r["seed"])] = float(r["utility"])
    return per


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), int(ok.sum())
    return float(stats.spearmanr(x[ok], y[ok]).statistic), int(ok.sum())


def boot_ci(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        vals.append(stats.spearmanr(x[idx], y[idx]).statistic)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def analyse(domain: str, cfg: dict, scores: dict) -> dict:
    per = load_utility(cfg["csv"])
    seeds = sorted({s for v in per.values() for s in v})
    rows = []
    for key, by_seed in per.items():
        sc = scores.get((domain, cfg["split"], key))
        if sc is None:
            continue
        rows.append(dict(
            key=key,
            u=np.mean(list(by_seed.values())),
            u_by_seed={s: by_seed[s] for s in sorted(by_seed)},
            **sc,
        ))
    keys = [r["key"] for r in rows]
    U = np.array([r["u"] for r in rows])
    res = {
        "domain": domain,
        "label": cfg["label"],
        "n_cases": len(rows),
        "seeds": seeds,
        "n_utility_rows": int(sum(len(v) for v in per.values())),
        "unmatched_case_keys": int(len(per) - len(rows)),
        "dist": {
            "mean": float(U.mean()),
            "sd": float(U.std()),
            "min": float(U.min()),
            "max": float(U.max()),
            "pos_frac": float((U > 0).mean()),
        },
        "dist_by_seed": {
            str(s): {
                "mean": float(np.mean([r["u_by_seed"][s] for r in rows])),
                "sd": float(np.std([r["u_by_seed"][s] for r in rows])),
                "pos_frac": float(np.mean([r["u_by_seed"][s] > 0 for r in rows])),
            }
            for s in seeds
        },
        "missing_scores": missing_counts(domain, cfg["split"]),
        "corr": {},
        "topk": {},
    }
    for label, col in SCORES:
        x = [r[col] for r in rows]
        rho, n = spearman(x, U)
        ci = boot_ci(x, U) if np.isfinite(rho) else (float("nan"), float("nan"))
        res["corr"][label] = {"col": col, "rho": rho, "n": n,
                              "ci95": [ci[0], ci[1]]}

    # 按零样本 Dice 从低到高（最难在前）
    order = np.argsort([r["dice_mean"] for r in rows])
    pool_mean = float(U.mean())
    for k in (5, 10, 19, 20):
        if k > len(rows):
            continue
        sel = U[order[:k]]
        res["topk"][f"top{k}"] = {
            "mean_u": float(sel.mean()),
            "diff": float(sel.mean() - pool_mean),
            "per_seed_mean_u": {
                str(s): float(np.mean([rows[i]["u_by_seed"][s] for i in order[:k]]))
                for s in seeds
            },
        }
    # 每个种子的熵相关（看符号稳不稳）
    res["corr_by_seed"] = {}
    for s in seeds:
        rho, _ = spearman([r["s_entropy"] for r in rows],
                          [r["u_by_seed"][s] for r in rows])
        rho_d, _ = spearman([r["dice_mean"] for r in rows],
                            [r["u_by_seed"][s] for r in rows])
        res["corr_by_seed"][str(s)] = {"rho_entropy": rho, "rho_dice": rho_d}
    res["_keys"] = keys
    res["_U"] = U.tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="写 summary json")
    args = ap.parse_args()

    scores = load_scores()
    results = {d: analyse(d, c, scores) for d, c in DOMAINS.items()}

    print("=" * 100)
    print("逐例效用 U(x)：多域 / 多种子汇总")
    print("=" * 100)
    for d, r in results.items():
        dd = r["dist"]
        print(f"\n[{r['label']}]  例数 {r['n_cases']}，种子 {r['seeds']}，"
              f"效用格数 {r['n_utility_rows']}"
              + (f"，未匹配 {r['unmatched_case_keys']} 例" if r["unmatched_case_keys"] else ""))
        print(f"  U(x): 均值 {dd['mean']:+.4f}  SD {dd['sd']:.4f}  "
              f"范围 [{dd['min']:+.4f}, {dd['max']:+.4f}]  正值 {dd['pos_frac']:.1%}")
        for s, item in r["dist_by_seed"].items():
            print(f"    seed {s}: 均值 {item['mean']:+.4f}  SD {item['sd']:.4f}  "
                  f"正值 {item['pos_frac']:.1%}")
        miss = {k: v for k, v in r["missing_scores"].items() if v}
        if miss:
            print(f"    缺失值（已按 0 计入）：{miss}")
        print(f"  {'分数':<20}{'rho':>9}{'95%CI':>22}{'n':>6}")
        for label, item in r["corr"].items():
            ci = item["ci95"]
            print(f"  {label:<20}{item['rho']:>+9.3f}"
                  f"{f'[{ci[0]:+.3f}, {ci[1]:+.3f}]':>22}{item['n']:>6}")
        print("  按难度（零样本 Dice 最低）选最难的 k 例：")
        for k, item in r["topk"].items():
            ps = " ".join(f"s{s}:{v:+.4f}" for s, v in item["per_seed_mean_u"].items())
            print(f"    {k:<6} 平均 U {item['mean_u']:+.4f}  "
                  f"vs 全池 {item['diff']:+.4f}   ({ps})")
        print("  每个种子的秩相关：")
        for s, item in r["corr_by_seed"].items():
            print(f"    seed {s}: rho(熵,U) {item['rho_entropy']:+.3f}   "
                  f"rho(零样本Dice,U) {item['rho_dice']:+.3f}")

    if args.write:
        slim = {d: {k: v for k, v in r.items() if not k.startswith("_")}
                for d, r in results.items()}
        path = OUT / "utility_multiseed_summary.json"
        path.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n写入 {path}")


if __name__ == "__main__":
    main()
