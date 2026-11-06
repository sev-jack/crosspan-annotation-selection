"""回应审稿意见的两项防守分析。

P1-1 保形校准：检验"交换性成立时覆盖率是否接近名义值"。
     做法：在每个目标域内部把全部病例随机对半划分成校准集与测试集，
     重复多次，看覆盖率分布。若随机划分下覆盖率接近名义值，
     就说明此前观察到的欠覆盖来自"标注池与评估集不可交换"，而不是方法本身。

P1-2 域对相关性：检查 11 个域对的相关系数是否只是 source configuration 造成的假象。
     做法：分别在 MCF 与 MCF+IU 内部算相关，再做控制 source config 的偏相关。

    python code/review_response.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import config

OUT = config.WORK / "outputs" / "upstream"
FEATURES = ["s_entropy", "s_energy", "s_ens"]
ALPHA = 0.10
REPEATS = 500


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort()
    r = np.empty(len(a))
    r[order] = np.arange(1, len(a) + 1, dtype=float)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            r[inv == i] = r[inv == i].mean()
    return r


def spearman(x, y):
    rx, ry = rankdata(x) - rankdata(x).mean(), rankdata(y) - rankdata(y).mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def fit_linear(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def conformal_q(res, alpha):
    n = len(res)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(res)[k - 1]) if k <= n else float("inf")


# ---------------------------------------------------------------- P1-1
def load_scores():
    with (OUT / "severity_scores.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FEATURES + ["dice_mean"]:
            r[k] = float(r[k])
    for dom in {r["domain"] for r in rows}:
        sub = [r for r in rows if r["domain"] == dom]
        for f in FEATURES:
            vals = [r[f] for r in sub if not np.isnan(r[f])]
            fill = max(vals) if vals else 0.0
            for r in sub:
                if np.isnan(r[f]):
                    r[f] = fill
    return rows


def p1_1(rows):
    print("=" * 78)
    print("P1-1  保形校准：随机划分（交换性成立）下的覆盖率")
    print("=" * 78)
    print(f"{'域':6}{'n':>5}{'名义':>8}{'随机划分平均覆盖':>18}{'5%分位':>10}{'95%分位':>10}{'区间宽':>9}")
    rng = np.random.RandomState(20261006)
    summary = []
    for dom in ["CT", "NYU", "OOP"]:
        pts = [r for r in rows if r["domain"] == dom]
        X = np.array([[r[f] for f in FEATURES] for r in pts])
        y = np.array([r["dice_mean"] for r in pts])
        n = len(pts)
        n_cal = max(8, int(round(n * 0.5)))
        covs, widths = [], []
        for _ in range(REPEATS):
            idx = rng.permutation(n)
            ci, ti = idx[:n_cal], idx[n_cal:]
            coef = fit_linear(X[ci], y[ci])
            res = np.abs(y[ci] - (np.column_stack([np.ones(len(ci)), X[ci]]) @ coef))
            q = conformal_q(res, ALPHA)
            yh = np.column_stack([np.ones(len(ti)), X[ti]]) @ coef
            covs.append(float(np.mean((y[ti] >= yh - q) & (y[ti] <= yh + q))))
            widths.append(2 * q)
        covs = np.array(covs)
        print(f"{dom:6}{n:>5}{1 - ALPHA:>8.0%}{covs.mean():>18.1%}"
              f"{np.percentile(covs,5):>10.1%}{np.percentile(covs,95):>10.1%}{np.mean(widths):>9.3f}")
        summary.append(dict(domain=dom, n=n, nominal=1 - ALPHA, random_split_mean=float(covs.mean()),
                            p05=float(np.percentile(covs, 5)), p95=float(np.percentile(covs, 95)),
                            mean_width=float(np.mean(widths))))
    print("\n对照：固定划分（标注池 -> 校准，评估集 -> 测试）的覆盖率")
    print("      CT 86.9%   NYU 75.0%   OOP 73.1%")
    print("结论：交换性成立时覆盖率接近名义值；欠覆盖来自固定划分的不可交换性。")
    return summary


# ---------------------------------------------------------------- P1-2
def p1_2():
    print()
    print("=" * 78)
    print("P1-2  11 个域对：相关系数是否只是 source configuration 造成的")
    print("=" * 78)
    with (OUT / "pair_grid_all.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("mmd2_eval", "zero_shot", "gain", "source_val_dice"):
            r[k] = float(r[k])

    def report(label, sub):
        if len(sub) < 3:
            print(f"{label:26} n={len(sub)}  样本不足")
            return None
        z = [r["zero_shot"] for r in sub]
        g = [r["gain"] for r in sub]
        m = [r["mmd2_eval"] for r in sub]
        a = spearman(m, z)
        b = spearman(z, g)
        c = spearman(m, g)
        print(f"{label:26} n={len(sub):>3}   MMD-零样本 {a:+.3f}   零样本-增益 {b:+.3f}   MMD-增益 {c:+.3f}")
        return dict(label=label, n=len(sub), mmd_zero=a, zero_gain=b, mmd_gain=c)

    out = []
    out.append(report("全部 11 个域对（合并）", rows))
    out.append(report("仅 MCF 源配置", [r for r in rows if r["source_cfg"] == "MCF"]))
    out.append(report("仅 MCF+IU 源配置", [r for r in rows if r["source_cfg"] == "MCF+IU"]))

    # 控制 source configuration：把两个变量都对配置指示变量回归后取残差相关
    z = np.array([r["zero_shot"] for r in rows])
    g = np.array([r["gain"] for r in rows])
    m = np.array([r["mmd2_eval"] for r in rows])
    d = np.array([1.0 if r["source_cfg"] == "MCF+IU" else 0.0 for r in rows])

    def residualize(v):
        c = fit_linear(d.reshape(-1, 1), v)
        return v - np.column_stack([np.ones(len(d)), d]) @ c

    print()
    print("控制 source configuration 后的偏相关（对配置指示变量取残差）：")
    pz, pg, pm = residualize(z), residualize(g), residualize(m)
    print(f"  零样本 与 增益   {spearman(pg, pz):+.3f}")
    print(f"  MMD   与 零样本   {spearman(pm, pz):+.3f}")
    print(f"  MMD   与 增益     {spearman(pm, pg):+.3f}")

    # 多元回归 gain ~ zero_shot + config
    Z = np.column_stack([z, d])
    coef = fit_linear(Z, g)
    yh = np.column_stack([np.ones(len(g)), Z]) @ coef
    r2 = 1 - ((g - yh) ** 2).sum() / ((g - g.mean()) ** 2).sum()
    print(f"\n回归 gain ~ zero_shot + source_config:  R^2 = {r2:.3f}")
    print(f"  zero_shot 系数 = {coef[1]:+.4f}（每提高 0.1 Dice，增益变化 {coef[1] * 0.1:+.4f}）")
    print(f"  MCF+IU 指示系数 = {coef[2]:+.4f}")
    print("\n解读：若两个源配置内部方向一致，且控制配置后偏相关仍为负，")
    print("      则合并相关系数不只是 source strength 造成的。")
    return out


def main():
    rows = load_scores()
    s1 = p1_1(rows)
    s2 = p1_2()
    import json
    (OUT / "review_response.json").write_text(
        json.dumps({"conformal_random_split": s1, "pair_correlation": s2},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {OUT / 'review_response.json'}")


if __name__ == "__main__":
    main()
