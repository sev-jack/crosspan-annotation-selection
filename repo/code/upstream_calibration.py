"""上游任务第 3 阶段：把严重度分数校准成「带覆盖率保证的 Dice 预测」。

设计：
  校准集 = 标注池的病例（相当于"已经标过、知道真实 Dice"的那一批）
  测试集 = 评估集（从未参与任何拟合）
  方法   = split conformal：残差分位数给出预测区间，覆盖率有分布无关的保证

同时做留一域验证：在一个域上校准，在另一个域上检验覆盖率。

    python code/upstream_calibration.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import config

OUT = config.WORK / "outputs" / "upstream"
FEATURES = ["s_entropy", "s_energy", "s_ens"]
ALPHAS = [0.10, 0.20]


def load_rows():
    with (OUT / "severity_scores.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FEATURES + ["dice_mean"]:
            r[k] = float(r[k])
    # 少数病例模型完全没有预测出前景（OOP 上 zero-shot 只有 0.06），
    # 此时熵与能量没有定义。按"最严重"处理：用该域非缺失值的最大值填充。
    nan_filled = 0
    for dom in {r["domain"] for r in rows}:
        sub = [r for r in rows if r["domain"] == dom]
        for f in FEATURES:
            vals = [r[f] for r in sub if not np.isnan(r[f])]
            fill = max(vals) if vals else 0.0
            for r in sub:
                if np.isnan(r[f]):
                    r[f] = fill
                    nan_filled += 1
    print(f"[提示] 缺失分数按域内最大值填充：{nan_filled} 处")
    return rows


def fit_linear(X, y):
    X = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def predict(coef, X):
    X = np.column_stack([np.ones(len(X)), X])
    return X @ coef


def conformal_q(residuals, alpha):
    n = len(residuals)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(residuals)[k - 1])


def main():
    rows = load_rows()
    domains = ["CT", "NYU", "OOP"]
    report = {}

    print("=== 1. 分域校准（池 → 校准，评估集 → 测试）===")
    print(f"{'域':6}{'n_cal':>7}{'n_test':>7}{'特征':>12}{'alpha':>7}{'覆盖':>8}{'区间宽':>9}{'测试集rho':>10}")

    per_domain = {}
    for dom in domains:
        cal = [r for r in rows if r["domain"] == dom and r["split"] == "pool"]
        tst = [r for r in rows if r["domain"] == dom and r["split"] == "eval"]
        Xc = np.array([[r[f] for f in FEATURES] for r in cal])
        yc = np.array([r["dice_mean"] for r in cal])
        Xt = np.array([[r[f] for f in FEATURES] for r in tst])
        yt = np.array([r["dice_mean"] for r in tst])
        coef = fit_linear(Xc, yc)
        rc = np.abs(yc - predict(coef, Xc))
        yhat = predict(coef, Xt)
        rho = float(np.corrcoef(np.argsort(np.argsort(yhat)),
                                np.argsort(np.argsort(yt)))[0, 1])
        entry = dict(n_cal=len(cal), n_test=len(tst), test_spearman=round(rho, 3), alphas={})
        for a in ALPHAS:
            q = conformal_q(rc, a)
            lo, hi = yhat - q, yhat + q
            cov = float(np.mean((yt >= lo) & (yt <= hi)))
            width = float(np.mean(hi - lo))
            entry["alphas"][f"{a:.2f}"] = dict(q=round(q, 4), coverage=round(cov, 4),
                                               mean_width=round(width, 4))
            print(f"{dom:6}{len(cal):>7}{len(tst):>7}{'3 维':>12}{a:>7.2f}{cov:>8.2%}{width:>9.3f}{rho:>10.3f}")
        per_domain[dom] = entry
    report["within_domain"] = per_domain

    print("\n=== 2. 留一域：在 A 校准，在 B 测试覆盖率 ===")
    print(f"{'校准域':8}{'测试域':8}{'alpha':>7}{'覆盖':>8}{'区间宽':>9}{'参考(域内)':>11}")
    transfer = []
    for a_dom in domains:
        cal = [r for r in rows if r["domain"] == a_dom]
        Xc = np.array([[r[f] for f in FEATURES] for r in cal])
        yc = np.array([r["dice_mean"] for r in cal])
        coef = fit_linear(Xc, yc)
        rc = np.abs(yc - predict(coef, Xc))
        for b_dom in domains:
            if a_dom == b_dom:
                continue
            tst = [r for r in rows if r["domain"] == b_dom]
            Xt = np.array([[r[f] for f in FEATURES] for r in tst])
            yt = np.array([r["dice_mean"] for r in tst])
            yhat = predict(coef, Xt)
            ref = per_domain[b_dom]["alphas"]["0.10"]["coverage"]
            for a in ALPHAS:
                q = conformal_q(rc, a)
                cov = float(np.mean((yt >= yhat - q) & (yt <= yhat + q)))
                width = float(np.mean(2 * q))
                print(f"{a_dom:8}{b_dom:8}{a:>7.2f}{cov:>8.2%}{width:>9.3f}{ref:>11.2%}")
                transfer.append(dict(calibrate_on=a_dom, test_on=b_dom, alpha=a,
                                     coverage=round(cov, 4), mean_width=round(width, 4),
                                     within_domain_coverage=round(ref, 4)))
    report["leave_one_domain_out"] = transfer

    (OUT / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {OUT / 'calibration_report.json'}")


if __name__ == "__main__":
    main()
