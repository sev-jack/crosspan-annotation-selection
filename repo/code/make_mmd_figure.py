"""论文 Figure 2 候选：无标签的域移度量（MMD）能否预测零样本退化与标注收益。

三个面板按 MMD 从大到小排列同一个域顺序，让三者的排序对齐关系一眼可见。
数据全部从磁盘实测文件读取，不写死数字：
    outputs/radiology_execution/feature_mmd.csv   -> MMD² 与 bootstrap CI
    outputs/seeded_v2/summary.csv                  -> 各预算 Dice

用法：
    python make_mmd_figure.py
输出：
    实验DG/outputs/MMD三联图.png
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import config

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LABEL = {"OOP": "OOP\n跨序列", "CT": "CT\n跨模态", "NYU": "NYU\n跨中心"}
SHIFT = {"OOP": "跨序列", "CT": "跨模态 MRI→CT", "NYU": "跨中心"}


def read_mmd():
    p = config.WORK / "outputs" / "radiology_execution" / "feature_mmd.csv"
    out = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["target"]] = dict(
                mmd=float(row["rbf_mmd2_zscored_embedding"]),
                lo=float(row["bootstrap_ci_low"]),
                hi=float(row["bootstrap_ci_high"]),
            )
    return out


def read_curves():
    """返回 {target: [(n_label, dice_mean), ...]}，按 n 升序。"""
    p = config.WORK / "outputs" / "seeded_v2" / "summary.csv"
    out = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            t = row["target"]
            if row.get("eval_key") and row["eval_key"] not in ("eval", "eval_patient_disjoint"):
                continue
            key = "dice_mean"
            if key not in row:
                key = next((k for k in row if k.startswith("dice") and "mean" in k), None)
            if key is None:
                raise SystemExit(f"summary.csv 里找不到 Dice 均值列，实际列：{list(row)}")
            out.setdefault(t, []).append((int(row["n_label"]), float(row[key])))
    for t in out:
        # 同一 n 可能有多行（不同 eval_key），取最大 n 那条即可，但先去重保序
        seen = {}
        for n, d in out[t]:
            seen[n] = d
        out[t] = sorted(seen.items())
    return out


def main():
    mmd = read_mmd()
    curves = read_curves()

    order = sorted(mmd, key=lambda t: -mmd[t]["mmd"])          # 按 MMD 从大到小
    mmd_v = [mmd[t]["mmd"] for t in order]
    mmd_lo = [mmd[t]["mmd"] - mmd[t]["lo"] for t in order]
    mmd_hi = [mmd[t]["hi"] - mmd[t]["mmd"] for t in order]
    zero = [curves[t][0][1] for t in order]
    full = [curves[t][-1][1] for t in order]
    gain = [f - z for z, f in zip(zero, full)]
    full_n = [curves[t][-1][0] for t in order]
    x = range(len(order))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    colors = ["#C00000", "#1F4E79", "#548235"]

    # --- 面板 1：MMD ---
    ax = axes[0]
    ax.bar(x, mmd_v, yerr=[mmd_lo, mmd_hi], capsize=6, color=colors,
           edgecolor="white", linewidth=1.2)
    # 标签放在误差棒上帽之上，否则会被竖线穿过
    for i, t in enumerate(order):
        ax.text(i, mmd[t]["hi"] + 0.012, f"{mmd_v[i]:.3f}", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(mmd[t]["hi"] for t in order) * 1.22)
    ax.set_ylabel("特征空间 MMD²", fontsize=11)
    ax.set_title("① 无标签可算的域移度量\n（源模型特征上，源域 vs 目标域）", fontsize=11.5, pad=10)

    # --- 面板 2：零样本 ---
    ax = axes[1]
    ax.bar(x, zero, color=colors, edgecolor="white", linewidth=1.2)
    for i, v in enumerate(zero):
        ax.text(i, v + 0.018, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("零样本 3D Dice", fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.set_title("② 不做任何目标域微调\n（同一源模型直接推理）", fontsize=11.5, pad=10)

    # --- 面板 3：标注总增益 ---
    ax = axes[2]
    ax.bar(x, gain, color=colors, edgecolor="white", linewidth=1.2)
    for i, (v, n) in enumerate(zip(gain, full_n)):
        ax.text(i, v + 0.018, f"+{v:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.text(i, 0.02, f"补满 {n} 例", ha="center", fontsize=8.5, color="white")
    ax.set_ylabel("补满目标域标注后的 Δ Dice", fontsize=11)
    ax.set_ylim(0, 0.70)
    ax.set_title("③ 标注预算的投资回报\n（零样本 → 池满标注）", fontsize=11.5, pad=10)

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([LABEL[t] for t in order], fontsize=10.5)
        ax.grid(axis="y", alpha=0.25, ls="--")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("域移越严重（MMD 越大）→ 零样本越差 → 标注越值钱", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = config.WORK / "outputs" / "MMD三联图.png"
    fig.savefig(out, dpi=200)
    print(f"written {out}")
    for t in order:
        print(f"  {t:4s} MMD²={mmd[t]['mmd']:.4f}  零样本={curves[t][0][1]:.4f}  "
              f"增益={curves[t][-1][1] - curves[t][0][1]:+.4f}")


if __name__ == "__main__":
    main()
