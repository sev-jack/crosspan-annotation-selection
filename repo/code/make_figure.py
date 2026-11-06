"""生成三域预算曲线与域移回报图（均值 ± 跨种子标准差）。

数据直接读取 runs/<run_id>/per_case.json，并先验证病例数与有限数值。
输出：
    实验DG/outputs/核心图_域移与标注回报.png
    实验DG/outputs/表3_三域预算多种子汇总.md
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config

SEEDS = (0, 1, 2)
BUDGETS = (0, 1, 5, 10, 20, 100)
TARGETS = {
    "OOP": dict(label="OOP 跨序列（池 95）", color="#C00000", marker="o"),
    "CT": dict(label="CT 跨模态（池 220）", color="#1F4E79", marker="s"),
    "NYU": dict(label="NYU 跨中心（池 17）", color="#548235", marker="^"),
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_scores():
    splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
    series = {}
    for target, style in TARGETS.items():
        pool_size = len(splits["targets"][target]["pool"])
        eval_size = len(splits["targets"][target]["eval"])
        per_seed = {budget: {} for budget in BUDGETS}
        labels = {}
        for budget in BUDGETS:
            n_label = int(round((budget / 100) * pool_size))
            if budget > 0:
                n_label = max(1, n_label)
            labels[budget] = min(n_label, pool_size)
            for seed in SEEDS:
                run_id = f"T1W2{target}_{budget:03d}pct_s{seed}"
                path = config.RUNS / run_id / "per_case.json"
                if not path.is_file():
                    raise FileNotFoundError(f"缺少实验结果：{path}")
                cases = json.loads(path.read_text(encoding="utf-8"))
                values = np.asarray(list(cases.values()), dtype=float)
                if len(values) != eval_size:
                    raise ValueError(f"{run_id}: 逐病例数 {len(values)}，预期 {eval_size}")
                if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                    raise ValueError(f"{run_id}: Dice 含非有限值或超出 [0, 1]")
                per_seed[budget][seed] = float(values.mean())

        means = [float(np.mean(list(per_seed[b].values()))) for b in BUDGETS]
        stds = [float(np.std(list(per_seed[b].values()), ddof=0)) for b in BUDGETS]
        series[target] = dict(
            **style,
            pool_size=pool_size,
            eval_size=eval_size,
            labels=labels,
            per_seed=per_seed,
            means=means,
            stds=stds,
        )
    return series


def write_table(series):
    lines = [
        "# 表 3：目标域标注预算曲线（三种子）",
        "",
        "指标为评估病例 3D Dice 的宏平均，随后在 seed 0/1/2 间计算均值 ± 标准差。",
        "标准差使用总体口径（ddof=0）；每个 run 的逐病例结果来自 `runs/<run_id>/per_case.json`。",
        "",
        "| 目标域（评估例数） |" + "".join(f" {b}% |" for b in BUDGETS),
        "|---|" + "---|" * len(BUDGETS),
    ]
    for target, d in series.items():
        short_label = d["label"].split("（")[0]
        row = f"| {short_label}（池 {d['pool_size']} / 评估 {d['eval_size']} 例） |"
        for i, budget in enumerate(BUDGETS):
            row += f" {d['means'][i]:.4f} ± {d['stds'][i]:.3f}（标注 {d['labels'][budget]} 例） |"
        lines.append(row)

    lines += [
        "",
        "## 种子明细",
        "",
        "| 目标域 | 预算 | 标注例数 | seed 0 | seed 1 | seed 2 | mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target, d in series.items():
        for i, budget in enumerate(BUDGETS):
            vals = d["per_seed"][budget]
            lines.append(
                f"| {target} | {budget}% | {d['labels'][budget]} | "
                f"{vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | "
                f"{d['means'][i]:.4f} ± {d['stds'][i]:.3f} |"
            )
    lines += [
        "",
        "## 口径与限制",
        "",
        "- 每个种子同时决定源模型检查点和目标域病例抽样；标准差包含这两部分的种子波动。",
        "- NYU 的 1% 与 5% 均对应 1 个标注病例，两个点的病例数相同，不能解释为不同标注量的剂量效应。",
        "- 每次评估分别对病例 Dice 求平均；表格不是把所有像素合并后计算的 pooled Dice。",
    ]
    out = config.WORK / "outputs" / "表3_三域预算多种子汇总.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return out


def make_figure(series):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    gains_by_target = {}

    for target, d in series.items():
        gains = [d["per_seed"][100][s] - d["per_seed"][0][s] for s in SEEDS]
        gains_by_target[target] = gains
        gain_mean = float(np.mean(gains))
        ax1.errorbar(
            BUDGETS,
            d["means"],
            yerr=d["stds"],
            marker=d["marker"],
            color=d["color"],
            lw=2.2,
            ms=7,
            capsize=4,
            label=f"{d['label']}   总增益 {gain_mean:+.3f}",
        )
        zero_label_offset = {"OOP": (6, -14), "CT": (6, 10), "NYU": (6, -18)}[target]
        ax1.annotate(f"{d['means'][0]:.3f}", (BUDGETS[0], d["means"][0]),
                     textcoords="offset points", xytext=zero_label_offset,
                     fontsize=9, color=d["color"])
        ax1.annotate(f"{d['means'][-1]:.3f}", (BUDGETS[-1], d["means"][-1]),
                     textcoords="offset points", xytext=(-40, 8), fontsize=9, color=d["color"])

    ax1.axvspan(0, 10, color="#FFF2CC", alpha=0.75, zorder=0)
    ax1.text(5, 0.18, "前 10% 的预算\n拿走大部分收益", ha="center", fontsize=9,
             color="#7F6000", zorder=1)
    ax1.set_xlabel("目标域标注预算（占该域可标注池的比例，%）", fontsize=11)
    ax1.set_ylabel("目标域 3D Dice", fontsize=11)
    ax1.set_title("标注预算 → 性能（三域，mean ± SD，3 seeds）", fontsize=12.5, pad=12)
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_xticks(BUDGETS)
    ax1.set_xticklabels([f"{b}%" for b in BUDGETS])
    ax1.set_ylim(-0.03, 0.82)
    ax1.grid(alpha=0.25, ls="--")
    ax1.legend(loc="upper left", fontsize=9.2, framealpha=0.95)

    points = []
    for target, d in series.items():
        zero = [d["per_seed"][0][s] for s in SEEDS]
        gains = gains_by_target[target]
        x, y = float(np.mean(zero)), float(np.mean(gains))
        points.append((x, y, target, d))
        ax2.errorbar(
            x, y,
            xerr=float(np.std(zero, ddof=0)),
            yerr=float(np.std(gains, ddof=0)),
            fmt=d["marker"],
            color=d["color"],
            markersize=10,
            capsize=4,
            elinewidth=1.6,
            zorder=3,
        )
        ax2.annotate(d["label"].split("（")[0], (x, y), textcoords="offset points",
                     xytext=(10, 7), fontsize=10, color=d["color"], fontweight="bold")

    order = sorted(points, key=lambda p: p[0])
    ax2.plot([p[0] for p in order], [p[1] for p in order],
             color="#A6A6A6", ls="--", lw=1.6, zorder=1)
    ax2.set_xlabel("零样本 Dice（越低表示域移越严重）", fontsize=11)
    ax2.set_ylabel("100% 相对 0% 的 Dice 增益", fontsize=11)
    ax2.set_title("域移严重度与标注收益（配对种子）", fontsize=12.5, pad=12)
    ax2.grid(alpha=0.25, ls="--")
    ax2.set_xlim(-0.03, 0.68)
    ax2.set_ylim(-0.03, 0.68)

    fig.suptitle("胰腺分割 · 域移严重度与标注预算回报", fontsize=14.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = config.WORK / "outputs" / "核心图_域移与标注回报.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    data = read_scores()
    table_path = write_table(data)
    figure_path = make_figure(data)
    print(f"written {table_path}")
    print(f"written {figure_path}")
