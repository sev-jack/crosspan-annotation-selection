"""把 results.csv 变成那张表。同时写一份 UTF-8-BOM 的 Markdown，方便用记事本打开。

用法:
    python report.py
"""

import csv
from collections import defaultdict

import numpy as np

from config import RESULTS, WORK


def main():
    if not RESULTS.exists():
        raise SystemExit(f"还没有结果: {RESULTS}\n先跑 run_grid.py")

    with open(RESULTS, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("results.csv 是空的")

    budgets = sorted({round(float(r["budget"]), 4) for r in rows})
    cell = defaultdict(list)
    src_dice = {}
    for r in rows:
        cell[(r["source"], r["target"], round(float(r["budget"]), 4))].append(float(r["dice"]))
        if r.get("source_val_dice"):
            src_dice[(r["source"], r["seed"])] = float(r["source_val_dice"])

    pairs = sorted({(r["source"], r["target"]) for r in rows})
    L = []

    def add(s=""):
        L.append(s)

    add("# 跨序列胰腺分割 · 标注预算表")
    add()
    add("## 主表 · 目标域 3D Dice（mean ± std，跨种子）")
    add()
    add("| 源域→目标域 |" + "".join(f" {b:.0%} |" for b in budgets))
    add("|---|" + "---|" * len(budgets))
    for s, t in pairs:
        line = f"| {s}→{t} |"
        for b in budgets:
            v = cell.get((s, t, b))
            if not v:
                line += " -- |"
            elif len(v) == 1:
                line += f" {v[0]:.4f} |"
            else:
                line += f" {np.mean(v):.4f} ± {np.std(v):.3f} |"
        add(line)

    add()
    add("## 源域 val Dice（诊断用；接近 0 说明源模型废了）")
    add()
    add("| 源域 | seed | val Dice |")
    add("|---|---|---|")
    for (s, sd), d in sorted(src_dice.items()):
        add(f"| {s} | {sd} | {d:.4f} |")

    add()
    add("## 相对零样本的增益（每个域各自的 0% 作基线）")
    add()
    for s, t in pairs:
        base = cell.get((s, t, 0.0))
        if not base:
            add(f"- {s}→{t}: 没有 0% 这一格，无法算增益")
            continue
        b0 = float(np.mean(base))
        parts = []
        for b in budgets:
            v = cell.get((s, t, b))
            if v:
                parts.append(f"{b:.0%}: {np.mean(v) - b0:+.4f}")
        add(f"- {s}→{t}（零样本 {b0:.4f}）：" + "，".join(parts))

    n_seed = len({r["seed"] for r in rows})
    add()
    add(f"格数 {len(rows)}，种子数 {n_seed}")
    if n_seed < 2:
        add()
        add("> 注意：目前只有 1 个种子，表里的结论还不能写进论文。")

    text = "\n".join(L)
    print(text)

    out = WORK / "outputs" / "结果表.md"
    out.write_text(text, encoding="utf-8-sig")   # 带 BOM，记事本/Excel 不乱码
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
