"""选样集合刻画：difficulty 选出来的病例和 diversity 选出来的病例到底有什么不同。

回应评审提出的"为什么 diversity 能赢而 difficulty 不能"。
只用已有的无标签分数与逐例真值，不需要训练。

    python code/report_selection_characterization.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import config

OUT = config.WORK / "outputs" / "upstream"
FEATS = ["s_entropy", "s_energy", "s_ens", "s_knn5", "s_mmd_sim"]


def load():
    with (OUT / "severity_scores.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FEATS + ["dice_mean", "fg_fraction", "n_slices"]:
            r[k] = float(r[k])
    for dom in {r["domain"] for r in rows}:
        sub = [r for r in rows if r["domain"] == dom]
        for f in FEATS:
            vals = [r[f] for r in sub if not np.isnan(r[f])]
            fill = max(vals) if vals else 0.0
            for r in sub:
                if np.isnan(r[f]):
                    r[f] = fill
    index = {(r["domain"], r["case_key"]): r for r in rows}
    with (OUT / "selection_upstream.json").open(encoding="utf-8") as f:
        sel = json.load(f)
    return rows, index, sel


def main():
    rows, index, sel = load()
    pool_stats = {}
    for dom in ["OOP", "CT", "NYU"]:
        sub = [r for r in rows if r["domain"] == dom and r["split"] == "pool"]
        pool_stats[dom] = {f: (float(np.mean([r[f] for r in sub])),
                               float(np.std([r[f] for r in sub]))) for f in FEATS}
        pool_stats[dom]["dice_mean"] = (float(np.mean([r["dice_mean"] for r in sub])),
                                        float(np.std([r["dice_mean"] for r in sub])))

    print("=" * 96)
    print("选样集合刻画：被选中的病例在无标签分数空间里长什么样")
    print("=" * 96)
    print("列含义：mean = 该策略选中病例的分数均值；disp = 选中集合的标准差 / 整个池的标准差")
    print("disp < 1 表示选出来的病例彼此更相似（集中），disp > 1 表示更分散（覆盖广）\n")

    out = []
    for dom in ["OOP", "CT", "NYU"]:
        for k in (5, 10):
            base = [s for s in sel if s["target"] == dom and s["n_label"] == k]
            if not base:
                continue
            print(f"--- {dom}  k={k}  (池 n={len([r for r in rows if r['domain']==dom and r['split']=='pool'])}) ---")
            print(f"{'策略':10}{'熵 mean':>10}{'熵 disp':>10}{'分歧 mean':>11}{'分歧 disp':>11}"
                  f"{'kNN mean':>10}{'kNN disp':>10}{'脏值 Dice mean':>15}")
            for strat in ["random", "entropy", "diversity", "contrast", "hard"]:
                subs = [s for s in base if s["strategy"] == strat]
                if not subs:
                    continue
                means, disps = {}, {}
                for f in FEATS:
                    vals, dd = [], []
                    for s in subs:
                        pts = [index[(dom, c)][f] for c in s["selected_keys"] if (dom, c) in index]
                        if pts:
                            vals.append(np.mean(pts))
                            dd.append(np.std(pts) / (pool_stats[dom][f][1] + 1e-9))
                    means[f], disps[f] = float(np.mean(vals)), float(np.mean(dd))
                dice = []
                for s in subs:
                    d = [index[(dom, c)]["dice_mean"] for c in s["selected_keys"] if (dom, c) in index]
                    if d:
                        dice.append(np.mean(d))
                print(f"{strat:10}{means['s_entropy']:>10.3f}{disps['s_entropy']:>10.2f}"
                      f"{means['s_ens']:>11.3f}{disps['s_ens']:>11.2f}"
                      f"{means['s_knn5']:>10.2f}{disps['s_knn5']:>10.2f}{np.mean(dice):>15.3f}")
                out.append(dict(domain=dom, k=k, strategy=strat,
                                mean_entropy=round(means["s_entropy"], 4),
                                disp_entropy=round(disps["s_entropy"], 4),
                                mean_ens=round(means["s_ens"], 4),
                                disp_ens=round(disps["s_ens"], 4),
                                mean_knn5=round(means["s_knn5"], 4),
                                disp_knn5=round(disps["s_knn5"], 4),
                                mean_selected_dice=round(float(np.mean(dice)), 4)))
            print()

    with (OUT / "selection_characterization.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"写入 {OUT / 'selection_characterization.csv'}")
    print("\n读法：若 difficulty 类策略的 disp 明显小于 1（集中），而 diversity 的 disp 接近或大于 1，")
    print("      就说明 hard/entropy 把预算压在了一小撮相似的极端病例上，")
    print("      而 diversity 覆盖了更广的输入分布——这正是后者在严重域移下更有效的可能原因。")


if __name__ == "__main__":
    main()
