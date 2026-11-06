"""逐例标注效用 U(x)：对 OOP 标注池的每一例单独微调（k=1），测评估集上的提升。

回应 20 号文档 Q1：评审认为个体效用不可测，实际只需 N 次（而非 N^2 次）微调。
OOP 池 68 例 x 3 种子 = 204 次，单次约 20 至 40 秒。

输出：outputs/upstream/individual_utility.csv（增量写入，可中断续跑）

    python code/report_individual_utility.py
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

import config
import train_eval as TE

WORK = config.WORK
OUT = WORK / "outputs" / "upstream"
FIELDS = ["seed", "case_key", "base_dice", "dice", "utility"]

# 每个域的配置：切分文件、runs 目录、源模型后缀、评估列表键
CONFIG = {
    "OOP": dict(splits="splits_oop_pd.json", runs="runs_oop_pd", tag="_oop_pd",
                eval_key="eval_patient_disjoint", run_prefix="T1W2OOP_n000case_pd"),
    "CT": dict(splits="splits.json", runs="runs_seeded_v2", tag="_seeded_v2",
               eval_key="eval", run_prefix="T1W2CT_n000case"),
    "NYU": dict(splits="splits.json", runs="runs_seeded_v2", tag="_seeded_v2",
                eval_key="eval", run_prefix="T1W2NYU_n000case"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="OOP", choices=list(CONFIG))
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    cfg = CONFIG[args.domain]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    splits = json.loads((WORK / cfg["splits"]).read_text(encoding="utf-8"))
    pool = splits["targets"][args.domain]["pool"]
    ev = splits["targets"][args.domain][cfg["eval_key"]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"individual_utility_{args.domain.lower()}.csv"
    runs = WORK / cfg["runs"]

    done = set()
    rows = []
    if path.exists():
        with path.open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        done = {(int(r["seed"]), r["case_key"]) for r in rows}
        print(f"已完成 {len(done)} 格，续跑")

    total = len(seeds) * len(pool)
    print(f"域 {args.domain}：目标 {total} 格（{len(pool)} 例 x {len(seeds)} 种子），"
          f"评估 {len(ev)} 例，device={dev}", flush=True)

    for seed in seeds:
        ckpt = runs / f"source_T1W_s{seed}{cfg['tag']}.pt"
        base = json.loads((runs / f"{cfg['run_prefix']}_s{seed}" / "per_case.json").read_text(encoding="utf-8"))
        base_mean = float(np.mean(list(base.values())))
        todo = [k for k in pool if (seed, k) not in done]
        if not todo:
            continue
        print(f"\n=== seed {seed}：{len(todo)} 例（零样本基线 {base_mean:.4f}）===", flush=True)
        t0 = time.time()
        for i, key in enumerate(todo, 1):
            tuned = TE.train([key], config.FINETUNE_EPOCHS, config.FINETUNE_LR, dev,
                             init_ckpt=ckpt, tag=f"utility_{args.domain}_{i}", seed=seed)
            dice, _ = TE.eval_volumes(tuned, ev, dev)
            del tuned
            if dev == "cuda":
                torch.cuda.empty_cache()
            rows.append(dict(seed=seed, case_key=key, base_dice=round(base_mean, 6),
                             dice=round(float(dice), 6),
                             utility=round(float(dice) - base_mean, 6)))
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                eta = el / i * (len(todo) - i) / 60
                print(f"  [{i}/{len(todo)}] 已用 {el/60:.1f} 分钟，预计还需 {eta:.1f} 分钟", flush=True)

    print(f"\n全部完成，写入 {path}")

    # 汇总：U(x) 的分布
    u = np.array([float(r["utility"]) for r in rows])
    print(f"\nU(x) 统计（n={len(u)}）：均值 {u.mean():+.4f}，SD {u.std():.4f}，"
          f"范围 [{u.min():+.4f}, {u.max():+.4f}]，正值比例 {(u > 0).mean():.1%}")


if __name__ == "__main__":
    main()
