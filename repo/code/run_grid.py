"""把整张表跑完。

用法:
    python run_grid.py                      # 全部
    python run_grid.py --targets OOP        # 只跑 OOP
    python run_grid.py --budgets 0,0.1,1    # 只跑几个档位
    python run_grid.py --seeds 0            # 先单种子试水

可中断、可续跑：已经写进 results.csv 的格子会自动跳过。
"""

import argparse
import csv
import json
import re
import time

import torch

import config
from config import RESULTS, RUNS, SPLITS, BUDGETS, SEEDS, SOURCES, TARGETS, run_id
from train_eval import run_one

FIELDS = [
    "run_id", "source", "target", "budget", "n_label", "pool_size",
    "seed", "dice", "source_val_dice", "n_eval", "seconds", "timestamp",
]


def load_done():
    if not RESULTS.exists():
        return set()
    done = set()
    with open(RESULTS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_id"):
                done.add(row["run_id"])
    return done


def append_row(row):
    new = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(SOURCES.keys()))
    ap.add_argument("--targets", default="")
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--eval-key", default="eval",
                    help="目标域中的评估列表；例如 eval_patient_disjoint")
    ap.add_argument("--label-counts", default="",
                    help="按绝对病例数跑预算，如 1,5,10；设置后忽略 --budgets")
    ap.add_argument("--surface-metrics", action="store_true",
                    help="同时保存逐病例 HD95/ASSD（mm）；缓存索引须有 spacing_zyx")
    args = ap.parse_args()
    config.start_log(config.LOG)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    spins = json.loads(SPLITS.read_text(encoding="utf-8"))
    # 目标域以 splits.json 为准，这样 prepare_ct.py 后加进来的 CT 会自动进网格
    targets = [s.strip() for s in args.targets.split(",") if s.strip()] \
        or list(spins["targets"].keys())
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    label_counts = [int(s) for s in args.label_counts.split(",") if s.strip()]
    budgets = [] if label_counts else [float(s) for s in args.budgets.split(",") if s.strip()]

    splits = spins
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")

    done = load_done()

    def eval_tag(key):
        if key == "eval":
            return ""
        if key == "eval_patient_disjoint":
            return "_pd"
        return "_" + re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_")

    todo = []
    for s in sources:
        for t in targets:
            if t not in splits["targets"]:
                raise KeyError(f"splits.json 中没有目标域 {t}")
            target_split = splits["targets"][t]
            if args.eval_key not in target_split:
                raise KeyError(f"目标域 {t} 没有评估列表 {args.eval_key!r}")
            if label_counts:
                n_pool = len(target_split["pool"])
                for n in label_counts:
                    if not 0 <= n <= n_pool:
                        raise ValueError(f"{t} 标注病例数 {n} 超出 pool={n_pool}")
                    for sd in seeds:
                        rid = f"{s}2{t}_n{n:03d}case{eval_tag(args.eval_key)}_s{sd}"
                        if rid not in done:
                            todo.append((s, t, n / n_pool, n, sd, rid))
            else:
                for b in budgets:
                    for sd in seeds:
                        base = run_id(s, t, b, sd)
                        rid = base + eval_tag(args.eval_key)
                        if rid not in done:
                            todo.append((s, t, b, None, sd, rid))
    print(f"待跑 {len(todo)} 格（已完成 {len(done)} 格）\n")

    t_start = time.time()
    for i, (s, t, b, n_label, sd, rid) in enumerate(todo, 1):
        print(f"\n########## [{i}/{len(todo)}] {rid} ##########")
        t0 = time.time()
        run_dir = RUNS / rid
        try:
            res = run_one(s, t, b, sd, splits=splits, dev=dev,
                          eval_key=args.eval_key, n_label_override=n_label,
                          checkpoint_path=run_dir / "model.pt",
                          include_surface_metrics=args.surface_metrics)
        except Exception as e:
            print(f"!! {rid} 失败: {type(e).__name__}: {e}")
            continue
        res["run_id"] = rid
        res["seconds"] = round(time.time() - t0, 1)
        res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        (RUNS / rid).mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        per_case_metrics = res.pop("per_case_metrics")
        (run_dir / "per_case.json").write_text(
            json.dumps(res.pop("per_case"), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if per_case_metrics is not None:
            (run_dir / "per_case_metrics.json").write_text(
                json.dumps(per_case_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        metadata = {
            "run_id": rid,
            "eval_key": res.pop("eval_key"),
            "label_keys": res.pop("label_keys"),
            "seed_protocol": res.pop("seed_protocol"),
            "eval_keys": splits["targets"][t][args.eval_key],
            "surface_metrics_saved": per_case_metrics is not None,
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        append_row(res)
        print(f"  -> 写入 results.csv: dice={res['dice']}  ({res['seconds']}s)")

    print(f"\n全部结束，总用时 {(time.time() - t_start) / 60:.1f} 分钟")
    print(f"结果表: {RESULTS}")
    print("下一步: python report.py")


if __name__ == "__main__":
    main()
