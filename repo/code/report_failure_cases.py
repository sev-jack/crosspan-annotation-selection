"""Summarize per-patient persistent failures and boundary-metric availability."""

import csv
import json
from collections import defaultdict

import numpy as np

import config


OUT = config.WORK / "outputs" / "radiology_execution"


def run_id(target, n_label, seed):
    if target == "OOP":
        pct = {0: 0, 10: 10, 19: 20, 95: 100}[n_label]
        return f"T1W2OOP_{pct:03d}pct_s{seed}_pd"
    return f"T1W2{target}_n{n_label:03d}case_s{seed}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets = ["OOP", "CT", "NYU", "AMOS"]
    rows = []
    report = []
    for target in targets:
        candidate_counts = {
            "OOP": [0, 10, 95],
            "CT": [0, 10, 220, 100],
            "NYU": [0, 10, 17],
            "AMOS": [0, 10, 20],
        }[target]
        available = [n for n in candidate_counts if all(
            (config.RUNS / run_id(target, n, s) / "per_case.json").is_file() for s in (0, 1, 2)
        )]
        if 0 not in available or 10 not in available:
            continue
        max_n = max(available)
        counts = [0, 10, max_n] if max_n != 10 else [0, 10]
        metrics_by_count = {}
        for n in counts:
            case_dice = defaultdict(list)
            case_surface = defaultdict(lambda: defaultdict(list))
            for seed in (0, 1, 2):
                run_dir = config.RUNS / run_id(target, n, seed)
                dice_path = run_dir / "per_case.json"
                if not dice_path.exists():
                    continue
                for case, score in json.loads(dice_path.read_text(encoding="utf-8")).items():
                    case_dice[case].append(float(score))
                metric_path = run_dir / "per_case_metrics.json"
                if metric_path.exists():
                    surface = json.loads(metric_path.read_text(encoding="utf-8"))
                    for case, vals in surface.items():
                        for metric in ("hd95_mm", "assd_mm"):
                            if vals.get(metric) is not None:
                                case_surface[case][metric].append(float(vals[metric]))
            metrics_by_count[n] = {
                case: float(np.mean(scores)) for case, scores in case_dice.items()
            }
            if n == max_n:
                for case, score in metrics_by_count[n].items():
                    row = {"target": target, "case_key": case,
                           "dice_zero_seed_mean": metrics_by_count.get(0, {}).get(case, ""),
                           "dice_10case_seed_mean": metrics_by_count.get(10, {}).get(case, ""),
                           "dice_highest_budget_seed_mean": score,
                           "highest_budget_n_label": max_n,
                           "hd95_mm_highest_budget_valid_seeds": len(case_surface[case]["hd95_mm"]),
                           "hd95_mm_highest_budget_mean": float(np.mean(case_surface[case]["hd95_mm"])) if case_surface[case]["hd95_mm"] else "",
                           "assd_mm_highest_budget_valid_seeds": len(case_surface[case]["assd_mm"]),
                           "assd_mm_highest_budget_mean": float(np.mean(case_surface[case]["assd_mm"])) if case_surface[case]["assd_mm"] else ""}
                    rows.append(row)
        # 这里是 dict 列表，不能直接 sorted()；下一行已经按 dice 排序，所以只做过滤。
        failures = [rows_for_target for rows_for_target in rows if rows_for_target["target"] == target]
        failures.sort(key=lambda r: r["dice_highest_budget_seed_mean"])
        persistent = [r for r in failures if r["dice_highest_budget_seed_mean"] < 0.5]
        report.append((target, max_n, len(failures), len(persistent), failures[:5]))

    fields = ["target", "case_key", "dice_zero_seed_mean", "dice_10case_seed_mean",
              "dice_highest_budget_seed_mean", "highest_budget_n_label",
              "hd95_mm_highest_budget_valid_seeds", "hd95_mm_highest_budget_mean",
              "assd_mm_highest_budget_valid_seeds", "assd_mm_highest_budget_mean"]
    with (OUT / "failure_case_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# Persistent failure-case summary",
        "",
        "Case scores are averaged across the available three seeds. The highest-budget checkpoint is used only to identify persistent failures, not to tune the model. Dice <0.50 is an exploratory descriptive flag, not a clinical threshold.",
        "",
        "| Target | Highest budget (cases) | Evaluation cases | Cases with mean Dice <0.50 | Five lowest-Dice cases (mean Dice) |",
        "|---|---:|---:|---:|---|",
    ]
    for target, n, total, n_bad, worst in report:
        brief = ", ".join(f"{r['case_key']} ({r['dice_highest_budget_seed_mean']:.3f})" for r in worst)
        lines.append(f"| {target} | {n} | {total} | {n_bad} | {brief} |")
    lines += [
        "",
        "HD95/ASSD availability is reported per case. Empty one-sided masks make surface distances undefined; those cases remain in the Dice summaries and must be reported with the valid denominator for surface metrics.",
        "",
        "This is a quantitative screen for selecting qualitative examples. A radiologist should review representative successes and failures before making anatomical or clinical claims.",
    ]
    (OUT / "failure_case_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Wrote {len(rows)} case-level failure records across {len(report)} targets.")


if __name__ == "__main__":
    main()
