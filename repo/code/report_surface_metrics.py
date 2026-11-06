"""Summarize saved per-case HD95 / ASSD across seeds with patient bootstrap CIs."""

import csv
import json

import numpy as np

import config
from config import RESULTS, RUNS, WORK
from protocol import patient_id_from_key


BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261003
METRICS = ("hd95_mm", "assd_mm")


def main():
    if not RESULTS.exists():
        raise FileNotFoundError(RESULTS)
    with RESULTS.open(newline="", encoding="utf-8") as f:
        result_rows = list(csv.DictReader(f))

    groups = {}
    missing = []
    for row in result_rows:
        run_id = row.get("run_id", "")
        run_dir = RUNS / run_id
        metadata_path = run_dir / "run_metadata.json"
        metrics_path = run_dir / "per_case_metrics.json"
        if not metrics_path.exists():
            missing.append(run_id)
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        eval_key = metadata["eval_key"]
        n_label = int(row["n_label"])
        key = (row["target"], eval_key, n_label)
        scores = json.loads(metrics_path.read_text(encoding="utf-8"))
        groups.setdefault(key, {})[int(row["seed"])] = scores

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = []
    for (target, eval_key, n_label), seed_scores in sorted(groups.items()):
        seeds = sorted(seed_scores)
        for metric in METRICS:
            by_patient_seed = {}
            for seed in seeds:
                patient_values = {}
                for case, value in seed_scores[seed].items():
                    score = value.get(metric)
                    if score is not None:
                        patient_values.setdefault(patient_id_from_key(case), []).append(float(score))
                by_patient_seed[seed] = patient_values
            patients = sorted(set.intersection(*[
                set(by_patient_seed[seed]) for seed in seeds
            ]))
            matrix = np.asarray([
                [np.mean(by_patient_seed[seed][patient]) for seed in seeds]
                for patient in patients
            ], dtype=float)
            if matrix.size == 0:
                continue
            seed_means = matrix.mean(axis=0)
            patient_means = matrix.mean(axis=1)
            sampled = rng.integers(0, len(patient_means), size=(BOOTSTRAP_REPS, len(patient_means)))
            boot = patient_means[sampled].mean(axis=1)
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            output.append({
                "target": target,
                "eval_key": eval_key,
                "n_label": n_label,
                "metric": metric,
                "valid_patients": len(patients),
                "n_seeds": len(seeds),
                "seed_mean": float(seed_means.mean()),
                "seed_sd": float(seed_means.std(ddof=0)),
                "patient_bootstrap_ci_low": float(ci_low),
                "patient_bootstrap_ci_high": float(ci_high),
            })

    out_dir = config._work_path_env("EXP_OUTPUT_DIR", "outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "surface_metrics_summary.csv"
    fields = list(output[0]) if output else [
        "target", "eval_key", "n_label", "metric", "valid_patients", "n_seeds",
        "seed_mean", "seed_sd", "patient_bootstrap_ci_low", "patient_bootstrap_ci_high",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    lines = [
        "# HD95 / ASSD summary",
        "",
        f"Patient bootstrap replicates: {BOOTSTRAP_REPS:,}; bootstrap seed: {BOOTSTRAP_SEED}.",
        "The bootstrap resamples cases and keeps the paired seed scores for each case together.",
        "",
        "| Target | Evaluation split | Labeled cases | Metric | Valid patients | Seeds | Mean ± SD | Patient bootstrap 95% CI |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in output:
        lines.append(
            f"| {row['target']} | {row['eval_key']} | {row['n_label']} | {row['metric']} | "
            f"{row['valid_patients']} | {row['n_seeds']} | "
            f"{row['seed_mean']:.3f} ± {row['seed_sd']:.3f} | "
            f"[{row['patient_bootstrap_ci_low']:.3f}, {row['patient_bootstrap_ci_high']:.3f}] |"
        )
    if missing:
        lines += ["", f"Runs without `per_case_metrics.json` (not summarized): {len(missing)}."]
        lines += [f"- `{run_id}`" for run_id in missing]
    md_path = out_dir / "surface_metrics_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"写入: {csv_path}")
    print(f"写入: {md_path}")
    print(f"汇总 {len(output)} 个组；缺少 surface metrics 的 run {len(missing)} 个")


if __name__ == "__main__":
    main()
