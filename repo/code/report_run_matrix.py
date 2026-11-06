"""Summarize seeded_v2 run matrices with patient-paired bootstrap intervals.

Reads the results table and each run's per_case.json / run_metadata.json. Writes
machine-readable tables, a Markdown report, and a label-count response plot.
Patient bootstrap resamples evaluation patients and keeps all seeds paired.
"""

import csv
import json
from collections import defaultdict

import numpy as np

import config
from config import RESULTS, RUNS, WORK
from protocol import patient_id_from_key


BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261003


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    boot_means = values[draws].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def load_runs():
    if not RESULTS.exists():
        raise FileNotFoundError(RESULTS)
    with RESULTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(dict)
    problems = []
    for row in rows:
        run_id = row.get("run_id", "")
        run_dir = RUNS / run_id
        metadata_path = run_dir / "run_metadata.json"
        cases_path = run_dir / "per_case.json"
        if not metadata_path.exists() or not cases_path.exists():
            problems.append(f"{run_id}: missing metadata or per_case.json")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("seed_protocol") != "seeded_v2":
            continue
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
        group_key = (row["target"], metadata["eval_key"], int(row["n_label"]))
        seed = int(row["seed"])
        groups[group_key][seed] = {
            "run_id": run_id,
            "cases": {str(k): float(v) for k, v in cases.items() if v is not None},
            "expected_cases": list(metadata.get("eval_keys", [])),
            "row": row,
        }
    return groups, problems


def common_case_matrix(seed_runs):
    seeds = sorted(seed_runs)
    common = set.intersection(*[
        set(seed_runs[seed]["cases"]) for seed in seeds
    ]) if seeds else set()
    cases = sorted(common)
    matrix = np.asarray([
        [seed_runs[seed]["cases"][case] for seed in seeds]
        for case in cases
    ], dtype=float)
    return seeds, cases, matrix


def summarize(groups):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries = []
    by_target_cases = defaultdict(dict)
    for (target, eval_key, n_label), seed_runs in sorted(groups.items()):
        seeds, cases, matrix = common_case_matrix(seed_runs)
        if matrix.size == 0:
            continue
        patient_rows = defaultdict(list)
        for i, case in enumerate(cases):
            patient_rows[patient_id_from_key(case)].append(matrix[i, :])
        patients = sorted(patient_rows)
        patient_seed_matrix = np.asarray([
            np.mean(patient_rows[patient], axis=0) for patient in patients
        ], dtype=float)
        seed_means = patient_seed_matrix.mean(axis=0)
        pooled_patient_means = patient_seed_matrix.mean(axis=1)
        lo, hi = bootstrap_ci(pooled_patient_means, rng)
        expected = max(len(seed_runs[s]["expected_cases"]) for s in seeds)
        summaries.append({
            "target": target,
            "eval_key": eval_key,
            "n_label": n_label,
            "n_seeds": len(seeds),
            "seeds": ",".join(map(str, seeds)),
            "n_eval_common": len(cases),
            "n_eval_expected": expected,
            "n_patients_common": len(patients),
            "dice_mean": float(seed_means.mean()),
            "dice_seed_sd": float(seed_means.std(ddof=0)),
            "patient_bootstrap_ci_low": lo,
            "patient_bootstrap_ci_high": hi,
            "complete_three_seeds": len(seeds) == 3 and seeds == [0, 1, 2],
            "complete_eval_cases": len(cases) == expected,
        })
        by_target_cases[(target, eval_key)][n_label] = (seeds, cases, patient_seed_matrix)

    gains = []
    marginal_gains = []
    for (target, eval_key), count_groups in sorted(by_target_cases.items()):
        if 0 not in count_groups:
            continue
        base_seeds, base_cases, base_matrix = count_groups[0]
        base_lookup = {case: i for i, case in enumerate(base_cases)}
        for n_label, (seeds, cases, matrix) in sorted(count_groups.items()):
            if n_label == 0:
                continue
            common_seeds = sorted(set(seeds) & set(base_seeds))
            current_seed_idx = [seeds.index(s) for s in common_seeds]
            base_seed_idx = [base_seeds.index(s) for s in common_seeds]
            current_lookup = {case: i for i, case in enumerate(cases)}
            paired_cases = sorted(set(cases) & set(base_cases))
            deltas_by_patient = defaultdict(list)
            for case in paired_cases:
                ci = current_lookup[case]
                bi = base_lookup[case]
                # Keep per-case differences paired by seed before averaging.
                deltas_by_patient[patient_id_from_key(case)].append(
                    float(np.mean(matrix[ci, current_seed_idx] - base_matrix[bi, base_seed_idx]))
                )
            patient_deltas = [float(np.mean(v)) for _, v in sorted(deltas_by_patient.items())]
            lo, hi = bootstrap_ci(patient_deltas, rng)
            mean_gain = float(np.mean(patient_deltas)) if patient_deltas else None
            gains.append({
                "target": target,
                "eval_key": eval_key,
                "n_label": n_label,
                "baseline_n_label": 0,
                "n_paired_seeds": len(common_seeds),
                "n_paired_cases": len(paired_cases),
                "n_paired_patients": len(patient_deltas),
                "dice_gain_vs_zero": mean_gain,
                "dice_gain_per_labeled_case": (mean_gain / n_label) if mean_gain is not None else None,
                "patient_bootstrap_ci_low": lo,
                "patient_bootstrap_ci_high": hi,
            })
        ordered_counts = sorted(count_groups)
        for previous_count, n_label in zip(ordered_counts, ordered_counts[1:]):
            prev_seeds, prev_cases, prev_matrix = count_groups[previous_count]
            seeds, cases, matrix = count_groups[n_label]
            common_seeds = sorted(set(seeds) & set(prev_seeds))
            current_seed_idx = [seeds.index(s) for s in common_seeds]
            prev_seed_idx = [prev_seeds.index(s) for s in common_seeds]
            current_lookup = {case: i for i, case in enumerate(cases)}
            prev_lookup = {case: i for i, case in enumerate(prev_cases)}
            paired_cases = sorted(set(cases) & set(prev_cases))
            deltas_by_patient = defaultdict(list)
            for case in paired_cases:
                ci = current_lookup[case]
                pi = prev_lookup[case]
                deltas_by_patient[patient_id_from_key(case)].append(
                    float(np.mean(matrix[ci, current_seed_idx] - prev_matrix[pi, prev_seed_idx]))
                )
            patient_deltas = [float(np.mean(v)) for _, v in sorted(deltas_by_patient.items())]
            lo, hi = bootstrap_ci(patient_deltas, rng)
            mean_gain = float(np.mean(patient_deltas)) if patient_deltas else None
            added = n_label - previous_count
            marginal_gains.append({
                "target": target,
                "eval_key": eval_key,
                "from_n_label": previous_count,
                "to_n_label": n_label,
                "additional_labeled_cases": added,
                "n_paired_seeds": len(common_seeds),
                "n_paired_cases": len(paired_cases),
                "n_paired_patients": len(patient_deltas),
                "dice_gain": mean_gain,
                "dice_gain_per_additional_case": (mean_gain / added) if mean_gain is not None else None,
                "patient_bootstrap_ci_low": lo,
                "patient_bootstrap_ci_high": hi,
            })
    return summaries, gains, marginal_gains


def write_outputs(summaries, gains, marginal_gains, problems):
    out_dir = config._work_path_env("EXP_OUTPUT_DIR", "outputs/seeded_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = list(summaries[0]) if summaries else [
        "target", "eval_key", "n_label", "n_seeds", "seeds", "n_eval_common",
        "n_eval_expected", "n_patients_common", "dice_mean", "dice_seed_sd",
        "patient_bootstrap_ci_low", "patient_bootstrap_ci_high",
        "complete_three_seeds", "complete_eval_cases",
    ]
    gain_fields = list(gains[0]) if gains else [
        "target", "eval_key", "n_label", "baseline_n_label", "n_paired_seeds",
        "n_paired_cases", "n_paired_patients", "dice_gain_vs_zero", "dice_gain_per_labeled_case",
        "patient_bootstrap_ci_low", "patient_bootstrap_ci_high",
    ]
    marginal_fields = list(marginal_gains[0]) if marginal_gains else [
        "target", "eval_key", "from_n_label", "to_n_label", "additional_labeled_cases",
        "n_paired_seeds", "n_paired_cases", "n_paired_patients", "dice_gain",
        "dice_gain_per_additional_case", "patient_bootstrap_ci_low", "patient_bootstrap_ci_high",
    ]
    for filename, rows, fields in (
        ("summary.csv", summaries, summary_fields),
        ("paired_gains.csv", gains, gain_fields),
        ("marginal_gains.csv", marginal_gains, marginal_fields),
    ):
        with (out_dir / filename).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Seeded v2 supplementary experiment summary",
        "",
        f"Patient bootstrap: {BOOTSTRAP_REPS:,} resamples; seed {BOOTSTRAP_SEED}.",
        "Dice is averaged over seeds and cases; seed SD is descriptive. Confidence intervals resample patients.",
        "A group is marked complete only when all three seeds and the full expected evaluation set are present.",
        "",
        "## Dice by target-domain label count",
        "",
        "| Target | Evaluation | Labeled cases | Seeds | Eval cases | Patients | Dice mean ± seed SD | Patient bootstrap 95% CI | Complete |",
        "|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in summaries:
        complete = "yes" if r["complete_three_seeds"] and r["complete_eval_cases"] else "in progress / incomplete"
        lines.append(
            f"| {r['target']} | {r['eval_key']} | {r['n_label']} | {r['seeds']} | "
            f"{r['n_eval_common']}/{r['n_eval_expected']} | {r['n_patients_common']} | "
            f"{r['dice_mean']:.4f} ± {r['dice_seed_sd']:.4f} | "
            f"[{r['patient_bootstrap_ci_low']:.4f}, {r['patient_bootstrap_ci_high']:.4f}] | {complete} |"
        )
    lines += [
        "",
        "## Paired gain versus zero target labels",
        "",
        "| Target | Evaluation | Labeled cases | Paired patients | Dice gain | Gain per labeled case | Patient bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in gains:
        lines.append(
            f"| {r['target']} | {r['eval_key']} | {r['n_label']} | {r['n_paired_patients']} | "
            f"{r['dice_gain_vs_zero']:.4f} | {r['dice_gain_per_labeled_case']:.4f} | "
            f"[{r['patient_bootstrap_ci_low']:.4f}, {r['patient_bootstrap_ci_high']:.4f}] |"
        )
    lines += [
        "",
        "## Marginal gain between consecutive annotation budgets",
        "",
        "| Target | Evaluation | From → to labeled cases | Paired patients | Dice gain | Gain per added case | Patient bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in marginal_gains:
        lines.append(
            f"| {r['target']} | {r['eval_key']} | {r['from_n_label']} → {r['to_n_label']} | "
            f"{r['n_paired_patients']} | {r['dice_gain']:.4f} | "
            f"{r['dice_gain_per_additional_case']:.4f} | "
            f"[{r['patient_bootstrap_ci_low']:.4f}, {r['patient_bootstrap_ci_high']:.4f}] |"
        )
    if problems:
        lines += ["", "## Incomplete or unreadable runs", ""]
        lines += [f"- {p}" for p in problems]
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        targets = sorted({r["target"] for r in summaries})
        if targets:
            fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.3), squeeze=False)
            for ax, target in zip(axes[0], targets):
                vals = sorted([r for r in summaries if r["target"] == target], key=lambda r: r["n_label"])
                x = np.asarray([r["n_label"] for r in vals], dtype=float)
                y = np.asarray([r["dice_mean"] for r in vals], dtype=float)
                low = np.asarray([r["patient_bootstrap_ci_low"] for r in vals], dtype=float)
                high = np.asarray([r["patient_bootstrap_ci_high"] for r in vals], dtype=float)
                ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), marker="o", capsize=3)
                ax.set_xscale("symlog", linthresh=1, linscale=1)
                ax.set_xticks(x, [str(int(v)) for v in x])
                ax.set_xlabel("Labeled target-domain cases")
                ax.set_ylabel("Mean Dice")
                ax.set_title(target)
                ax.grid(True, alpha=0.25)
            fig.suptitle("Target annotation response (seeded v2)")
            fig.tight_layout()
            fig.savefig(out_dir / "annotation_response.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        targets = sorted({r["target"] for r in marginal_gains})
        if targets:
            fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.3), squeeze=False)
            for ax, target in zip(axes[0], targets):
                vals = sorted([r for r in marginal_gains if r["target"] == target], key=lambda r: r["to_n_label"])
                x = np.asarray([r["to_n_label"] for r in vals], dtype=float)
                y = np.asarray([r["dice_gain_per_additional_case"] for r in vals], dtype=float)
                low = np.asarray([r["patient_bootstrap_ci_low"] / r["additional_labeled_cases"] for r in vals], dtype=float)
                high = np.asarray([r["patient_bootstrap_ci_high"] / r["additional_labeled_cases"] for r in vals], dtype=float)
                ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), marker="o", capsize=3)
                ax.set_xscale("symlog", linthresh=1, linscale=1)
                ax.set_xticks(x, [str(int(v)) for v in x])
                ax.set_xlabel("Labeled cases at upper budget")
                ax.set_ylabel("ΔDice per additional labeled case")
                ax.set_title(target)
                ax.grid(True, alpha=0.25)
            fig.suptitle("Marginal annotation gain (seeded v2)")
            fig.tight_layout()
            fig.savefig(out_dir / "marginal_gain_per_case.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

        targets = sorted({r["target"] for r in gains})
        if targets:
            fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.3), squeeze=False)
            for ax, target in zip(axes[0], targets):
                vals = sorted([r for r in gains if r["target"] == target], key=lambda r: r["n_label"])
                x = np.asarray([r["n_label"] for r in vals], dtype=float)
                y = np.asarray([r["dice_gain_per_labeled_case"] for r in vals], dtype=float)
                low = np.asarray([r["patient_bootstrap_ci_low"] / r["n_label"] for r in vals], dtype=float)
                high = np.asarray([r["patient_bootstrap_ci_high"] / r["n_label"] for r in vals], dtype=float)
                ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), marker="o", capsize=3)
                ax.set_xscale("symlog", linthresh=1, linscale=1)
                ax.set_xticks(x, [str(int(v)) for v in x])
                ax.set_xlabel("Labeled target-domain cases")
                ax.set_ylabel("(Dice(N) − Dice(0)) / N")
                ax.set_title(target)
                ax.grid(True, alpha=0.25)
            fig.suptitle("Cumulative annotation efficiency (seeded v2)")
            fig.tight_layout()
            fig.savefig(out_dir / "cumulative_efficiency.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        lines.append("\nPlot skipped: matplotlib is unavailable.")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"写入: {out_dir / 'summary.csv'}")
    print(f"写入: {out_dir / 'paired_gains.csv'}")
    print(f"写入: {out_dir / 'marginal_gains.csv'}")
    print(f"写入: {md_path}")
    print(f"汇总组数={len(summaries)}；相对零标注比较={len(gains)}；异常={len(problems)}")


def main():
    groups, problems = load_runs()
    summaries, gains, marginal_gains = summarize(groups)
    write_outputs(summaries, gains, marginal_gains, problems)


if __name__ == "__main__":
    main()
