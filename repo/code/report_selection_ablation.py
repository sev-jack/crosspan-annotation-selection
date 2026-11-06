"""Summarize the 1% selection ablation using paired patient-level bootstrap CIs."""

import csv
import json
from collections import defaultdict

import numpy as np

from config import WORK
from protocol import patient_id_from_key


BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261003


def ci(values, rng):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None, None
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def main():
    input_path = WORK / "outputs" / "seeded_v2" / "selection_1pct.json"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["target"], row["eval_key"], row["strategy"])][int(row["seed"])] = {
            case: float(value) for case, value in row.get("per_case", {}).items()
            if value is not None
        }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summaries = []
    patient_seed_scores = {}
    for (target, eval_key, strategy), seeds_to_cases in sorted(grouped.items()):
        seeds = sorted(seeds_to_cases)
        common_cases = sorted(set.intersection(*[
            set(seeds_to_cases[seed]) for seed in seeds
        ])) if seeds else []
        by_patient = defaultdict(list)
        for case in common_cases:
            by_patient[patient_id_from_key(case)].append([
                seeds_to_cases[seed][case] for seed in seeds
            ])
        patients = sorted(by_patient)
        patient_matrix = np.asarray([
            np.mean(by_patient[patient], axis=0) for patient in patients
        ], dtype=float)
        if patient_matrix.size == 0:
            continue
        seed_means = patient_matrix.mean(axis=0)
        patient_means = patient_matrix.mean(axis=1)
        lo, hi = ci(patient_means, rng)
        summaries.append({
            "target": target,
            "eval_key": eval_key,
            "strategy": strategy,
            "n_seeds": len(seeds),
            "seeds": ",".join(map(str, seeds)),
            "n_eval_cases": len(common_cases),
            "n_patients": len(patients),
            "dice_mean": float(seed_means.mean()),
            "dice_seed_sd": float(seed_means.std(ddof=0)),
            "patient_bootstrap_ci_low": lo,
            "patient_bootstrap_ci_high": hi,
        })
        patient_seed_scores[(target, eval_key, strategy)] = (seeds, common_cases, seeds_to_cases)

    contrasts = []
    target_splits = sorted({(target, eval_key) for target, eval_key, _ in patient_seed_scores})
    for target, eval_key in target_splits:
        if (target, eval_key, "random") not in patient_seed_scores:
            continue
        random_seeds, random_cases, random_data = patient_seed_scores[(target, eval_key, "random")]
        strategies = sorted(s for t, e, s in patient_seed_scores if t == target and e == eval_key and s != "random")
        for strategy in strategies:
            seeds, cases, strategy_data = patient_seed_scores[(target, eval_key, strategy)]
            common_seeds = sorted(set(seeds) & set(random_seeds))
            common_cases = sorted(set(cases) & set(random_cases))
            if not common_seeds or not common_cases:
                continue
            by_patient = defaultdict(list)
            for case in common_cases:
                delta = np.mean([
                    strategy_data[seed][case] - random_data[seed][case]
                    for seed in common_seeds
                ])
                by_patient[patient_id_from_key(case)].append(float(delta))
            patient_deltas = [float(np.mean(v)) for _, v in sorted(by_patient.items())]
            lo, hi = ci(patient_deltas, rng)
            contrasts.append({
                "target": target,
                "eval_key": eval_key,
                "strategy": strategy,
                "reference": "random",
                "n_paired_seeds": len(common_seeds),
                "n_paired_cases": len(common_cases),
                "n_paired_patients": len(patient_deltas),
                "dice_gain_vs_random": float(np.mean(patient_deltas)) if patient_deltas else None,
                "patient_bootstrap_ci_low": lo,
                "patient_bootstrap_ci_high": hi,
            })

    out_dir = WORK / "outputs" / "seeded_v2"
    summary_fields = list(summaries[0]) if summaries else [
        "target", "eval_key", "strategy", "n_seeds", "seeds", "n_eval_cases", "n_patients",
        "dice_mean", "dice_seed_sd", "patient_bootstrap_ci_low", "patient_bootstrap_ci_high",
    ]
    contrast_fields = list(contrasts[0]) if contrasts else [
        "target", "eval_key", "strategy", "reference", "n_paired_seeds", "n_paired_cases",
        "n_paired_patients", "dice_gain_vs_random", "patient_bootstrap_ci_low", "patient_bootstrap_ci_high",
    ]
    for name, data, fields in (
        ("selection_summary.csv", summaries, summary_fields),
        ("selection_vs_random.csv", contrasts, contrast_fields),
    ):
        with (out_dir / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)

    lines = [
        "# 1% sampling-strategy ablation",
        "",
        f"Patient bootstrap: {BOOTSTRAP_REPS:,} resamples; seed {BOOTSTRAP_SEED}.",
        "Hard/oracle selection uses target labels and is an upper-bound analysis only.",
        "",
        "| Target | Eval split | Strategy | Seeds | Cases | Patients | Dice mean ± seed SD | Patient bootstrap 95% CI |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['target']} | {row['eval_key']} | {row['strategy']} | {row['seeds']} | "
            f"{row['n_eval_cases']} | {row['n_patients']} | {row['dice_mean']:.4f} ± {row['dice_seed_sd']:.4f} | "
            f"[{row['patient_bootstrap_ci_low']:.4f}, {row['patient_bootstrap_ci_high']:.4f}] |"
        )
    lines += [
        "",
        "## Paired strategy contrasts versus random",
        "",
        "| Target | Eval split | Strategy | Paired patients | ΔDice vs random | Patient bootstrap 95% CI |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['target']} | {row['eval_key']} | {row['strategy']} | {row['n_paired_patients']} | "
            f"{row['dice_gain_vs_random']:.4f} | "
            f"[{row['patient_bootstrap_ci_low']:.4f}, {row['patient_bootstrap_ci_high']:.4f}] |"
        )
    (out_dir / "selection_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        targets = sorted({row["target"] for row in summaries})
        if targets:
            fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4.2), squeeze=False)
            for ax, target in zip(axes[0], targets):
                vals = [row for row in summaries if row["target"] == target]
                x = np.arange(len(vals))
                y = np.asarray([row["dice_mean"] for row in vals])
                low = np.asarray([row["patient_bootstrap_ci_low"] for row in vals])
                high = np.asarray([row["patient_bootstrap_ci_high"] for row in vals])
                ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), fmt="o", capsize=3)
                ax.set_xticks(x, [row["strategy"] for row in vals], rotation=35, ha="right")
                ax.set_ylabel("Mean Dice")
                ax.set_title(target)
                ax.grid(True, axis="y", alpha=0.25)
            fig.suptitle("1% target-case selection strategies")
            fig.tight_layout()
            fig.savefig(out_dir / "selection_ablation.png", dpi=180, bbox_inches="tight")
            plt.close(fig)
    except ImportError:
        pass

    print(f"Selection rows={len(summaries)}; paired contrasts={len(contrasts)}")


if __name__ == "__main__":
    main()
