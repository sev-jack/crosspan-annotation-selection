"""Secondary same-patient OOP analysis using the primary saved checkpoints."""

import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import train_eval as TE
from protocol import patient_id_from_key


OUT = config.WORK / "outputs" / "radiology_execution" / "oop_same_patient"
COUNTS = (0, 1, 5, 10, 19, 95)
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261004


def primary_run_id(n_label, seed):
    pct = {0: 0, 1: 1, 5: 5, 10: 10, 19: 20, 95: 100}[n_label]
    return f"T1W2OOP_{pct:03d}pct_s{seed}_pd"


def paired_patient_summary(per_seed, patient_ids):
    seeds = sorted(per_seed)
    patients = sorted(patient_ids)
    matrix = np.asarray([
        [np.mean([per_seed[s][case] for case in per_seed[s]
                  if patient_id_from_key(case) == patient]) for s in seeds]
        for patient in patients
    ], dtype=float)
    seed_means = matrix.mean(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(patients), size=(BOOTSTRAP_REPS, len(patients)))
    draws = matrix.mean(axis=1)[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(seed_means.mean()), float(seed_means.std(ddof=0)), float(lo), float(hi)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
    all_eval = splits["targets"]["OOP"]["eval"]
    pd_eval = splits["targets"]["OOP"]["eval_patient_disjoint"]
    source_patients = {patient_id_from_key(k) for k in splits["source"]["train"] + splits["source"]["val"]}
    pd_patients = {patient_id_from_key(k) for k in pd_eval}
    same_patients = {patient_id_from_key(k) for k in all_eval}
    overlap = sorted(same_patients & source_patients)
    strict_cases = {k for k in all_eval if patient_id_from_key(k) not in source_patients}
    if set(pd_eval) != strict_cases:
        raise RuntimeError("The configured patient-disjoint OOP set does not equal the strict subset of the 41-case evaluation")

    per_budget = defaultdict(dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for n_label in COUNTS:
        for seed in (0, 1, 2):
            rid = primary_run_id(n_label, seed)
            ckpt = config.RUNS / rid / "model.pt"
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            model = TE.build_model()
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            model.to(device).eval()
            dice, per_case, surface = TE.eval_volumes(
                model, all_eval, device, include_surface_metrics=True
            )
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            per_budget[n_label][seed] = per_case
            payload = {"run_id": rid, "checkpoint_reused": str(ckpt.relative_to(config.WORK)),
                       "eval_key": "eval", "seed": seed, "n_label": n_label,
                       "per_case": per_case, "per_case_metrics": surface}
            (OUT / f"n{n_label:03d}_s{seed}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"{rid}: same-patient eval Dice={dice:.4f}", flush=True)

    summaries = []
    for n_label, seed_runs in per_budget.items():
        for eval_name, case_set in (("all_41", set(all_eval)), ("patient_disjoint_26", strict_cases)):
            scores = {s: {k: v for k, v in case_vals.items() if k in case_set}
                      for s, case_vals in seed_runs.items()}
            patients = {patient_id_from_key(k) for k in case_set}
            mean, sd, lo, hi = paired_patient_summary(scores, patients)
            summaries.append({"eval_group": eval_name, "n_label": n_label, "n_eval_cases": len(case_set),
                              "n_patients": len(patients), "seeds": "0,1,2", "dice_mean": mean,
                              "dice_seed_sd": sd, "patient_bootstrap_ci_low": lo,
                              "patient_bootstrap_ci_high": hi})

    fields = list(summaries[0])
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)
    lines = [
        "# OOP same-patient secondary analysis",
        "",
        "The same source/target-adapted checkpoints used for the primary analysis were evaluated on all 41 OOP cases. The patient-disjoint 26-case subset is recomputed from the same predictions as a consistency check. No model selection or hyperparameter tuning used these secondary results.",
        "",
        f"The 41-case evaluation contains {len(overlap)} patients also present in source train/validation; these cases are not independent external testing. The strict 26-case subset excludes them.",
        "",
        "| Evaluation subset | Labeled cases | Cases | Patients | Dice mean ± seed SD | Patient bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(f"| {r['eval_group']} | {r['n_label']} | {r['n_eval_cases']} | {r['n_patients']} | "
                     f"{r['dice_mean']:.4f} ± {r['dice_seed_sd']:.4f} | "
                     f"[{r['patient_bootstrap_ci_low']:.4f}, {r['patient_bootstrap_ci_high']:.4f}] |")
    lines += [
        "",
        "This comparison helps separate same-patient sequence transfer from patient-disjoint generalization. The same-patient result is secondary and must not be described as independent external validation.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for group, marker in (("all_41", "o"), ("patient_disjoint_26", "s")):
        vals = sorted((r for r in summaries if r["eval_group"] == group), key=lambda r: r["n_label"])
        x = np.asarray([r["n_label"] for r in vals])
        y = np.asarray([r["dice_mean"] for r in vals])
        low = np.asarray([r["patient_bootstrap_ci_low"] for r in vals])
        high = np.asarray([r["patient_bootstrap_ci_high"] for r in vals])
        ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), marker=marker, capsize=3,
                    label=group.replace("_", " "))
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(list(COUNTS), [str(v) for v in COUNTS])
    ax.set_xlabel("Labeled target OOP cases")
    ax.set_ylabel("Mean 3D Dice")
    ax.set_title("Primary patient-disjoint and secondary same-patient OOP evaluation")
    ax.grid(alpha=0.23)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "same_patient_vs_disjoint.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Source-overlapping OOP patients: {len(overlap)}; strict subset: {len(pd_patients)}.")


if __name__ == "__main__":
    main()
