"""Build a traceable registry and reviewer-facing audit from completed runs."""

import csv
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import config
from protocol import patient_id_from_key


OUT = config.WORK / "outputs" / "radiology_execution"


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_registry():
    rows = []
    run_tables = [
        ("seeded_v2_absolute_budget", config.WORK / "results_seeded_v2.csv"),
        ("AMOS22_external_budget", config.WORK / "results_amos_external.csv"),
    ]
    for experiment_line, result_path in run_tables:
        for result in read_csv(result_path):
            run_id = result.get("run_id", "")
            run_dir = config.RUNS / run_id
            meta_path = run_dir / "run_metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            case_path = run_dir / "per_case.json"
            surface_path = run_dir / "per_case_metrics.json"
            checkpoint = run_dir / "model.pt"
            required = [meta_path, case_path, checkpoint]
            if meta.get("surface_metrics_saved"):
                required.append(surface_path)
            rows.append({
                "run_id": run_id,
                "experiment_line": experiment_line,
                "target": result.get("target", ""),
                "eval_key": meta.get("eval_key", ""),
                "n_label": result.get("n_label", ""),
                "pool_size": result.get("pool_size", ""),
                "budget_fraction": result.get("budget", ""),
                "seed": result.get("seed", ""),
                "dice": result.get("dice", ""),
                "n_eval": result.get("n_eval", ""),
                "seconds": result.get("seconds", ""),
                "timestamp": result.get("timestamp", ""),
                "seed_protocol": meta.get("seed_protocol", ""),
                "checkpoint": str(checkpoint.relative_to(config.WORK)) if checkpoint.exists() else "",
                "metadata": str(meta_path.relative_to(config.WORK)) if meta_path.exists() else "",
                "per_case": str(case_path.relative_to(config.WORK)) if case_path.exists() else "",
                "surface_metrics": str(surface_path.relative_to(config.WORK)) if surface_path.exists() else "",
                "artifact_complete": all(p.is_file() for p in required),
                "status": "complete" if all(p.is_file() for p in required) else "missing_artifact",
            })

    selection_path = OUT.parent / "seeded_v2" / "selection_1pct.json"
    if selection_path.exists():
        for r in json.loads(selection_path.read_text(encoding="utf-8")):
            run_id = f"select_{r['target']}_n{int(r['n_label']):03d}_{r['strategy']}_s{r['seed']}"
            rows.append({
                "run_id": run_id,
                "experiment_line": "selection_ablation_1pct",
                "target": r["target"],
                "eval_key": r["eval_key"],
                "n_label": r["n_label"],
                "pool_size": r["pool_size"],
                "budget_fraction": r["budget"],
                "seed": r["seed"],
                "dice": r["dice"],
                "n_eval": len(r.get("per_case", {})),
                "seconds": "",
                "timestamp": "",
                "seed_protocol": r.get("seed_protocol", ""),
                "checkpoint": "",
                "metadata": str(selection_path.relative_to(config.WORK)),
                "per_case": str(selection_path.relative_to(config.WORK)),
                "surface_metrics": "",
                "artifact_complete": bool(r.get("selected_keys") and r.get("per_case")),
                "status": "complete" if r.get("selected_keys") and r.get("per_case") else "missing_artifact",
            })

    fields = [
        "run_id", "experiment_line", "target", "eval_key", "n_label", "pool_size",
        "budget_fraction", "seed", "dice", "n_eval", "seconds", "timestamp",
        "seed_protocol", "checkpoint", "metadata", "per_case", "surface_metrics",
        "artifact_complete", "status",
    ]
    write_csv(OUT / "experiment_registry.csv", rows, fields)
    return rows


def audit_splits():
    split_path = config.WORK / "splits_amos.json"
    splits = json.loads((split_path if split_path.exists() else config.SPLITS).read_text(encoding="utf-8"))
    source = splits["source"]
    source_train = {patient_id_from_key(k) for k in source["train"]}
    source_val = {patient_id_from_key(k) for k in source["val"]}
    report = {
        "source_train_cases": len(source["train"]),
        "source_val_cases": len(source["val"]),
        "source_train_patients": len(source_train),
        "source_val_patients": len(source_val),
        "source_train_val_patient_overlap": sorted(source_train & source_val),
        "targets": {},
    }
    for target, d in splits["targets"].items():
        pool_patients = {patient_id_from_key(k) for k in d["pool"]}
        entry = {
            "pool_cases": len(d["pool"]),
            "pool_patients": len(pool_patients),
            "evaluations": {},
        }
        for eval_key in ("eval", "eval_patient_disjoint"):
            if eval_key not in d:
                continue
            eval_patients = {patient_id_from_key(k) for k in d[eval_key]}
            entry["evaluations"][eval_key] = {
                "cases": len(d[eval_key]),
                "patients": len(eval_patients),
                "pool_patient_overlap": sorted(pool_patients & eval_patients),
                "source_train_overlap": sorted(source_train & eval_patients),
                "source_val_overlap": sorted(source_val & eval_patients),
            }
        report["targets"][target] = entry

    report["source_train_val_patient_disjoint"] = not report["source_train_val_patient_overlap"]
    report["target_pool_eval_patient_disjoint"] = all(
        not e["pool_patient_overlap"]
        for d in report["targets"].values()
        for e in d["evaluations"].values()
    )
    oop_pd = report["targets"].get("OOP", {}).get("evaluations", {}).get("eval_patient_disjoint", {})
    report["oop_patient_disjoint_from_source"] = not (
        oop_pd.get("source_train_overlap") or oop_pd.get("source_val_overlap")
    )
    (OUT / "split_seed_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    protocol = {
        "seed_protocol": "seeded_v2",
        "python_random_numpy_torch_cpu_cuda_seeded": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "dataloader_workers": config.NUM_WORKERS,
        "torch_deterministic_algorithms": bool(__import__("torch").are_deterministic_algorithms_enabled()),
        "interpretation": "Controlled random seeds and deterministic cuDNN settings are recorded. PyTorch deterministic-algorithm enforcement is not enabled; bitwise reproducibility is therefore not claimed.",
    }
    report["reproducibility"] = protocol
    report["software"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    try:
        import torch
        import monai
        report["software"]["torch"] = torch.__version__
        report["software"]["cuda_available"] = torch.cuda.is_available()
        report["software"]["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        report["software"]["monai"] = monai.__version__
    except Exception as e:
        report["software_error"] = f"{type(e).__name__}: {e}"

    lines = [
        "# Split and reproducibility audit",
        "",
        f"Source train/validation are patient-disjoint: **{report['source_train_val_patient_disjoint']}**.",
        f"All target label pools and evaluation sets are patient-disjoint: **{report['target_pool_eval_patient_disjoint']}**.",
        f"Primary patient-disjoint OOP evaluation is disjoint from source train/validation: **{report['oop_patient_disjoint_from_source']}**.",
        "",
        "## Patient overlap counts",
        "",
        "| Target evaluation | Patients | Source train overlap | Source validation overlap | Pool overlap |",
        "|---|---:|---:|---:|---:|",
    ]
    for target, d in report["targets"].items():
        for eval_key, e in d["evaluations"].items():
            lines.append(
                f"| {target} / {eval_key} | {e['patients']} | {len(e['source_train_overlap'])} | "
                f"{len(e['source_val_overlap'])} | {len(e['pool_patient_overlap'])} |"
            )
    lines += [
        "",
        "## Seed and software controls",
        "",
        f"- Seed protocol: `{protocol['seed_protocol']}`; Python `random`, NumPy, PyTorch CPU/CUDA are seeded.",
        f"- cuDNN deterministic: `{protocol['cudnn_deterministic']}`; benchmark: `{protocol['cudnn_benchmark']}`.",
        f"- DataLoader worker count: `{protocol['dataloader_workers']}`; training augmentation uses seeded PyTorch RNG on GPU.",
        f"- `torch.use_deterministic_algorithms`: `{protocol['torch_deterministic_algorithms']}`. We do not claim bitwise reproducibility.",
        f"- Runtime versions: Python {report['software'].get('python')}, NumPy {report['software'].get('numpy')}, "
        f"PyTorch {report['software'].get('torch')}, MONAI {report['software'].get('monai')}, "
        f"GPU {report['software'].get('gpu')}.",
        "",
        "OOP's original 41-case same-patient sequence evaluation is retained only as secondary/mechanistic analysis; it overlaps source patients and must not be described as patient-independent external testing.",
    ]
    (OUT / "split_seed_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return report


def report_thresholds():
    summary = read_csv(OUT.parent / "seeded_v2" / "summary.csv")
    summary += read_csv(OUT.parent / "amos_external" / "summary.csv")
    split_path = config.WORK / "splits_amos.json"
    splits = json.loads((split_path if split_path.exists() else config.SPLITS).read_text(encoding="utf-8"))
    rows = []
    for r in summary:
        target = r["target"]
        pool_n = len(splits["targets"][target]["pool"])
        r["n_label"] = int(r["n_label"])
        r["dice_mean"] = float(r["dice_mean"])
        r["pool_size"] = pool_n
        rows.append(r)
    thresholds = (0.60, 0.65, 0.70)
    table = []
    for target in sorted({r["target"] for r in rows}):
        group = sorted((r for r in rows if r["target"] == target), key=lambda x: x["n_label"])
        for thr in thresholds:
            hit = next((r for r in group if r["dice_mean"] >= thr), None)
            table.append({
                "target": target,
                "dice_threshold": thr,
                "first_observed_n_label": hit["n_label"] if hit else "not reached",
                "budget_fraction": hit["n_label"] / hit["pool_size"] if hit else "",
                "observed_dice_mean": hit["dice_mean"] if hit else "",
                "interpretation": "exploratory descriptive milestone; not a clinically validated acceptability threshold",
            })

    curve_rows = []
    for target in sorted({r["target"] for r in rows}):
        group = sorted((r for r in rows if r["target"] == target), key=lambda x: x["n_label"])
        for prev, cur in zip(group, group[1:]):
            added = cur["n_label"] - prev["n_label"]
            curve_rows.append({
                "target": target,
                "from_n_label": prev["n_label"],
                "to_n_label": cur["n_label"],
                "additional_cases": added,
                "dice_gain": cur["dice_mean"] - prev["dice_mean"],
                "dice_gain_per_additional_case": (cur["dice_mean"] - prev["dice_mean"]) / added if added > 0 else "",
                "from_seed_sd": prev["dice_seed_sd"],
                "to_seed_sd": cur["dice_seed_sd"],
            })

    threshold_fields = ["target", "dice_threshold", "first_observed_n_label", "budget_fraction", "observed_dice_mean", "interpretation"]
    write_csv(OUT / "exploratory_thresholds.csv", table, threshold_fields)
    curve_fields = ["target", "from_n_label", "to_n_label", "additional_cases", "dice_gain", "dice_gain_per_additional_case", "from_seed_sd", "to_seed_sd"]
    write_csv(OUT / "marginal_dice_by_observed_interval.csv", curve_rows, curve_fields)

    lines = [
        "# Exploratory label-count milestones",
        "",
        "The Dice cutoffs below are descriptive milestones from the analysis plan. No radiologist or clinical deployment team prespecified them, so they are not clinical acceptability thresholds. The first observed budget is reported without interpolation.",
        "",
        "| Target | Dice milestone | First observed labeled cases | Pool fraction | Mean Dice at that point |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in table:
        n = r["first_observed_n_label"]
        frac = f"{float(r['budget_fraction']):.1%}" if r["budget_fraction"] != "" else "—"
        dice = f"{float(r['observed_dice_mean']):.3f}" if r["observed_dice_mean"] != "" else "not reached"
        lines.append(f"| {r['target']} | {r['dice_threshold']:.2f} | {n} | {frac} | {dice} |")
    lines += [
        "",
        "## Marginal Dice per additional labeled case",
        "",
        "| Target | Budget interval | Added cases | ΔDice | ΔDice per added case |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in curve_rows:
        lines.append(
            f"| {r['target']} | {r['from_n_label']}→{r['to_n_label']} | {r['additional_cases']} | "
            f"{r['dice_gain']:+.4f} | {r['dice_gain_per_additional_case']:+.5f} |"
        )
    lines += [
        "",
        "Intervals are descriptive; the target cases are nested, selected by a deterministic random ordering, and evaluated with only three training seeds. Values must not be interpreted as a universal annotation law or actual annotation-time ROI.",
    ]
    (OUT / "exploratory_thresholds.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    registry = build_registry()
    audit = audit_splits()
    report_thresholds()
    counts = Counter((r["experiment_line"], r["status"]) for r in registry)
    lines = [
        "# Radiology plan execution artifacts",
        "",
        f"Current seeded-v2 core runs registered: {sum(1 for r in registry if r['experiment_line'] == 'seeded_v2_absolute_budget')}; "
        f"selection-ablation runs registered: {sum(1 for r in registry if r['experiment_line'] == 'selection_ablation_1pct') }.",
        f"Registered rows with complete artifacts: {sum(bool(r['artifact_complete']) for r in registry)}/{len(registry)}.",
        "",
        "The older mixed-protocol `results.csv` and legacy ablations are intentionally not merged into the seeded-v2 registry or its inferential summaries.",
        "",
        "| Experiment line | Status | Rows |",
        "|---|---|---:|",
    ]
    for (line, status), n in sorted(counts.items()):
        lines.append(f"| {line} | {status} | {n} |")
    lines += [
        "",
        f"Patient split checks: source train/validation disjoint = {audit['source_train_val_patient_disjoint']}; "
        f"target pool/evaluation disjoint = {audit['target_pool_eval_patient_disjoint']}; "
        f"patient-disjoint OOP/source disjoint = {audit['oop_patient_disjoint_from_source']}.",
        "",
        "See `experiment_registry.csv`, `split_seed_audit.md`, `exploratory_thresholds.md`, and `marginal_dice_by_observed_interval.csv`.",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Wrote {len(registry)} registry rows; {sum(bool(r['artifact_complete']) for r in registry)} complete artifacts.")
    print(f"Split checks: source={audit['source_train_val_patient_disjoint']}, pools={audit['target_pool_eval_patient_disjoint']}, OOP primary={audit['oop_patient_disjoint_from_source']}.")


if __name__ == "__main__":
    main()
