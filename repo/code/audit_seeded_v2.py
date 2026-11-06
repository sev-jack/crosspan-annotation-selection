"""Audit completeness, split isolation, saved artifacts, and nested labels."""

import csv
import json
from pathlib import Path

from config import RESULTS, RUNS, SPLITS, WORK
from protocol import patient_id_from_key


def expected_runs(splits):
    expected = {}
    for seed in (0, 1, 2):
        for pct in (0, 1, 5, 10, 20, 100):
            run_id = f"T1W2OOP_{pct:03d}pct_s{seed}_pd"
            expected[run_id] = ("OOP", "eval_patient_disjoint", seed)
        for target, counts in (
            ("CT", (0, 1, 5, 10, 20, 100, len(splits["targets"]["CT"]["pool"]))),
            ("NYU", (0, 1, 5, 10, len(splits["targets"]["NYU"]["pool"]))),
        ):
            for n_label in counts:
                run_id = f"T1W2{target}_n{n_label:03d}case_s{seed}"
                expected[run_id] = (target, "eval", seed)
    return expected


def main():
    problems = []
    if not RESULTS.exists():
        raise FileNotFoundError(RESULTS)
    with RESULTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {}
    for row in rows:
        run_id = row.get("run_id", "")
        if run_id in by_id:
            problems.append(f"duplicate result row: {run_id}")
        by_id[run_id] = row

    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    expected = expected_runs(splits)
    missing = sorted(set(expected) - set(by_id))
    unexpected = sorted(set(by_id) - set(expected))
    problems.extend(f"missing result row: {run_id}" for run_id in missing)
    problems.extend(f"unexpected result row: {run_id}" for run_id in unexpected)

    label_sets = {}
    for run_id in sorted(set(expected) & set(by_id)):
        target, eval_key, seed = expected[run_id]
        row = by_id[run_id]
        run_dir = RUNS / run_id
        paths = {
            "metadata": run_dir / "run_metadata.json",
            "per-case Dice": run_dir / "per_case.json",
            "surface metrics": run_dir / "per_case_metrics.json",
            "checkpoint": run_dir / "model.pt",
        }
        absent = [name for name, path in paths.items() if not path.is_file()]
        if absent:
            problems.append(f"{run_id}: missing artifacts: {', '.join(absent)}")
            continue

        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        per_case = json.loads(paths["per-case Dice"].read_text(encoding="utf-8"))
        surface = json.loads(paths["surface metrics"].read_text(encoding="utf-8"))
        target_split = splits["targets"][target]
        eval_keys = set(target_split[eval_key])
        if metadata.get("eval_key") != eval_key:
            problems.append(f"{run_id}: metadata eval_key mismatch")
        if metadata.get("seed_protocol") != "seeded_v2":
            problems.append(f"{run_id}: seed protocol mismatch")
        if set(metadata.get("eval_keys", [])) != eval_keys:
            problems.append(f"{run_id}: metadata evaluation cases mismatch")
        if set(per_case) != eval_keys or set(surface) != eval_keys:
            problems.append(f"{run_id}: saved per-case rows do not match evaluation cases")
        try:
            n_label = int(row["n_label"])
        except (TypeError, ValueError):
            n_label = -1
        selected = list(metadata.get("label_keys", []))
        if len(selected) != n_label:
            problems.append(f"{run_id}: selected {len(selected)} labels but row reports {n_label}")
        label_sets[(target, seed, n_label)] = selected
        if int(row["n_eval"]) != len(eval_keys):
            problems.append(f"{run_id}: result row n_eval mismatch")

        pool_patients = {patient_id_from_key(k) for k in target_split["pool"]}
        eval_patients = {patient_id_from_key(k) for k in eval_keys}
        if pool_patients & eval_patients:
            problems.append(f"{run_id}: pool/evaluation patient overlap")
        if eval_key == "eval_patient_disjoint":
            source_patients = {
                patient_id_from_key(k)
                for k in splits["source"]["train"] + splits["source"]["val"]
            }
            if source_patients & eval_patients:
                problems.append(f"{run_id}: patient-disjoint evaluation overlaps source patients")

    for target, pool, counts in (
        ("OOP", "pool", (0, 1, 5, 10, 19, 95)),
        ("CT", "pool", (0, 1, 5, 10, 20, 100, len(splits["targets"]["CT"]["pool"]))),
        ("NYU", "pool", (0, 1, 5, 10, len(splits["targets"]["NYU"]["pool"]))),
    ):
        for seed in (0, 1, 2):
            previous = None
            for n_label in counts:
                current = label_sets.get((target, seed, n_label))
                if current is None:
                    continue
                if previous is not None and current[:len(previous)] != previous:
                    problems.append(f"{target} seed {seed}: label subsets are not nested at n={n_label}")
                previous = current

    output = {
        "expected_runs": len(expected),
        "result_rows": len(rows),
        "unique_result_rows": len(by_id),
        "missing_run_ids": missing,
        "unexpected_run_ids": unexpected,
        "problems": problems,
        "passed": not problems and len(expected) == len(by_id),
    }
    out_dir = WORK / "outputs" / "seeded_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Seeded v2 run audit",
        "",
        f"Expected runs: {len(expected)}; result rows: {len(rows)}; unique IDs: {len(by_id)}.",
        f"Audit status: {'PASS' if output['passed'] else 'ISSUES FOUND'}.",
        "",
        "Checks include run completeness, patient isolation, evaluation-case alignment, saved checkpoints and metrics, and nested target-label subsets.",
    ]
    if problems:
        lines += ["", "## Problems", ""]
        lines += [f"- {problem}" for problem in problems]
    (out_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Audit {'PASS' if output['passed'] else 'FAILED'}: {len(expected)} expected, {len(rows)} rows, {len(problems)} problems")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
