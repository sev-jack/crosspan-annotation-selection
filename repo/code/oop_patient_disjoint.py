"""Re-analyze saved OOP per-case Dice on a patient-disjoint test subset.

This does not retrain models: it filters the already saved 41-case predictions
to patients absent from both source train and source validation.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS, RUNS, SPLITS, WORK
from protocol import patient_id_from_key


BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 20261003


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_runs():
    with RESULTS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["target"] == "OOP"]
    lookup = {}
    for row in rows:
        lookup[(round(float(row["budget"]), 4), int(row["seed"]))] = row
    return lookup


def _bootstrap_mean_ci(matrix, rng):
    """Patient bootstrap CI for the mean after averaging model seeds per case."""
    # matrix is patients x seeds. Resample patients, preserving each patient's
    # paired seed results, then average seeds within each sampled patient.
    per_patient = np.mean(matrix, axis=1)
    n = len(per_patient)
    sampled = rng.integers(0, n, size=(BOOTSTRAP_REPS, n))
    means = per_patient[sampled].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]).tolist())


def main():
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    source = splits["source"]
    target = splits["targets"]["OOP"]
    original_eval = target["eval"]
    source_patients = {
        patient_id_from_key(k) for k in source["train"] + source["val"]
    }
    train_patients = {patient_id_from_key(k) for k in source["train"]}
    val_patients = {patient_id_from_key(k) for k in source["val"]}
    pool_patients = {patient_id_from_key(k) for k in target["pool"]}
    if train_patients & val_patients:
        raise ValueError("source train / val 存在患者重叠")
    if pool_patients & {patient_id_from_key(k) for k in original_eval}:
        raise ValueError("OOP pool / eval 存在患者重叠")
    if len({patient_id_from_key(k) for k in original_eval}) != len(original_eval):
        raise ValueError("OOP 原始 eval 中同一患者有多条记录，请先明确患者聚合规则")
    overlap_train = {patient_id_from_key(k) for k in original_eval} & train_patients
    overlap_val = {patient_id_from_key(k) for k in original_eval} & val_patients
    disjoint_eval = [k for k in original_eval
                     if patient_id_from_key(k) not in source_patients]

    # Keep the audit split in sync with the code-generated split. This makes
    # the evaluation list visible to later runs without changing legacy eval.
    target["eval_patient_disjoint"] = disjoint_eval
    SPLITS.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")

    run_rows = _read_runs()
    seeds = sorted({seed for _, seed in run_rows})
    budgets = sorted({budget for budget, _ in run_rows})
    if len(seeds) != 3 or len(budgets) != 6:
        raise ValueError(f"预期 OOP 六档预算×三种子，实际 seeds={seeds}, budgets={budgets}")

    all_values = {b: {} for b in budgets}
    for budget in budgets:
        for seed in seeds:
            row = run_rows.get((budget, seed))
            if row is None:
                raise ValueError(f"缺少 OOP budget={budget} seed={seed} 汇总行")
            run_id = row["run_id"]
            path = RUNS / run_id / "per_case.json"
            if not path.exists():
                raise FileNotFoundError(path)
            scores = json.loads(path.read_text(encoding="utf-8"))
            if set(scores) != set(original_eval):
                raise ValueError(f"{run_id}: per_case 与原始 OOP eval 不一致")
            all_values[budget][seed] = scores

    # Add the field to splits only after confirming that the saved cases and
    # the current original evaluation list agree.
    out_dir = WORK / "outputs" / "OOP患者独立分析"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    cohort_keys = {
        "患者独立主分析": disjoint_eval,
        "原始41例配对分析": original_eval,
    }
    rows = []
    tidy = []
    matrices = {}
    for cohort, keys in cohort_keys.items():
        keys = sorted(keys)
        for budget in budgets:
            matrix = np.array([
                [all_values[budget][seed][key] for seed in seeds]
                for key in keys
            ], dtype=float)
            matrices[(cohort, budget)] = matrix
            seed_means = matrix.mean(axis=0)
            ci_low, ci_high = _bootstrap_mean_ci(matrix, rng)
            n_label = int(run_rows[(budget, seeds[0])]["n_label"])
            rows.append({
                "cohort": cohort,
                "budget_pct": round(budget * 100, 4),
                "n_label": n_label,
                "n_eval_patients": len(keys),
                "seed_mean_dice": float(seed_means.mean()),
                "seed_sd_dice": float(seed_means.std(ddof=0)),
                "patient_bootstrap_ci_low": ci_low,
                "patient_bootstrap_ci_high": ci_high,
            })
            for i, key in enumerate(keys):
                for j, seed in enumerate(seeds):
                    tidy.append({
                        "cohort": cohort,
                        "patient_id": patient_id_from_key(key),
                        "case_key": key,
                        "budget_pct": round(budget * 100, 4),
                        "n_label": n_label,
                        "seed": seed,
                        "dice": float(matrix[i, j]),
                    })

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "per_case.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(tidy[0]))
        writer.writeheader()
        writer.writerows(tidy)

    primary = "患者独立主分析"
    report = [
        "# OOP 患者独立重分析",
        "",
        "本报告复用现有 18 组模型的逐病例 Dice，不重新训练。主分析从原始 41 例中排除与源训练或源验证患者重叠者；原始 41 例保留作同患者跨序列的次要分析。",
        "",
        f"- 原始 OOP eval：{len(original_eval)} 例。",
        f"- 与 source train 重叠：{len(overlap_train)} 位患者。",
        f"- 与 source val 重叠：{len(overlap_val)} 位患者。",
        f"- patient-disjoint 主分析：{len(disjoint_eval)} 例。",
        f"- bootstrap：按患者成组重采样 {BOOTSTRAP_REPS:,} 次；对每位患者先平均 3 个 seed。区间反映患者抽样不确定性，不包含 seed 不确定性。",
        "",
        "## 患者独立主分析",
        "",
        "| 标注预算 | 标注病例数 | 评估患者数 | Dice（3 seed 均值 ± SD） | 患者 bootstrap 95% CI |",
        "|---:|---:|---:|---:|---:|",
    ]
    primary_rows = [r for r in rows if r["cohort"] == primary]
    for r in primary_rows:
        report.append(
            f"| {r['budget_pct']:g}% | {r['n_label']} | {r['n_eval_patients']} | "
            f"{r['seed_mean_dice']:.4f} ± {r['seed_sd_dice']:.4f} | "
            f"[{r['patient_bootstrap_ci_low']:.4f}, {r['patient_bootstrap_ci_high']:.4f}] |"
        )

    base = matrices[(primary, budgets[0])]
    base_n = int(run_rows[(budgets[0], seeds[0])]["n_label"])
    report += [
        "",
        "## 相对零标注的配对收益",
        "",
        "| 标注预算 | 新增标注数 | ΔDice（均值） | 患者 bootstrap 95% CI | ΔDice / 新增病例 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget in budgets[1:]:
        current = matrices[(primary, budget)]
        delta_by_patient = (current - base).mean(axis=1)
        n = int(run_rows[(budget, seeds[0])]["n_label"])
        boot_idx = rng.integers(0, len(delta_by_patient), size=(BOOTSTRAP_REPS, len(delta_by_patient)))
        ci_low, ci_high = np.percentile(delta_by_patient[boot_idx].mean(axis=1), [2.5, 97.5])
        gain = float(delta_by_patient.mean())
        report.append(
            f"| {budget:.0%} | {n - base_n} | {gain:+.4f} | "
            f"[{ci_low:+.4f}, {ci_high:+.4f}] | {gain / max(n - base_n, 1):+.4f} |"
        )

    report += [
        "",
        "## 阈值分析（探索性）",
        "",
        "阈值仅用于帮助解释曲线；尚无临床预先规定的可接受 Dice 阈值。",
        "",
        "| Dice 阈值 | 首个达到阈值的预算 |",
        "|---:|---:|",
    ]
    for threshold in (0.60, 0.70):
        reached = next((r for r in primary_rows if r["seed_mean_dice"] >= threshold), None)
        value = "未达到" if reached is None else f"{reached['n_label']} 例（{reached['budget_pct']:g}%）"
        report.append(f"| {threshold:.2f} | {value} |")

    report += [
        "",
        "## 解释边界",
        "",
        "- 横轴仍是数据集内病例数/比例，不能称为真实标注工时或经济 ROI；项目没有逐例标注耗时记录。",
        "- 主分析排除了源 train 和 val 患者，但 26 例来自同一 OOP 来源，不能单独据此声称外部多中心验证。",
        "- patient-disjoint 子集按患者 ID 规则筛出，与分数无关；但其旧结果已被查看。需审计此前是否曾据 OOP eval 调整模型/超参；若有，这只能算回顾性敏感性分析，不是未触碰的最终测试集。",
        "- 旧结果仅保存 Dice，没有 HD95/ASSD，也没有微调后权重；这些指标需按新版评估流程重新推理。",
        "- 训练种子与病例重采样不等同于临床标注人群抽样；区间需按上述 bootstrap 口径解读。",
        "",
        "## 次要分析：原始 41 例同患者跨序列",
        "",
        "详细数值见 `summary.csv`；该分析用于控制患者差异下观察序列迁移，不作为独立患者泛化结果。",
        "",
        "文件：`summary.csv`、`per_case.csv`、`OOP患者独立曲线.png`。",
    ]
    (out_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    for cohort, color, marker, style in [
        (primary, "#5b3c88", "o", "-"),
        ("原始41例配对分析", "#858b95", "s", "--"),
    ]:
        series = [r for r in rows if r["cohort"] == cohort]
        xs = [r["n_label"] for r in series]
        ys = [r["seed_mean_dice"] for r in series]
        if cohort == primary:
            lo = np.array([r["patient_bootstrap_ci_low"] for r in series])
            hi = np.array([r["patient_bootstrap_ci_high"] for r in series])
            yerr = np.vstack([np.array(ys) - lo, hi - np.array(ys)])
            ax.errorbar(xs, ys, yerr=yerr, color=color, marker=marker, linestyle=style,
                        capsize=3, linewidth=2,
                        label="Patient-disjoint primary (patient bootstrap 95% CI)")
        else:
            sd = [r["seed_sd_dice"] for r in series]
            ax.errorbar(xs, ys, yerr=sd, color=color, marker=marker, linestyle=style,
                        capsize=3, linewidth=1.5,
                        label="Original 41-case same-patient analysis (seed SD)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Target-domain labeled cases (symlog scale)")
    ax.set_ylabel("Mean patient-level 3D Dice")
    ax.set_title("OOP annotation response: patient-disjoint primary analysis")
    ax.set_ylim(0, 1)
    ax.set_xticks([r["n_label"] for r in primary_rows],
                  [str(r["n_label"]) for r in primary_rows])
    ax.tick_params(axis="x", labelrotation=18)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(out_dir / "OOP患者独立曲线.png", dpi=200)
    plt.close(fig)

    input_files = [
        SPLITS, RESULTS,
        WORK / "code" / "config.py",
        WORK / "code" / "prepare.py",
        WORK / "code" / "train_eval.py",
        WORK / "code" / "run_grid.py",
        WORK / "code" / "protocol.py",
        WORK / "code" / "oop_patient_disjoint.py",
    ]
    input_files.extend(
        RUNS / run_rows[(budget, seed)]["run_id"] / "per_case.json"
        for budget in budgets for seed in seeds
    )
    manifest = {
        "analysis": "OOP patient-disjoint reanalysis of saved per-case Dice",
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "source_train_overlap_patients": sorted(overlap_train),
        "source_val_overlap_patients": sorted(overlap_val),
        "original_eval_n": len(original_eval),
        "patient_disjoint_eval_n": len(disjoint_eval),
        "files_sha256": {str(p.relative_to(WORK)): _sha256(p) for p in input_files},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"source train overlap={len(overlap_train)}, source val overlap={len(overlap_val)}")
    print(f"OOP patient-disjoint eval={len(disjoint_eval)} / {len(original_eval)}")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
