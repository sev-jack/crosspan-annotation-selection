"""Summarize raw acquisition geometry and per-scan intensity distributions."""

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

import config


OUT = config.WORK / "outputs" / "radiology_execution"


def raw_image_path(key):
    parts = key.split("_", 2)
    seq, split, stem = parts
    if seq == "AMOS":
        root = Path(r"<DATA_ROOT>\AMOS22\amos22")
        directory = "imagesTr" if split.lower() == "tr" else "imagesVa"
        candidates = [root / directory / f"{stem}.nii.gz"]
    elif seq == "CT":
        root = config.CT_ROOT
        # MSD 原始格式是 pancreas_012.nii.gz（无 _0000 通道后缀），转成 nnU-Net 格式后才带。
        # 两种命名都要试，否则 CT 全部找不到原始影像。
        candidates = [
            root / "imagesTr" / f"{stem}_0000.nii.gz",
            root / "imagesTr" / f"{stem}.nii.gz",
            root / "imagesTs" / f"{stem}_0000.nii.gz",
            root / "imagesTs" / f"{stem}.nii.gz",
        ]
    else:
        root = config.ROOT
        candidates = [root / seq / split / "images" / f"{stem}_0000.nii.gz"]
    return next((p for p in candidates if p.is_file()), None)


def case_metrics(key):
    path = raw_image_path(key)
    if path is None:
        raise FileNotFoundError(f"Raw image not found for {key}")
    image = nib.as_closest_canonical(nib.load(str(path)))
    values = np.asarray(image.dataobj, dtype=np.float32)
    if values.ndim == 4:
        values = values[..., 0]
    values = values[np.isfinite(values)]
    seq = key.split("_", 1)[0]
    if seq in {"CT", "AMOS"}:
        body = values[values > -500.0]
        mask_name = "HU > -500"
    else:
        body = values[values != 0]
        mask_name = "nonzero voxels"
    if body.size == 0:
        body = values
        mask_name = "all finite voxels (fallback)"
    # Keep quantile calculation bounded while preserving deterministic coverage.
    if body.size > 500_000:
        body = body[:: int(np.ceil(body.size / 500_000))]
    q05, q50, q95 = np.percentile(body, [5, 50, 95])
    zooms = image.header.get_zooms()[:3]
    return {
        "case_key": key,
        "sequence": seq,
        "raw_shape_x": image.shape[0],
        "raw_shape_y": image.shape[1],
        "raw_shape_z": image.shape[2],
        "spacing_x_mm": zooms[0],
        "spacing_y_mm": zooms[1],
        "spacing_z_mm": zooms[2],
        "voxel_count": int(np.prod(image.shape[:3])),
        "intensity_mask": mask_name,
        "intensity_mean": float(np.mean(body)),
        "intensity_sd": float(np.std(body)),
        "intensity_p05": float(q05),
        "intensity_p50": float(q50),
        "intensity_p95": float(q95),
    }


def summarize(rows, groups):
    output = []
    metrics = [
        "raw_shape_x", "raw_shape_y", "raw_shape_z", "spacing_x_mm", "spacing_y_mm",
        "spacing_z_mm", "voxel_count", "intensity_mean", "intensity_sd",
        "intensity_p05", "intensity_p50", "intensity_p95",
    ]
    for group_name, keys in groups.items():
        selected = [r for r in rows if r["case_key"] in set(keys)]
        if not selected:
            continue
        line = {"domain": group_name, "n_cases": len(selected), "intensity_mask": selected[0]["intensity_mask"]}
        for m in metrics:
            vals = np.asarray([r[m] for r in selected], dtype=float)
            line[f"{m}_median"] = float(np.median(vals))
            line[f"{m}_q25"] = float(np.percentile(vals, 25))
            line[f"{m}_q75"] = float(np.percentile(vals, 75))
        output.append(line)
    return output


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
    groups = {
        "Source T1W reference": splits["source"]["train"] + splits["source"]["val"],
        "OOP patient-disjoint evaluation": splits["targets"]["OOP"]["eval_patient_disjoint"],
        "CT evaluation": splits["targets"]["CT"]["eval"],
        "NYU evaluation": splits["targets"]["NYU"]["eval"],
    }
    if "AMOS" in splits["targets"]:
        groups["AMOS22 external CT evaluation"] = splits["targets"]["AMOS"]["eval"]
    all_keys = list(dict.fromkeys(k for keys in groups.values() for k in keys))
    case_rows = [case_metrics(k) for k in all_keys]
    case_fields = list(case_rows[0])
    with (OUT / "domain_case_characteristics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=case_fields)
        w.writeheader()
        w.writerows(case_rows)
    summary = summarize(case_rows, groups)
    summary_fields = list(summary[0])
    with (OUT / "domain_characteristics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["domain"] for r in summary]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(summary))
    width = 0.22
    for i, axis in enumerate(("x", "y", "z")):
        med = [r[f"spacing_{axis}_mm_median"] for r in summary]
        lo = [r[f"spacing_{axis}_mm_q25"] for r in summary]
        hi = [r[f"spacing_{axis}_mm_q75"] for r in summary]
        axes[0].errorbar(x + (i - 1) * width, med,
                         yerr=[np.asarray(med) - lo, np.asarray(hi) - med],
                         fmt="o", capsize=3, label=f"{axis.upper()} axis")
    axes[0].set_xticks(x, names, rotation=20, ha="right")
    axes[0].set_ylabel("Voxel spacing (mm)")
    axes[0].set_title("Acquisition geometry")
    axes[0].legend(frameon=False)
    for i, r in enumerate(summary):
        axes[1].plot([r["intensity_p05_median"], r["intensity_p95_median"]], [i, i], lw=3)
        axes[1].plot(r["intensity_p50_median"], i, "o", color="black", ms=5)
    axes[1].set_yticks(x, names)
    axes[1].set_xlabel("Per-case raw intensity quantiles (median across scans)")
    axes[1].set_title("Intensity range within each domain")
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "domain_characterization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Raw domain characterization",
        "",
        "Geometry comes from canonical raw NIfTI headers. Per-case intensity quantiles use nonzero voxels for MRI and voxels above −500 HU for CT; each scan is capped at a deterministic 500,000-voxel stride sample. Values are summarized as the median across scans.",
        "",
        "| Domain | Cases | Median volume shape (X×Y×Z) | Median spacing X/Y/Z (mm) | Median within-mask p05 / p50 / p95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        shape = "×".join(str(round(r[f"raw_shape_{a}_median"])) for a in "xyz")
        spacing = "/".join(f"{r[f'spacing_{a}_mm_median']:.2f}" for a in "xyz")
        intensity = "/".join(f"{r[f'intensity_p{p}_median']:.2f}" for p in ("05", "50", "95"))
        lines.append(f"| {r['domain']} | {r['n_cases']} | {shape} | {spacing} | {intensity} |")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- MRI intensities are arbitrary scanner-dependent units; cross-sequence and cross-center intensity ranges are descriptive only.",
        "- CT uses Hounsfield units and a body-like threshold, so its numeric range is not directly comparable with MRI.",
        "- The current NIfTI files do not provide verified scanner vendor/model, protocol identifiers, or acquisition dates for all cases. Those fields remain unavailable and must not be inferred from intensity or spacing.",
        "- This is a dataset characterization, not an independent measure of domain-shift severity; there are only three target domains.",
    ]
    (OUT / "domain_characterization.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Characterized {len(case_rows)} raw scans across {len(summary)} domains.")


if __name__ == "__main__":
    main()
