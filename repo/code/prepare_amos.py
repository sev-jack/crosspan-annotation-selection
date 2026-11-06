"""Prepare the labeled AMOS22 CT cohort with the established MSD-CT pipeline."""

import argparse
import json
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

import config


AMOS_ROOT = Path(r"<DATA_ROOT>\AMOS22\amos22")
PANCREAS_LABEL = 10


def cases(split):
    image_dir = AMOS_ROOT / f"images{split}"
    label_dir = AMOS_ROOT / f"labels{split}"
    found = []
    for image in sorted(image_dir.glob("amos_*.nii.gz")):
        m = re.fullmatch(r"amos_(\d{4})\.nii\.gz", image.name)
        if not m or int(m.group(1)) >= 500:
            continue  # AMOS IDs >= 500 are MRI; retain CT only.
        label = label_dir / image.name
        if not label.is_file():
            raise FileNotFoundError(f"AMOS CT case lacks label: {label}")
        stem = image.name[:-len(".nii.gz")]
        found.append({"key": f"AMOS_{split.lower()}_{stem}", "image": image, "label": label,
                      "stem": stem})
    return found


def fit_to_size(t):
    _, _, h, w = t.shape
    ph, pw = max(0, config.SIZE - h), max(0, config.SIZE - w)
    if ph or pw:
        t = F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
    h, w = t.shape[-2:]
    h0, w0 = (h - config.SIZE) // 2, (w - config.SIZE) // 2
    return t[:, :, h0:h0 + config.SIZE, w0:w0 + config.SIZE]


def process_case(case):
    image = nib.as_closest_canonical(nib.load(str(case["image"])))
    label = nib.as_closest_canonical(nib.load(str(case["label"])))
    x = np.ascontiguousarray(np.asarray(image.dataobj, dtype=np.float32))
    y = np.ascontiguousarray(np.asarray(label.dataobj))
    if x.ndim == 4:
        x = x[..., 0]
    if y.ndim == 4:
        y = y[..., 0]
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(f"{case['key']}: image/label mismatch {x.shape} vs {y.shape}")

    zooms = image.header.get_zooms()[:3]
    xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(1)
    yt = torch.from_numpy((y == PANCREAS_LABEL).astype(np.float32)).permute(2, 0, 1).unsqueeze(1)
    h = max(1, int(round(x.shape[0] * float(zooms[0]) / config.TARGET_SPACING)))
    w = max(1, int(round(x.shape[1] * float(zooms[1]) / config.TARGET_SPACING)))
    xt = F.interpolate(xt, size=(h, w), mode="bilinear", align_corners=False)
    yt = F.interpolate(yt, size=(h, w), mode="nearest")
    xt = fit_to_size(xt)
    yt = fit_to_size(yt)

    hu = xt[:, 0].numpy()
    lo, hi = config.CT_HU_WINDOW
    xi = np.clip((hu - lo) / (hi - lo), 0.0, 1.0)
    xi = (xi * 255.0).round().astype(np.uint8)
    yi = (yt[:, 0].numpy() > 0.5).astype(np.uint8)
    spacing_zyx = [float(zooms[2]), config.TARGET_SPACING, config.TARGET_SPACING]
    return xi, yi, spacing_zyx, [float(v) for v in zooms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT = config.WORK / "outputs" / "amos_external"
    OUT.mkdir(parents=True, exist_ok=True)
    train = cases("Tr")
    validation = cases("Va")
    if len(train) != 200 or len(validation) != 100:
        raise RuntimeError(f"Expected 200 labeled AMOS CT training and 100 validation scans, got {len(train)}/{len(validation)}")
    train_ids = {c["stem"] for c in train}
    val_ids = {c["stem"] for c in validation}
    if train_ids & val_ids:
        raise RuntimeError(f"AMOS train/validation patient overlap: {sorted(train_ids & val_ids)}")

    index_path = config.CACHE / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    rows = []
    started = time.time()
    for split, group in (("Tr", train), ("Va", validation)):
        for i, case in enumerate(group, 1):
            out = config.CACHE / f"{case['key']}.npz"
            if out.exists() and not args.force:
                with np.load(out) as d:
                    img, lab = d["img"], d["lab"]
                    spacing_zyx = d["spacing_zyx"].astype(float).tolist()
                spacing_xyz = None
            else:
                img, lab, spacing_zyx, spacing_xyz = process_case(case)
                np.savez_compressed(out, img=img, lab=lab,
                                    spacing_zyx=np.asarray(spacing_zyx, dtype=np.float32))
            entry = {
                "seq": "AMOS",
                "split": "train" if split == "Tr" else "val",
                "stem": case["stem"],
                "center": "AMOS",
                "z": int(img.shape[0]),
                "fg_pct": float(lab.mean() * 100),
                "spacing_zyx": [round(float(s), 4) for s in spacing_zyx],
            }
            if spacing_xyz is not None:
                entry["spacing"] = [round(float(s), 4) for s in spacing_xyz]
            index[case["key"]] = entry
            rows.append(case["key"])
            if i % 20 == 0 or i == len(group):
                print(f"{split}: {i}/{len(group)} prepared", flush=True)

    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    splits = json.loads(config.SPLITS.read_text(encoding="utf-8"))
    splits["targets"]["AMOS"] = {
        "pool": [c["key"] for c in train],
        "eval": [c["key"] for c in validation],
        "eval_description": "AMOS22 labeled validation split, CT only; held out from adaptation; not a blind test split",
        "modality": "CT",
        "foreground_label": PANCREAS_LABEL,
    }
    split_path = config.WORK / "splits_amos.json"
    split_path.write_text(json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "dataset": "AMOS22",
        "cohort": "CT only; case IDs below 500 per local dataset README",
        "foreground_label_id": PANCREAS_LABEL,
        "pool_cases": len(train),
        "evaluation_cases": len(validation),
        "evaluation_caveat": "The local AMOS README describes this labeled validation split as a model-selection split. Treat the present analysis as external validation, not a blind external test.",
        "pool_evaluation_id_overlap": sorted(train_ids & val_ids),
        "cache": str(config.CACHE.resolve()),
        "split_file": str(split_path.relative_to(config.WORK)),
        "image_preprocessing": f"HU window {config.CT_HU_WINDOW}; in-plane resampled to {config.TARGET_SPACING} mm; centered {config.SIZE}x{config.SIZE}; label nearest-neighbor",
        "finished_unix_seconds": round(time.time() - started, 1),
        "generated_keys": len(rows),
    }
    (OUT / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(rows)} AMOS cache files and {split_path}; train/eval IDs are disjoint.")


if __name__ == "__main__":
    main()
