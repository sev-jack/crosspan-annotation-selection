"""Create a code/protocol snapshot and SHA-256 manifest without copying images."""

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import config


OUT = config.WORK / "outputs" / "radiology_execution"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for p in (config.WORK / "code").glob("*.py"):
        files.append(p)
    for name in (
        "environment.yml", "splits.json", "splits_amos.json", "results_seeded_v2.csv",
        "results_amos_external.csv", "00_实验总报告.md", "01_论文实验章节提纲.md",
        "02_下一步行动建议.md",
    ):
        p = config.WORK / name
        if p.is_file():
            files.append(p)
    for subdir in ("seeded_v2", "amos_external", "radiology_execution", "OOP患者独立分析"):
        root = config.WORK / "outputs" / subdir
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and p.name != "patient_features_internal.json")
    for p in config.RUNS.glob("*/run_metadata.json"):
        files.append(p)
    for pattern in ("*/model.pt", "*/per_case.json", "*/per_case_metrics.json"):
        files.extend(config.RUNS.glob(pattern))
    files = sorted(set(p.resolve() for p in files if p.is_file()))

    manifest_rows = []
    for p in files:
        manifest_rows.append({"path": str(p.relative_to(config.WORK)),
                              "size_bytes": p.stat().st_size,
                              "sha256": digest(p)})
    with (OUT / "sha256_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["path", "size_bytes", "sha256"])
        w.writeheader()
        w.writerows(manifest_rows)

    archive = OUT / "analysis_protocol_snapshot.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            rel = p.relative_to(config.WORK)
            # Checkpoints and raw per-case payloads remain in the run tree and are hashed,
            # but the compact portable archive contains code, protocol, tables, and figures.
            if p.suffix == ".pt" or p.name == "per_case.json" or p.name == "per_case_metrics.json":
                continue
            z.write(p, arcname=str(rel))

    import numpy as np
    import torch
    import monai
    note = {
        "python": __import__("platform").python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "monai": monai.__version__,
        "image_data_included": False,
        "model_checkpoints_in_archive": False,
        "checkpoint_and_per_case_hashes_in_manifest": True,
        "runs_directory": str(config.RUNS.relative_to(config.WORK)),
        "cache_directory": str(config.CACHE.relative_to(config.WORK)),
        "source_dataset_root": str(config.ROOT),
        "ct_dataset_root": str(config.CT_ROOT),
        "reproduction_note": "The archive stores analysis code/protocol/results and figures. Public/local image data and large model checkpoints stay in their existing directories; verify them against sha256_manifest.csv.",
    }
    (OUT / "environment_manifest.json").write_text(json.dumps(note, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Hashed {len(manifest_rows)} files; wrote {archive.name} ({archive.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
