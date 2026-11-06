"""Compare unlabeled patient embeddings under the fixed source model."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import train_eval as TE
from protocol import patient_id_from_key


OUT = config.WORK / "outputs" / "radiology_execution"
FEATURE_MODULE = "model.1.submodule.1.submodule.1.submodule.1.submodule"
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261004


def embed_cases(model, keys, device, batch=12):
    module = dict(model.named_modules())[FEATURE_MODULE]
    features = []
    def capture(_module, _inputs, output):
        features.append(output.detach().mean(dim=(-2, -1)).float().cpu().numpy())
    handle = module.register_forward_hook(capture)
    result = {}
    try:
        for key in keys:
            image, _ = TE.load_case(key)
            volume = TE.volume_to_25d(image)
            zs = np.unique(np.linspace(0, len(image) - 1, 5).round().astype(int))
            for i in range(0, len(zs), batch):
                selected = zs[i:i + batch]
                xt = torch.from_numpy(TE.norm(volume[selected])).to(device)
                with torch.no_grad():
                    with TE.amp_ctx(device):
                        model(xt)
            per_slice = np.concatenate(features, axis=0)
            result[key] = per_slice.mean(axis=0)
            features.clear()
    finally:
        handle.remove()
    return result


def rbf_mmd2(x, y, gamma):
    def kernel(a, b):
        d2 = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)
        return np.exp(-gamma * d2)
    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean())


def main():
    split_file = config.WORK / "splits_amos.json"
    splits = json.loads((split_file if split_file.exists() else config.SPLITS).read_text(encoding="utf-8"))
    groups = {"Source T1W reference": splits["source"]["train"] + splits["source"]["val"]}
    for target, desc in (("OOP", "eval_patient_disjoint"), ("CT", "eval"), ("NYU", "eval"), ("AMOS", "eval")):
        if target in splits["targets"]:
            groups[target] = splits["targets"][target][desc]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TE.build_model()
    ckpt = config.RUNS / "source_T1W_s0_seeded_v2.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()
    case_features = {name: embed_cases(model, keys, device) for name, keys in groups.items()}
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    case_rows = []
    for domain, values in case_features.items():
        for key, vector in values.items():
            case_rows.append({"domain": domain, "case_key": key,
                              "patient_id": patient_id_from_key(key),
                              "feature": vector.astype(float).tolist()})
    with (OUT / "patient_features_internal.json").open("w", encoding="utf-8") as f:
        json.dump(case_rows, f, ensure_ascii=False)

    source_raw = np.asarray(list(case_features["Source T1W reference"].values()), dtype=float)
    all_raw = np.concatenate([source_raw] + [np.asarray(list(v.values()), dtype=float)
                                             for k, v in case_features.items() if k != "Source T1W reference"])
    scale = all_raw.std(axis=0)
    scale[scale < 1e-8] = 1.0
    means = all_raw.mean(axis=0)
    source = (source_raw - means) / scale
    target_arrays = {k: (np.asarray(list(v.values()), dtype=float) - means) / scale
                     for k, v in case_features.items() if k != "Source T1W reference"}

    pooled = np.concatenate([source] + list(target_arrays.values()))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    max_n = min(len(pooled), 800)
    sample = pooled[rng.choice(len(pooled), max_n, replace=False)] if len(pooled) > max_n else pooled
    dist = np.sum(sample * sample, axis=1)[:, None] + np.sum(sample * sample, axis=1)[None, :] - 2 * sample @ sample.T
    nonzero = dist[np.triu_indices(len(sample), 1)]
    median_d2 = float(np.median(nonzero[nonzero > 0])) if np.any(nonzero > 0) else 1.0
    gamma = 1.0 / median_d2

    stats = []
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    for target, values in target_arrays.items():
        mmd = rbf_mmd2(source, values, gamma)
        draws = []
        for _ in range(BOOTSTRAP_REPS):
            a = source[bootstrap_rng.integers(0, len(source), len(source))]
            b = values[bootstrap_rng.integers(0, len(values), len(values))]
            draws.append(rbf_mmd2(a, b, gamma))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        stats.append({"target": target, "source_patients": len(source), "target_patients": len(values),
                      "rbf_mmd2_zscored_embedding": mmd, "bootstrap_ci_low": float(lo),
                      "bootstrap_ci_high": float(hi), "rbf_gamma_median_heuristic": gamma,
                      "feature_layer": FEATURE_MODULE})
    with (OUT / "feature_mmd.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0]))
        w.writeheader()
        w.writerows(stats)

    # PCA is fit jointly and serves as a visualization only.
    matrix = np.concatenate([source] + list(target_arrays.values()))
    center = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - center, full_matrices=False)
    coords = (matrix - center) @ vt[:2].T
    names = ["Source T1W reference"] + list(target_arrays)
    sizes = [len(source)] + [len(v) for v in target_arrays.values()]
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    palette = ["#64748B", "#E67E22", "#2878B5", "#3A9D5D", "#8E63B6"]
    start = 0
    for name, size, color in zip(names, sizes, palette):
        pts = coords[start:start + size]
        ax.scatter(pts[:, 0], pts[:, 1], s=25, alpha=0.72, label=f"{name} (n={size})", color=color,
                   edgecolors="white", linewidths=0.3)
        start += size
    ax.set_xlabel("PC1 of z-scored bottleneck embeddings")
    ax.set_ylabel("PC2 of z-scored bottleneck embeddings")
    ax.set_title("Fixed source-model feature space (descriptive)")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "feature_space_pca.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Fixed-source-model feature-space analysis",
        "",
        f"Embeddings are 512-channel global-average-pooled activations from `{FEATURE_MODULE}`. Five evenly spaced slices per volume were averaged into one patient-level vector. One fixed source checkpoint (seed 0) was used; target labels and target-adapted checkpoints were not used.",
        "",
        "| Target | Source patients | Target patients | RBF MMD² (z-scored features) | Patient bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in stats:
        lines.append(f"| {r['target']} | {r['source_patients']} | {r['target_patients']} | "
                     f"{r['rbf_mmd2_zscored_embedding']:.4f} | "
                     f"[{r['bootstrap_ci_low']:.4f}, {r['bootstrap_ci_high']:.4f}] |")
    lines += [
        "",
        "MMD uses a pooled median-distance RBF bandwidth and patient-level bootstrap resampling. The PCA projection is descriptive. These distances are not a calibrated clinical severity scale and cannot establish a general relationship from only a few target datasets.",
    ]
    (OUT / "feature_space.md").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"Computed feature embeddings for {sum(sizes)} patients; wrote {len(stats)} source-target MMD comparisons.")


if __name__ == "__main__":
    main()
