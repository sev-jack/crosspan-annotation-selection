"""Render prespecified representative target cases from saved checkpoints."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import config
import train_eval as TE


OUT = config.WORK / "outputs" / "radiology_execution"


def run_id(target, budget, seed, eval_key):
    if target == "OOP":
        # OOP checkpoints use percentage IDs, with the exact pool fraction encoded in the split.
        pct = {0: 0, 10: 10, 95: 100}[budget]
        tag = "_pd"
        return f"T1W2OOP_{pct:03d}pct_s{seed}{tag}"
    return f"T1W2{target}_n{budget:03d}case_s{seed}"


def predict_case(case_key, checkpoint, device):
    model = TE.build_model()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    img, lab = TE.load_case(case_key)
    vol = TE.volume_to_25d(img)
    pred = np.zeros_like(lab, dtype=np.uint8)
    with torch.no_grad():
        for z0 in range(0, len(img), 16):
            xt = torch.from_numpy(TE.norm(vol[z0:z0 + 16])).to(device)
            logits = TE.forward_tta(model, xt, device) if config.USE_TTA else TE.forward_plain(model, xt, device)
            pred[z0:z0 + len(xt)] = logits.argmax(1).cpu().numpy().astype(np.uint8)
    del model, state
    if device == "cuda":
        torch.cuda.empty_cache()
    return img, lab, pred


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split_path = config.WORK / "splits_amos.json"
    split_file = split_path if split_path.exists() else config.SPLITS
    splits = json.loads(split_file.read_text(encoding="utf-8"))
    targets = [t for t in ("OOP", "CT", "NYU", "AMOS") if t in splits["targets"]]
    fig, axes = plt.subplots(len(targets), 3, figsize=(12, 3.45 * len(targets)), squeeze=False)
    selected = []
    for row_idx, target in enumerate(targets):
        d = splits["targets"][target]
        eval_key = "eval_patient_disjoint" if target == "OOP" else "eval"
        if target == "OOP":
            budget_counts = [0, 10, 95]
        else:
            candidates = [20, 100, 220] if target == "CT" else \
                         ([17, 10, 5] if target == "NYU" else [20, 10, 5])
            available_max = next((n for n in sorted(candidates, reverse=True)
                                  if (config.RUNS / run_id(target, n, 0, eval_key) / "model.pt").is_file()), None)
            if available_max is None:
                raise FileNotFoundError(f"No completed high-budget checkpoint found for {target}")
            budget_counts = [0, 10, available_max]

        baseline_id = run_id(target, 0, 0, eval_key)
        baseline_scores = json.loads((config.RUNS / baseline_id / "per_case.json").read_text(encoding="utf-8"))
        case_vals = [(case, float(baseline_scores[case])) for case in d[eval_key] if case in baseline_scores]
        med = float(np.median([v for _, v in case_vals]))
        case_key, baseline_dice = min(case_vals, key=lambda kv: abs(kv[1] - med))
        img, gt, baseline_pred = predict_case(case_key, config.RUNS / baseline_id / "model.pt",
                                              "cuda" if torch.cuda.is_available() else "cpu")
        z_foreground = np.where(gt.reshape(gt.shape[0], -1).sum(axis=1) > 0)[0]
        z = int(z_foreground[len(z_foreground) // 2]) if len(z_foreground) else img.shape[0] // 2
        selected.append({"target": target, "case_key": case_key, "baseline_dice_seed0": baseline_dice,
                         "median_baseline_dice_seed0": med, "display_slice_0based": z,
                         "eval_key": eval_key, "display_budgets": ",".join(map(str, budget_counts))})

        for col, n_label in enumerate(budget_counts):
            if col == 0:
                pred_img, pred_gt, prediction = img, gt, baseline_pred
            else:
                rid = run_id(target, n_label, 0, eval_key)
                checkpoint = config.RUNS / rid / "model.pt"
                pred_img, pred_gt, prediction = predict_case(case_key, checkpoint,
                                                              "cuda" if torch.cuda.is_available() else "cpu")
            ax = axes[row_idx, col]
            ax.imshow(pred_img[z], cmap="gray", vmin=0, vmax=255)
            if pred_gt[z].any():
                ax.contour(pred_gt[z], levels=[0.5], colors=["#22C55E"], linewidths=1.0)
            if prediction[z].any():
                ax.contour(prediction[z], levels=[0.5], colors=["#F43F5E"], linewidths=1.0)
            budget_label = "0 labeled" if col == 0 else f"{n_label} labeled"
            ax.set_title(f"{target} · {budget_label}", fontsize=11)
            ax.axis("off")
            ax.text(0.02, 0.03, "GT green · prediction red", transform=ax.transAxes,
                    fontsize=7.5, color="white", ha="left", va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"})

    fig.suptitle("Representative cases selected from the median zero-label Dice case (seed 0)",
                 y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "qualitative_cases.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    with (OUT / "qualitative_case_selection.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader()
        w.writerows(selected)
    print(f"Wrote qualitative figure for {len(selected)} target domains.")


if __name__ == "__main__":
    main()
