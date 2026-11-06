"""Create the study-design schematic from the locked cohort split file."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config


OUT = config.WORK / "outputs" / "radiology_execution"


def box(ax, xy, width, height, title, body, facecolor, edgecolor="#44546A"):
    x, y = xy
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.025",
                           linewidth=1.3, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#1F2937")
    ax.text(x + width / 2, y + height * 0.36, body, ha="center", va="center",
            fontsize=9.4, color="#263445", linespacing=1.4)


def arrow(ax, start, end, color="#52677F"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=color, connectionstyle="arc3,rad=0"))


def main():
    split_path = config.WORK / "splits_amos.json"
    splits = json.loads((split_path if split_path.exists() else config.SPLITS).read_text(encoding="utf-8"))
    source_n = len(splits["source"]["train"]) + len(splits["source"]["val"])
    target_order = ["OOP", "CT", "NYU", "AMOS"]
    colors = {"OOP": "#FCE8D5", "CT": "#DBEAFE", "NYU": "#E4F4E7", "AMOS": "#F1E8FB"}
    names = {"OOP": "OOP · cross-sequence", "CT": "MSD CT · cross-modality",
             "NYU": "NYU · cross-center", "AMOS": "AMOS22 CT · external dataset"}
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, "Study design: target-domain annotation response",
            ha="center", va="top", fontsize=18, fontweight="bold", color="#172B4D")

    box(ax, (0.035, 0.57), 0.22, 0.23, "Source cohort · T1W",
        f"Train + source validation\n{source_n} cases\n3 controlled seeds", "#E7EEF7")
    box(ax, (0.34, 0.57), 0.25, 0.23, "Source 2.5D U-Net",
        "One source model per seed\nFixed training schedule\nNo target-test tuning", "#DDE7F4")
    arrow(ax, (0.255, 0.685), (0.335, 0.685))

    ys = [0.745, 0.575, 0.405, 0.235]
    for target, y in zip(target_order, ys):
        if target not in splits["targets"]:
            continue
        d = splits["targets"][target]
        pool_n = len(d["pool"])
        eval_key = "eval_patient_disjoint" if target == "OOP" else "eval"
        eval_n = len(d[eval_key])
        label = "0 / 1 / 5 / 10 / 19 / 95" if target == "OOP" else \
                ("0 / 1 / 5 / 10 / 20 / 100 / 220" if target == "CT" else
                 ("0 / 1 / 5 / 10 / 17" if target == "NYU" else "0 / 1 / 5 / 10 / 20"))
        eval_desc = "patient-disjoint eval" if target == "OOP" else "held-out evaluation"
        box(ax, (0.655, y), 0.325, 0.14, names[target],
            f"Pool {pool_n} cases\n{label} labeled\n{eval_n} cases · {eval_desc}", colors[target])
        arrow(ax, (0.59, 0.685), (0.65, y + 0.07))

    box(ax, (0.035, 0.12), 0.54, 0.25, "Evaluation and reporting",
        "Dice · HD95 · ASSD (mm)\nPatient-level bootstrap 95% CI\n"
        "Absolute labeled-case count + pool fraction\nNested random label subsets; 3 training seeds",
        "#F4F5F7")
    ax.text(0.5, 0.035,
            "Clinical acceptability thresholds and real annotation time/cost are not established by this experiment.",
            ha="center", va="center", fontsize=9, color="#6B4E16")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "study_design.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / "study_design.png")


if __name__ == "__main__":
    main()
