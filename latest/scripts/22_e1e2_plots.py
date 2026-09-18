"""E0/E1/E2 v3: plots from already-computed results. Reads only the CSVs
and JSONs written by 19_e1e2_person_finetune.py and 21_e1e2_confusion.py --
no model, no GPU, no crops needed. Safe and fast to run on any machine,
including a laptop with no CUDA.

Outputs, under --out-dir (default docs/plots/ next to this script's project
root):
    {e0,e1,e2}_confusion_overall.png   row-normalized 30x31 heatmap
    per_occlusion_accuracy.png         grouped bar chart, E0 vs E1 vs E2
    outcome_breakdown.png              stacked bar: accepted/rejected/misID
    epoch_curves.png                   val accuracy per epoch, all 6 folds, E1 vs E2

Usage:
    python 22_e1e2_plots.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPS = ["e0", "e1", "e2"]
LABELS = {"e0": "E0 (untrained)", "e1": "E1 (block only)", "e2": "E2 (block+fc)"}
COLORS = {"e0": "#9e9e9e", "e1": "#1f77b4", "e2": "#d62728"}
OCC_ORDER = ["none", "cap", "clearglass", "sunglass", "scarf", "mask", "mask_clearglass", "mask_sunglass"]


def plot_confusion(analysis_dir: Path, out_dir: Path):
    for exp in EXPS:
        with open(analysis_dir / f"{exp}_confusion_overall.csv", newline="") as fh:
            rows = list(csv.reader(fh))
        persons = rows[0][1:-1]
        cm = np.array([[int(x) for x in r[1:]] for r in rows[1:]], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(persons) + 1))
        ax.set_xticklabels(persons + ["REJECT"], rotation=90, fontsize=6)
        ax.set_yticks(range(len(persons)))
        ax.set_yticklabels(persons, fontsize=6)
        ax.set_xlabel("Predicted (last column = rejected, score < 0.3)")
        ax.set_ylabel("True person")
        ax.set_title(f"{LABELS[exp]} — pooled confusion matrix (row-normalized), n=5818")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="row-normalized rate")
        fig.tight_layout()
        fig.savefig(out_dir / f"{exp}_confusion_overall.png", dpi=150)
        plt.close(fig)
        print(f"saved {out_dir / f'{exp}_confusion_overall.png'}")


def plot_per_occlusion(metrics: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(OCC_ORDER))
    width = 0.25
    for i, exp in enumerate(EXPS):
        accs = [metrics[exp]["per_occlusion"][o]["accuracy"] * 100 for o in OCC_ORDER]
        ax.bar(x + (i - 1) * width, accs, width, label=LABELS[exp], color=COLORS[exp])
    ax.set_xticks(x)
    ax.set_xticklabels(OCC_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Accepted & correct (%)")
    ax.set_title("Per-occlusion accuracy, pooled over 6 folds (0.3 acceptance threshold)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_occlusion_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir / 'per_occlusion_accuracy.png'}")


def plot_outcome_breakdown(results_dir: Path, out_dir: Path):
    stats = {}
    for exp in EXPS:
        total_n = total_correct = total_rej = total_mis = 0
        for k in range(1, 7):
            with open(results_dir / f"fold{k}" / f"{exp}_person_results.json") as fh:
                r = json.load(fh)
            total_n += r["test_n"]
            total_correct += r["test_correct_n"]
            total_rej += r["test_rejected_correct_n"]
            total_mis += r["test_misidentified_accepted_n"]
        stats[exp] = {"acc": total_correct / total_n * 100, "rej": total_rej / total_n * 100,
                      "mis": total_mis / total_n * 100}

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    bottoms = np.zeros(3)
    for key, color, label in [("acc", "#2ca02c", "Accepted & correct"),
                               ("rej", "#ff7f0e", "Rejected (< 0.3)"),
                               ("mis", "#d62728", "Misidentified")]:
        vals = [stats[e][key] for e in EXPS]
        ax.bar(x, vals, 0.6, bottom=bottoms, label=label, color=color)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[e] for e in EXPS])
    ax.set_ylabel("% of pooled test queries (n=5818)")
    ax.set_title("Outcome breakdown, pooled over 6 folds")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "outcome_breakdown.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir / 'outcome_breakdown.png'}")


def plot_epoch_curves(results_dir: Path, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, exp in zip(axes, ["e1", "e2"]):
        for k in range(1, 7):
            with open(results_dir / f"fold{k}" / f"{exp}_person_model_history.json") as fh:
                hist = json.load(fh)["history"]
            ax.plot([r["epoch"] for r in hist], [r["val_acc"] * 100 for r in hist],
                    marker="o", markersize=3, label=f"fold {k}")
        ax.set_xlabel("Epoch")
        ax.set_title(LABELS[exp])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Val accuracy at 0.3 threshold (%)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Validation accuracy per epoch, all 6 folds")
    fig.tight_layout()
    fig.savefig(out_dir / "epoch_curves.png", dpi=150)
    plt.close(fig)
    print(f"saved {out_dir / 'epoch_curves.png'}")


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=str(root / "runs" / "e1e2" / "analysis"))
    ap.add_argument("--results-dir", default=str(root / "runs" / "e1e2" / "folds"))
    ap.add_argument("--out-dir", default=str(root / "docs" / "plots"))
    args = ap.parse_args()

    analysis_dir, results_dir, out_dir = Path(args.analysis_dir), Path(args.results_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {exp: json.load(open(analysis_dir / f"{exp}_metrics.json")) for exp in EXPS}

    plot_confusion(analysis_dir, out_dir)
    plot_per_occlusion(metrics, out_dir)
    plot_outcome_breakdown(results_dir, out_dir)
    plot_epoch_curves(results_dir, out_dir)


if __name__ == "__main__":
    main()
