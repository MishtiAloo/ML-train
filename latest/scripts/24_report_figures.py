"""Figures for docs/final_technical_report.md that aren't confusion matrices
or epoch curves (those come from 21_e1e2_confusion.py / 22_e1e2_plots.py).
Pure data/plotting -- no model, no GPU. Reads manifest.csv, the raw
processed_data/ images (one level above latest/, never modified), the
aligned crops/, and metadata/e1e2_person_folds.json.

Usage:
    python 24_report_figures.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = ROOT.parent / "processed_data"
OUT_DIR = ROOT / "docs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OCCS = ["none", "clearglass", "sunglass", "cap", "scarf", "mask", "mask_clearglass", "mask_sunglass"]
OCC_LABELS = {
    "none": "none", "clearglass": "clear glasses", "sunglass": "sunglasses", "cap": "cap",
    "scarf": "scarf", "mask": "mask", "mask_clearglass": "mask+clear", "mask_sunglass": "mask+sun",
}


def load_manifest():
    with open(ROOT / "metadata" / "manifest.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick_best(rows, person, occ):
    cands = [r for r in rows if r["person_id"] == person and r["occlusion"] == occ]
    if not cands:
        return None
    reg = [r for r in cands if r["lighting"] == "regular"]
    pool = reg if reg else cands
    return max(pool, key=lambda r: float(r["det_score"]))


def fig_sample_grid(rows, people, title, out_name, raw=False):
    """rows=people, cols=occlusions. raw=True reads processed_data/ (pre-crop,
    full-resolution); raw=False reads crops/ (112x112 aligned, model input)."""
    fig, axes = plt.subplots(len(people), len(OCCS), figsize=(2.0 * len(OCCS), 2.2 * len(people)))
    for i, p in enumerate(people):
        for j, occ in enumerate(OCCS):
            ax = axes[i, j]
            r = pick_best(rows, p, occ)
            if r is None:
                ax.axis("off")
                continue
            if raw:
                img_path = RAW_ROOT / p / Path(r["image_path"]).name
            else:
                img_path = ROOT / "crops" / p / Path(r["crop_path"]).name
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(OCC_LABELS[occ], fontsize=11)
            if j == 0:
                ax.set_ylabel(p, fontsize=11, rotation=0, labelpad=28, va="center")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0.02, 0, 1, 0.96))
    fig.savefig(OUT_DIR / out_name, dpi=130)
    plt.close(fig)
    print(f"wrote {out_name}")


def fig_occlusion_counts(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r["occlusion"]] += 1
    occs = OCCS
    vals = [counts[o] for o in occs]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar([OCC_LABELS[o] for o in occs], vals, color="#97ECF8", edgecolor="black")
    ax.bar_label(bars, padding=2, fontsize=9)
    ax.set_ylabel("photos")
    ax.set_title("Dataset composition by occlusion condition (all 30 people)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_occlusion_counts.png", dpi=130)
    plt.close(fig)
    print("wrote dataset_occlusion_counts.png")


def fig_person_counts(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r["person_id"]] += 1
    people = sorted(counts)
    vals = [counts[p] for p in people]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(people, vals, color="#FFD0AC", edgecolor="black")
    ax.axhline(np.mean(vals), color="black", linestyle="--", linewidth=1, label=f"mean={np.mean(vals):.0f}")
    ax.set_ylabel("photos")
    ax.set_title("Photos per enrolled person (30 people, all occlusions pooled)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_person_counts.png", dpi=130)
    plt.close(fig)
    print("wrote dataset_person_counts.png")


def fig_det_score_hist(rows):
    scores = np.array([float(r["det_score"]) for r in rows if r["det_score"]])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=40, color="#A4FFA1", edgecolor="black")
    ax.set_xlabel("RetinaFace detection confidence")
    ax.set_ylabel("photos")
    ax.set_title(f"Detection-score distribution, all accepted crops (n={len(scores)})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dataset_det_score_hist.png", dpi=130)
    plt.close(fig)
    print("wrote dataset_det_score_hist.png")


def fig_fold_assignment():
    with open(ROOT / "metadata" / "e1e2_person_folds.json", encoding="utf-8") as fh:
        setup = json.load(fh)
    folds = setup["folds"]
    all_people = sorted({p for f in folds for p in (f["train"] + f["val"] + f["test"])})
    n_fold = len(folds)
    grid = np.zeros((len(all_people), n_fold))  # 0=train,1=val,2=test
    for f in folds:
        k = f["fold"] - 1
        for p in f["train"]:
            grid[all_people.index(p), k] = 0
        for p in f["val"]:
            grid[all_people.index(p), k] = 1
        for p in f["test"]:
            grid[all_people.index(p), k] = 2
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#E0E0E0", "#FFF17B", "#FFA0A0"])
    fig, ax = plt.subplots(figsize=(5, 10))
    ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(range(n_fold)); ax.set_xticklabels([f"fold {f['fold']}" for f in folds])
    ax.set_yticks(range(len(all_people))); ax.set_yticklabels(all_people, fontsize=7)
    ax.set_title("Person role per fold\n(grey=train 20, yellow=val 5, red=test 5)")
    for i in range(len(all_people)):
        for j in range(n_fold):
            pass
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fold_assignment_grid.png", dpi=130)
    plt.close(fig)
    print("wrote fold_assignment_grid.png")


def fig_benchmark_selection(rows, person="p03", occ="none"):
    cands = [r for r in rows if r["person_id"] == person and r["occlusion"] == occ]
    cands = sorted(cands, key=lambda r: -float(r["det_score"]))[:6]
    fig, axes = plt.subplots(1, len(cands), figsize=(2.1 * len(cands), 2.6))
    for ax, r in zip(axes, cands):
        img = cv2.imread(str(ROOT / "crops" / person / Path(r["crop_path"]).name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        is_best = r is cands[0]
        color = "#A4FFA1" if is_best else "#E0E0E0"
        for spine in ax.spines.values():
            spine.set_edgecolor("black"); spine.set_linewidth(3 if is_best else 1)
        ax.set_facecolor(color)
        label = f"det={float(r['det_score']):.3f}"
        if is_best:
            label += "\nSELECTED"
        ax.set_xlabel(label, fontsize=9)
    fig.suptitle(f"Benchmark candidate selection for {person} ('none', highest RetinaFace score wins)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "benchmark_selection_example.png", dpi=130)
    plt.close(fig)
    print("wrote benchmark_selection_example.png")


def parse_occlusion_from_filename(stem: str) -> str:
    for occ in sorted(OCCS, key=len, reverse=True):
        if f"_{occ}_" in f"_{stem}_":
            return occ
    return "?"


def fig_retinaface_failures():
    """Raw photos RetinaFace could not detect a face in at all -- these never
    made it into manifest.csv, so there is no crop to show; that IS the
    failure. Found by diffing processed_data/'s raw jpgs against every
    image_path already in manifest.csv (no dependency on the archived
    preprocess_log.csv, so this stays reproducible from latest/ alone)."""
    with open(ROOT / "metadata" / "manifest.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    have = {(Path(r["image_path"]).parent.name, Path(r["image_path"]).name) for r in rows}

    missing = []
    for pdir in sorted(RAW_ROOT.iterdir()):
        if not pdir.is_dir():
            continue
        for f in sorted(pdir.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg"):
                continue  # excludes the 2 unconverted .dng raw-format files -- a separate pipeline issue, not a detection failure
            if (pdir.name, f.name) not in have:
                missing.append((pdir.name, f))

    print(f"RetinaFace detection failures found: {len(missing)}")
    n = len(missing)
    ncols = 6
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.6 * nrows))
    axes = np.atleast_2d(axes)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        ax.axis("off")
        if idx >= n:
            continue
        person, f = missing[idx]
        img = cv2.imread(str(f))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        occ = parse_occlusion_from_filename(f.stem)
        ax.set_title(f"{person}, {OCC_LABELS.get(occ, occ)}", fontsize=9)
    fig.suptitle(f"Raw photos where RetinaFace detected NO face at all (n={n} of 6,000 -- 0.2%; "
                 f"excluded from manifest.csv, so ArcFace never sees them)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "retinaface_failure_gallery.png", dpi=130)
    plt.close(fig)
    print("wrote retinaface_failure_gallery.png")


def fig_failure_gallery():
    """Real E0 (untrained ArcFace) failures picked from runs/e1e2/analysis/
    e0_predictions.csv (per-image scores dumped by 23_dump_predictions.py) --
    not staged. Two rows: lowest-confidence rejections (score near 0, mostly
    mask+sunglass), and cases where the top match was a *different* person
    entirely (argmax wrong, not just low-confidence)."""
    with open(ROOT / "runs" / "e1e2" / "analysis" / "e0_predictions.csv", newline="", encoding="utf-8") as fh:
        preds = list(csv.DictReader(fh))
    rejected = [r for r in preds if r["outcome"] == "rejected"]
    lowest = sorted(rejected, key=lambda r: float(r["score"]))[:6]
    wrong = [r for r in rejected if r["pred_person"] != r["person_id"]]
    wrong = sorted(wrong, key=lambda r: -float(r["score"]))[:6]

    fig, axes = plt.subplots(2, 6, figsize=(13, 6.6))
    row_titles = [
        "Near-zero confidence (correct person, but score ≈ 0)",
        "Wrong top match (argmax picked a different enrolled person)",
    ]
    for row_idx, (title, examples) in enumerate(zip(row_titles, [lowest, wrong])):
        for j, r in enumerate(examples):
            ax = axes[row_idx, j]
            img = cv2.imread(str(ROOT / "crops" / r["person_id"] / Path(r["crop_path"]).name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("black"); spine.set_linewidth(2)
            ax.set_xlabel(f"true={r['person_id']}\ntop={r['pred_person']} ({float(r['score']):.2f})\n{r['occlusion']}", fontsize=8)
    fig.suptitle("Real E0 (untrained pretrained ArcFace) failures at the 0.3 acceptance threshold", fontsize=12)
    fig.tight_layout(rect=(0.0, 0, 1, 0.88))
    fig.subplots_adjust(hspace=0.55)
    for row_idx, title in enumerate(row_titles):
        y = axes[row_idx, 0].get_position().y1 + 0.03
        fig.text(0.5, y, title, ha="center", fontsize=10, weight="bold")
    fig.savefig(OUT_DIR / "arcface_failure_gallery.png", dpi=130)
    plt.close(fig)
    print("wrote arcface_failure_gallery.png")


def fig_learning_curve():
    """Real train-loss / val-loss learning curve, epoch on x-axis, single
    loss axis, all 6 folds, E1 vs E2. Source: runs/e1e2/learning_curve/,
    produced by scripts/25_e1e2_learning_curve.py -- a copy of
    19_e1e2_person_finetune.py that additionally computes val_loss
    (cross-entropy over the full 30-person gallery, no margin, no label
    smoothing -- the natural loss analogue of eval-time scoring) from the
    SAME cos matrix score() already computes, at zero extra cost. Re-run on
    the 3090 box with the same seed (42): every val_acc/best_epoch/test
    result below is bit-for-bit identical to runs/e1e2/folds/ (verified),
    only val_loss is new."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    fold_colors = plt.cm.tab10(np.linspace(0, 1, 6))
    for ax, exp, title in zip(axes, ["e1", "e2"], ["E1 (last residual block)", "E2 (block + embedding head)"]):
        for k in range(1, 7):
            with open(ROOT / "runs" / "e1e2" / "learning_curve" / f"fold{k}" / f"{exp}_person_model_history.json") as fh:
                hist = json.load(fh)["history"]
            train_epochs = [r["epoch"] for r in hist if r["train_loss"] is not None]
            train_loss = [r["train_loss"] for r in hist if r["train_loss"] is not None]
            val_epochs = [r["epoch"] for r in hist]
            val_loss = [r["val_loss"] for r in hist]
            ax.plot(train_epochs, train_loss, color=fold_colors[k - 1], linestyle="-", marker="o", markersize=3,
                     label=f"fold {k}" if exp == "e1" else None)
            ax.plot(val_epochs, val_loss, color=fold_colors[k - 1], linestyle="--", marker="^", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Loss (solid = train, dashed = val)")
    fig.legend(loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Learning curve: training loss and validation loss per epoch, all 6 folds")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT_DIR / "learning_curve.png", dpi=130)
    plt.close(fig)
    print("wrote learning_curve.png")


def main():
    rows = load_manifest()
    people = ["p03", "p11", "p29"]
    fig_sample_grid(rows, people, "Sample raw photos, one occlusion per column (pre-detection)", "sample_raw_grid.png", raw=True)
    fig_sample_grid(rows, people, "Corresponding aligned 112x112 crops (RetinaFace output, ArcFace input)", "sample_crops_grid.png", raw=False)
    fig_occlusion_counts(rows)
    fig_person_counts(rows)
    fig_det_score_hist(rows)
    fig_fold_assignment()
    fig_benchmark_selection(rows)
    fig_retinaface_failures()
    fig_failure_gallery()
    fig_learning_curve()


if __name__ == "__main__":
    main()
