"""Loss-only learning curve from runs/e1e2/learning_curve/.

Training loss and validation loss are DIFFERENT objectives here, so they are
drawn in separate panels rather than on one axis:

  training loss  : cross-entropy over the 20-class locked head (training
                   people only), with the angular margin (0 -> 0.20 during
                   epochs 3-6) and label smoothing 0.1, on augmented crops.
                   Label smoothing alone keeps it above 0.594 (the entropy
                   of the smoothed target over 20 classes), so it can never
                   approach zero.
  validation loss: cross-entropy over all 30 gallery benchmarks, no margin,
                   no label smoothing, clean crops. Identically defined at
                   every epoch, so only this one is comparable across epochs.

Reading the vertical distance between the two as a generalisation gap would
be wrong; each panel is meaningful only in its own shape.

Usage (from the project root):
    python scripts/26_loss_curve.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#bcbd22", "#17becf"]
EXPS = [("e1", "E1 (last residual block)"), ("e2", "E2 (block + embedding head)")]


def load(exp: str, fold: int, runs_dir: Path):
    path = runs_dir / f"fold{fold}" / f"{exp}_person_model_history.json"
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return data["history"], data.get("best_epoch")


def plot_matched(runs_dir: Path, out: Path, folds: int):
    """True learning curve: training and validation loss are the SAME
    objective here (clean crops, no margin, no label smoothing,
    cross-entropy over the 30-benchmark gallery), so they belong on one
    axis and their distance is a real generalisation gap."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (exp, title) in zip(axes, EXPS):
        for k in range(1, folds + 1):
            hist, best = load(exp, k, runs_dir)
            colour = COLORS[(k - 1) % len(COLORS)]
            ep = [h["epoch"] for h in hist]
            ax.plot(ep, [h["train_loss_matched"] for h in hist], "-o", ms=3, color=colour, label=f"fold {k}")
            ax.plot(ep, [h["val_loss"] for h in hist], "--^", ms=3.5, color=colour, alpha=0.85)
            if best is not None:
                bh = next(h for h in hist if h["epoch"] == best)
                ax.plot([best], [bh["val_loss"]], "*", ms=13, color=colour,
                        markeredgecolor="black", markeredgewidth=0.4, zorder=5)
        ax.axvspan(2.5, 6.5, color="#f0c000", alpha=0.13, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("epoch  (0 = untrained backbone)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cross-entropy over the 30 benchmarks\n(clean crops, no margin, no smoothing)")
    handles, labels = axes[0].get_legend_handles_labels()
    extra = [plt.Line2D([], [], color="#444", ls="-", marker="o", ms=3, label="train people"),
             plt.Line2D([], [], color="#444", ls="--", marker="^", ms=3.5, label="validation people"),
             plt.Line2D([], [], marker="*", ls="none", ms=11, color="#444", label="kept epoch")]
    fig.legend(handles + extra, labels + [h.get_label() for h in extra], loc="lower center",
               ncol=9, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Learning curve: same loss measured on training people and on unseen validation people")
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")

    print("\nmatched objective (same loss both sides)")
    print("exp fold  train e0 -> last   val e0 -> last    val min @ epoch   cross?")
    for exp, _ in EXPS:
        for k in range(1, folds + 1):
            hist, _ = load(exp, k, runs_dir)
            tr = [h["train_loss_matched"] for h in hist]
            vl = [h["val_loss"] for h in hist]
            sign = [v > t for t, v in zip(tr, vl)]
            cross = "yes at epoch " + str(sign.index(True)) if (sign[0] != sign[-1] and True in sign) else "no"
            print(f"{exp}  {k}    {tr[0]:.4f} -> {tr[-1]:.4f}   {vl[0]:.4f} -> {vl[-1]:.4f}   "
                  f"{min(vl):.4f} @ {vl.index(min(vl))}    {cross}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs" / "e1e2" / "learning_curve"))
    ap.add_argument("--matched-dir", default=str(ROOT / "runs" / "e1e2" / "learning_curve_matched"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "plots" / "loss_curve.png"))
    ap.add_argument("--matched-out", default=str(ROOT / "docs" / "plots" / "loss_curve_matched.png"))
    ap.add_argument("--folds", type=int, default=6)
    args = ap.parse_args()
    runs_dir = Path(args.runs_dir)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, (exp, title) in enumerate(EXPS):
        ax_val, ax_train = axes[0][col], axes[1][col]
        for k in range(1, args.folds + 1):
            hist, best = load(exp, k, runs_dir)
            colour = COLORS[(k - 1) % len(COLORS)]
            ep = [h["epoch"] for h in hist]
            ax_val.plot(ep, [h["val_loss"] for h in hist], "-o", ms=3, color=colour, label=f"fold {k}")
            if best is not None:
                bh = next(h for h in hist if h["epoch"] == best)
                ax_val.plot([best], [bh["val_loss"]], "*", ms=13, color=colour,
                            markeredgecolor="black", markeredgewidth=0.4, zorder=5)
            ax_train.plot(ep[1:], [h["train_loss"] for h in hist[1:]], "-o", ms=3, color=colour)

        for ax in (ax_val, ax_train):
            ax.axvspan(2.5, 6.5, color="#f0c000", alpha=0.13, zorder=0)
            ax.grid(alpha=0.25)
        ax_val.set_title(title)
        ax_train.set_xlabel("epoch  (0 = untrained backbone)")
        if col == 0:
            ax_val.set_ylabel("validation loss\n30-benchmark cross-entropy\nno margin, no smoothing")
            ax_train.set_ylabel("training loss\n20-class head, margin + smoothing 0.1\naugmented crops")

    # smallest cross-entropy reachable with label smoothing 0.1 over 20
    # classes: entropy of the smoothed target distribution
    floor = 0.594
    for ax in (axes[1][0], axes[1][1]):
        ax.axhline(floor, ls=":", lw=1, color="#666")
        ax.set_ylim(bottom=min(floor - 0.02, ax.get_ylim()[0]))
    axes[1][1].text(0.98, floor, " lowest possible with label smoothing ≈ 0.59",
                    transform=axes[1][1].get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7, color="#666")
    axes[0][1].text(4.5, axes[0][1].get_ylim()[1], "margin ramp 0 → 0.20 ",
                    ha="center", va="top", fontsize=7, color="#8a6d00")

    handles, labels = axes[0][0].get_legend_handles_labels()
    star = plt.Line2D([], [], marker="*", ls="none", ms=11, color="#444", label="kept epoch")
    fig.legend(handles + [star], labels + ["kept epoch"], loc="lower center", ncol=7, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Loss per epoch — training and validation objectives shown separately (they are not comparable)")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    matched = Path(args.matched_dir)
    if (matched / "fold1" / "e1_person_model_history.json").exists():
        plot_matched(matched, Path(args.matched_out), args.folds)

    print("\nvalidation loss (identical definition every epoch)")
    print("exp  fold  epoch0   min     at epoch   last")
    for exp, _ in EXPS:
        for k in range(1, args.folds + 1):
            hist, best = load(exp, k, runs_dir)
            vl = [h["val_loss"] for h in hist]
            lo = min(vl)
            print(f"{exp}   {k}    {vl[0]:.4f}  {lo:.4f}   {vl.index(lo):>2d}        {vl[-1]:.4f}   (kept epoch {best})")


if __name__ == "__main__":
    main()
