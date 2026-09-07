"""E0: nearest-centroid probe on frozen pretrained embeddings. Zero
training. Decides whether the project's story is "accuracy" (if this
already scores very high) or "per-occlusion breakdown" (if there's real
headroom). See docs/training_plan.md Section 5.

Usage:
    python 04_e0_probe.py --embeddings /home/tahmid/metadata/embeddings.npz
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="/home/tahmid/metadata/embeddings.npz")
    args = ap.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    emb = data["embeddings"]
    pid = data["person_ids"]
    occ = data["occlusions"]
    split = data["splits"]

    train_mask = split == "train"
    test_mask = split == "test"
    val_mask = split == "val"

    persons = sorted(set(pid))
    print(f"Persons: {len(persons)}   train: {train_mask.sum()}   "
          f"val: {val_mask.sum()}   test: {test_mask.sum()}")

    # ---- build centroids from train only ----
    centroids = np.zeros((len(persons), emb.shape[1]), dtype=np.float32)
    pid_to_idx = {p: i for i, p in enumerate(persons)}
    for p in persons:
        m = train_mask & (pid == p)
        v = emb[m].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        centroids[pid_to_idx[p]] = v

    def evaluate(mask, label):
        X = emb[mask]
        y_true = pid[mask]
        y_occ = occ[mask]
        sims = X @ centroids.T  # cosine, since both sides are L2-normalized
        pred_idx = sims.argmax(axis=1)
        y_pred = np.array([persons[i] for i in pred_idx])
        correct = (y_pred == y_true)

        acc = correct.mean()
        lo, hi = wilson_ci(int(correct.sum()), len(correct))
        print(f"\n=== {label} (n={len(correct)}) ===")
        print(f"Top-1 accuracy: {acc*100:.2f}%  (95% CI: {lo*100:.1f}-{hi*100:.1f}%)")

        # macro-F1 (precision/recall per person, then averaged) -- simple
        # implementation, no sklearn dependency needed
        f1s = []
        for p in persons:
            tp = np.sum((y_pred == p) & (y_true == p))
            fp = np.sum((y_pred == p) & (y_true != p))
            fn = np.sum((y_pred != p) & (y_true == p))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        print(f"Macro-F1: {np.mean(f1s)*100:.2f}%")

        # per-occlusion breakdown
        print("\nPer-occlusion accuracy:")
        for o in sorted(set(y_occ)):
            m = y_occ == o
            n = m.sum()
            if n == 0:
                continue
            a = correct[m].mean()
            lo_o, hi_o = wilson_ci(int(correct[m].sum()), int(n))
            print(f"  {o:20s} n={n:4d}  acc={a*100:5.1f}%  (CI {lo_o*100:.1f}-{hi_o*100:.1f}%)")

        return acc, y_pred, y_true, correct

    acc_val, *_ = evaluate(val_mask, "VALIDATION")
    acc_test, y_pred_test, y_true_test, correct_test = evaluate(test_mask, "TEST")

    # per-person recall on test, flagged if notably low
    print("\n=== Per-person recall (test) ===")
    low = []
    for p in persons:
        m = y_true_test == p
        if m.sum() == 0:
            print(f"  {p}: NO TEST IMAGES")
            continue
        r = correct_test[m].mean()
        flag = "  <-- LOW" if r < 0.7 else ""
        if flag:
            low.append(p)
        print(f"  {p}: n={m.sum():3d}  recall={r*100:5.1f}%{flag}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"E0 (nearest-centroid, no training) test top-1: {acc_test*100:.2f}%")
    if acc_test > 0.97:
        print("-> Accuracy is likely saturated already. The report's main")
        print("   finding should be the per-occlusion breakdown above, not")
        print("   headline accuracy improvements from E1/E2/E3.")
    elif acc_test > 0.85:
        print("-> Solid baseline with real headroom. E1 vs E2 vs E3 should")
        print("   show a meaningful, reportable difference.")
    else:
        print("-> Substantial headroom. Occlusion is genuinely breaking the")
        print("   pretrained embeddings for this dataset -- E2 (mixed-")
        print("   condition training) is expected to matter a lot.")
    if low:
        print(f"\nPersons with low recall, worth a closer look: {low}")


if __name__ == "__main__":
    main()
