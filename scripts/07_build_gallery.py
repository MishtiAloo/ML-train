"""Build the deployed gallery (one centroid per person) and calibrate the
UNKNOWN-rejection threshold, per docs/training_plan.md Section 8.

Two different data uses, kept deliberately separate:
  - threshold calibration uses ONLY train-built centroids vs held-out val
    scores (no leakage) -- this is what makes the threshold trustworthy.
  - the final deployed gallery uses ALL data (train+val+test), since for
    an actually-deployed tool there is no reason to hold anything back:
    test's job (an unbiased accuracy estimate) is already done, and every
    test image is a genuine photo of a genuinely enrolled person.

Usage:
    python 07_build_gallery.py --embeddings /home/tahmid/metadata/embeddings.npz \
        --out /home/tahmid/runs/gallery.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="/home/tahmid/metadata/embeddings.npz")
    ap.add_argument("--out", default="/home/tahmid/runs/gallery.npz")
    ap.add_argument("--percentile", type=float, default=1.0,
                     help="threshold = this percentile of genuine (correct-match) "
                          "validation scores -- low value keeps false-rejects rare")
    args = ap.parse_args()

    data = np.load(args.embeddings, allow_pickle=True)
    emb = data["embeddings"]
    pid = data["person_ids"]
    split = data["splits"]

    # rec.get_feat() (insightface) does NOT return L2-normalized vectors --
    # normalize per-row now so every score below is a true cosine
    # similarity in [-1, 1], not a magnitude-scaled dot product. This did
    # not affect E0-E3's accuracy numbers (argmax over classes for a fixed
    # row is unchanged by a positive per-row scalar), but it matters a
    # great deal here, where the threshold is an absolute cutoff.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-9, None)

    persons = sorted(set(pid))
    pid_to_idx = {p: i for i, p in enumerate(persons)}

    def build_centroids(mask):
        c = np.zeros((len(persons), emb.shape[1]), dtype=np.float32)
        for p in persons:
            m = mask & (pid == p)
            v = emb[m].mean(axis=0)
            c[pid_to_idx[p]] = v / (np.linalg.norm(v) + 1e-9)
        return c

    # ---- calibration: train-only centroids, scored against held-out val ----
    calib_centroids = build_centroids(split == "train")
    val_mask = split == "val"
    val_emb = emb[val_mask]
    val_pid = pid[val_mask]
    sims = val_emb @ calib_centroids.T
    pred_idx = sims.argmax(axis=1)
    genuine_scores = []
    for i, p in enumerate(val_pid):
        true_idx = pid_to_idx[p]
        if pred_idx[i] == true_idx:  # only correct matches count as "genuine"
            genuine_scores.append(sims[i, true_idx])
    genuine_scores = np.array(genuine_scores)
    threshold = float(np.percentile(genuine_scores, args.percentile))

    print(f"Calibration: {len(genuine_scores)} genuine-match scores from val "
          f"(of {val_mask.sum()} val images)")
    print(f"Genuine score distribution: min={genuine_scores.min():.4f} "
          f"p1={np.percentile(genuine_scores,1):.4f} "
          f"p5={np.percentile(genuine_scores,5):.4f} "
          f"median={np.median(genuine_scores):.4f}")
    print(f"Rejection threshold (tau, p{args.percentile}): {threshold:.4f}")
    print("NOTE: this dataset has no true impostor/unknown examples, so tau is "
          "calibrated only to keep false-rejects of KNOWN people rare. It has "
          "not been validated against real unknown-person rejection -- treat "
          "the UNKNOWN behaviour as a reasonable default, not a proven guarantee.")

    # ---- final deployed gallery: all data ----
    deploy_centroids = build_centroids(np.ones(len(pid), dtype=bool))

    np.savez_compressed(
        args.out,
        centroids=deploy_centroids,
        persons=np.array(persons),
        threshold=threshold,
    )
    with open(args.out.replace(".npz", "_meta.json"), "w") as fh:
        json.dump({
            "persons": persons,
            "threshold": threshold,
            "calibration_percentile": args.percentile,
            "n_calibration_scores": len(genuine_scores),
        }, fh, indent=2)

    print(f"\nSaved gallery: {args.out}  ({len(persons)} persons, "
          f"built from all {len(pid)} images)")


if __name__ == "__main__":
    main()
