"""Per-image prediction dump for the final technical report's sample-image
sections ("sample crops", "ArcFace-failed crops"). None of the existing
analysis outputs (confusion CSVs, metrics.json) keep image-level detail --
they're already aggregated. This re-runs inference exactly like
21_e1e2_confusion.py (same preprocessing, same benchmark gallery, same
--accept-threshold rule) but writes one row per query photo instead of a
confusion matrix, so specific failure/success examples can be picked by
score.

Usage:
    python 23_dump_predictions.py --exp e0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from importlib import import_module

_m21 = import_module("21_e1e2_confusion")
default_rec_model_path = _m21.default_rec_model_path
preprocess = _m21.preprocess
local_path = _m21.local_path


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e0", "e1", "e2"], default="e0")
    ap.add_argument("--setup", default=str(root / "metadata" / "e1e2_person_folds.json"))
    ap.add_argument("--manifest", default=str(root / "metadata" / "manifest.csv"))
    ap.add_argument("--ckpt-dir", default=str(root / "runs" / "e1e2" / "folds"))
    ap.add_argument("--out-dir", default=str(root / "runs" / "e1e2" / "analysis"))
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument("--accept-threshold", type=float, default=0.3)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.setup, encoding="utf-8") as fh:
        setup = json.load(fh)
    excluded = setup["excluded"]
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    bench_file = np.load(local_path(setup["benchmarks"]), allow_pickle=True)
    persons = [str(p) for p in bench_file["persons"]]
    pid_to_idx = {p: i for i, p in enumerate(persons)}
    gallery = torch.tensor(bench_file["benchmarks"], dtype=torch.float32)
    gallery_dev = F.normalize(gallery, dim=1).to(device)

    onnx_path = args.onnx_path or default_rec_model_path()
    import onnx
    from onnx2torch import convert
    backbone = convert(onnx.load(onnx_path)).to(device)
    backbone.eval()

    out_path = f"{args.out_dir}/{args.exp}_predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as out_fh:
        w = csv.writer(out_fh)
        w.writerow(["fold", "crop_path", "person_id", "occlusion", "lighting", "det_score",
                    "pred_person", "score", "outcome"])

        for fold in setup["folds"]:
            k = fold["fold"]
            test_rows = [r for r in rows if r["person_id"] in set(fold["test"])
                         and r["crop_path"] not in set(excluded[r["person_id"]])]

            ckpt_path = f"{args.ckpt_dir}/fold{k}/{args.exp}_person_model.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert ckpt["persons"] == persons and ckpt["test_people"] == fold["test"]
            backbone.load_state_dict(ckpt["state_dict"])
            backbone.eval()

            imgs = []
            for r in test_rows:
                img = cv2.imread(local_path(r["crop_path"]))
                if img is None:
                    raise FileNotFoundError(f"could not read crop: {r['crop_path']}")
                imgs.append(preprocess(img))
            X = torch.tensor(np.stack(imgs))

            outs = []
            with torch.no_grad():
                for i in range(0, len(X), args.eval_batch_size):
                    xb = X[i:i + args.eval_batch_size].to(device)
                    emb = backbone(xb)
                    emb = emb[0] if isinstance(emb, (tuple, list)) else emb
                    emb = F.normalize(emb, dim=1)
                    outs.append((emb @ gallery_dev.t()).cpu())
            cos = torch.cat(outs).numpy()
            pred = cos.argmax(axis=1)
            top_score = cos.max(axis=1)

            for r, p, s in zip(test_rows, pred, top_score):
                true_idx = pid_to_idx[r["person_id"]]
                accepted = s >= args.accept_threshold
                if accepted and p == true_idx:
                    outcome = "correct"
                elif accepted and p != true_idx:
                    outcome = "misidentified"
                else:
                    outcome = "rejected"
                w.writerow([k, r["crop_path"], r["person_id"], r["occlusion"], r["lighting"],
                            r["det_score"], persons[p], f"{s:.6f}", outcome])
            print(f"fold {k}: dumped {len(test_rows)} rows", flush=True)
            torch.cuda.empty_cache() if device == "cuda" else None

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
