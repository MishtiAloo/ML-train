"""E0/E1/E2 v3 analysis: pooled confusion matrix (all 6 folds) and
per-class / per-occlusion precision, recall, macro-F1, accuracy, for one
--exp at a time. Reuses the checkpoints already saved by
19_e1e2_person_finetune.py --fold N; does not retrain anything.

The "class" is the identified person (30 classes) PLUS a 31st column,
REJECT, for any query whose top cosine fell below --accept-threshold --
so accuracy here matches the accepted-and-correct rate already reported
in <exp>_person_results.json exactly. Precision/recall for a person are
computed the standard way (REJECT counts against recall, never against
another person's precision, since a rejection is not an accusation).

Outputs, under --out-dir (default runs/e1e2/analysis/):
    <exp>_confusion_overall.csv       30 (true) x 31 (pred + REJECT) counts
    <exp>_confusion_<occlusion>.csv   same, restricted to that occlusion's queries
    <exp>_metrics.json                overall + per-person + per-occlusion P/R/F1/accuracy

Usage:
    CUDA_VISIBLE_DEVICES=0 python 21_e1e2_confusion.py --exp e0
    CUDA_VISIBLE_DEVICES=0 python 21_e1e2_confusion.py --exp e1
    CUDA_VISIBLE_DEVICES=0 python 21_e1e2_confusion.py --exp e2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def default_rec_model_path() -> str:
    """ArcFace recognizer. latest/models/ is flat (no buffalo_l/ subfolder);
    still checks models/buffalo_l/ and ~/.insightface for other layouts."""
    root = Path(__file__).resolve().parent.parent / "models"
    for candidate in (root / "w600k_r50.onnx", root / "buffalo_l" / "w600k_r50.onnx"):
        if candidate.exists():
            return str(candidate)
    found = glob.glob(str(Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"))
    if not found:
        raise FileNotFoundError("w600k_r50.onnx not found in models/, models/buffalo_l/ or ~/.insightface/models/buffalo_l/")
    return found[0]


def default_det_model_path() -> str:
    """RetinaFace detector -- not used by this script (it reads already-
    cropped images), but resolved the same way for anything that needs it
    (e.g. a future demo run from latest/)."""
    root = Path(__file__).resolve().parent.parent / "models"
    for candidate in (root / "det_10g.onnx", root / "buffalo_l" / "det_10g.onnx"):
        if candidate.exists():
            return str(candidate)
    found = glob.glob(str(Path.home() / ".insightface" / "models" / "buffalo_l" / "det_10g.onnx"))
    if not found:
        raise FileNotFoundError("det_10g.onnx not found in models/, models/buffalo_l/ or ~/.insightface/models/buffalo_l/")
    return found[0]


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 127.5
    return np.transpose(img, (2, 0, 1))


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


def per_person_prf(cm, persons):
    """cm: (30, 31) true x (30 persons + REJECT). Standard multiclass P/R/F1
    per person; REJECT never contributes to any person's precision."""
    n = len(persons)
    rows = []
    for c in range(n):
        tp = int(cm[c, c])
        support = int(cm[c, :].sum())
        fn = support - tp
        fp = int(cm[:, c].sum()) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"person": persons[c], "support": support, "precision": prec, "recall": rec, "f1": f1})
    return rows


def summarize(cm, persons):
    per_person = per_person_prf(cm, persons)
    n_total = int(cm.sum())
    n_correct = int(np.trace(cm[:, :len(persons)]))
    return {
        "n": n_total,
        "accuracy": n_correct / n_total if n_total else 0.0,
        "macro_precision": float(np.mean([r["precision"] for r in per_person])),
        "macro_recall": float(np.mean([r["recall"] for r in per_person])),
        "macro_f1": float(np.mean([r["f1"] for r in per_person])),
        "per_person": per_person,
    }


def local_path(p: str) -> str:
    """Manifest/setup files store /home/tahmid/... paths written on the GPU
    box. Make them resolve on any machine (including this script copied into
    latest/, which has its own crops/, metadata/, runs/) by taking the path
    from its top-level data folder onward and joining with this script's own
    project root -- same idea as this project's rec-model path lookup."""
    root = Path(__file__).resolve().parent.parent
    p_norm = p.replace("\\", "/")
    for anchor in ("crops/", "runs/", "metadata/", "processed_data/"):
        idx = p_norm.find(anchor)
        if idx != -1:
            return str(root / p_norm[idx:])
    return p


def save_cm_csv(cm, persons, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + persons + ["REJECT"])
        for i, p in enumerate(persons):
            w.writerow([p] + [int(x) for x in cm[i]])


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e0", "e1", "e2"], required=True)
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
    n_persons = len(persons)

    onnx_path = args.onnx_path or default_rec_model_path()
    import onnx
    from onnx2torch import convert
    # pass an already-loaded ModelProto, not a path string: onnx2torch's
    # safe_shape_inference opens a second handle on the same tempfile path
    # when given a path, which Windows refuses (PermissionError) -- passing
    # the loaded model takes the pure in-memory branch instead.
    backbone = convert(onnx.load(onnx_path)).to(device)
    backbone.eval()

    all_y, all_dec, all_occ = [], [], []

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
                raise FileNotFoundError(f"could not read crop: {r['crop_path']} "
                                         f"(resolved to {local_path(r['crop_path'])})")
            imgs.append(preprocess(img))
        X = torch.tensor(np.stack(imgs))
        y = np.array([pid_to_idx[r["person_id"]] for r in test_rows])
        occ = np.array([r["occlusion"] for r in test_rows])

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
        decision = np.where(top_score >= args.accept_threshold, pred, n_persons)  # n_persons index = REJECT

        all_y.append(y)
        all_dec.append(decision)
        all_occ.append(occ)
        print(f"fold {k}: n={len(test_rows)}  acc={(decision == y).mean()*100:.2f}%", flush=True)
        torch.cuda.empty_cache()

    y_all = np.concatenate(all_y)
    dec_all = np.concatenate(all_dec)
    occ_all = np.concatenate(all_occ)

    cm_overall = np.zeros((n_persons, n_persons + 1), dtype=np.int64)
    for t, d in zip(y_all, dec_all):
        cm_overall[t, d] += 1
    overall = summarize(cm_overall, persons)
    save_cm_csv(cm_overall, persons, f"{args.out_dir}/{args.exp}_confusion_overall.csv")

    print(f"\n=== {args.exp} OVERALL (n={overall['n']}) ===")
    print(f"accuracy={overall['accuracy']*100:.2f}%  macro_P={overall['macro_precision']*100:.2f}%  "
          f"macro_R={overall['macro_recall']*100:.2f}%  macro_F1={overall['macro_f1']*100:.2f}%")

    per_occlusion = {}
    for o in sorted(set(occ_all)):
        m = occ_all == o
        cm_o = np.zeros((n_persons, n_persons + 1), dtype=np.int64)
        for t, d in zip(y_all[m], dec_all[m]):
            cm_o[t, d] += 1
        s = summarize(cm_o, persons)
        per_occlusion[o] = s
        save_cm_csv(cm_o, persons, f"{args.out_dir}/{args.exp}_confusion_{o}.csv")
        print(f"  {o:20s} n={s['n']:4d}  acc={s['accuracy']*100:5.1f}%  macroF1={s['macro_f1']*100:5.1f}%")

    metrics = {"exp": args.exp, "accept_threshold": args.accept_threshold, "overall": overall,
               "per_occlusion": {o: {k: v for k, v in s.items() if k != "per_person"}
                                  for o, s in per_occlusion.items()}}
    with open(f"{args.out_dir}/{args.exp}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved: {args.out_dir}/{args.exp}_metrics.json and confusion CSVs", flush=True)


if __name__ == "__main__":
    main()
