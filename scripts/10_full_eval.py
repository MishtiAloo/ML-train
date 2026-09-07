"""Full evaluation: confusion matrix, per-class precision/recall/F1, and
per-occlusion precision/recall/F1, for one of E0/E1/E2/E3. Fills the gap
left by 04_e0_probe.py / 05_train_head.py / 06_e3_finetune.py, which only
printed top-1 accuracy and macro-F1 to stdout with no persisted artifacts.

Usage:
    python 10_full_eval.py --mode e0 --embeddings /home/tahmid/metadata/embeddings.npz
    python 10_full_eval.py --mode e1 --embeddings ... --head /home/tahmid/runs/e1_head.pt
    python 10_full_eval.py --mode e2 --embeddings ... --head /home/tahmid/runs/e2_head.pt
    python 10_full_eval.py --mode e3 --manifest /home/tahmid/metadata/manifest.csv \
        --checkpoint /home/tahmid/runs/e3_model.pt

Writes, under --out-dir (default /home/tahmid/runs/eval/<mode>/):
    metrics.json          top-1, macro-F1, per-class P/R/F1, per-occlusion P/R/F1
    confusion_matrix.csv  raw 30x30 counts, row=true, col=predicted
    confusion_matrix.png  row-normalized heatmap
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np


def default_rec_model_path() -> str:
    """Prefers this repo's local models/buffalo_l/w600k_r50.onnx (see
    README); falls back to insightface's own cache on whichever machine
    this runs on (this project's own PC, or the remote GPU box)."""
    local = Path(__file__).resolve().parent.parent / "models" / "buffalo_l" / "w600k_r50.onnx"
    if local.exists():
        return str(local)
    candidates = (
        glob.glob(str(Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"))
        or glob.glob("/home/tahmid/.insightface/models/buffalo_l/w600k_r50.onnx")
    )
    if not candidates:
        raise FileNotFoundError("w600k_r50.onnx not found in models/buffalo_l/ or ~/.insightface/models/buffalo_l/")
    return candidates[0]


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


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(y_true, y_pred, n_classes):
    rows = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        support = int(np.sum(y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"class": c, "support": support, "precision": prec, "recall": rec, "f1": f1})
    return rows


def group_prf(mask_fn, groups, y_true, y_pred, correct, n_classes=None):
    """Per-group (e.g. per-occlusion) accuracy, plus -- if n_classes is given --
    macro precision/recall/F1 computed on the row-filtered subset: restrict
    to rows belonging to this group, build a conditional confusion matrix
    over just those rows, and average per-class P/R/F1 over the identities
    that actually appear as ground truth in the subset. This is a real,
    well-defined multi-class P/R/F1 -- not just accuracy relabelled -- since
    precision here penalises a class for being over-predicted specifically
    within this occlusion's images, not across the whole test set."""
    out = {}
    for g in groups:
        m = mask_fn(g)
        n = int(m.sum())
        if n == 0:
            continue
        acc = float(correct[m].mean())
        lo, hi = wilson_ci(int(correct[m].sum()), n)
        entry = {"n": n, "accuracy": acc, "ci_lo": lo, "ci_hi": hi}
        if n_classes is not None:
            yt, yp = y_true[m], y_pred[m]
            present = sorted(set(yt.tolist()))
            precs, recs, f1s = [], [], []
            for c in present:
                tp = int(np.sum((yp == c) & (yt == c)))
                fp = int(np.sum((yp == c) & (yt != c)))
                fn = int(np.sum((yp != c) & (yt == c)))
                prec = tp / (tp + fp) if (tp + fp) else 0.0
                rec = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                precs.append(prec); recs.append(rec); f1s.append(f1)
            entry["macro_precision"] = float(np.mean(precs))
            entry["macro_recall"] = float(np.mean(recs))
            entry["macro_f1"] = float(np.mean(f1s))
        out[str(g)] = entry
    return out


def save_confusion_png(cm, labels, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_and_save(mode, persons, occ_test, y_test, test_pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_classes = len(persons)
    correct = (test_pred == y_test)
    acc = float(correct.mean())
    lo, hi = wilson_ci(int(correct.sum()), len(correct))

    cm = confusion_matrix(y_test, test_pred, n_classes)
    per_class = per_class_prf(y_test, test_pred, n_classes)
    macro_p = float(np.mean([r["precision"] for r in per_class]))
    macro_r = float(np.mean([r["recall"] for r in per_class]))
    macro_f1 = float(np.mean([r["f1"] for r in per_class]))

    per_occ = group_prf(lambda o: occ_test == o, sorted(set(occ_test)), y_test, test_pred, correct,
                         n_classes=n_classes)

    # top confusions: (true, pred) off-diagonal pairs sorted by count
    confusions = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusions.append({"true": persons[i], "pred": persons[j], "count": int(cm[i, j])})
    confusions.sort(key=lambda r: -r["count"])

    metrics = {
        "mode": mode,
        "n_test": int(len(y_test)),
        "top1_accuracy": acc,
        "top1_ci": [lo, hi],
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": [
            {"person": persons[r["class"]], **{k: v for k, v in r.items() if k != "class"}}
            for r in per_class
        ],
        "per_occlusion": per_occ,
        "top_confusions": confusions[:20],
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    with open(os.path.join(out_dir, "confusion_matrix.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + persons)
        for i, p in enumerate(persons):
            w.writerow([p] + list(cm[i]))

    save_confusion_png(cm, persons, os.path.join(out_dir, "confusion_matrix.png"),
                        f"{mode.upper()} test confusion matrix (row-normalized)")

    print(f"[{mode}] top1={acc*100:.2f}%  macro-P={macro_p*100:.2f}%  "
          f"macro-R={macro_r*100:.2f}%  macro-F1={macro_f1*100:.2f}%")
    print(f"[{mode}] wrote metrics.json, confusion_matrix.csv, confusion_matrix.png -> {out_dir}")
    if confusions:
        print(f"[{mode}] top confusion: true={confusions[0]['true']} "
              f"predicted={confusions[0]['pred']} count={confusions[0]['count']}")
    else:
        print(f"[{mode}] zero off-diagonal confusions (perfect on test).")


def run_e0(args):
    data = np.load(args.embeddings, allow_pickle=True)
    emb = data["embeddings"].astype(np.float64)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    pid = data["person_ids"]
    occ = data["occlusions"]
    split = data["splits"]

    persons = sorted(set(pid))
    pid_to_idx = {p: i for i, p in enumerate(persons)}
    y_all = np.array([pid_to_idx[p] for p in pid])

    train_mask = split == "train"
    test_mask = split == "test"

    centroids = np.stack([
        emb[train_mask & (pid == p)].mean(axis=0) for p in persons
    ])
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)

    sims = emb[test_mask] @ centroids.T
    test_pred = sims.argmax(axis=1)
    y_test = y_all[test_mask]
    occ_test = occ[test_mask]

    evaluate_and_save("e0", persons, occ_test, y_test, test_pred,
                       os.path.join(args.out_dir, "e0"))


def run_head(args, mode):
    import torch
    import torch.nn.functional as F

    data = np.load(args.embeddings, allow_pickle=True)
    emb = data["embeddings"]
    pid = data["person_ids"]
    occ = data["occlusions"]
    split = data["splits"]

    persons = sorted(set(pid))
    pid_to_idx = {p: i for i, p in enumerate(persons)}
    y_all = np.array([pid_to_idx[p] for p in pid])
    test_mask = split == "test"

    ckpt = torch.load(args.head, map_location="cpu", weights_only=False)
    weight = ckpt["state_dict"]["weight"]  # (n_classes, dim)
    persons_ckpt = ckpt["persons"]
    assert persons_ckpt == persons, "person ordering mismatch between checkpoint and embeddings"

    x = torch.tensor(emb[test_mask], dtype=torch.float32)
    x = F.normalize(x, dim=1)
    w = F.normalize(weight, dim=1)
    with torch.no_grad():
        logits = x @ w.t()
    test_pred = logits.argmax(dim=1).numpy()
    y_test = y_all[test_mask]
    occ_test = occ[test_mask]

    evaluate_and_save(mode, persons, occ_test, y_test, test_pred,
                       os.path.join(args.out_dir, mode))


def run_e3(args):
    import csv as csv_mod
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class PartialFineTuneModel(nn.Module):
        def __init__(self, backbone, n_classes, unfreeze_frac=0.15, s=32.0):
            super().__init__()
            self.backbone = backbone
            self.backbone.eval()
            self.weight = nn.Parameter(torch.randn(n_classes, 512) * 0.01)
            self.s = s

        def embed(self, x):
            out = self.backbone(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return F.normalize(out, dim=1)

        def forward(self, x):
            emb = self.embed(x)
            w = F.normalize(self.weight, dim=1)
            return emb @ w.t() * self.s

    def preprocess(img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        return np.transpose(img, (2, 0, 1))

    def batched_forward(model, x, batch_size=64):
        outs = []
        with torch.no_grad():
            for i in range(0, x.size(0), batch_size):
                outs.append(model(x[i:i + batch_size]).cpu())
        return torch.cat(outs, dim=0)

    onnx_path = args.onnx_path or default_rec_model_path()
    from onnx2torch import convert
    backbone = convert(onnx_path).to(device)
    backbone.eval()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv_mod.DictReader(fh))
    persons = sorted(set(r["person_id"] for r in rows))
    pid_to_idx = {p: i for i, p in enumerate(persons)}
    test_rows = [r for r in rows if r["split"] == "test"]

    model = PartialFineTuneModel(backbone, len(persons)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    assert ckpt["persons"] == persons, "person ordering mismatch between checkpoint and manifest"
    model.eval()

    imgs = [preprocess(cv2.imread(r["crop_path"])) for r in test_rows]
    x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
    y_test = np.array([pid_to_idx[r["person_id"]] for r in test_rows])
    occ_test = np.array([r["occlusion"] for r in test_rows])

    logits = batched_forward(model, x)
    test_pred = logits.argmax(dim=1).numpy()

    evaluate_and_save("e3", persons, occ_test, y_test, test_pred,
                       os.path.join(args.out_dir, "e3"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["e0", "e1", "e2", "e3"], required=True)
    ap.add_argument("--embeddings", default="/home/tahmid/metadata/embeddings.npz")
    ap.add_argument("--head", default=None, help="path to e1_head.pt / e2_head.pt (e1/e2 only)")
    ap.add_argument("--manifest", default="/home/tahmid/metadata/manifest.csv")
    ap.add_argument("--checkpoint", default="/home/tahmid/runs/e3_model.pt")
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument("--out-dir", default="/home/tahmid/runs/eval")
    args = ap.parse_args()

    if args.mode == "e0":
        run_e0(args)
    elif args.mode in ("e1", "e2"):
        run_head(args, args.mode)
    elif args.mode == "e3":
        run_e3(args)


if __name__ == "__main__":
    main()
