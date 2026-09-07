"""E1/E2: train a lightweight ArcFace classification head on top of the
frozen, cached embeddings (see docs/training_plan.md Section 4 and 5).

Backbone stays frozen throughout (no gradient reaches it, and indeed can't --
the recognition model is packaged as ONNX). This trains only the new
512->30 head with an angular-margin loss.

--mode normal : train on 'none'-occlusion train images only (E1, the
                 proposal's baseline)
--mode mixed  : train on all train images regardless of occlusion (E2)

Usage:
    python 05_train_head.py --embeddings /home/tahmid/metadata/embeddings.npz \
        --mode mixed --out /home/tahmid/runs/e2_head.pt
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ArcFaceHead(nn.Module):
    """cos(theta + m) margin head, per docs/training_plan.md Section 4."""

    def __init__(self, in_dim, n_classes, s=32.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, in_dim) * 0.01)
        self.s = s

    def forward(self, x, labels=None, m=0.0):
        # insightface's get_feat() does NOT return L2-normalized vectors --
        # normalize here or `cos` below is an unbounded dot product, not a
        # true cosine, which breaks the arccos margin math (see 07_build_gallery.py).
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        cos = x @ w.t()  # (batch, n_classes)
        if labels is None or m == 0.0:
            return cos * self.s
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + m)
        one_hot = F.one_hot(labels, cos.size(1)).float()
        out = cos * (1 - one_hot) + target_logit * one_hot
        return out * self.s


def macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default="/home/tahmid/metadata/embeddings.npz")
    ap.add_argument("--mode", choices=["normal", "mixed"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.embeddings, allow_pickle=True)
    emb = data["embeddings"]
    pid = data["person_ids"]
    occ = data["occlusions"]
    split = data["splits"]

    persons = sorted(set(pid))
    pid_to_idx = {p: i for i, p in enumerate(persons)}
    y_all = np.array([pid_to_idx[p] for p in pid])

    train_mask = split == "train"
    if args.mode == "normal":
        train_mask = train_mask & (occ == "none")
    val_mask = split == "val"
    test_mask = split == "test"

    X_train = torch.tensor(emb[train_mask], dtype=torch.float32, device=device)
    y_train = torch.tensor(y_all[train_mask], dtype=torch.long, device=device)
    X_val = torch.tensor(emb[val_mask], dtype=torch.float32, device=device)
    y_val_np = y_all[val_mask]
    X_test = torch.tensor(emb[test_mask], dtype=torch.float32, device=device)
    y_test_np = y_all[test_mask]
    occ_test = occ[test_mask]

    print(f"[{args.mode}] train n={len(y_train)}  val n={len(y_val_np)}  test n={len(y_test_np)}")

    head = ArcFaceHead(emb.shape[1], len(persons)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    n = X_train.size(0)
    best_f1 = -1.0
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        head.train()
        m = 0.0 if epoch <= args.warmup_epochs else \
            args.margin * min(1.0, (epoch - args.warmup_epochs) / 5.0)

        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = X_train[idx], y_train[idx]
            logits = head(xb, yb, m=m)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        sched.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(X_val)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        val_f1 = macro_f1(y_val_np, val_pred, list(range(len(persons))))
        val_acc = (val_pred == y_val_np).mean()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  m={m:.3f}  loss={total_loss/n:.4f}  "
                  f"val_acc={val_acc*100:.2f}%  val_macroF1={val_f1*100:.2f}%")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch} (best val macroF1={best_f1*100:.2f}%)")
                break

    head.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "persons": persons, "mode": args.mode}, args.out)
    print(f"Saved head: {args.out}")

    # ---- final test evaluation ----
    head.eval()
    with torch.no_grad():
        test_logits = head(X_test)
        test_pred = test_logits.argmax(dim=1).cpu().numpy()
    correct = (test_pred == y_test_np)
    acc = correct.mean()
    lo, hi = wilson_ci(int(correct.sum()), len(correct))
    f1 = macro_f1(y_test_np, test_pred, list(range(len(persons))))

    print(f"\n=== TEST [{args.mode}] (n={len(correct)}) ===")
    print(f"Top-1 accuracy: {acc*100:.2f}%  (95% CI: {lo*100:.1f}-{hi*100:.1f}%)")
    print(f"Macro-F1: {f1*100:.2f}%")
    print("\nPer-occlusion accuracy:")
    results = {"mode": args.mode, "test_acc": float(acc), "test_macro_f1": f1, "per_occlusion": {}}
    for o in sorted(set(occ_test)):
        m_ = occ_test == o
        n_ = m_.sum()
        if n_ == 0:
            continue
        a = correct[m_].mean()
        lo_o, hi_o = wilson_ci(int(correct[m_].sum()), int(n_))
        print(f"  {o:20s} n={n_:4d}  acc={a*100:5.1f}%  (CI {lo_o*100:.1f}-{hi_o*100:.1f}%)")
        results["per_occlusion"][o] = {"n": int(n_), "acc": float(a)}

    with open(args.out.replace(".pt", "_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
