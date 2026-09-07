"""E3: partial fine-tuning of the ArcFace backbone (docs/training_plan.md
Section 4, Stage 2).

buffalo_l ships as ONNX (inference-only), so there is no ready-made
trainable PyTorch checkpoint. Rather than sourcing a separate ArcFace
PyTorch weight file (which would NOT be guaranteed identical to the
weights E0/E1/E2 were measured against, making the comparison unfair),
this converts the exact same ONNX graph to a trainable torch.nn.Module
via onnx2torch. Step 1 below verifies the conversion is numerically
faithful before anything is trained.

BatchNorm stays in eval mode throughout, per the training plan -- running
stats are never updated, only the unfrozen conv/fc weights receive
gradients.

Usage:
    python 06_e3_finetune.py --manifest /home/tahmid/metadata/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_rec_model_path() -> str:
    """Prefers this repo's local models/buffalo_l/w600k_r50.onnx (see
    README); falls back to insightface's own cache on whichever machine
    this runs on (this project's own PC, or the remote GPU box)."""
    from pathlib import Path
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


def batched_forward(model, X, batch_size=64):
    """Run inference in chunks -- forwarding the whole val/test set at
    once through the onnx2torch-converted graph OOMs even under
    no_grad(), since the converted graph is less memory-efficient than
    a native torchvision implementation."""
    outs = []
    with torch.no_grad():
        for i in range(0, X.size(0), batch_size):
            outs.append(model(X[i:i + batch_size]).cpu())
    return torch.cat(outs, dim=0)


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def preprocess(img_bgr):
    """Mirrors insightface's ArcFaceONNX preprocessing exactly:
    BGR->RGB, (x-127.5)/127.5, CHW."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 127.5
    return np.transpose(img, (2, 0, 1))  # CHW


def light_augment(img_bgr, rng):
    """The 'live-capture' augmentation group from training_plan.md
    Section 4, applied at image level (train split only)."""
    img = img_bgr.copy()
    h, w = img.shape[:2]

    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    if rng.random() < 0.3:
        s = rng.randint(64, 112)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.15:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.2:
        noise = rng.normal(0, rng.uniform(2, 6), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        q = rng.randint(50, 95)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.2:
        alpha = rng.uniform(0.8, 1.2)
        beta = rng.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img


class PartialFineTuneModel(nn.Module):
    def __init__(self, backbone: nn.Module, n_classes: int, unfreeze_frac: float = 0.15, s: float = 32.0):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()  # BN stays in eval mode for the whole run

        params = list(self.backbone.named_parameters())
        n_unfreeze = max(1, int(len(params) * unfreeze_frac))
        unfrozen_names = set(name for name, _ in params[-n_unfreeze:])
        for name, p in params:
            p.requires_grad = name in unfrozen_names
        self.unfrozen_names = unfrozen_names

        self.weight = nn.Parameter(torch.randn(n_classes, 512) * 0.01)
        self.s = s

    def train(self, mode: bool = True):
        # override so backbone always stays in eval() (frozen BN stats),
        # even though its unfrozen conv weights still receive gradients
        super().train(mode)
        self.backbone.eval()
        return self

    def embed(self, x):
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return F.normalize(out, dim=1)

    def forward(self, x, labels=None, m=0.0):
        emb = self.embed(x)
        w = F.normalize(self.weight, dim=1)
        cos = emb @ w.t()
        if labels is None or m == 0.0:
            return cos * self.s
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + m)
        one_hot = F.one_hot(labels, cos.size(1)).float()
        out = cos * (1 - one_hot) + target_logit * one_hot
        return out * self.s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/tahmid/metadata/manifest.csv")
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument("--out", default="/home/tahmid/runs/e3_model.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    onnx_path = args.onnx_path or default_rec_model_path()

    # ---- 1. convert and verify numerical equivalence ----
    print(f"Converting {onnx_path} via onnx2torch...", flush=True)
    from onnx2torch import convert
    backbone = convert(onnx_path).to(device)
    backbone.eval()

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng_check = np.random.RandomState(0)
    test_input = rng_check.randn(2, 3, 112, 112).astype(np.float32)

    onnx_out = sess.run(None, {input_name: test_input})[0]
    with torch.no_grad():
        torch_out = backbone(torch.tensor(test_input, device=device))
        if isinstance(torch_out, (tuple, list)):
            torch_out = torch_out[0]
        torch_out = torch_out.cpu().numpy()

    max_diff = np.abs(onnx_out - torch_out).max()
    cos_sim = float(np.mean([
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
        for a, b in zip(onnx_out, torch_out)
    ]))
    print(f"Conversion check: max abs diff={max_diff:.6f}  mean cosine sim={cos_sim:.6f}")
    if cos_sim < 0.999:
        print("*** WARNING: converted model does not match ONNX output closely. "
              "E3 results below would not be a fair comparison to E0/E1/E2 -- "
              "treat with caution. ***")
    else:
        print("OK: converted backbone matches the ONNX model used for E0/E1/E2.")

    # ---- 2. data ----
    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    persons = sorted(set(r["person_id"] for r in rows))
    pid_to_idx = {p: i for i, p in enumerate(persons)}

    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"train={len(train_rows)}  val={len(val_rows)}  test={len(test_rows)}")

    def load_batch(rows_subset, augment, rng):
        imgs = []
        for r in rows_subset:
            img = cv2.imread(r["crop_path"])
            if augment:
                img = light_augment(img, rng)
            imgs.append(preprocess(img))
        x = torch.tensor(np.stack(imgs), dtype=torch.float32, device=device)
        y = torch.tensor([pid_to_idx[r["person_id"]] for r in rows_subset], dtype=torch.long, device=device)
        return x, y

    # pre-load val/test once (no augmentation)
    print("Loading val/test crops...", flush=True)
    X_val, y_val = load_batch(val_rows, augment=False, rng=None)
    X_test, y_test = load_batch(test_rows, augment=False, rng=None)
    occ_test = np.array([r["occlusion"] for r in test_rows])

    # ---- 3. model ----
    model = PartialFineTuneModel(backbone, len(persons), unfreeze_frac=0.15).to(device)
    n_trainable_backbone = sum(1 for n_, p in model.backbone.named_parameters() if p.requires_grad)
    n_total_backbone = sum(1 for _ in model.backbone.named_parameters())
    print(f"Backbone: {n_trainable_backbone}/{n_total_backbone} parameter tensors unfrozen "
          f"(last {n_trainable_backbone}, approximating the final residual stage)")

    head_params = [model.weight]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr_head, "weight_decay": 5e-4},
        {"params": backbone_params, "lr": args.lr_backbone, "weight_decay": 1e-4},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    rng = np.random.RandomState(args.seed)
    best_f1 = -1.0
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        m = 0.0 if epoch <= args.warmup_epochs else \
            args.margin * min(1.0, (epoch - args.warmup_epochs) / 4.0)

        order = list(range(len(train_rows)))
        rng.shuffle(order)
        total_loss, n_seen = 0.0, 0

        for i in range(0, len(order), args.batch_size):
            batch_idx = order[i:i + args.batch_size]
            batch_rows = [train_rows[j] for j in batch_idx]
            xb, yb = load_batch(batch_rows, augment=True, rng=rng)

            logits = model(xb, yb, m=m)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch_idx)
            n_seen += len(batch_idx)
        sched.step()

        model.eval()
        val_logits = batched_forward(model, X_val)
        val_pred = val_logits.argmax(dim=1).numpy()
        torch.cuda.empty_cache()
        y_val_np = y_val.cpu().numpy()
        val_f1 = macro_f1(y_val_np, val_pred, len(persons))
        val_acc = (val_pred == y_val_np).mean()

        print(f"  epoch {epoch:3d}  m={m:.3f}  loss={total_loss/n_seen:.4f}  "
              f"val_acc={val_acc*100:.2f}%  val_macroF1={val_f1*100:.2f}%", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch} (best val macroF1={best_f1*100:.2f}%)")
                break

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "persons": persons}, args.out)
    print(f"Saved: {args.out}")

    # ---- 4. final test evaluation ----
    model.eval()
    test_logits = batched_forward(model, X_test)
    test_pred = test_logits.argmax(dim=1).numpy()
    y_test_np = y_test.cpu().numpy()
    correct = (test_pred == y_test_np)
    acc = correct.mean()
    lo, hi = wilson_ci(int(correct.sum()), len(correct))
    f1 = macro_f1(y_test_np, test_pred, len(persons))

    print(f"\n=== TEST [E3 partial fine-tune] (n={len(correct)}) ===")
    print(f"Top-1 accuracy: {acc*100:.2f}%  (95% CI: {lo*100:.1f}-{hi*100:.1f}%)")
    print(f"Macro-F1: {f1*100:.2f}%")
    print("\nPer-occlusion accuracy:")
    for o in sorted(set(occ_test)):
        m_ = occ_test == o
        n_ = m_.sum()
        if n_ == 0:
            continue
        a = correct[m_].mean()
        lo_o, hi_o = wilson_ci(int(correct[m_].sum()), int(n_))
        print(f"  {o:20s} n={n_:4d}  acc={a*100:5.1f}%  (CI {lo_o*100:.1f}-{hi_o*100:.1f}%)")


if __name__ == "__main__":
    main()
