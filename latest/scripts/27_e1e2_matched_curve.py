"""Same trainer as 25_e1e2_learning_curve.py, with one addition: the
TRAINING photos are also scored each epoch with exactly the rule used for
validation -- clean crops (no augmentation), no angular margin, no label
smoothing, cross-entropy over the full 30-benchmark gallery.

25's train_loss is the optimisation objective (20-class locked head,
margin, label smoothing, augmented crops), so it is not comparable with
its val_loss. The `train_loss_matched` added here is the same objective as
`val_loss` measured on the training people, giving a genuine train/validation
learning curve on one axis.

Nothing about the optimisation changes: the extra scoring pass uses no
random numbers and no gradients, so the augmentation stream, best_epoch and
every reported result stay identical to 25 / 19. Writes to its own
--out-dir so no existing run is overwritten.

Usage:
    CUDA_VISIBLE_DEVICES=0 python 27_e1e2_matched_curve.py --exp e1 --fold 1 \\
        --accept-threshold 0.3 --setup metadata/e1e2_person_folds.json \\
        --out-dir runs/e1e2/learning_curve_matched
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UNFREEZE_PREFIXES = {
    "e0": (),
    "e1": ("BatchNormalization_121.", "Conv_122.", "Conv_124."),
    "e2": ("BatchNormalization_121.", "Conv_122.", "Conv_124.",
           "BatchNormalization_126.", "Gemm_128.", "BatchNormalization_129."),
    "e3": None,
}


def default_rec_model_path() -> str:
    local = Path(__file__).resolve().parent.parent / "models" / "buffalo_l" / "w600k_r50.onnx"
    if local.exists():
        return str(local)
    flat = Path(__file__).resolve().parent.parent / "models" / "w600k_r50.onnx"
    if flat.exists():
        return str(flat)
    candidates = glob.glob(str(Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"))
    if not candidates:
        raise FileNotFoundError("w600k_r50.onnx not found in models/, models/buffalo_l/ or ~/.insightface/models/buffalo_l/")
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


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = (img - 127.5) / 127.5
    return np.transpose(img, (2, 0, 1))


def light_augment(img_bgr, rng):
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


def benchmark_gap(cos, y):
    own = cos[np.arange(len(y)), y]
    other = cos.copy()
    other[np.arange(len(y)), y] = -np.inf
    return own - other.max(axis=1)


class EmbeddingModel(nn.Module):
    def __init__(self, backbone: nn.Module, head_benchmarks: torch.Tensor, unfreeze: str, s: float = 32.0):
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        prefixes = UNFREEZE_PREFIXES[unfreeze]
        self.unfrozen_names = []
        for name, p in self.backbone.named_parameters():
            p.requires_grad = True if prefixes is None else name.startswith(prefixes)
            if p.requires_grad:
                self.unfrozen_names.append(name)
        self.register_buffer("head_bench", F.normalize(head_benchmarks, dim=1))
        self.s = s

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def embed(self, x):
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return F.normalize(out, dim=1)

    def forward(self, x, labels=None, m=0.0):
        cos = self.embed(x) @ self.head_bench.t()
        if labels is None or m == 0.0:
            return cos * self.s
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + m)
        one_hot = F.one_hot(labels, cos.size(1)).float()
        return (cos * (1 - one_hot) + target_logit * one_hot) * self.s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(UNFREEZE_PREFIXES), required=True)
    ap.add_argument("--setup", default="/home/tahmid/metadata/e1e2_person_split.json")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--manifest", default="/home/tahmid/metadata/manifest.csv")
    ap.add_argument("--out-dir", default="/home/tahmid/runs/e1e2/learning_curve_matched")
    ap.add_argument("--onnx-path", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--ramp-epochs", type=int, default=4)
    ap.add_argument("--scale", type=float, default=32.0)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accept-threshold", type=float, default=0.0)
    args = ap.parse_args()

    unfreeze = args.exp
    if args.fold is not None:
        out_path = f"{args.out_dir}/fold{args.fold}/{args.exp}_person_model.pt"
    else:
        out_path = f"{args.out_dir}/{args.exp}_person_model.pt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  exp={args.exp}  unfreeze={unfreeze}", flush=True)

    onnx_path = args.onnx_path or default_rec_model_path()
    from onnx2torch import convert
    import onnx as onnx_mod
    import onnxruntime as ort

    backbone = convert(onnx_mod.load(onnx_path)).to(device)
    backbone.eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_input = np.random.RandomState(0).randn(2, 3, 112, 112).astype(np.float32)
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: test_input})[0]
    with torch.no_grad():
        torch_out = backbone(torch.tensor(test_input, device=device))
        torch_out = (torch_out[0] if isinstance(torch_out, (tuple, list)) else torch_out).cpu().numpy()
    cos_sim = float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
                             for a, b in zip(onnx_out, torch_out)]))
    print(f"Conversion check: mean cosine sim={cos_sim:.6f}", flush=True)
    assert cos_sim >= 0.999, "converted backbone does not match the ONNX model"

    with open(args.setup, encoding="utf-8") as fh:
        setup = json.load(fh)
    if args.fold is not None:
        fold_entry = next(f for f in setup["folds"] if f["fold"] == args.fold)
        train_people, val_people, test_people = fold_entry["train"], fold_entry["val"], fold_entry["test"]
        print(f"fold {args.fold}: test={test_people}  val={val_people}  train={len(train_people)} people",
              flush=True)
    else:
        train_people, val_people, test_people = setup["train"], setup["val"], setup["test"]
    excluded = setup["excluded"]

    bench_file = np.load(setup["benchmarks"], allow_pickle=True)
    assert str(bench_file["kind"]) == "best"
    persons = [str(p) for p in bench_file["persons"]]
    gallery = torch.tensor(bench_file["benchmarks"], dtype=torch.float32)
    pid_to_idx = {p: i for i, p in enumerate(persons)}
    head_benchmarks = gallery[[pid_to_idx[p] for p in train_people]]
    head_idx = {p: i for i, p in enumerate(train_people)}

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    train_rows = [r for r in rows if r["person_id"] in set(train_people)]
    val_rows = [r for r in rows if r["person_id"] in set(val_people)
                and r["crop_path"] not in set(excluded[r["person_id"]])]
    test_rows = [r for r in rows if r["person_id"] in set(test_people)
                 and r["crop_path"] not in set(excluded[r["person_id"]])]
    print(f"train people={len(train_people)} ({len(train_rows)} photos)  "
          f"val people={len(val_people)} ({len(val_rows)} query photos)  "
          f"test people={len(test_people)} ({len(test_rows)} query photos)  "
          f"gallery={len(persons)}", flush=True)

    print("Loading crops into memory...", flush=True)
    train_imgs = [cv2.imread(r["crop_path"]) for r in train_rows]
    assert all(im is not None for im in train_imgs), "unreadable train crop"
    y_train = np.array([head_idx[r["person_id"]] for r in train_rows])
    X_val = torch.tensor(np.stack([preprocess(cv2.imread(r["crop_path"])) for r in val_rows]))
    y_val = np.array([pid_to_idx[r["person_id"]] for r in val_rows])
    X_test = torch.tensor(np.stack([preprocess(cv2.imread(r["crop_path"])) for r in test_rows]))
    y_test = np.array([pid_to_idx[r["person_id"]] for r in test_rows])
    occ_test = np.array([r["occlusion"] for r in test_rows])
    gallery_dev = F.normalize(gallery, dim=1).to(device)

    # clean (un-augmented) view of the training photos, with gallery labels --
    # scored exactly like validation, purely for the learning curve
    X_train_clean = torch.tensor(np.stack([preprocess(im) for im in train_imgs]))
    y_train_gallery = np.array([pid_to_idx[r["person_id"]] for r in train_rows])

    model = EmbeddingModel(backbone, head_benchmarks, unfreeze, s=args.scale).to(device)
    n_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"Unfrozen backbone tensors: {model.unfrozen_names}", flush=True)
    print(f"Unfrozen backbone parameters: {n_params:,}", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if trainable:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    else:
        opt = sched = None

    tau = args.accept_threshold

    def score(X, y):
        """Identical to 25's score(): cosine similarities against the full
        30-benchmark gallery, cross-entropy at scale=args.scale with no
        margin and no label smoothing."""
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), args.eval_batch_size):
                outs.append((model.embed(X[i:i + args.eval_batch_size].to(device)) @ gallery_dev.t()).cpu())
        cos = torch.cat(outs).numpy()
        pred = cos.argmax(axis=1)
        top_score = cos.max(axis=1)
        accepted = top_score >= tau
        raw_correct = pred == y
        val_loss = float(F.cross_entropy(torch.tensor(cos) * args.scale, torch.tensor(y, dtype=torch.long)))
        torch.cuda.empty_cache()
        return {
            "acc": float((raw_correct & accepted).mean()),
            "raw_acc": float(raw_correct.mean()),
            "rejected_correct": float((raw_correct & ~accepted).mean()),
            "misidentified_accepted": float((~raw_correct & accepted).mean()),
            "gap": float(benchmark_gap(cos, y).mean()),
            "loss": val_loss,
        }, pred, cos, accepted

    history = []
    v0, _, _, _ = score(X_val, y_val)
    t0_clean, _, _, _ = score(X_train_clean, y_train_gallery)
    history.append({"epoch": 0, "margin": 0.0, "train_loss": None, "train_acc": None,
                     "train_loss_matched": t0_clean["loss"], "train_acc_matched": t0_clean["raw_acc"],
                     "val_acc": v0["acc"], "val_gap": v0["gap"], "val_loss": v0["loss"]})
    print(f"  epoch   0 (untrained)  val_acc={v0['acc']*100:.2f}%  val_gap={v0['gap']:.4f}  "
          f"val_loss={v0['loss']:.4f}  train_loss_matched={t0_clean['loss']:.4f}", flush=True)
    best_key = (v0["acc"], v0["gap"])
    best_epoch = 0
    best_state = {k: t.detach().clone().cpu() for k, t in model.backbone.state_dict().items()}
    patience_left = args.patience

    rng = np.random.RandomState(args.seed)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        m = 0.0 if epoch <= args.warmup_epochs else \
            args.margin * min(1.0, (epoch - args.warmup_epochs) / args.ramp_epochs)

        order = rng.permutation(len(train_rows))
        total_loss, total_correct = 0.0, 0
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            xb = torch.tensor(np.stack([preprocess(light_augment(train_imgs[j], rng)) for j in b]),
                              device=device)
            yb = torch.tensor(y_train[b], dtype=torch.long, device=device)
            logits = model(xb, yb, m=m)
            loss = F.cross_entropy(logits, yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(b)
            total_correct += int((logits.argmax(dim=1) == yb).sum())
        sched.step()
        torch.cuda.empty_cache()

        v, _, _, _ = score(X_val, y_val)
        tc, _, _, _ = score(X_train_clean, y_train_gallery)
        rec = {"epoch": epoch, "margin": m, "train_loss": total_loss / len(order),
               "train_acc": total_correct / len(order),
               "train_loss_matched": tc["loss"], "train_acc_matched": tc["raw_acc"],
               "val_acc": v["acc"], "val_gap": v["gap"], "val_loss": v["loss"]}
        history.append(rec)
        print(f"  epoch {epoch:3d}  m={m:.3f}  loss={rec['train_loss']:.4f}  "
              f"train_acc={rec['train_acc']*100:.2f}%  val_acc={v['acc']*100:.2f}%  "
              f"val_gap={v['gap']:.4f}  val_loss={v['loss']:.4f}  "
              f"train_loss_matched={tc['loss']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

        key = (v["acc"], v["gap"])
        if key > best_key:
            best_key, best_epoch = key, epoch
            best_state = {k: t.detach().clone().cpu() for k, t in model.backbone.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  early stopping at epoch {epoch}", flush=True)
                break

    print(f"Best epoch: {best_epoch}  val_acc={best_key[0]*100:.2f}%  val_gap={best_key[1]:.4f}", flush=True)
    model.backbone.load_state_dict(best_state)

    with open(out_path.replace(".pt", "_history.json"), "w") as fh:
        json.dump({"best_epoch": best_epoch, "history": history, "args": vars(args)}, fh, indent=2)
    print(f"Saved history: {out_path.replace('.pt', '_history.json')}", flush=True)

    v_test, test_pred, cos_test, accepted_test = score(X_test, y_test)
    raw_correct = (test_pred == y_test)
    correct = raw_correct & accepted_test
    acc = correct.mean()
    lo, hi = wilson_ci(int(correct.sum()), len(correct))
    print(f"\n=== TEST [{args.exp}] unseen people {test_people} (n={len(correct)})  "
          f"accept-threshold={tau} ===", flush=True)
    print(f"Top-1 accuracy (accepted AND correct): {acc*100:.2f}%  (95% CI: {lo*100:.1f}-{hi*100:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
