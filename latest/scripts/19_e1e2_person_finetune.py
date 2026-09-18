"""E1/E2 v2: partial ArcFace-backbone fine-tuning on the person-disjoint
20/5/5 split (18_e1e2_person_split.py). Replaces 17_e1e2_finetune.py's
closed-set version -- no classifier-head variants (a fixed 30-way head
does not make sense when val/test people are never trained on), and no
leakage (train/val/test people are disjoint, so no image can appear in
more than one split).

Two experiments in normal use, chosen with --exp (e3 = full unfreeze also
exists in the code but is not part of the current comparison):
  e1 : unfreeze the last residual block only
  e2 : unfreeze the last residual block + the embedding (fc) layer
       (same scope as B1 / protocol 3's b1)
Both output an embedding matched to a FIXED benchmark matrix (ArcFace
loss, class weights locked to the "one best" benchmarks) -- same recipe
as 13_benchmark_finetune.py / 17_e1e2_finetune.py's e1a/e2a. BatchNorm
running statistics stay frozen throughout regardless of unfreeze scope
(backbone.eval() always; only affine weight/bias tensors ever train).

The locked training head holds ONLY the 20 train people's benchmarks
(they are the only classes the model is pulled toward). Val and test
people, never trained on, are scored against ALL 30 benchmarks: a
correct answer means their query photo landed nearest their own
(excluded-from-training) benchmark rather than any of the other 29,
including the 20 people the model did train on.

--accept-threshold tau (default 0.0 = disabled): a query only counts as
correct if its top cosine score is ALSO >= tau -- same idea as the
demo's UNKNOWN cutoff (scripts/09_server.py). Since every val/test
person genuinely has a true match in the 30-benchmark gallery (there
are no strangers here), a rejection below tau is always a miss
("correct person, but the model wasn't confident enough"), reported
separately from a genuine misidentification.

Model selection: val accepted-correct rate (accuracy under the same
--accept-threshold; not macro-F1, since only 5 of 30 classes are
present in val truth), ties broken by mean val benchmark gap.

--fold N (1-6): reads a {"folds": [...]} setup file built by
20_e1e2_person_folds.py instead of a flat train/val/test one, and trains
on that fold's 20/5/5 split. Outputs go to <out-dir>/fold<N>/ so folds
never collide. Every person is a test person in exactly one fold, so
pooling all 6 folds' results covers all 30 people once each.

Usage:
    CUDA_VISIBLE_DEVICES=0 python 19_e1e2_person_finetune.py --exp e1 --accept-threshold 0.3
    CUDA_VISIBLE_DEVICES=0 python 19_e1e2_person_finetune.py --exp e2 --accept-threshold 0.3
    CUDA_VISIBLE_DEVICES=0 python 19_e1e2_person_finetune.py --exp e1 --accept-threshold 0.3 \\
        --setup metadata/e1e2_person_folds.json --fold 3 --out-dir runs/e1e2/folds
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
    "e0": (),  # nothing unfrozen -- untrained baseline (use with --epochs 0)
    "e1": ("BatchNormalization_121.", "Conv_122.", "Conv_124."),
    "e2": ("BatchNormalization_121.", "Conv_122.", "Conv_124.",
           "BatchNormalization_126.", "Gemm_128.", "BatchNormalization_129."),
    "e3": None,  # full unfreeze -- every backbone parameter trains
}


def default_rec_model_path() -> str:
    local = Path(__file__).resolve().parent.parent / "models" / "buffalo_l" / "w600k_r50.onnx"
    if local.exists():
        return str(local)
    candidates = glob.glob(str(Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"))
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
    ap.add_argument("--fold", type=int, default=None,
                     help="pick fold N (1-6) from a --setup file built by 20_e1e2_person_folds.py "
                          "(a {\"folds\": [...]} file) instead of that file's flat train/val/test")
    ap.add_argument("--manifest", default="/home/tahmid/metadata/manifest.csv")
    ap.add_argument("--out-dir", default="/home/tahmid/runs/e1e2")
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
    ap.add_argument("--accept-threshold", type=float, default=0.0,
                     help="a match only counts as correct if its cosine score is >= this "
                          "(0.0 = no threshold, plain argmax). Applied to model selection AND reporting.")
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

    # ---- 1. convert and verify numerical equivalence ----
    onnx_path = args.onnx_path or default_rec_model_path()
    from onnx2torch import convert
    import onnxruntime as ort

    backbone = convert(onnx_path).to(device)
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

    # ---- 2. setup + data ----
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

    # ---- 3. model ----
    model = EmbeddingModel(backbone, head_benchmarks, unfreeze, s=args.scale).to(device)
    n_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"Unfrozen backbone tensors: {model.unfrozen_names}", flush=True)
    print(f"Unfrozen backbone parameters: {n_params:,}", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if trainable:
        opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    else:
        opt = sched = None  # e.g. --exp e0: nothing unfrozen, no training possible

    tau = args.accept_threshold

    def score(X, y):
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
        torch.cuda.empty_cache()
        return {
            "acc": float((raw_correct & accepted).mean()),  # accepted-correct: the model-selection metric
            "raw_acc": float(raw_correct.mean()),            # plain argmax accuracy, no threshold
            "rejected_correct": float((raw_correct & ~accepted).mean()),  # right person, but below tau
            "misidentified_accepted": float((~raw_correct & accepted).mean()),  # wrong person, above tau
            "gap": float(benchmark_gap(cos, y).mean()),
        }, pred, cos, accepted

    history = []
    v0, _, _, _ = score(X_val, y_val)
    history.append({"epoch": 0, "margin": 0.0, "train_loss": None, "train_acc": None,
                     "val_acc": v0["acc"], "val_gap": v0["gap"]})
    print(f"  epoch   0 (untrained)  val_acc={v0['acc']*100:.2f}%  val_gap={v0['gap']:.4f}", flush=True)
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
        rec = {"epoch": epoch, "margin": m, "train_loss": total_loss / len(order),
               "train_acc": total_correct / len(order), "val_acc": v["acc"], "val_gap": v["gap"]}
        history.append(rec)
        print(f"  epoch {epoch:3d}  m={m:.3f}  loss={rec['train_loss']:.4f}  "
              f"train_acc={rec['train_acc']*100:.2f}%  val_acc={v['acc']*100:.2f}%  "
              f"val_gap={v['gap']:.4f}  ({time.time()-t0:.0f}s)", flush=True)

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

    torch.save({
        "state_dict": best_state,
        "persons": persons,
        "train_people": train_people, "val_people": val_people, "test_people": test_people,
        "exp": args.exp, "unfreeze": unfreeze, "unfrozen_names": model.unfrozen_names,
        "best_epoch": best_epoch, "args": vars(args),
    }, out_path)
    with open(out_path.replace(".pt", "_history.json"), "w") as fh:
        json.dump({"best_epoch": best_epoch, "history": history, "args": vars(args)}, fh, indent=2)
    print(f"Saved: {out_path}", flush=True)

    # ---- 4. final test evaluation (against the full 30-benchmark gallery) ----
    v_test, test_pred, cos_test, accepted_test = score(X_test, y_test)
    raw_correct = (test_pred == y_test)
    correct = raw_correct & accepted_test  # "correct" now means accepted AND right, per --accept-threshold
    acc = correct.mean()
    lo, hi = wilson_ci(int(correct.sum()), len(correct))

    print(f"\n=== TEST [{args.exp}] unseen people {test_people} (n={len(correct)})  "
          f"accept-threshold={tau} ===", flush=True)
    print(f"Top-1 accuracy (accepted AND correct): {acc*100:.2f}%  (95% CI: {lo*100:.1f}-{hi*100:.1f}%)", flush=True)
    print(f"  of which: correct-but-rejected (score < {tau}) = {v_test['rejected_correct']*100:.2f}%  "
          f"misidentified-and-accepted = {v_test['misidentified_accepted']*100:.2f}%", flush=True)
    print(f"Plain argmax accuracy (no threshold): {raw_correct.mean()*100:.2f}%", flush=True)
    print(f"Benchmark gap (mean): {v_test['gap']:.4f}", flush=True)

    results = {"exp": args.exp, "unfreeze": unfreeze, "best_epoch": best_epoch, "accept_threshold": tau,
               "fold": args.fold,
               "train_people": train_people, "val_people": val_people, "test_people": test_people,
               "test_n": int(len(correct)), "test_correct_n": int(correct.sum()),
               "test_acc": float(acc), "test_acc_ci": [lo, hi], "test_gap": v_test["gap"],
               "test_raw_acc": float(raw_correct.mean()),
               "test_rejected_correct": v_test["rejected_correct"],
               "test_rejected_correct_n": int((raw_correct & ~accepted_test).sum()),
               "test_misidentified_accepted": v_test["misidentified_accepted"],
               "test_misidentified_accepted_n": int((~raw_correct & accepted_test).sum()),
               "per_occlusion": {}, "per_person": {}}
    print("\nPer-occlusion accuracy (accepted AND correct):", flush=True)
    for o in sorted(set(occ_test)):
        m_ = occ_test == o
        n_ = int(m_.sum())
        if n_ == 0:
            continue
        n_correct = int(correct[m_].sum())
        a = float(correct[m_].mean())
        lo_o, hi_o = wilson_ci(n_correct, n_)
        n_rej = int((raw_correct[m_] & ~accepted_test[m_]).sum())
        rej = float((raw_correct[m_] & ~accepted_test[m_]).mean())
        print(f"  {o:20s} n={n_:4d}  acc={a*100:5.1f}%  (CI {lo_o*100:.1f}-{hi_o*100:.1f}%)  "
              f"rejected={rej*100:4.1f}%", flush=True)
        results["per_occlusion"][o] = {"n": n_, "correct_n": n_correct, "acc": a,
                                        "rejected_correct_n": n_rej, "rejected_correct": rej}

    print("\nPer-person accuracy (test people, accepted AND correct):", flush=True)
    for p in test_people:
        m_ = np.array([r["person_id"] for r in test_rows]) == p
        n_ = int(m_.sum())
        n_correct = int(correct[m_].sum()) if n_ else 0
        a = float(correct[m_].mean()) if n_ else None
        print(f"  {p}  n={n_:4d}  acc={a*100:5.1f}%" if n_ else f"  {p}  n=0", flush=True)
        results["per_person"][p] = {"n": n_, "correct_n": n_correct, "acc": a}

    results_path = f"{Path(out_path).parent}/{args.exp}_person_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
