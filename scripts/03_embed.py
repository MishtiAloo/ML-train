"""Extract a 512-D ArcFace embedding for every crop in the manifest and
cache it to disk. Every later experiment (E0-E4) reads from this cache
instead of re-running the recognition network, so this only needs to run
once (or once whenever the manifest changes).

Usage:
    python 03_embed.py \
        --manifest /home/tahmid/metadata/manifest.csv \
        --out /home/tahmid/metadata/embeddings.npz
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import cv2
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


def build_recognizer():
    from insightface.model_zoo import get_model

    rec = get_model(default_rec_model_path(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    rec.prepare(ctx_id=0)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/tahmid/metadata/manifest.csv")
    ap.add_argument("--out", default="/home/tahmid/metadata/embeddings.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Embedding {len(rows)} crops...", flush=True)

    rec = build_recognizer()

    embeddings = np.zeros((len(rows), 512), dtype=np.float32)
    batch_imgs, batch_idx = [], []

    def flush():
        if not batch_imgs:
            return
        feats = rec.get_feat(batch_imgs)  # NOT L2-normalized -- normalize downstream before any absolute-score use (e.g. a rejection threshold)
        for j, idx in enumerate(batch_idx):
            embeddings[idx] = feats[j]
        batch_imgs.clear()
        batch_idx.clear()

    for i, r in enumerate(rows):
        img = cv2.imread(r["crop_path"])
        if img is None:
            print(f"WARNING: could not read crop {r['crop_path']}", flush=True)
            continue
        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) >= args.batch_size:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)
    flush()

    person_ids = np.array([r["person_id"] for r in rows])
    occlusions = np.array([r["occlusion"] for r in rows])
    lightings = np.array([r["lighting"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    crop_paths = np.array([r["crop_path"] for r in rows])

    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        person_ids=person_ids,
        occlusions=occlusions,
        lightings=lightings,
        splits=splits,
        crop_paths=crop_paths,
    )
    print(f"Saved: {args.out}  shape={embeddings.shape}")


if __name__ == "__main__":
    main()
