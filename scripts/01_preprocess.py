"""Stage B2 of the preprocessing plan: detect, align, crop every image in
processed_data/p##/*.jpg to a 112x112 ArcFace-aligned crop.

Usage (on the training machine):
    python 01_preprocess.py \
        --input /home/tahmid/processed_data \
        --output /home/tahmid/crops \
        --metadata /home/tahmid/metadata \
        --log /home/tahmid/metadata/preprocess_log.csv

Writes:
    <output>/p##/<same filename>.jpg   (112x112 aligned crop, RGB, quality 95)
    <log>                              one row per input image, det_ok etc.

Never silently drops an image: every failure is logged with a reason.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import align_crop, default_model_root, load_rename_map, parse_filename  # noqa: E402


def build_detector(det_thresh: float, det_size: tuple[int, int]):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", root=default_model_root(), allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
    return app


def pick_best_face(faces):
    """Largest bounding box wins (most likely to be the intended subject)."""
    if not faces:
        return None
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)


def detect_with_fallback(img_bgr: np.ndarray, primary_app, loose_app):
    """Try primary detector, then a looser threshold, then a 2x upscale
    at the loose threshold. Returns (face_or_None, stage_used)."""
    faces = primary_app.get(img_bgr)
    face = pick_best_face(faces)
    if face is not None:
        return face, "primary"

    faces = loose_app.get(img_bgr)
    face = pick_best_face(faces)
    if face is not None:
        return face, "loose"

    h, w = img_bgr.shape[:2]
    upscaled = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    faces = loose_app.get(upscaled)
    face = pick_best_face(faces)
    if face is not None:
        face.bbox = face.bbox / 2.0
        face.kps = face.kps / 2.0
        return face, "upscaled"

    return None, "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/tahmid/processed_data")
    ap.add_argument("--output", default="/home/tahmid/crops")
    ap.add_argument("--rename-txt", default="/home/tahmid/metadata/rename.txt")
    ap.add_argument("--log", default="/home/tahmid/metadata/preprocess_log.csv")
    ap.add_argument("--det-size", type=int, default=640)
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rename_map = {}
    rp = Path(args.rename_txt)
    if rp.exists():
        rename_map = load_rename_map(rp)
    else:
        print(f"WARNING: rename.txt not found at {rp}; source_stem will be blank", file=sys.stderr)

    print("Loading detectors...", flush=True)
    primary_app = build_detector(det_thresh=0.5, det_size=(args.det_size, args.det_size))
    loose_app = build_detector(det_thresh=0.3, det_size=(args.det_size, args.det_size))

    person_dirs = sorted(d for d in in_root.iterdir() if d.is_dir())
    all_files = []
    for pdir in person_dirs:
        for f in sorted(pdir.iterdir()):
            if f.suffix.lower() == ".jpg":
                all_files.append(f)

    print(f"Found {len(all_files)} images across {len(person_dirs)} people.", flush=True)

    rows = []
    ok_count = 0
    fail_count = 0
    t0 = time.time()

    for i, f in enumerate(all_files, start=1):
        pid_dir = f.parent.name
        parsed = parse_filename(f.name)
        occ = parsed["occ"] if parsed else ""
        light = parsed["light"] if parsed else ""
        pid = parsed["pid"] if parsed else pid_dir

        raw_bytes = f.read_bytes()
        md5 = hashlib.md5(raw_bytes).hexdigest()

        img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            rows.append(dict(
                image_path=str(f), person_id=pid, occlusion=occ, lighting=light,
                source_stem=rename_map.get(f.name, ""), md5=md5,
                det_ok=False, det_score="", bbox_area_frac="", det_stage="decode_failed",
                crop_path="",
            ))
            fail_count += 1
            continue

        h, w = img.shape[:2]
        face, stage = detect_with_fallback(img, primary_app, loose_app)

        if face is None:
            rows.append(dict(
                image_path=str(f), person_id=pid, occlusion=occ, lighting=light,
                source_stem=rename_map.get(f.name, ""), md5=md5,
                det_ok=False, det_score="", bbox_area_frac="", det_stage=stage,
                crop_path="",
            ))
            fail_count += 1
        else:
            crop = align_crop(img, face.kps, image_size=112)
            out_dir = out_root / pid_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            crop_path = out_dir / f.name
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            x1, y1, x2, y2 = face.bbox
            bbox_area_frac = max(0.0, x2 - x1) * max(0.0, y2 - y1) / (w * h)

            rows.append(dict(
                image_path=str(f), person_id=pid, occlusion=occ, lighting=light,
                source_stem=rename_map.get(f.name, ""), md5=md5,
                det_ok=True, det_score=float(face.det_score), bbox_area_frac=bbox_area_frac,
                det_stage=stage, crop_path=str(crop_path),
            ))
            ok_count += 1

        if i % 200 == 0 or i == len(all_files):
            elapsed = time.time() - t0
            print(f"[{i}/{len(all_files)}] ok={ok_count} fail={fail_count} "
                  f"({elapsed:.0f}s, {i/elapsed:.1f} img/s)", flush=True)

    fieldnames = [
        "image_path", "person_id", "occlusion", "lighting", "source_stem", "md5",
        "det_ok", "det_score", "bbox_area_frac", "det_stage", "crop_path",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {ok_count} crops written, {fail_count} detection failures "
          f"({100*fail_count/len(all_files):.1f}%).")
    print(f"Log: {log_path}")

    if fail_count:
        print("\nFailures by occlusion:")
        from collections import Counter
        c = Counter(r["occlusion"] for r in rows if not r["det_ok"])
        for occ, n in c.most_common():
            print(f"  {occ or '(unparsed)'}: {n}")


if __name__ == "__main__":
    main()
