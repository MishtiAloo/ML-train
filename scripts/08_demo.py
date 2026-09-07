"""Live/still-image demo: detect -> align -> embed -> match against the
gallery -> UNKNOWN rejection. Per docs/training_plan.md Section 8.

Uses the IDENTICAL align_crop() from common.py that built the training
crops -- do not reimplement alignment here.

Two modes:
  --source <int|url>   live loop (webcam index, or a phone-as-webcam /
                        IP-camera stream URL e.g. from DroidCam/Iriun)
  --image <path>       single still image (the safer capture path per
                        the training plan -- prefer this for the graded
                        demo if at all possible)

Run wherever the camera is attached. If that's not this GPU machine,
copy scripts/common.py, scripts/08_demo.py, runs/gallery.npz, and
metadata/person_id_mapping.txt to that machine, install insightface +
opencv-python there (CPU-only onnxruntime is fine for single-frame
inference), and run with --source 0.

Usage:
    python 08_demo.py --gallery /home/tahmid/runs/gallery.npz --source 0
    python 08_demo.py --gallery /home/tahmid/runs/gallery.npz --image photo.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import align_crop, default_model_root, default_rec_model_path  # noqa: E402


def build_models(use_gpu: bool):
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]

    det = FaceAnalysis(name="buffalo_l", root=default_model_root(),
                        allowed_modules=["detection"], providers=providers)
    det.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))

    rec = get_model(default_rec_model_path(), providers=providers)
    rec.prepare(ctx_id=0 if use_gpu else -1)
    return det, rec


def pick_best_face(faces):
    if not faces:
        return None
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)


def load_gallery(path):
    data = np.load(path, allow_pickle=True)
    return data["centroids"], list(data["persons"]), float(data["threshold"])


def load_display_names(mapping_path):
    """Optional: p## -> 'member/person' for friendlier on-screen labels.
    Off by default (docs/preprocessing_plan.md keeps IDs pseudonymous)."""
    names = {}
    if not mapping_path or not Path(mapping_path).exists():
        return names
    import re
    for line in Path(mapping_path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(p\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|", line)
        if m:
            names[m.group(1).strip().lower()] = f"{m.group(2).strip()}/{m.group(3).strip()}"
    return names


def classify(rec, centroids, persons, threshold, crop_bgr, emb_buffer=None):
    feat = rec.get_feat([crop_bgr])[0]
    feat = feat / (np.linalg.norm(feat) + 1e-9)

    if emb_buffer is not None:
        emb_buffer.append(feat)
        avg = np.mean(emb_buffer, axis=0)
        feat = avg / (np.linalg.norm(avg) + 1e-9)

    sims = centroids @ feat
    best = int(sims.argmax())
    score = float(sims[best])
    if score < threshold:
        return "UNKNOWN", score
    return persons[best], score


def run_still(args, det, rec, centroids, persons, threshold, names):
    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}")
        return
    faces = det.get(img)
    face = pick_best_face(faces)
    if face is None:
        print("No face detected.")
        return
    crop = align_crop(img, face.kps, image_size=112)
    label, score = classify(rec, centroids, persons, threshold, crop)
    display = names.get(label, label) if label != "UNKNOWN" else "UNKNOWN"
    print(f"Prediction: {display}   cosine score: {score:.4f}   threshold: {threshold:.4f}")

    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{display} ({score:.2f})", (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    out_path = str(Path(args.image).with_suffix("")) + "_result.jpg"
    cv2.imwrite(out_path, img)
    print(f"Annotated image saved: {out_path}")


def run_live(args, det, rec, centroids, persons, threshold, names):
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open video source: {source}")
        return

    emb_buffer = deque(maxlen=args.buffer)
    print("Press 'q' to quit.")
    last_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed, stopping.")
            break

        faces = det.get(frame)
        face = pick_best_face(faces)

        if face is not None:
            crop = align_crop(frame, face.kps, image_size=112)
            label, score = classify(rec, centroids, persons, threshold, crop, emb_buffer)
            display = names.get(label, label) if label != "UNKNOWN" else "UNKNOWN"
            x1, y1, x2, y2 = map(int, face.bbox)
            color = (0, 255, 0) if label != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{display} ({score:.2f})", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            emb_buffer.clear()  # face lost -- don't average across a gap
            cv2.putText(frame, "no face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        now = time.time()
        fps = 1.0 / max(1e-6, now - last_t)
        last_t = now
        cv2.putText(frame, f"{fps:.1f} fps", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Occlusion-robust face ID", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="/home/tahmid/runs/gallery.npz")
    ap.add_argument("--source", default=None, help="camera index (e.g. 0) or stream URL")
    ap.add_argument("--image", default=None, help="path to a single still image")
    ap.add_argument("--threshold", type=float, default=None, help="override calibrated threshold")
    ap.add_argument("--buffer", type=int, default=8, help="rolling frame-average size (live mode)")
    ap.add_argument("--names", default=None, help="optional path to person_id_mapping.txt for friendlier labels")
    ap.add_argument("--cpu", action="store_true", help="force CPU (use on a machine with no CUDA GPU)")
    args = ap.parse_args()

    if not args.source and not args.image:
        ap.error("pass either --source (live) or --image (still)")

    centroids, persons, threshold = load_gallery(args.gallery)
    if args.threshold is not None:
        threshold = args.threshold
    names = load_display_names(args.names)

    use_gpu = False
    if not args.cpu and _has_torch():
        import torch
        use_gpu = torch.cuda.is_available()
    det, rec = build_models(use_gpu)

    if args.image:
        run_still(args, det, rec, centroids, persons, threshold, names)
    else:
        run_live(args, det, rec, centroids, persons, threshold, names)


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
