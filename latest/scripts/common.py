"""Shared constants and the alignment function used by every stage.

IMPORTANT: align_crop() here must be the exact function imported by the
live demo (05_demo.py). Never reimplement alignment separately.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

OCCLUSIONS = [
    "none", "cap", "mask", "scarf",
    "clearglass", "sunglass", "mask_clearglass", "mask_sunglass",
]
LIGHTINGS = ["regular", "low"]

# p##_<occlusion>_<lighting>_<seq>.jpg  (occlusion may itself contain "_")
FILENAME_RE = re.compile(
    r"^(?P<pid>p\d{2})_(?P<occ>none|mask_clearglass|mask_sunglass|clearglass|sunglass|mask|scarf|cap)"
    r"_(?P<light>regular|low)_(?P<seq>\d+)\.jpg$",
    re.IGNORECASE,
)


def parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    d["occ"] = d["occ"].lower()
    d["light"] = d["light"].lower()
    d["pid"] = d["pid"].lower()
    return d


def align_crop(bgr_image: np.ndarray, kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Align a face to the standard ArcFace 112x112 template.

    kps: 5x2 array of landmarks in order [left_eye, right_eye, nose,
    left_mouth, right_mouth], as returned by insightface's detector
    (face.kps).
    """
    from insightface.utils import face_align

    return face_align.norm_crop(bgr_image, landmark=kps, image_size=image_size)


def default_model_root() -> str:
    """Root directory to hand insightface's FaceAnalysis(root=...): it
    expects <root>/models/<name>/*.onnx. Prefers the project-local copy
    at models/buffalo_l/ (see README) so this repo is self-contained and
    does not depend on insightface's auto-download cache; falls back to
    the default ~/.insightface cache if the local copy isn't present."""
    project_root = Path(__file__).resolve().parent.parent
    if (project_root / "models" / "buffalo_l").exists():
        return str(project_root)
    return str(Path.home() / ".insightface")


def default_rec_model_path() -> str:
    """Path to the ArcFace recognition model (w600k_r50.onnx). Prefers the
    project-local copy at models/buffalo_l/, falls back to insightface's
    auto-download cache."""
    local = Path(__file__).resolve().parent.parent / "models" / "buffalo_l" / "w600k_r50.onnx"
    if local.exists():
        return str(local)
    import glob
    candidates = glob.glob(str(Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"))
    if not candidates:
        raise FileNotFoundError(
            "w600k_r50.onnx not found under models/buffalo_l/ or ~/.insightface/models/buffalo_l/ "
            "-- run any insightface script once to auto-download it, or copy it into models/buffalo_l/."
        )
    return candidates[0]


def load_rename_map(rename_txt: Path) -> dict[str, str]:
    """new-filename (basename) -> original-filename (basename), from
    metadata/rename.txt (format: 'orig\\path | new\\path' per line)."""
    mapping: dict[str, str] = {}
    for line in rename_txt.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        left, _, right = line.partition("|")
        left, right = left.strip(), right.strip()
        if "\\" not in right:
            continue
        orig_base = left.split("\\")[-1]
        new_base = right.split("\\")[-1]
        mapping[new_base] = orig_base
    return mapping
