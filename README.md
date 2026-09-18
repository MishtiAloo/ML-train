# Occlusion-Robust Face Identification Using a Pretrained ArcFace Model

CSE 4112 Machine Learning Laboratory project (KUET) — identification of 30 enrolled people under masks, sunglasses, scarves, caps and combined occlusions, using a pretrained ArcFace model (`buffalo_l`, iResNet50) and a RetinaFace detector.

**All current work lives in [`latest/`](latest/), which is fully self-contained** — its own crops, models, metadata, checkpoints, results, plots, docs and scripts. Everything this project explored before it (trained-classifier heads, the benchmark-embedding protocol, the unseen-people protocol, the live phone demo, earlier versions of this same experiment, and the dataset-preprocessing tooling) has been moved — not deleted — to `D:\ML train archive\`. See "Archive" below.

## The final experiment (`latest/`)

Partial ArcFace-backbone fine-tuning, evaluated on people the model never trained on, using a 6-fold **person-disjoint** cross-validation (20 train / 5 val / 5 test people per fold, every one of the 30 people tested exactly once) and a 0.3 cosine acceptance threshold (a match only counts as correct if it's the right person **and** confident enough — same rule as a real enrollment system's UNKNOWN cutoff).

| Experiment | Backbone unfrozen | Pooled accuracy (accepted & correct, n=5,818) |
|---|---|---|
| E0 | none (untrained baseline) | 84.98% |
| **E1** | last residual block only | **86.18%** |
| E2 | last residual block + embedding (fc) layer | 84.72% |

**E1 is the only configuration that reliably beats doing nothing** — it improved on the untrained baseline in every fold; E2 did not.

| Document | Contents |
|---|---|
| [`latest/docs/final_technical_report.md`](latest/docs/final_technical_report.md) | **Full technical report**: dataset/sample galleries, data-splitting and benchmark-selection diagrams, verified block-level ArcFace/RetinaFace architectures, E1/E2 training mechanics (loss, hyperparameters, backprop, epoch stopping), and the complete result set, all in one self-contained document |
| [`latest/docs/report.md`](latest/docs/report.md) | Shorter write-up: objective, method, results, conclusion |
| [`latest/docs/confusion_matrices.md`](latest/docs/confusion_matrices.md) | Per-person and per-occlusion precision/recall/macro-F1/accuracy, confusion-matrix heatmaps |
| [`latest/docs/epoch_curves.md`](latest/docs/epoch_curves.md) | Full epoch-by-epoch training/validation curve, every fold and experiment |

## Repository layout

```
D:\ML train\
├── processed_data/       raw captured images, p01..p30 (frozen dataset -- do not edit; unrelated tooling archived)
├── latest/                the final experiment, self-contained
│   ├── crops/             112x112 aligned face crops (5,988 images)
│   ├── models/            flat: w600k_r50.onnx (recognizer), det_10g.onnx (detector)
│   ├── metadata/          manifest.csv, e1e2_person_folds.json
│   ├── runs/e1e2/          benchmarks, 18 checkpoints (E0/E1/E2 x 6 folds), results, confusion matrices
│   ├── scripts/           18-22: fold setup, training, confusion analysis, plotting
│   └── docs/              report.md, confusion_matrices.md, epoch_curves.md, plots/
└── requirements-gpu-box.txt   exact package set of the training environment (3090 box)
```

That's the whole local project now. `processed_data/` is the only other thing kept at the top level, per an explicit decision to never move the raw dataset regardless of any other cleanup.

### Archive

Everything else was moved (not deleted) to **`D:\ML train archive\`**, keeping original relative paths:

| What | Where |
|---|---|
| Trained-classifier heads (E1–E3), benchmark-embedding protocol (B0/B1), unseen-people protocol (6 folds, UNKNOWN calibration), the live phone demo, and all their scripts/checkpoints/docs | `archive/scripts/`, `archive/runs/`, `archive/docs/`, `archive/metadata/` |
| An earlier, leaky (image-level split) version of this same E1/E2 experiment, and its single-fold (not 6-fold) successor | `archive/scripts/16_e1e2_split.py`, `17_e1e2_finetune.py`; `archive/latest_single_split_03/` |
| Dataset-preprocessing tooling: `common.py`, `01_preprocess.py`, `02_split.py`, `03_embed.py`, `models/buffalo_l/*.onnx`, `metadata/manifest.csv`, `embeddings.npz`, `preprocess_log.csv`, `person_id_mapping.txt`, `rename.txt` | `archive/scripts/`, `archive/models/buffalo_l/`, `archive/metadata/` |
| Consent form (`consent.pdf`, `consent.tex`) | `archive/docs/` |
| Older material from before that (draft slides/charts, GPU-box script snapshots, rejected photos, an abandoned E0 probe) | `archive/_scratch/`, `archive/box_snapshot/`, `archive/put_asides/`, `archive/runs/eval/e0/` |
| The main project's `crops/` (duplicate of `latest/crops/`) | `archive/crops/` |

Copy anything back to its original path to restore it — nothing was deleted. If `processed_data/` ever needs reprocessing (new raw photos, a re-crop), restore `archive/scripts/{common.py,01_preprocess.py,02_split.py,03_embed.py}` and `archive/models/buffalo_l/` first.

## Setup

- **3090 GPU box** (`tahmid@100.73.147.96`) — the heavier half of `latest/`'s training. Never train on this laptop.
- **Laptop** — `latest/`'s analysis/plotting scripts also run here on CPU (slower, but confirmed working).

```bash
# GPU box: exact environment (Python 3.12.3, torch 2.6.0+cu124)
pip install -r requirements-gpu-box.txt --extra-index-url https://download.pytorch.org/whl/cu124
# laptop (for latest/'s scripts)
pip install opencv-python-headless numpy torch onnx onnx2torch onnxruntime matplotlib
```

`latest/models/` is flat (`w600k_r50.onnx`, `det_10g.onnx`, no `buffalo_l/` subfolder); `latest/scripts/*.py` resolve paths relative to their own project root, so they work unchanged whether run on the box or here.

> **onnxruntime-gpu note:** installing plain `onnxruntime` after `onnxruntime-gpu` silently overwrites it with a CPU-only build. Install `onnxruntime-gpu` last, with `--no-deps`, pinned to your CUDA runtime (`onnxruntime-gpu==1.20.2` for CUDA 12.4).

> **Windows + onnx2torch:** passing a file path to `onnx2torch.convert()` triggers a Windows-only tempfile bug (`PermissionError`, reopening a handle it still holds). `latest/scripts/21_e1e2_confusion.py` works around this by passing a pre-loaded `onnx.load()` model instead — reuse that pattern in anything new.

## Key design notes

- **The dataset is frozen.** `processed_data/` is a fixed input; the 429 byte-identical duplicates were kept as independent images, not merged.
- **Embeddings are not pre-normalised** by insightface's `get_feat()`. Always L2-normalise before cosine similarity or a threshold.
- **Person-disjoint splits, not image-level ones, for any "unseen person" question.** An earlier version of `latest/`'s experiment split by image and leaked near-duplicate photos across train/test, inflating accuracy to ~100%; splitting by identity (as `latest/` now does) is what makes the result trustworthy.
- **All training runs on the 3090 box's GPU 0**; the laptop can also run `latest/`'s analysis/plotting scripts on CPU when convenient.

## Privacy / consent

Every participant signed a consent form before being photographed (archived at `archive/docs/consent.pdf`). Raw images and crops are restricted to the six project members and are not published outside the group.
