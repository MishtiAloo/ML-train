# Occlusion-Robust Face Identification Using a Pretrained ArcFace Model

CSE 4112 Machine Learning Laboratory project (KUET) — identification of 30 enrolled people under masks, sunglasses, scarves, caps and combined occlusions, using a pretrained ArcFace model (`buffalo_l`, iResNet50) and a RetinaFace detector.

**All current work lives in [`latest/`](latest/), which is fully self-contained** — its own crops, models, metadata, checkpoints, results, plots, docs and scripts. The preprocessing scripts needed to rebuild the current data artifacts are also retained in `latest/scripts/`. Older experiments and historical copies were moved — not deleted — to `D:\ML train archive\`. See "Archive" below.

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
├── processed_data/       raw captured images, p01..p30 (frozen dataset -- do not edit)
├── latest/                the final experiment, self-contained
│   ├── crops/             112x112 aligned face crops (5,988 images)
│   ├── models/            flat: w600k_r50.onnx (recognizer), det_10g.onnx (detector)
│   ├── metadata/          manifest.csv, e1e2_person_folds.json
│   ├── runs/e1e2/          benchmarks, 18 checkpoints (E0/E1/E2 x 6 folds), results, confusion matrices
│   │                      (+ demo_enrollments.npz, enrolled_people/ once someone registers in the demo)
│   ├── scripts/           01-03: crop preprocessing, manifest grouping, cached embeddings
│   │                      18-27: person splits, training, evaluation, plotting, learning curves
│   │                      28_demo_server.py: the live phone/browser camera demo (E0-E2), runs on this laptop
│   │                      29_webcam_demo.py: the same demo on this laptop's own webcam, desktop window
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
| Historical copies of dataset-preprocessing code, older model layout, and intermediate metadata such as `preprocess_log.csv` and `rename.txt` | `scripts/`, `models/buffalo_l/`, `metadata/` inside `D:\ML train archive\` |
| Consent form (`consent.pdf`, `consent.tex`) | `archive/docs/` |
| Older material from before that (draft slides/charts, GPU-box script snapshots, rejected photos, an abandoned E0 probe) | `archive/_scratch/`, `archive/box_snapshot/`, `archive/put_asides/`, `archive/runs/eval/e0/` |
| The main project's `crops/` (duplicate of `latest/crops/`) | `archive/crops/` |

The active preprocessing path is now `latest/scripts/{01_preprocess.py,02_split.py,03_embed.py}` plus `latest/scripts/common.py`; these use the model files already stored under `latest/models/`.

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

## Live demo (runs on this laptop, CPU only)

```bash
python latest/scripts/28_demo_server.py --show-names    # add --port 5000 to change the port
```

It prints an `https://<LAN-IP>:5000` address. Open it on a phone on the same Wi-Fi, accept the self-signed certificate warning once (the certificate is generated into `latest/runs/certs/` and reused), then tap **Start**. `https://localhost:5000` works in this machine's browser too. HTTPS is required because browsers only expose the camera on a secure context.

Each frame goes through detection → alignment → embedding → cosine match against the 30 fixed benchmarks, with the same **0.3 acceptance threshold** used in evaluation: below it, the answer is **UNKNOWN**. The last 6 frames' embeddings are averaged before matching. Measured here: about 0.3 s per frame for E0 and 0.5 s for E1/E2.

**Model dropdown:**

| Mode | Weights served | Notes |
|---|---|---|
| E0 | pretrained backbone | no fine-tuning; the fold selector doesn't apply |
| E1 | `runs/e1e2/folds/fold<k>/e1_person_model.pt` | last residual block fine-tuned |
| E2 | `runs/e1e2/folds/fold<k>/e2_person_model.pt` | block + embedding head fine-tuned |

A second dropdown picks the fold for E1/E2, and the page states whether the person currently named was **unseen**, a validation person, or trained on in that fold — fold *k* is the honest setting for its own 5 test people. Switching model or fold clears the frame buffer, because a fine-tuned backbone produces a different embedding space.

**Offline check:** correct on every stored photo tried across E0/E1/E2 and two folds, including masked and sunglasses shots.

**Webcam version (no phone, no browser):**

```bash
python latest/scripts/29_webcam_demo.py --show-names    # --camera 1 to use the other camera
```

This opens a desktop window on the laptop's webcam, with the same model and fold dropdowns, live box and name, and registration (a name box plus **Register New Person**). It reuses `28_demo_server.py`'s code in-process, so results, quality checks and registered people are identical to the phone demo and shared with it. Tested here: recognized p30 live (score 0.66), and registering a new person from the webcam saved them and then recognized them (score 0.86).

**Registering a new person:** press **Start**, then **Register New Person** and type a name. The page keeps sending frames until 5 of them pass the quality gate. A frame passes only if:
- exactly one face is visible and all 5 landmarks are inside the frame in a frontal arrangement;
- the head is upright (eye-line tilt ≤ 15°);
- the detector score is ≥ 0.65;
- the aligned crop is sharp (Laplacian variance ≥ 80);
- the crop is neither too dark nor too bright (mean brightness 55–205).

The highest-quality of the 5 is kept as that person's benchmark, embedded with **E0** (the same model that built the fixed 30-person gallery, so it works with E0, E1 and E2). Registered people are saved permanently and reloaded on every start:
- `latest/runs/e1e2/demo_enrollments.npz` holds the embeddings and names;
- `latest/runs/e1e2/enrolled_people/enrolled_###.jpg` holds the chosen crop.

`person_benchmarks.npz` and the experiment results are never modified. Every threshold is a command-line option (`--enroll-samples`, `--enroll-min-blur`, `--enroll-min-light`, `--enroll-max-light`, `--enroll-max-tilt`, `--enroll-min-det-score`).

**People list and deleting:** both demos have a **People** button (**People...** in the webcam window). It shows the original 30, which cannot be deleted, and everyone registered in the demo, each with a delete option and a confirmation prompt. A deletion takes effect immediately and permanently: the person's row, their saved crop and their entry in `demo_enrollments.npz` are removed, and the file itself is removed once nobody is left. The server refuses deletes for the original 30 even if the request is hand-made. To remove everyone at once, delete the two paths above.

Offline check of registration: p05 was removed from the gallery, registered from 5 of their clean photos, and the server restarted. p05's other photos, many of them occluded, were then recognized under the new name in 8/12 cases with E0 and 11/12 with E1 (fold 1).

## Key design notes

- **The dataset is frozen.** `processed_data/` is a fixed input; the 429 byte-identical duplicates were kept as independent images, not merged.
- **Embeddings are not pre-normalised** by insightface's `get_feat()`. Always L2-normalise before cosine similarity or a threshold.
- **Person-disjoint splits, not image-level ones, for any "unseen person" question.** An earlier version of `latest/`'s experiment split by image and leaked near-duplicate photos across train/test, inflating accuracy to ~100%; splitting by identity (as `latest/` now does) is what makes the result trustworthy.
- **All training runs on the 3090 box's GPU 0**; the laptop can also run `latest/`'s analysis/plotting scripts on CPU when convenient.

## Privacy / consent

Every participant signed a consent form before being photographed (archived at `archive/docs/consent.pdf`). Raw images and crops are restricted to the six project members and are not published outside the group.
