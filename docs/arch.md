# Architecture

Four diagrams: manual data curation, the constrained train/val/test split,
the training pipeline for each experiment (E1-E3), and the inference
pipeline for each experiment as used by the live demo. File/folder names
match the actual repo layout (see `README.md`).

---

## 1. Data labeling, cleaning, adding, removing (manual)

Everything here is a human decision, made once, before any script runs.
Once `processed_data/` is accepted, it is treated as **frozen** -- no
further adding, removing, or relabeling.

```mermaid
flowchart TD
    A[Phone capture by 6 team members<br/>5 people each, 8 occlusions x 2 lightings] --> B[Consent form signed<br/>docs/consent.pdf]
    B --> C[Raw photos, original camera filenames]
    C --> D{Manual quality review}
    D -->|reject: blurry, too dark,<br/>fully covered, mis-cropped,<br/>accidental duplicate shot| E["put_asides/&lt;member&gt;/&lt;person&gt;/<br/>(kept, with reason -- never silently deleted)"]
    D -->|accept| F[Rename to standard scheme<br/>p##_occlusion_lighting_seq.jpg]
    F --> G[metadata/rename.txt<br/>original filename to new filename map]
    F --> H["processed_data/p01..p30/<br/>FROZEN dataset (6,000 images)"]
    C --> I[metadata/person_id_mapping.txt<br/>p## to team member to real person]
    H --> J[metadata/initial_count.txt<br/>metadata/count_after_trim.txt<br/>manual bookkeeping]

    style H fill:#1f4e7d,color:#fff
    style E fill:#5a2020,color:#fff
```

**Notes:**
- 429 images turned out to be byte-identical duplicates (repeat shutter presses). These were **kept**, not merged or removed -- see diagram 2 for how splitting handles that honestly.
- `person_id_mapping.txt` and `rename.txt` are identity-linking metadata (real names, original filenames) -- kept out of git (see `.gitignore`), unlike everything downstream of them.

---

## 2. Data splitting strategy (constrained)

Runs once (`scripts/02_split.py`), after automated detection has produced
`preprocess_log.csv` (see diagram 3 for that step). Two hard constraints
drive the design: no near-duplicate leakage across splits, and every
identity present in all three splits.

```mermaid
flowchart TD
    A[metadata/preprocess_log.csv<br/>md5, det_ok, crop_path per image] --> B[02_split.py]
    B --> C[Group near-duplicates<br/>union-find]
    C --> C1["same source stem<br/>(e.g. a darkened low-light copy)"]
    C --> C2["same person + occlusion,<br/>captured within an 8s burst window"]
    C --> D["429 byte-identical duplicates<br/>kept as INDEPENDENT images<br/>(not grouped -- a deliberate, disclosed choice)"]
    C1 --> E[Deficit-based greedy allocator]
    C2 --> E
    D --> E
    E --> E1["per (person, occlusion) cell:<br/>place largest groups first,<br/>into whichever split has<br/>the biggest shortfall vs 60/20/20"]
    E1 --> F{Hard assertions}
    F -->|"a person missing<br/>from a split"| G["Repair pass:<br/>force-move whole groups<br/>train -> val/test<br/>(not triggered in the final run)"]
    F -->|"no group spans 2 splits,<br/>every person in all 3 splits<br/>-- PASS"| H["metadata/manifest.csv<br/>image_path, person_id, occlusion,<br/>lighting, split<br/><br/>train 3,546 (59%) / val 1,444 (24%) / test 998 (17%)"]
    G --> F

    style H fill:#1f4e7d,color:#fff
    style D fill:#5a4a1f,color:#fff
```

**Note:** per-occlusion coverage inside each split is best-effort, not a hard assertion -- 36 of 240 `(person, occlusion)` cells are absent from the test split specifically; this is informational, reported in the final report, and not treated as a failure.

---

## 3. Training pipelines (E1-E3)

E1 and E2 share one automated preprocessing + embedding-extraction stage
and never touch raw pixels again after that. E3 is the exception -- it
needs raw pixels to backpropagate through the backbone, so it branches
off before embedding extraction.

```mermaid
flowchart TD
    A["processed_data/ + metadata/manifest.csv"] --> B["01_preprocess.py<br/>RetinaFace detect + 5-point align + crop"]
    B --> C["crops/p01..p30/*.jpg<br/>112x112 aligned"]
    C --> D["03_embed.py<br/>frozen ArcFace (models/buffalo_l/w600k_r50.onnx)"]
    D --> E["metadata/embeddings.npz<br/>512-d embeddings + person_id + occlusion + split"]

    E --> G["05_train_head.py --mode normal<br/>30-way ArcFace head, TRAIN=none-occlusion only"]
    G --> G1["runs/e1_head.pt<br/>runs/e1_head_results.json"]

    E --> H["05_train_head.py --mode mixed<br/>30-way ArcFace head, TRAIN=all 8 occlusions"]
    H --> H1["runs/e2_head.pt<br/>runs/e2_head_results.json"]

    C --> I["06_e3_finetune.py<br/>onnx2torch-convert w600k_r50.onnx (verified numerically identical)<br/>unfreeze last 15% of backbone + new head, BatchNorm frozen"]
    I --> I1["runs/e3_model.pt<br/>(fine-tuned backbone weights + head, ~166MB)"]

    G1 & H1 & I1 --> J["10_full_eval.py<br/>per experiment: confusion matrix,<br/>per-class + per-occlusion P/R/F1"]
    J --> K["runs/eval/e1../e3/<br/>metrics.json, confusion_matrix.csv/.png"]

    style E fill:#1f4e7d,color:#fff
    style C fill:#1f4e7d,color:#fff
    style K fill:#2d5a2d,color:#fff
```

---

## 4. Inference pipelines (E1-E3, as used by the live demo)

`scripts/09_server.py` loads E1/E2 at startup and E3 lazily (on first
selection), switching between them via a dropdown without restarting the
server. Detection and alignment are shared; only the embedding source
and the classifier differ per mode.

```mermaid
flowchart TD
    A[Phone camera frame, POSTed to /predict] --> B["RetinaFace (models/buffalo_l/det_10g.onnx)<br/>pick largest detected face"]
    B --> C["align_crop() -- common.py<br/>112x112 aligned crop<br/>(same function used to build training crops)"]
    C --> D{"active_mode<br/>(dropdown)"}

    D -->|e1| M1Emb["Frozen ArcFace (w600k_r50.onnx)<br/>get_feat() + L2 normalize"]
    M1Emb --> M1Buf["rolling buffer avg"]
    M1Buf --> M1Cls["argmax vs runs/e1_head.pt<br/>(30x512 weight matrix)"]
    M1Cls --> M1Out["label -- always one of 30,<br/>no UNKNOWN (closed-set only)"]

    D -->|e2| M2Emb["same frozen ArcFace embedding"]
    M2Emb --> M2Buf["rolling buffer avg"]
    M2Buf --> M2Cls["argmax vs runs/e2_head.pt"]
    M2Cls --> M2Out["label -- closed-set only"]

    D -->|e3| M3Bb["E3 backbone -- onnx2torch-converted,<br/>fine-tuned (runs/e3_model.pt)<br/>loaded lazily on first selection"]
    M3Bb --> M3Emb["own embed_fn(crop)<br/>-- different embedding space,<br/>buffer cleared on mode switch"]
    M3Emb --> M3Cls["argmax vs e3_model.pt head weight"]
    M3Cls --> M3Out["label -- closed-set only"]

    M1Out & M2Out & M3Out --> J["JSON: label, display_name, score, mode, bbox<br/>(no UNKNOWN in any currently active mode)"]
    J --> K[Phone browser renders result]

    style C fill:#1f4e7d,color:#fff
    style J fill:#2d5a2d,color:#fff
```

**Note:** switching modes clears the rolling embedding buffer, since E3's fine-tuned backbone produces a different (incompatible) embedding space from the one E1/E2 share.
