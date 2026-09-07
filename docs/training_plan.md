# Training Plan — Occlusion-Robust Face Identification

**Project:** CSE 4112 Machine Learning Laboratory, KUET
**Task:** Closed-set identification of 30 enrolled people under masks, sunglasses, scarves, caps, combined occlusions, and lighting variation.
**Dataset:** 6,000 images — 30 identities (`p01`–`p30`) × 200 images, flat at `processed_data/p##/`.
**Compute:** remote box `tahmid@100.73.147.96` — 2× RTX 3090 (24 GB), 24 cores, 62 GB RAM.

---

## 0. Deviations from the proposal (state these in the report)

| Proposal said | What we do | Why |
|---|---|---|
| Three capture sessions; S1=train, S2=val, S3=test | **One session.** Group-aware stratified split instead | Time shortage during collection — only one sitting per person was possible |
| Accuracy reported by pose (frontal / left / right) | **Dropped** | Pose was never labelled during capture |
| Conditions include `mask+hat`, `partial object` | Conditions are `none, cap, mask, scarf, clearglass, sunglass, mask_clearglass, mask_sunglass` | What was actually collected; `clearglass` and `mask_clearglass` were added instead |
| 10–15 participants | **30 participants** | Six team members × 5 people each |

The session change is the important one. A single-session split cannot prove generalisation across days, clothing, and setting the way the proposal intended. Section 3 describes the strongest split available without a second session, and Section 7 adds a cheap out-of-distribution check that partly covers the gap.

---

## 1. Environment setup (do this first)

Nothing is installed on the remote box yet — `torch` and `insightface` both fail to import.

```bash
ssh tahmid@100.73.147.96

sudo apt-get update && sudo apt-get install -y python3-venv python3-dev build-essential
python3 -m venv ~/venv && source ~/venv/bin/activate
pip install --upgrade pip

# CUDA 12.x wheels for the 3090s
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install insightface onnxruntime-gpu opencv-python-headless
pip install numpy pandas scikit-learn matplotlib seaborn tqdm

python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Only ~20 GB free on `/` — keep checkpoints small and delete failed runs. Store outputs under `~/runs/`.

---

## 2. Preprocessing — detect, align, crop

This step is what makes the model work on **different backgrounds**, which is the main practical requirement. A 112×112 face crop aligned to fixed landmark geometry contains almost no background, so the classifier physically cannot key on the wall behind the person. Do not skip alignment and feed raw photos.

**Script:** `scripts/01_preprocess.py`

1. Load each image from `processed_data/p##/`.
2. **RetinaFace detection** (`insightface.app.FaceAnalysis`, `det_size=(640,640)`). Keep the largest detected face.
3. If nothing is detected, retry at `det_thresh=0.3`, then at 2× upscale. Log every failure — never drop silently.
4. **Similarity-transform align** the 5 landmarks (2 eyes, nose, 2 mouth corners) onto the standard ArcFace 112×112 reference template.
5. Write the crop to `crops/p##/<same filename>.jpg` at quality 95.
6. Append a row to `metadata/manifest.csv`:

```
image_path, person_id, occlusion, lighting, source_stem, det_ok, det_score, bbox_area_frac, split
```

`occlusion` and `lighting` parse directly out of the filename (`p07_mask_sunglass_low_014.jpg` → `p07`, `mask_sunglass`, `low`). `source_stem` is the original camera filename from `metadata/rename.txt` — Section 3 needs it.

**Acceptance gate:** detection rate should exceed 97%. If a whole condition (likely `mask_sunglass` or `scarf`) detects far worse than the rest, that is a result worth reporting, not a bug to hide. Record the per-condition detection rate — it feeds the end-to-end metric in Section 6.

---

## 3. The split

With one capture session there is no session boundary to split on, so the danger is near-duplicate frames — shots taken seconds apart, or one image derived from another — landing on opposite sides and making test accuracy look better than it is.

**Rule: near-duplicates travel together.** Build groups before splitting:

- Two images are in the same group if their `source_stem` shares a base name (one is derived from the other).
- Two images are in the same group if they came from the same camera and their capture timestamps are within 10 seconds.
- Groups are transitive — union-find over the pairs.

The 429 byte-identical duplicates are, by project decision, **not** grouped — each counts as an independent image. See preprocessing plan §B1 for the size of the effect and the one-line check that keeps the reported number honest.

Then split **by group**, stratified per `person × occlusion × lighting` cell, at **60 / 20 / 20**:

| Split | Per person | Total | Purpose |
|---|---|---|---|
| train | ~120 | ~3,600 | fitting |
| val | ~40 | ~1,200 | early stopping, hyperparameter choice |
| test | ~40 | ~1,200 | reported once, at the very end |

Fix `seed=42`, write the assignment into `manifest.csv`, and **never regenerate it**. Every experiment reads the same column.

At ~1,200 test images across 8 conditions, each condition holds ~150 images. That supports a per-condition table. It does **not** support a per-person-per-condition table (≈5 images per cell) — do not report one.

**Assert before training:** no group id appears in two splits; every person appears in all three splits; every `person × occlusion` pair appears in test. Additionally *record* (do not assert) how many test images share an md5 with a training image — that number goes in the report.

---

## 4. Model

Pretrained **ArcFace iResNet50** (`insightface` `buffalo_l` recognition model, or `glint360k_r50`), 512-D L2-normalised embedding, plus a 30-class head.

### Head and loss — corrected from the proposal

The proposal quotes ArcFace with the default margin. **Do not use `m=0.5` here.** That value is tuned for 85k+ identities and millions of images; on 30 classes with ~120 images each it over-constrains the objective and commonly stalls or collapses.

```python
ArcFaceHead(in=512, classes=30, s=32.0, m=0.20)
# margin warm-up: m = 0 for epochs 1-3, then linear ramp to 0.20 by epoch 8
```

Label smoothing `0.1` on the cross-entropy. If it still trains unstably, fall back to a plain cosine head (`s=32, m=0`) — with 30 classes the margin is a refinement, not a necessity, and a clean run beats a fashionable loss.

### BatchNorm — the failure mode to avoid

The proposal says "frozen backbone" without saying what happens to BatchNorm. If BN layers stay in train mode they keep updating running statistics on 3,600 images and silently destroy the pretrained features, which looks like "fine-tuning made it worse."

```python
backbone.eval()                      # keeps BN in inference mode
for p in backbone.parameters():
    p.requires_grad = False          # Stage 1
```

Keep `backbone.eval()` in **Stage 2 as well**. Unfreeze the last residual stage's *weights* only; leave every BN running statistic frozen throughout.

### Optimisation

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| LR — head | `1e-3` |
| LR — unfrozen backbone (Stage 2) | `1e-5` |
| Weight decay | `5e-4` head, `1e-4` backbone |
| Schedule | cosine, 3-epoch linear warm-up |
| Batch size | 64 (3090 has ample headroom) |
| Epochs | 30 stage 1, 15 stage 2 |
| Precision | AMP (`torch.autocast`) |
| Early stopping | val macro-F1, patience 7 |
| Seed | 42, and log `torch.backends.cudnn.deterministic` |

Single GPU is enough — the whole run is minutes. Use the second 3090 to run experiments in parallel, not to shard one model.

### Augmentation — training split only

Training data is phone photography and the demo runs on a phone, so the domain gap is small. Augment for **live-capture conditions** — handheld movement, autofocus, preview-stream compression — not for a foreign sensor. Over-augmenting here costs accuracy by spending model capacity on a domain that never occurs.

**Geometric / photometric:** horizontal flip `p=0.5`; rotation `±10°`; brightness/contrast jitter `0.2`; hue/saturation jitter `0.05/0.2`.

**Live-capture group:**

| Augmentation | Setting | Simulates |
|---|---|---|
| Resolution degradation | downscale to a random 64–112 px, back to 112, `p=0.3` | phone-as-webcam preview stream |
| Motion blur | kernel 3–7 px, random angle, `p=0.2` | handheld capture, person moving |
| Gaussian blur | σ 0.3–1.0, `p=0.15` | autofocus not fully settled |
| Sensor noise | Gaussian σ 2–6 / 255, `p=0.2` | indoor light on a live stream |
| JPEG re-compression | quality 50–95, `p=0.3` | preview/video codec |
| Random erasing | `p=0.25`, scale 2–20% | extra synthetic occlusion |

Resolution degradation is kept but **deliberately mild** (64–112 px, not 32–96). It is insurance for the phone-as-webcam path described in the preprocessing plan §B5, where a 720p preview stream drops the face to ~250–350 px. If the demo takes full still photos instead, this augmentation is nearly free and does no harm; if it streams preview frames, it is what saves the demo.

JPEG re-compression and noise do a second job worth keeping regardless of camera. Each team member shot with a different phone (Pixel, iPhone, Samsung, Android), and camera identity is confounded with the group of 5 people that member photographed. Washing out per-device sensor and quantisation signatures stops the model shortcutting "this is a Pixel photo, so it's one of p06–p10."

No augmentation on val or test, ever.

---

## 5. Experiments

Four runs. Each is minutes on a 3090.

| # | Name | Backbone | Train data | Answers |
|---|---|---|---|---|
| **E0** | Nearest-centroid probe | frozen, no training | all conditions | What do the pretrained embeddings already do? |
| **E1** | Normal-only | frozen + head | `none` condition only | The proposal's baseline |
| **E2** | Mixed-condition | frozen + head | all conditions | Does seeing real occlusion help? |
| **E3** | Partial fine-tune | last stage unfrozen | all conditions | Does adapting the backbone help further? |
| **E4** *(contingent)* | Live-adapted | best of E2/E3 | all + rehearsal frames | Only if the live rehearsal score falls short |

**Run E0 first.** Extract embeddings once, average them per identity, classify test images by cosine similarity to the 30 centroids. Zero training, ~10 minutes.

E0 is not a formality — it decides what the project is about. If nearest-centroid already reaches ~99% top-1, then the headline accuracy is saturated and no amount of fine-tuning will produce an interesting number. The real finding then becomes **the per-occlusion breakdown** — which occlusions still break a strong pretrained face model — and the report should be written around that. If E0 lands nearer 85–92%, there is genuine headroom and E1→E2→E3 becomes the story. Either way you learn it on day one instead of week three.

**E1 vs E2 is the headline comparison** the proposal promised. E1 trains only on `none` images (~15 per person after splitting — small, so expect noise) and is tested on everything. Expect E2 to win clearly on occluded conditions and roughly tie on `none`.

**E4 is contingent and will probably not be needed.** Score E2/E3 on the held-out live rehearsal set (preprocessing plan §B5) first. Because both the dataset and the demo are phone cameras, that number should land close to the test number — if it does, skip E4 entirely. Run it only if live accuracy has visibly dropped, in which case add part of the rehearsal set to training and hold the rest back. Report both numbers either way; the gap between still-test and live accuracy is a result worth a paragraph.

---

## 6. Evaluation

Compute once, on test, after hyperparameters are frozen. `scripts/04_evaluate.py` emits one CSV and one confusion-matrix PNG per experiment.

**Report:**

- **Top-1 accuracy** and **macro-F1** (they will be close — classes are balanced by construction, 200 each).
- **Per-person recall** — 30 rows. Flags an identity the model cannot handle.
- **Confusion matrix**, 30×30, normalised by row.
- **Per-occlusion accuracy** — 8 rows, the primary scientific table.
- **Per-lighting accuracy** — `regular` vs `low`.
- **Robustness gap** — `acc(none) − acc(occluded)`, per model. This single number is the project's headline result.
- **Detection failure rate** and **end-to-end accuracy** = correct predictions ÷ *all raw test images*, counting detection failures as errors. Report crop-level and end-to-end separately, as the proposal requires.
- **Top-1 on the duplicate-free test subset** — filter out test images whose md5 also appears in train, then recompute. One line of pandas; quote it beside the headline number.
- **Live rehearsal accuracy** on the held-out phone capture set (preprocessing plan §B5), reported separately and never merged with the main test figure.
- **Efficiency** — parameter count, model size on disk, mean inference ms/image at batch 1.

**Report 95% Wilson confidence intervals on every accuracy.** With ~1,200 test images, 1% is 12 images; with ~150 per condition, a condition-level CI is roughly ±5–7 points. Without CIs the report will over-claim on differences that are noise. If E2 beats E1 by 0.4% overall, the correct sentence is that they are indistinguishable.

---

## 7. Background robustness check

The stated goal is that the system identifies people **in a different background**, and nothing in the formal test set proves that — every image was shot in one session, in one set of locations.

The live rehearsal set (preprocessing plan §B5) already covers most of this, since it is captured in a different room on a different day. Get the coverage for free by shooting part of it **somewhere clearly different** — another room, a corridor, outdoors — and, where possible, on a phone that did **not** capture that person's training images.

That second detail is worth the small effort. Each team member photographed their own five people with one phone, so device signature is confounded with a group of five identities. Testing a person on someone else's phone is the cleanest way to find out whether the model learned the face or the camera. If accuracy holds, the confound is not being exploited and you can say so with evidence. If it drops sharply, that is a genuine finding and belongs in the report.

Report these numbers separately and label them clearly, never merged into the main test figure.

---

## 8. Demonstration — this is the acceptance test

The project is judged by a team member sitting in front of a **live phone camera** wearing an occlusion and being named correctly. Build for that directly.

`scripts/05_demo.py` — capture loop: detect → align → embed → classify → draw `p##` and confidence.

**Decide the capture path first**, because it changes the risk more than any hyperparameter (see preprocessing plan §B5):

- **Still photo from the phone, transferred to the laptop** — face at ~1,200–1,600 px, identical to the training domain. Safest, and what I would demo with.
- **Phone as webcam** (DroidCam / Iriun over USB or Wi-Fi) — more impressive live, but a preview stream at 720p drops the face to ~250–350 px and adds codec artifacts. If you take this path, **set the app to 1080p or higher**; that one dropdown matters more than another training run.

**Four things decide whether this works:**

**1. Identical preprocessing.** Import the very same `align_crop()` used to build the training crops. Do not reimplement it for the live path. Mismatched alignment between training and inference is the most common cause of a demo failing after a clean test result.

**2. Use a gallery, not just a trained head.** Keep both inference paths:

- *classifier head* — the trained 30-way ArcFace/cosine head;
- *gallery* — the mean L2-normalised embedding per identity, compared by cosine.

The gallery path is the safety net. If anything is off on demo day — an unfamiliar phone, harsh room lighting — you can capture ten frames of that person live, add them to their centroid, and the system improves **immediately, with no retraining**. A fixed classifier head cannot do that. Pick whichever scores better on the rehearsal set, but ship both.

**3. Temporal aggregation** (live-stream path only). Never classify on a single frame. Keep a rolling buffer of the last 5–10 frames, average the L2-normalised embeddings, renormalise, then classify. A live camera gives you many looks at the same face for free, and averaging removes most per-frame noise, blur and blink artifacts. This alone typically converts a jittery demo into a stable one. On the still-photo path it does not apply — though taking three shots and averaging costs nothing and helps.

**4. Rejection threshold.** If the top cosine score falls below `τ`, print `UNKNOWN` rather than a person ID. Pick `τ` from the *validation* score distribution, never the test one. The system is closed-set, so an unenrolled visitor is otherwise silently forced into one of the 30 identities — this is the difference between a demo that looks honest and one that confidently misnames a stranger who walks past.

**Rehearse on the real hardware.** Run the demo on the actual phone, in the actual room, under the actual lighting, with the actual masks and sunglasses, before the day. Note the per-condition behaviour. If `sunglass` or `mask_sunglass` fails at the *detection* stage rather than the classification stage, the fix is detector settings (lower `det_thresh`, larger `det_size`), not more training — and you want to discover that in advance, not in front of the examiners.

---

## 9. Repository layout

```
D:\ML train\
├── processed_data/          p01..p30, 200 images each (source)
├── crops/                   112x112 aligned faces (generated)
├── metadata/
│   ├── manifest.csv         authoritative: paths, labels, groups, split
│   ├── rename.txt           original -> current filename map
│   └── person_id_mapping.txt
├── scripts/
│   ├── 01_preprocess.py     detect, align, crop, build manifest
│   ├── 02_split.py          group-aware stratified split
│   ├── 03_train.py          E0/E1/E2/E3 driven by a config flag
│   ├── 04_evaluate.py       metrics, CIs, confusion matrices
│   └── 05_demo.py           still-image / live-phone inference
├── runs/                    checkpoints, logs, results per experiment
└── docs/
    ├── Occlusion_Robust_Face_Identification_Proposal_Updated.pdf
    └── training_plan.md
```

---

## 10. Order of work

1. Set up the remote environment; verify both GPUs are visible.
2. Hash all files, record md5 (preprocessing plan §B1).
3. Run detection/alignment/cropping; check the detection-failure log before anything else.
4. Build the split; run the assertions in Section 3.
5. **Run E0.** Read the result. Decide whether the report's story is "accuracy" or "per-occlusion breakdown."
6. Train E1 and E2; compare on validation only.
7. Train E3.
8. **Collect the live rehearsal set** (preprocessing plan §B5) while training runs — real demo phone, different room.
9. Score E2/E3 on it. Run E4 only if live accuracy fell short.
10. Freeze hyperparameters. Evaluate on test **once**.
11. Build the demo: identical `align_crop`, gallery + head, frame averaging, rejection threshold.
12. Rehearse end-to-end on the real phone with the real accessories.
13. Write up, including the deviations table from Section 0.

---

## 11. Known limitations to state in the report

- **Single session.** Train, validation, and test all come from one sitting per person. Same clothing, same setting, same day. This inflates accuracy relative to a genuine multi-session protocol, and the OOD set in Section 7 is a partial mitigation, not a substitute.
- **Closed set.** Every input is forced into one of 30 identities unless the rejection threshold fires. No open-set evaluation is performed.
- **Camera confound.** Each team member photographed their 5 people with one phone, so device signature correlates with a group of 5 identities. Augmentation mitigates it; it is not eliminated.
- **Mixed provenance in the low-light condition.** Not all `low` images are original low-light captures — some were produced by reducing the brightness of a corresponding `regular` image. The grouping rule in Section 3 keeps each such image with its source so the split stays clean, but low-light accuracy should be read as robustness to reduced brightness generally, not exclusively to genuine low-light capture.
- **429 exact duplicates are present and are treated as independent images.** Consequently about 8% of test images are byte-identical to a training image, concentrated in p21–p25 and p05, and the headline accuracy reads slightly optimistic. The duplicate-free test accuracy is reported alongside it (Section 6).
- **Camera confound.** Device signature correlates with a group of five identities; the cross-phone check in Section 7 tests whether the model exploits it.
- **Live capture is not identical to the dataset.** The demo phone is the same class of device as the training cameras, so the gap is small, but preview-stream resolution and handheld motion still differ from deliberate still shots. Quote the live rehearsal number alongside the test number rather than only the latter.
- **No anti-spoofing or liveness.** This is a laboratory classification prototype, not an access-control system.
- **30 identities.** Results will not extrapolate to hundreds or thousands of enrolled people.
