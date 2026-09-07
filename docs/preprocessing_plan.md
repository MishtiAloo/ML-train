# Preprocessing Plan

**Acceptance test for this project:** a team member sits in front of a **live phone camera**, wearing an occlusion, and the system names them correctly. Every decision below is judged against that, not against a test-set number.

**The dataset is frozen.** The 6,002 files now at `/home/tahmid/processed_data` are the dataset. Nothing below adds, removes, regenerates or re-balances an image. Where the collected data has a defect, it is handled at the *split* or *sampling* level instead of by editing the set.

---

## Part A — What has already been done

All steps below are complete. Nothing was deleted at any stage; every removal moved files to `put_asides/`, and `metadata/rename.txt` maps every current filename back to its original camera filename.

| # | Step | Result |
|---|---|---|
| 1 | **Brightness sorting** — images whose original filename contained `brightness` (output of `brt.py`) separated from the rest | 870 files moved; 1 misfiled image corrected |
| 2 | **Low-light generation** for images that had no dark counterpart (PIL brightness ×0.4–0.6) | 78 images created |
| 3 | **Folder taxonomy normalised** — `hat`/`cap`, `clear_glass`/`clearglass`, `scraf`/`scarf`, `Bright Light`/`Lighter`/`normal`/`regular`, etc. collapsed to a single scheme | 446 folders renamed → 8 occlusions × {`regular`, `low`} |
| 4 | **Count audit** | `metadata/initial_count.txt` |
| 5 | **Balanced to 200 images per person** — greedy "trim the largest bucket", floor of 6 per bucket, random choice within a bucket | 1,117 images moved to `put_asides/` |
| 6 | **Padded 4 short people** (Ansh, Ullash, Arpita, Rajorshi) with generated low-brightness images, filling the smallest bucket first | 36 images created |
| 7 | **HEIC → JPG** conversion, quality 95, EXIF orientation applied, decode verified before deleting each source | 1,863 files converted |
| 8 | **Canonical rename** to `p##_<occlusion>_<lighting>_<NNN>.jpg` | 6,000 images + 2 `.dng`; `metadata/rename.txt` written |
| 9 | **Flattened** to `processed_data/p##/` with no sub-folders | 6,002 files, 30 folders |
| 10 | **Transferred** to the training box | `/home/tahmid/processed_data`, 13 GB |

**Current state:** 30 identities × 200 images, filenames carry person, occlusion and lighting; occlusion/lighting metadata is recoverable by parsing, and original provenance by `rename.txt`.

**What has *not* happened yet: no face has been detected, aligned, or cropped.** The files are still whole 12 MP phone photographs of a head and shoulders. That is the largest remaining task and it is step B3.

---

## Part B — What still needs to be done

### B1. Hash every file — 429 exact duplicates exist

A hash pass over the 6,000 JPGs finds **5,571 unique images and 429 byte-identical duplicates** (419 clusters: 409 pairs, 10 triples). They are copy artifacts from the original transfer — `IMG_20260822_114440.jpg` and `IMG_20260822_114440(1).jpg` are the same bytes under two names.

They are concentrated:

| Person | Duplicate files | Distinct images |
|---|---|---|
| p21–p25 (Rafi's five) | 67, 68, 69, 66, 61 | ~132–139 each |
| p05 | 55 | 145 |
| p04 | 17 | 183 |
| p11 | 12 | 188 |
| p01 | 11 | 189 |
| p13 | 3 | 197 |
| all others | 0 | 200 |

**Nothing is deleted, and — per project decision — each duplicate is treated as an independent image.** They are not grouped, and they may land in different splits. Still compute the `md5` column in the manifest, because it is what lets you quantify the effect below rather than guess at it.

Two further facts from the pass:

- **No duplicate cluster spans two different people.** No identity label is corrupted.
- **8 clusters span `low` and `regular`** — the identical file is filed under both lighting conditions (e.g. `p05_cap_low_007.jpg == p05_cap_low_008.jpg == p05_cap_regular_015.jpg`). One of the two lighting labels is necessarily wrong, since a file cannot be a darkened version of itself. Record these 8 in `metadata/known_label_issues.csv` and exclude them from the *per-lighting* table only. They stay in training and in the overall accuracy figure.

**What this costs, stated so the report can be accurate.** With a 60/20/20 split, a duplicate pair lands train/test about 24% of the time, so roughly **100 of the ~1,200 test images (~8%) will be byte-identical to an image the model trained on**, concentrated in p21–p25 and p05. Those are free correct predictions, so overall top-1 will read a little higher than it would on fully distinct data. After evaluation, use the `md5` column to compute accuracy on the duplicate-free subset of the test set and quote both — it takes one line of pandas and turns a hidden bias into a documented one.

Related: p21–p25 have roughly 135 genuinely distinct looks rather than 200, so they contribute less variety than their file count suggests. A class-balanced sampler and macro-F1 (both already in the training plan) cover this.

The two `.dng` files are RAW sidecars of JPGs already in the set and cannot be decoded by the image pipeline. Filter on extension when building the manifest — they are simply never listed.

### B2. Detect, align, crop — the main step

This is what makes the system work against an unfamiliar background, which is a stated requirement. A 112×112 crop aligned on facial landmarks contains essentially no background, so the classifier cannot key on the room.

For each image:

1. **Detect** with RetinaFace (`insightface.app.FaceAnalysis`, `det_size=(640,640)`). Keep the highest-scoring face; if several are found, keep the largest box.
2. On failure, retry at `det_thresh=0.3`, then on a 2× upscale. Log every failure — never drop an image silently.
3. **Align** the five landmarks (both eyes, nose, both mouth corners) onto the standard ArcFace 112×112 reference template by similarity transform.
4. **Write** to `crops/p##/<same filename>.jpg`, quality 95.
5. Record `det_ok`, `det_score`, and `bbox_area_frac`.

**Expect `sunglass`, `mask_sunglass` and `scarf` to detect worst** — RetinaFace leans on eye landmarks, and dark glasses remove them. If a condition detects far below the rest, that is a finding for the report, not a bug to paper over. It is also exactly what the end-to-end metric in the training plan is designed to capture.

Freeze this as a single function, `align_crop(bgr_image) -> 112x112 crop`. **The live demo must import the identical function.** Training on crops produced one way and inferring on crops produced another way is the most common reason a demo fails after a clean test result.

### B3. Build the manifest

One row per successfully cropped image, written to `metadata/manifest.csv`:

```
image_path, person_id, occlusion, lighting, source_stem, md5, group_id,
det_ok, det_score, bbox_area_frac, split
```

`occlusion` and `lighting` parse out of the filename. `source_stem` comes from `rename.txt`. `md5` is the hash of the **original** file from B1, not the crop. The CSV is the only thing training reads — never let a script walk the directory tree directly, or an experiment will silently disagree with the manifest.

### B4. Group-aware split

Assign `group_id` before splitting, so near-identical images do not straddle the boundary. Two rules, then take the transitive closure:

1. same `source_stem` base, i.e. one image derived from another → same group;
2. same camera and capture timestamps within 10 s (burst neighbours) → same group.

Exact duplicates are **not** a grouping rule — by project decision each is treated as an independent image (see B1).

Then split **by group**, stratified per `person × occlusion × lighting`, **60 / 20 / 20**, `seed=42`. Write the result into the `split` column and never regenerate it.

**Assertions that must pass before any training starts:**
- no `group_id` appears in more than one split;
- all 30 people appear in all three splits;
- every `person × occlusion` pair appears in test;
- split proportions are within a couple of points of 60/20/20 for every person.

Also record, for the report: the count of test images whose `md5` also appears in train.

### B5. Live rehearsal set — small, but do it

Because the demo runs on a **phone** and the dataset was shot on **phones**, the domain gap is small. Same sensor class, same optics, same colour science. This is the single luckiest fact about the project and it removes most of the risk that would exist with a webcam.

One residual gap remains, and it depends entirely on how the phone feeds the model:

| Capture path | Face size at desk distance | Gap |
|---|---|---|
| Phone takes a **still photo**, transferred to the laptop | ~1,200–1,600 px | essentially none — identical to training |
| Phone as webcam over USB/Wi-Fi (DroidCam, Iriun), **1080p** | ~400–600 px | mild — video compression, some softness |
| Phone as webcam, **720p** | ~250–350 px | moderate — the case the augmentation is insuring against |

**Prefer the still-photo path for the graded demo**, and if you use phone-as-webcam, set the app to the highest resolution it offers. A dropdown changed from 720p to 1080p is worth more than any amount of extra training.

The rehearsal set is therefore a pipeline check, not a domain-adaptation exercise. Capture with the actual demo phone, in the actual room:

- the six team members who will sit in front of the camera (all 30 is unnecessary);
- all 8 occlusion conditions;
- normal lighting plus one dimmer setting;
- ~5 frames each.

That is roughly 500 images, about half an hour for the team. Run them through the identical `align_crop`, hold them out entirely, and score the trained model on them. If that number tracks the test number, the demo will work. If it collapses, you have found out a week early and the training plan's E4 exists for exactly that case.

### B6. Quality review

Before training, view a contact sheet of ~200 random crops. Look for: wrong person cropped from a photo containing two faces, crops that are mostly hand or phone, upside-down or 90°-rotated crops from EXIF that did not apply, and near-black crops where the brightness reduction went too far. Fix what is broken, log it, and move on — a handful of bad crops is normal; a systematic pattern is not.

---

## Part C — Order of execution

```
B1  hash all files, record md5          ~5 min
B2  detect + align + crop 6,000 imgs    ~20-40 min on the 3090
B6  quality review of crops             30 min, human
B3  build manifest                      minutes
B4  group-aware split + assertions      minutes
B5  live rehearsal capture (parallel)   ~30 min, six people
```

Nothing here modifies the dataset. B1 must precede B4 because the md5 grouping rule depends on it. B5 can be collected while the GPU work runs.

---

## Part D — Numbers to carry into the report

| Quantity | Value |
|---|---|
| Dataset as delivered | 6,000 JPG (+2 RAW, not used) |
| Identities | 30, 200 images each |
| Byte-identical duplicates present | 429 (kept, but grouped so they cannot cross the split) |
| Distinct image contents | 5,571 |
| Least-diverse identities | p21–p25, ~135 distinct looks each |
| Median source resolution | 12.2 MP (3024×4032) |
| Face-detection success rate | to be measured in B2 |
| Model input | 112×112 aligned RGB crop |
