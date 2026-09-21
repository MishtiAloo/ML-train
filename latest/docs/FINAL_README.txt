FINAL DATASET REPORT

final.tex is the editable LaTeX source; final.pdf is the compiled report.
SOURCE_SECTION_COVERAGE.md maps the source sections
and explains the revised figure numbering.
The user lifted the earlier 22-page cap to allow larger figures and full coverage.
Keep final_assets/ alongside final.tex: its PDF figures are required.

Compile in this directory with XeLaTeX, twice:
  xelatex -interaction=nonstopmode -halt-on-error final.tex
  xelatex -interaction=nonstopmode -halt-on-error final.tex

Fonts: Times New Roman, Arial, Consolas (installed on the source Windows machine).
The source uses A4 and the DOCX template's actual margins, 11-point body text,
1.15 line spacing, and the template's numbered sections and appendices.
Page 1 follows its visual-synopsis requirement, with figures 1(a) and 1(b).

Content precedence: final_technical_report.md/.pdf and saved final experiment
records control methods/results; prev_report.pdf supplies the requested original
submission, related-paper, collection, consent and quality-review information.
Required administrative facts absent from the sources are marked not supplied.

build_final_figures.py regenerates vector charts from the parent project's
saved CSV/JSON data and crops. It does not train models or change source data.
check_final.py renders previews and checks pagination, references and counts.
These optional tools require the original project data, Python, matplotlib,
NumPy, Pillow and PyMuPDF; compiling the supplied .tex does not require Python.

Latest revision: p11 raw photos respect EXIF orientation; detector rejection
gallery retains original examples 1, 2, 5, 6, 7; architectures have spaced boxes;
training and validation losses share taller axes in the main results; six-fold
rotation and full assignment diagrams are included; the former Figure 10
validation-accuracy plot is in Appendix B. The Drive dataset URL is embedded.

Section 10.4 now starts with the complete ArcFace/IResNet-50 architecture,
verified against models/w600k_r50.onnx: input/output shapes, all residual stages,
embedding projection, and external L2 normalization. Residual-block internals
are omitted as requested; the main input-to-output diagram is centered.

DATASET VERSION 1 | GROUP 5 | SUBMISSION 22/09/2026
Dataset: https://drive.google.com/drive/u/1/folders/1XN55OGpozy5Rhbs5QRFMatlCMTAo8sPu
Code: https://github.com/MishtiAloo/ML-train.git
Request permission from the Drive owner if necessary. Consent is academic,
non-commercial and withdrawable. Faces remain identifiable. Do not redistribute
raw images or identity mappings without authorization. Public redistribution
license: not supplied. Respect pretrained-model licensing conditions.

FILES (relative to project root, the parent of docs/)
../processed_data/                 Raw photos, separate from processed crops
crops/p01/ ... crops/p30/           5,988 aligned 112 x 112 RGB crops
metadata/manifest.csv              Labels, provenance and detector quality
metadata/e1e2_person_folds.json     Six folds, references and query exclusions
runs/e1e2/person_benchmarks.npz     30 fixed pretrained 512-D gallery vectors
runs/e1e2/folds/fold*/              Checkpoints, histories and result JSONs
runs/e1e2/analysis/                 Predictions, metrics and confusion matrices
runs/e1e2/learning_curve/           Histories with validation-loss logging
models/; scripts/                  ONNX models; preparation/training/analysis
docs/final.tex; final.pdf          Editable source and compiled dataset report
docs/final_assets/                 PDF figures required for compilation
docs/CONTRIBUTIONS.md              Shared responsibilities and evidence index

VARIABLES: metadata/manifest.csv
All 13 columns are populated. Each row describes one crop. Check required
values before reuse; no imputation policy is defined.
image_path: text, source path; remap training-machine paths to local raw files.
crop_path: text, crop path; basename resolves in the participant directory.
person_id: text, p01-p30; identity grouping key, not a person's name.
occlusion: none, clearglass, sunglass, cap, scarf, mask, mask_clearglass,
           mask_sunglass.
lighting: regular or low; qualitative, not measured illuminance.
source_stem: text, original filename/stem linking source provenance.
md5: text, recorded 32-character hash; repeated hashes are retained.
group_id: text, related-frame/burst group used for leakage exclusions.
split: legacy train/val/test label; superseded by the six-fold JSON.
det_ok: Boolean text, True in retained rows; detector failures are absent.
det_score: numeric [0,1], detector confidence, not calibrated image quality.
bbox_area_frac: numeric [0,1], face bounding-box area / source frame area.
det_stage: primary or loose (5,928 / 60 crops).
See report Section 7 and Appendix A for the complete dictionary.

SETUP AND REPRODUCTION
Clone into a new directory:
  git clone https://github.com/MishtiAloo/ML-train.git
Obtain authorized files from Drive and arrange the folders above. Inspect each
script's --help and remap absolute paths before execution. Several defaults
refer to the original Linux training machine. Preserve supplied folds, gallery
and histories when auditing; use separate output locations for reruns.
Use an isolated Python environment with PyTorch, ONNX Runtime, onnx2torch,
InsightFace, NumPy, Pillow and OpenCV. Plotting also uses matplotlib and PyMuPDF.
Exact dependency versions/lockfile: not supplied. Select compatible GPU packages
for the local system. Reported hardware: RTX 3090; seed: 42. The local snapshot
contains crops and saved records, not a complete raw-to-crop preparation
environment. Full raw-data or bitwise reproduction is not claimed.

Workflow from scripts/ (inspect arguments and supply local paths):
1. 20_e1e2_person_folds.py prepares gallery, exclusions and folds. It requires
   manifest and cached embeddings. Use supplied JSON/NPZ for exact evaluation.
2. 19_e1e2_person_finetune.py runs E0/E1/E2 for each fold (max epochs 0/20/20).
3. 21_e1e2_confusion.py and 23_dump_predictions.py generate evaluation records.
4. 25_e1e2_learning_curve.py repeats E1/E2 with validation loss.
5. 22_e1e2_plots.py and 24_report_figures.py generate project figures.
From docs/, build_final_figures.py rebuilds report figures using saved records
and images, without retraining. Compile instructions are at the top of this file.

REUSE AND QUALITY
6,000 photos from 30 people (200 each); 12 detector failures leave 5,988 crops.
After reference/burst exclusions, 5,818 queries remain. Each fold uses 20 train,
5 validation and 5 test identities. All 30 have a fixed enrollment reference.
Never substitute the legacy image-level split. Augmentation is training-only.
Correct means right top identity AND cosine >= 0.30. All queries are enrolled;
UNKNOWN is rejection, not validated unseen-stranger recognition. Retain saved
GPU aggregates: later CPU export can move one borderline query across 0.30.
Ages: 23-24 (26 people); 28, 39, 44 and 78 (one each). This indoor, single-sitting
sample is young-adult-heavy. Scarf covers only 24 people. The manifest retains
429 repeated MD5 occurrences within identities. Some synthetic content was
added, including digital coverings; method, proportion and per-image flags:
not supplied. Rater-agreement checking was completed; method and coefficient:
not supplied. Do not infer perfect agreement or a fully natural dataset.
Reuse for occlusion analysis, fixed-gallery recognition or adaptation must
preserve consent, identity grouping, provenance and these limitations.

TEAM AND CITATION
2107031 Adiba Tahsin; 2107044 Arafat Islam; 2107046 Arman Rahman Rafi;
2107047 Dip Shekhor Datta; 2107057 Megha Tania; 2107059 Farhan Tahmid.
Each captured and annotated own five people's data. Other work is shared
equally. CONTRIBUTIONS.md records supplied figures and artifact evidence.
Suggested citation: the six authors above, Occlusion-Robust Face Identification
Dataset: Evaluation with a Partially Fine-Tuned ArcFace Model, version 1,
Group 5, CSE 4112, KUET, submission 22 September 2026, dataset URL above.
DOI, release date and code commit: not supplied.

VERSION 1 DOCUMENTATION CHANGE RECORD
Submission 22/09/2026: group, roster, email pattern, URLs, demographics,
completed rater agreement, synthetic disclosure and accountability updated;
README expanded for files, variables, setup and reuse. This documentation
update does not alter images, folds, weights or results. Submission date is
not asserted as release date.
