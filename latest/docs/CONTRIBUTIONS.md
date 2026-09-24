# Group 5 — accountability

Dataset version 1; submission 22/09/2026. These are team-declared contributions,
not an inferred Git authorship audit.

| Roll | Student | Contribution | Individual role |
|---|---|---:|---|
| 2107031 | Adiba Tahsin | 17.50% | Captured and annotated own five participants |
| 2107044 | Arafat Islam | 15.83% | Captured and annotated own five participants |
| 2107046 | Arman Rahman Rafi | 16.67% | Captured and annotated own five participants |
| 2107047 | Dip Shekhor Datta | 15.83% | Captured and annotated own five participants |
| 2107057 | Megha Tania | 16.67% | Captured and annotated own five participants |
| 2107059 | Farhan Tahmid | 17.50% | Captured and annotated own five participants |

The original relative weights were proportionally normalized and rounded to
two decimal places. The displayed contribution percentages total exactly 100%.

## Specific project responsibilities

Each member captured and annotated their own five participants. Beyond this
common collection role, Farhan Tahmid handles E1/E2 training, fine-tuning and
training/evaluation pipeline checking. Implementation, integration, metric analysis and documentation are
distributed among the other five members. This records the requested allocation; artifact
references support the work products rather than independently proving authorship.

| Student | Project work | Supporting evidence |
|---|---|---|
| Adiba Tahsin (2107031) | Capture-protocol and manual quality-review coordination; label and age-distribution summaries; data dictionary and consent documentation | E1; E6; report Sections 6–7, 9 and 13 |
| Arafat Islam (2107044) | Frozen E0 baseline implementation; RetinaFace alignment/crop and detection-failure review; manifest schema, hash and duplicate checks | E1; E3; E5; `docs/final_assets/retinaface_rejected.pdf`; report Sections 8–9 and 10.4 |
| Arman Rahman Rafi (2107046) | Six-fold identity partitioning; fixed-gallery selection and near-burst exclusions; trainable-scope configuration; end-to-end training/evaluation pipeline integration and leakage-safety checks | E2; E3; E4; `runs/e1e2/person_benchmarks.npz`; report Sections 10.2, 10.4 and 10.6 |
| Dip Shekhor Datta (2107047) | Validation-based checkpoint selection; prediction export and metric analysis; confusion matrices; per-occlusion comparisons and rejection-error interpretation | E3; E4; E5; `runs/e1e2/folds/`; report Sections 10.4, 10.5 and 10.7 |
| Megha Tania (2107057) | Learning-curve and architecture visualizations; base-paper comparison; report/README editing and citation checks | E5; E6; `runs/e1e2/learning_curve/`; report Sections 5, 9.1, 10.4 and 10.7 |
| Farhan Tahmid (2107059) | E1/E2 model training and fine-tuning; checking the training/evaluation pipeline | E3; E4; `scripts/25_e1e2_learning_curve.py`; `runs/e1e2/folds/`; `runs/e1e2/learning_curve/`; `runs/e1e2/analysis/` |

The balance refers to the overall scope of responsibilities, not identical tasks
or measured hours. The normalized percentages above preserve the relative weights.

## Shared evidence

Paths are relative to the project root. These artifacts support work products,
not exclusive individual authorship. No participant-to-student mapping or
historical authorship evidence is invented.

| ID | Work | Evidence |
|---|---|---|
| E1 | Capture and annotation | `metadata/manifest.csv`, `crops/`, `docs/prev_report.pdf` Section 4.1 |
| E2 | Methodology and splits | `metadata/e1e2_person_folds.json`, `scripts/20_e1e2_person_folds.py` |
| E3 | Software and training | `scripts/19_e1e2_person_finetune.py`, `runs/e1e2/folds/` |
| E4 | Validation and analysis | `scripts/21_e1e2_confusion.py`, `scripts/23_dump_predictions.py`, `runs/e1e2/analysis/` |
| E5 | Visualization | `docs/build_final_figures.py`, `docs/final_assets/`, `scripts/22_e1e2_plots.py` |
| E6 | Writing and administration | `docs/final.tex`, `docs/FINAL_README.txt`, `docs/CONTRIBUTIONS.md` |

Inter-rater agreement checking was completed, per the team declaration.
Agreement method, coefficient and rating sheet: not supplied.
