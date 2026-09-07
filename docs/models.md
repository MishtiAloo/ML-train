# Models

Every model actually used in this project, and where. "Training" means it
appears in one of `scripts/01_preprocess.py`-`10_full_eval.py`; "inference"
means it appears in `scripts/09_server.py` (the live demo) or `08_demo.py`
(the CLI equivalent).

| Model | File | Used in | Why |
|---|---|---|---|
| **RetinaFace** (detector + 5-point landmarks) | `models/buffalo_l/det_10g.onnx` | Preprocessing (`01_preprocess.py`); every experiment's inference path (E0-E3, all demo modes) | Purpose-built face detector that also outputs the 5 landmarks alignment needs, so every downstream model sees a consistently cropped face instead of a raw photo. |
| **ArcFace backbone, frozen** (iResNet50) | `models/buffalo_l/w600k_r50.onnx` | Embedding cache (`03_embed.py`); **E0** training + inference; **E1/E2** training (feature extractor) + inference | Already pretrained on large-scale face data, so it gives a strong 512-d identity representation for 30 people without training a backbone from scratch on a few hundred images each. |
| **E0 — nearest-centroid** (no learned parameters, just per-person mean embeddings) | in-memory during training; `runs/gallery.npz` for deployment | **E0** training (`04_e0_probe.py`); **gallery** mode in the demo | Zero-training baseline: measures how discriminative the pretrained embedding already is, before any task-specific optimization. |
| **E1 head** (512→30 linear layer, ArcFace angular-margin loss) | `runs/e1_head.pt` | **E1** training (`05_train_head.py --mode normal`); **e1** mode in the demo | Tests whether a head trained only on clean, unoccluded faces still generalizes to occluded test photos. |
| **E2 head** (same architecture as E1, different training data) | `runs/e2_head.pt` | **E2** training (`05_train_head.py --mode mixed`); **e2** mode in the demo | Tests whether training the same head on all 8 occlusion conditions closes the gap E1 leaves open. |
| **onnx2torch-converted backbone** (trainable copy of the frozen ArcFace backbone, numerically verified identical) | built at runtime from `w600k_r50.onnx`, not saved standalone | **E3** training (`06_e3_finetune.py`); rebuilt each time **e3** mode is first selected in the demo | `insightface` ships ArcFace as inference-only ONNX; converting the *same* weights (verified, not a substitute model) is what makes a genuine partial fine-tune possible without an unfair, differently-sourced backbone. |
| **E3 model** (last 15% of the converted backbone unfrozen + a new 512→30 head, BatchNorm frozen) | `runs/e3_model.pt` | **E3** training (`06_e3_finetune.py`); **e3** mode in the demo | Tests whether adapting backbone weights (not just fitting a new decision boundary on frozen features) closes any remaining accuracy gap. |

## Considered but not used

- **A small CNN trained from scratch**, originally planned as a baseline to show the benefit of pretraining. Dropped once E0 (99.6% with zero training) showed the pretrained embedding was already close to saturated — a from-scratch CNN was judged unlikely to add an informative data point, so effort went to the E1-E3 ablation ladder instead. See `docs/final_report.tex`, Section 9 and the deviations table.
- `1k3d68.onnx`, `2d106det.onnx`, `genderage.onnx` — bundled alongside `det_10g.onnx`/`w600k_r50.onnx` in the `buffalo_l` model pack, but never loaded (`allowed_modules=["detection"]` restricts `FaceAnalysis` to the detector only). Not copied into `models/buffalo_l/` in this repo.
