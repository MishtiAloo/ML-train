# Source-section coverage in the revised final report

The final report keeps the DOCX template's 17-section structure. The technical
report's eight main sections and their substantive subsections are incorporated
at the locations below, rather than replacing the required template numbering.
Final pagination: 26 pages. Figure numbers changed when diagrams were added.

| Technical report section | Location in final.tex / final.pdf |
|---|---|
| 1. Introduction & Objective | Section 5, “Introduction and objective”; Section 10.1 |
| 2.1 Collection & composition | Sections 6 and 7.3; Figure 2 |
| 2.2 Detection and alignment pipeline | Section 8; Figures 4–5 |
| 2.3 Sample data | Section 7.3; Figure 3 uses p11 raw/aligned photos, with EXIF orientation applied |
| 2.4 RetinaFace failure cases | Section 9; Figure 6 shows original examples 1, 2, 5, 6, 7; total failures remain 12 |
| 3.1 Why person-disjoint | Sections 8 and 10.2 |
| 3.2 Six-fold person-disjoint rotation | Section 10.2; fold table, rotation diagram (Figure 7), full assignment grid (Figure 8) |
| 3.3 Benchmark embedding selection | Section 10.3; Figure 9 |
| 3.4 Acceptance / UNKNOWN threshold | Sections 10.1 and 10.5; full inference diagram (Figure 15) |
| 4.1 ArcFace recognizer architecture | Section 10.4, page 12; full input-to-output architecture (Figure 10), without residual-block internals as requested; scope comparison (Figure 11), module/parameter table |
| 4.2 Detector architecture | Section 10.4 continued; Figure 12, three-scale tensor table; named RetinaFace as requested |
| 5.1 Common training protocol | Section 10.4 continued; training diagram (Figure 13), margin schedule (Figure 14), hyperparameters, frozen BN statistics, checkpoint selection |
| 5.2 E0 baseline | Section 10.4; E0 branch in Figure 11, zero-epoch and parameter rules |
| 5.3 E1 last residual block | Section 10.4; E1 branch in Figure 11, exact modules, gradient-flow explanation |
| 5.4 E2 block plus embedding head | Section 10.4; E2 branch in Figure 11, exact modules, expanded head and epoch-0 fallback |
| 6.1 Pooled accuracy and outcome types | Section 10.7; pooled metrics/outcome tables; rejected ArcFace examples in Appendix B.1, Figure 20 |
| 6.2 Per-fold detail | Section 10.7 per-fold table; validation accuracy curves in Appendix B.3, Figure 24 |
| 6.3 Per-occlusion breakdown | Section 10.7 table; chart in Appendix B.2, Figure 22 |
| 6.4 Learning curves | Main Section 10.7, page 19; Figures 16–17 overlay train/validation losses on shared axes, one colour per fold |
| 6.5 Confusion matrices and macro metrics | Section 10.7 macro table; Appendix B contains all three 30-person matrices (Figures 19, 21, 23) |
| 7. Discussion & Conclusion | Section 10.7, explicitly labeled “Discussion and conclusion”; Section 12 limits the interpretation |
| 8. Reproduction & File Index | Section 7.1 file index; Section 10.6 ordered script sequence and reproducibility details |

The requested prior-report material is also retained: original submission details
on page 1; related papers in Section 5 and references; recruitment, consent and
capture protocol in Sections 6 and 13; cleaning in Section 8. The four original
quality-review rejection photos from prior Section 5 are reproduced in Appendix B,
Figure 18. Earlier superseded experiment results are not reinstated.

The user-provided dataset link appears on the cover and in Sections 1, 3, 11 and
14. Permission status is not assumed from the existence of a folder URL.

The former Figure 10 validation-accuracy plot is now Figure 24 in Appendix B.3.
The former small loss plots are now the taller Figures 16 and 17 in the main report.
