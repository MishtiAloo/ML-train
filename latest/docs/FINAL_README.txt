FINAL DATASET REPORT

final.tex is the editable LaTeX source; final.pdf is the compiled report.
The revised PDF has 26 pages. SOURCE_SECTION_COVERAGE.md maps the source sections
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
Administrative facts absent from the sources are marked not reported.

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
