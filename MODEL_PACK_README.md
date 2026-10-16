# Model Pack for the Webcam and Phone Demos

This archive contains the large model files that are intentionally not uploaded to GitHub. It is designed to be extracted into the root of a fresh clone of this repository.

## What is in the archive

```text
latest/
├── models/
│   ├── det_10g.onnx
│   └── w600k_r50.onnx
└── runs/e1e2/folds/
    ├── fold1/{e1,e2}_person_model.pt
    ├── fold2/{e1,e2}_person_model.pt
    ├── fold3/{e1,e2}_person_model.pt
    ├── fold4/{e1,e2}_person_model.pt
    ├── fold5/{e1,e2}_person_model.pt
    └── fold6/{e1,e2}_person_model.pt
```

E0 does not need an E0 checkpoint: it uses `w600k_r50.onnx` directly. The E1 and E2 options need their checkpoint for the selected fold.

The repository already supplies the demo scripts, fixed benchmark gallery, fold definitions, and optional display-name mapping. They are not duplicated in this archive.

## Install the pack after cloning

1. Clone or pull the GitHub repository.
2. Put `modelPack.zip` in the repository root—the directory containing `README.md` and the `latest` folder.
3. From PowerShell in that directory, run:

```powershell
Expand-Archive -LiteralPath .\modelPack.zip -DestinationPath . -Force
```

The archive already contains the `latest/...` directory structure. Do not extract it inside `latest`, or the result will incorrectly become `latest/latest/...`.

Confirm the important files are in place:

```powershell
Test-Path .\latest\models\det_10g.onnx
Test-Path .\latest\models\w600k_r50.onnx
Test-Path .\latest\runs\e1e2\folds\fold1\e1_person_model.pt
Test-Path .\latest\runs\e1e2\folds\fold6\e2_person_model.pt
```

All four commands should print `True`.

## Create the Python environment

Python 3.11 is recommended because that is the version used to verify the demo.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements-demo.txt
```

Tkinter is required by the desktop webcam window. It is included with the standard Windows installer from python.org; if `import tkinter` fails, modify/reinstall Python and enable Tcl/Tk support.

## Run the desktop webcam demo

From the repository root:

```powershell
python .\latest\scripts\29_webcam_demo.py --show-names
```

If the wrong camera opens or camera 0 is unavailable:

```powershell
python .\latest\scripts\29_webcam_demo.py --show-names --camera 1
```

The model menu provides:

- E0: pretrained ONNX recognizer;
- E1: last residual block fine-tuned for the selected fold;
- E2: residual block and embedding head fine-tuned for the selected fold.

E1/E2 can be slower than E0 on a CPU because their ONNX backbone is converted to PyTorch and the selected checkpoint is loaded.

## Optional phone/browser demo

The same files also support the HTTPS phone demo:

```powershell
python .\latest\scripts\28_demo_server.py --show-names
```

Open the printed HTTPS address on a phone connected to the same local network and accept the self-signed certificate warning.

## Files created automatically

These are not required in the model pack:

- `latest/runs/certs/`: self-signed HTTPS certificate and key;
- `latest/runs/e1e2/demo_enrollments.npz`: people registered through either demo;
- `latest/runs/e1e2/enrolled_people/`: selected registration photos.

They are created when needed. Copy the enrollment NPZ and enrollment-photo directory separately only if registered people must be transferred to another computer.

