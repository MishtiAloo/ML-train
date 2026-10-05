"""Live demo: runs on this laptop, serves a web page that a phone (or this
machine's browser) opens over the LAN. The camera captures frames in the
browser and POSTs them here; the server runs detect -> align -> embed ->
match against the fixed 30-person benchmark gallery and returns JSON.

Models in the dropdown
    E0  pretrained ArcFace backbone, no fine-tuning
    E1  last residual block fine-tuned      (per-fold checkpoint)
    E2  last residual block + embedding head fine-tuned (per-fold checkpoint)

A second dropdown picks which fold's E1/E2 checkpoint to use. A person is
"unseen" only for the fold that holds them in its test split; the page shows
the fold's train/val/test people so you can say which case is on screen.

Recognition is a lookup, not a classifier: the model only produces a
512-d embedding, which is compared by cosine similarity with the 30 fixed
benchmark vectors (runs/e1e2/person_benchmarks.npz, built once from each
person's best clean photo by the pretrained model). The top match is
reported only if its similarity reaches --threshold (default 0.3), the same
acceptance rule used in evaluation; otherwise the answer is UNKNOWN.

Run (from the project root):
    python scripts/28_demo_server.py --show-names
Then open the printed https://<LAN-IP>:5000 on a phone on the same Wi-Fi,
accept the self-signed certificate warning once, and tap Start.

HTTPS is required: browsers only expose the camera on a secure context.
The certificate is generated once into runs/certs/ and reused.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import align_crop  # noqa: E402  -- the exact alignment used to build every training crop

ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__)

EXPERIMENTS = {
    "e0": "E0 - pretrained backbone (no fine-tuning)",
    "e1": "E1 - last residual block fine-tuned",
    "e2": "E2 - block + embedding head fine-tuned",
}
FOLD_EXPERIMENTS = ("e1", "e2")  # these load a per-fold checkpoint; e0 uses the pretrained model

STATE = {
    "det": None,
    "rec_onnx": None,          # pretrained ArcFace (ONNX) -- used by e0
    "backbone": None,          # one converted torch backbone, weights swapped on demand
    "loaded_key": None,        # (exp, fold) currently in `backbone`
    "benchmarks": None,        # (30, 512) unit vectors
    "persons": [],
    "names": {},
    "folds": {},
    "threshold": 0.3,
    "exp": "e0",
    "fold": 1,
    "emb_buffer": deque(maxlen=6),
    "model_paths": {},
}


# ---------------------------------------------------------------- models
def load_detector(det_path: str, det_size: int, det_thresh: float):
    from insightface.model_zoo import get_model

    det = get_model(det_path, providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=-1, input_size=(det_size, det_size), det_thresh=det_thresh)
    return det


def load_recognizer(rec_path: str):
    from insightface.model_zoo import get_model

    rec = get_model(rec_path, providers=["CPUExecutionProvider"])
    rec.prepare(ctx_id=-1)
    return rec


def fold_checkpoint(exp: str, fold: int) -> Path:
    return ROOT / "runs" / "e1e2" / "folds" / f"fold{fold}" / f"{exp}_person_model.pt"


def ensure_backbone(exp: str, fold: int):
    """Convert the ONNX backbone once, then swap in the requested fold's
    weights. Keeps memory at a single backbone instead of one per mode."""
    import torch
    from onnx2torch import convert
    import onnx

    key = (exp, fold)
    if STATE["loaded_key"] == key:
        return STATE["backbone"]
    if STATE["backbone"] is None:
        # convert(path) shape-infers through a temp file that Windows keeps
        # locked; passing a loaded ModelProto avoids that entirely
        STATE["backbone"] = convert(onnx.load(STATE["model_paths"]["rec"]))
    ckpt = torch.load(fold_checkpoint(exp, fold), map_location="cpu", weights_only=False)
    assert ckpt["exp"] == exp, f"checkpoint says {ckpt['exp']}, expected {exp}"
    STATE["backbone"].load_state_dict(ckpt["state_dict"])
    STATE["backbone"].eval()
    STATE["loaded_key"] = key
    return STATE["backbone"]


def embed(crop_bgr: np.ndarray, exp: str, fold: int) -> np.ndarray:
    """512-d unit embedding of one aligned 112x112 crop."""
    if exp == "e0":
        feat = STATE["rec_onnx"].get_feat([crop_bgr])[0]          # not unit length
        return feat / (np.linalg.norm(feat) + 1e-9)
    import torch

    backbone = ensure_backbone(exp, fold)
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.tensor(np.transpose((img - 127.5) / 127.5, (2, 0, 1))[None])
    with torch.no_grad():
        out = backbone(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
    feat = out.numpy()[0].astype(np.float64)
    return feat / (np.linalg.norm(feat) + 1e-9)


def classify(emb: np.ndarray):
    """Nearest benchmark, with the acceptance threshold applied."""
    sims = STATE["benchmarks"] @ emb
    best = int(sims.argmax())
    score = float(sims[best])
    person = STATE["persons"][best]
    accepted = score >= STATE["threshold"]
    runner_up = float(np.sort(sims)[-2])
    return (person if accepted else "UNKNOWN"), person, score, runner_up


def display_name(pid: str) -> str:
    return STATE["names"].get(pid, pid)


def person_role(pid: str, fold: int) -> str:
    f = STATE["folds"].get(fold)
    if not f:
        return ""
    if pid in f["test"]:
        return "unseen by this fold (test person)"
    if pid in f["val"]:
        return "validation person for this fold"
    if pid in f["train"]:
        return "trained on in this fold"
    return ""


def pick_largest(bboxes, kpss):
    if bboxes is None or len(bboxes) == 0:
        return None, None
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    i = int(np.argmax(areas))
    return bboxes[i], (kpss[i] if kpss is not None else None)


def load_names(path: Path) -> dict:
    names = {}
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(p\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|", line)
        if m:
            names[m.group(1).strip().lower()] = m.group(3).strip()
    return names


def ensure_cert(cert_dir: Path):
    cert_path, key_path = cert_dir / "demo_cert.pem", cert_dir / "demo_key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "face-id-demo")])
    cert = (x509.CertificateBuilder().subject_name(subject).issuer_name(issuer)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("*")]), critical=False)
            .sign(key, hashes.SHA256()))
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(encoding=serialization.Encoding.PEM,
                                           format=serialization.PrivateFormat.TraditionalOpenSSL,
                                           encryption_algorithm=serialization.NoEncryption()))
    return str(cert_path), str(key_path)


# ---------------------------------------------------------------- page
PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Occlusion-Robust Face ID</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family: system-ui, sans-serif; text-align:center; }
  video { width:100%; max-width:480px; background:#000; }
  #result { font-size:1.7em; margin:12px 0 2px; min-height:1.3em; }
  .ok { color:#4caf50; } .unknown { color:#ff5252; } .none { color:#888; }
  #score, #role { font-size:.9em; color:#aaa; }
  #modelNote { font-size:.8em; color:#888; margin-top:4px; padding:0 10px; }
  button { font-size:1.05em; padding:9px 20px; margin:6px; border-radius:8px; border:none; background:#2979ff; color:#fff; }
  button:disabled { background:#555; }
  select { font-size:1em; padding:6px; margin:5px; max-width:92%; }
</style>
</head>
<body>
<h3>Occlusion-Robust Face Identification</h3>
<video id="video" autoplay playsinline muted></video>
<div id="result" class="none">Press Start</div>
<div id="score"></div>
<div id="role"></div>
<div>
  <select id="modelSelect">{{ model_options|safe }}</select>
  <select id="foldSelect">{{ fold_options|safe }}</select>
</div>
<div id="modelNote"></div>
<div><select id="camSelect"></select></div>
<div>
  <button id="startBtn">Start</button>
  <button id="stopBtn" disabled>Stop</button>
  <button id="switchBtn">Switch Camera</button>
</div>
<canvas id="canvas" style="display:none;"></canvas>
<script>
const video=document.getElementById('video'), canvas=document.getElementById('canvas');
const resultEl=document.getElementById('result'), scoreEl=document.getElementById('score');
const roleEl=document.getElementById('role'), noteEl=document.getElementById('modelNote');
const camSelect=document.getElementById('camSelect'), modelSelect=document.getElementById('modelSelect');
const foldSelect=document.getElementById('foldSelect');
let running=false, stream=null, facingMode='user', frameIntervalMs=500;

async function setModel() {
  modelSelect.disabled = foldSelect.disabled = true;
  const wasRunning = running; running = false;
  noteEl.textContent = 'Loading model...';
  try {
    const res = await fetch('/set_model', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({exp: modelSelect.value, fold: parseInt(foldSelect.value)})});
    const data = await res.json();
    noteEl.textContent = data.ok ? data.note : ('Error: ' + data.error);
    frameIntervalMs = data.heavy ? 900 : 500;
  } catch (e) { noteEl.textContent = 'Error: ' + e; }
  modelSelect.disabled = foldSelect.disabled = false;
  if (wasRunning) { running = true; loop(); }
}

function constraints() {
  const base = {width:{ideal:1280}, height:{ideal:720}};
  if (camSelect.value) base.deviceId = {exact: camSelect.value}; else base.facingMode = {ideal: facingMode};
  return base;
}
async function listCams() {
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const cams = devs.filter(d => d.kind === 'videoinput');
    const prev = camSelect.value; camSelect.innerHTML = '';
    const auto = document.createElement('option'); auto.value=''; auto.textContent='Auto'; camSelect.appendChild(auto);
    cams.forEach((d,i) => { const o=document.createElement('option'); o.value=d.deviceId;
      o.textContent = d.label || ('Camera ' + (i+1)); camSelect.appendChild(o); });
    if (prev && cams.some(d => d.deviceId === prev)) camSelect.value = prev;
  } catch (e) {}
}
async function openStream() {
  stream = await navigator.mediaDevices.getUserMedia({video: constraints(), audio:false});
  video.srcObject = stream; await listCams();
}
async function start() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    resultEl.className='unknown'; resultEl.textContent='No camera API';
    scoreEl.textContent='Open this page over https:// for the camera to work.'; return;
  }
  try { await openStream(); } catch (e) {
    resultEl.className='unknown'; resultEl.textContent='Camera error'; scoreEl.textContent=e.name+': '+e.message; return; }
  running = true;
  document.getElementById('startBtn').disabled = true;
  document.getElementById('stopBtn').disabled = false;
  loop();
}
function stop() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  resultEl.className='none'; resultEl.textContent='Stopped'; scoreEl.textContent=''; roleEl.textContent='';
}
async function switchCam() {
  facingMode = (facingMode === 'user') ? 'environment' : 'user';
  camSelect.value = '';
  if (!running) return;
  if (stream) stream.getTracks().forEach(t => t.stop());
  try { await openStream(); } catch (e) { resultEl.textContent = 'Camera error'; }
}
camSelect.onchange = async () => {
  if (!running) return;
  if (stream) stream.getTracks().forEach(t => t.stop());
  try { await openStream(); } catch (e) {}
};
async function loop() {
  if (!running) return;
  const w = video.videoWidth, h = video.videoHeight;
  if (w && h) {
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);
    canvas.toBlob(async (blob) => {
      try {
        const form = new FormData(); form.append('frame', blob, 'f.jpg');
        const res = await fetch('/predict', {method:'POST', body: form});
        render(await res.json());
      } catch (e) { resultEl.className='unknown'; resultEl.textContent='Error: ' + e; }
      if (running) setTimeout(loop, frameIntervalMs);
    }, 'image/jpeg', 0.85);
  } else if (running) setTimeout(loop, 200);
}
function render(d) {
  if (!d.face_found) { resultEl.className='none'; resultEl.textContent='No face detected';
    scoreEl.textContent=''; roleEl.textContent=''; return; }
  if (d.label === 'UNKNOWN') {
    resultEl.className='unknown'; resultEl.textContent='UNKNOWN';
    scoreEl.textContent = 'best match ' + d.display_top + '  score ' + d.score.toFixed(3) +
      ' < threshold ' + d.threshold.toFixed(2);
  } else {
    resultEl.className='ok'; resultEl.textContent = d.display_name;
    scoreEl.textContent = '[' + d.exp.toUpperCase() + (d.uses_fold ? ' fold ' + d.fold : '') + ']  score ' +
      d.score.toFixed(3) + '  (next ' + d.runner_up.toFixed(3) + ', threshold ' + d.threshold.toFixed(2) + ')';
  }
  roleEl.textContent = d.role || '';
}
document.getElementById('startBtn').onclick = start;
document.getElementById('stopBtn').onclick = stop;
document.getElementById('switchBtn').onclick = switchCam;
modelSelect.onchange = setModel; foldSelect.onchange = setModel;
if (navigator.mediaDevices) listCams();
setModel();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    models = "\n".join(f'<option value="{k}"{" selected" if k == STATE["exp"] else ""}>{v}</option>'
                       for k, v in EXPERIMENTS.items())
    folds = "\n".join(f'<option value="{k}"{" selected" if k == STATE["fold"] else ""}>fold {k}</option>'
                      for k in sorted(STATE["folds"]))
    return render_template_string(PAGE, model_options=models, fold_options=folds)


@app.route("/set_model", methods=["POST"])
def set_model():
    payload = request.get_json(force=True, silent=True) or {}
    exp = payload.get("exp", "e0")
    fold = int(payload.get("fold", 1))
    if exp not in EXPERIMENTS:
        return jsonify({"ok": False, "error": f"unknown model {exp}"}), 400
    if exp in FOLD_EXPERIMENTS and not fold_checkpoint(exp, fold).exists():
        return jsonify({"ok": False, "error": f"no checkpoint for {exp} fold {fold}"}), 400
    t0 = time.time()
    try:
        if exp in FOLD_EXPERIMENTS:
            ensure_backbone(exp, fold)
    except Exception as e:  # noqa: BLE001 -- report to the page instead of a 500 page
        return jsonify({"ok": False, "error": f"failed to load {exp}: {e}"}), 500
    STATE["exp"], STATE["fold"] = exp, fold
    STATE["emb_buffer"].clear()  # fine-tuned backbones produce a different embedding space
    f = STATE["folds"].get(fold, {})
    if exp in FOLD_EXPERIMENTS:
        note = (f"{EXPERIMENTS[exp]} - fold {fold}. Unseen in this fold: "
                f"{', '.join(display_name(p) for p in f.get('test', []))}.")
    else:
        note = "E0 is the pretrained backbone with no fine-tuning; the fold selector does not apply."
    return jsonify({"ok": True, "note": note, "heavy": exp in FOLD_EXPERIMENTS,
                    "load_ms": int((time.time() - t0) * 1000)})


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame"}), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "decode failed"}), 400

    exp, fold = STATE["exp"], STATE["fold"]
    bboxes, kpss = STATE["det"].detect(img, max_num=0, metric="default")
    bbox, kps = pick_largest(bboxes, kpss)
    if bbox is None or kps is None:
        STATE["emb_buffer"].clear()
        return jsonify({"face_found": False, "exp": exp, "fold": fold})

    crop = align_crop(img, kps, image_size=112)
    STATE["emb_buffer"].append(embed(crop, exp, fold))
    avg = np.mean(STATE["emb_buffer"], axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-9)

    label, top_person, score, runner_up = classify(avg)
    return jsonify({
        "face_found": True, "exp": exp, "fold": fold,
        "uses_fold": exp in FOLD_EXPERIMENTS,
        "label": label, "top_person": top_person,
        "display_name": display_name(top_person) if label != "UNKNOWN" else "UNKNOWN",
        "display_top": display_name(top_person),
        "score": score, "runner_up": runner_up, "threshold": STATE["threshold"],
        "role": person_role(top_person, fold) if exp in FOLD_EXPERIMENTS else "",
        "bbox": [float(v) for v in bbox[:4]],
        "frames_averaged": len(STATE["emb_buffer"]),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--benchmarks", default=str(ROOT / "runs" / "e1e2" / "person_benchmarks.npz"))
    ap.add_argument("--folds-file", default=str(ROOT / "metadata" / "e1e2_person_folds.json"))
    ap.add_argument("--names", default=str(ROOT / "metadata" / "person_id_mapping.txt"))
    ap.add_argument("--show-names", action="store_true", help="show real names instead of p## ids")
    ap.add_argument("--threshold", type=float, default=0.3, help="acceptance threshold; below it the answer is UNKNOWN")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--det-thresh", type=float, default=0.5)
    ap.add_argument("--buffer", type=int, default=6, help="frames averaged before matching")
    ap.add_argument("--exp", default="e0", choices=list(EXPERIMENTS))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    det_path = models_dir / "det_10g.onnx"
    rec_path = models_dir / "w600k_r50.onnx"
    for p in (det_path, rec_path, Path(args.benchmarks), Path(args.folds_file)):
        if not p.exists():
            raise SystemExit(f"missing required file: {p}")
    STATE["model_paths"] = {"det": str(det_path), "rec": str(rec_path)}

    bench = np.load(args.benchmarks, allow_pickle=True)
    b = bench["benchmarks"].astype(np.float64)
    STATE["benchmarks"] = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    STATE["persons"] = [str(p) for p in bench["persons"]]
    STATE["threshold"] = args.threshold
    STATE["emb_buffer"] = deque(maxlen=args.buffer)
    STATE["exp"], STATE["fold"] = args.exp, args.fold

    setup = json.loads(Path(args.folds_file).read_text(encoding="utf-8"))
    STATE["folds"] = {f["fold"]: f for f in setup["folds"]}
    if args.show_names:
        STATE["names"] = load_names(Path(args.names))

    print("Loading detector and pretrained recognizer (CPU)...", flush=True)
    STATE["det"] = load_detector(str(det_path), args.det_size, args.det_thresh)
    STATE["rec_onnx"] = load_recognizer(str(rec_path))

    available = {e: (e not in FOLD_EXPERIMENTS or all(fold_checkpoint(e, k).exists() for k in STATE["folds"]))
                 for e in EXPERIMENTS}
    print(f"Gallery: {len(STATE['persons'])} people, threshold {STATE['threshold']}, "
          f"averaging {args.buffer} frames")
    print(f"Models: " + ", ".join(f"{e}{'' if ok else ' (checkpoints missing)'}" for e, ok in available.items()))

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
    except Exception:
        lan_ip = "<this machine's LAN IP>"
    finally:
        s.close()

    cert_path, key_path = ensure_cert(ROOT / "runs" / "certs")
    print(f"\nOpen on a phone on the same Wi-Fi:\n\n    https://{lan_ip}:{args.port}\n")
    print("Accept the self-signed certificate warning once, then tap Start.")
    print("On this machine you can also use https://localhost:%d\n" % args.port)

    from gevent.pywsgi import WSGIServer
    WSGIServer(("0.0.0.0", args.port), app, certfile=cert_path, keyfile=key_path).serve_forever()


if __name__ == "__main__":
    main()
