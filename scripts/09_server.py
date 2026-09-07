"""Local demo server: runs on the laptop, serves a web page a phone opens
over the LAN. The phone's camera captures frames in the browser and
POSTs them here; this runs detect -> align -> embed -> classify and
returns the result as JSON.

A dropdown on the page lets you switch between all four experiments
live, without restarting the server:
    gallery  E0-style nearest-centroid on frozen embeddings (the only
             mode with a calibrated UNKNOWN-rejection threshold)
    e1       trained head, clean-faces-only (94.29% test acc)
    e2       trained head, mixed-condition (99.90% test acc)
    e3       partial backbone fine-tune (100.00% test acc; runs its own
             onnx2torch-converted backbone, loaded lazily on first
             selection since it's ~166MB and slower on CPU)
Every mode except gallery is closed-set only: it always names one of
the 30 enrolled people, with no "this might be a stranger" rejection,
because E1/E2/E3 were never calibrated with a rejection threshold.

Uses the IDENTICAL align_crop() from common.py that built the training
crops -- required so predictions here match what was measured in
docs/training_plan.md.

Setup (already done if you followed along):
    pip install flask opencv-python-headless insightface onnxruntime numpy

Run:
    python 09_server.py --gallery "D:\ML train\runs\gallery.npz"

Then on the phone (same WiFi as this laptop):
    https://<this-laptop's-LAN-IP>:5000
    (find the IP with `ipconfig` -> IPv4 Address, e.g. 192.168.10.105)

MUST be https, not http -- mobile browsers only expose the camera
(getUserMedia) on a "secure context" (https:// or localhost). Plain
http on a LAN IP silently has no camera API at all: no prompt, no
error, just a black box. This server uses a self-signed certificate
(pip install pyopenssl), so the phone's browser will show a privacy/
security warning on first visit -- this is expected, not a real
problem. Tap through it:
    Chrome (Android): "Advanced" -> "Proceed to <ip> (unsafe)"
    Safari (iOS): "Show Details" -> "visit this website" -> "Visit Website"
You only need to do this once per phone.

Uses the buffalo_l detection+recognition models from models/buffalo_l/ in
this repo if present; otherwise falls back to insightface's own cache,
auto-downloading them (~280MB, one-time, needs internet) to ~/.insightface/models/.

If the phone can't connect: Windows Firewall likely needs an inbound
allow rule for this port on first launch -- accept the prompt, or add
one manually for python.exe / port 5000 on private networks.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import align_crop, default_model_root, default_rec_model_path  # noqa: E402

app = Flask(__name__)

STATE = {
    "det": None,
    "rec": None,               # frozen ArcFace recognizer -- used by gallery/e1/e2
    "gallery": None,           # {"centroids", "persons", "threshold"}
    "heads": {},                # {"e1": {"weight", "persons"}, "e2": {...}}
    "e3": None,                  # lazily built: {"embed_fn", "weight", "persons"}
    "e3_checkpoint_path": None,
    "names": {},
    "emb_buffer": deque(maxlen=6),
    "active_mode": None,  # set in main() to the best available mode once loaded
    "available_modes": [],      # [(value, label, rejects_unknown), ...]
}


def ensure_cert(cert_dir: Path):
    """Generate a persistent self-signed cert+key if one doesn't exist yet.
    Reused across restarts, so the phone only has to click through the
    browser's untrusted-certificate warning once per browser, not once
    per launch."""
    cert_path = cert_dir / "demo_cert.pem"
    key_path = cert_dir / "demo_key.pem"
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    import datetime

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "face-id-demo")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("*")]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    return str(cert_path), str(key_path)


def build_models():
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    print("Loading detector (from models/buffalo_l/ if present, else ~/.insightface)...")
    det = FaceAnalysis(name="buffalo_l", root=default_model_root(),
                        allowed_modules=["detection"], providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=-1, det_size=(640, 640))

    rec = get_model(default_rec_model_path(), providers=["CPUExecutionProvider"])
    rec.prepare(ctx_id=-1)
    print("Models ready.")
    return det, rec


def load_gallery(path):
    data = np.load(path, allow_pickle=True)
    return data["centroids"], list(data["persons"]), float(data["threshold"])


def load_head(path: str) -> dict:
    """Load an E1/E2 trained head (scripts/05_train_head.py) -- just a
    512x30 weight matrix. Normalized once here so /predict is a single
    matmul + argmax, matching what 10_full_eval.py measured."""
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    weight = ckpt["state_dict"]["weight"].numpy().astype(np.float64)
    weight = weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-9)
    return {"weight": weight, "persons": ckpt["persons"]}


def build_e3(checkpoint_path: str) -> dict:
    """Lazily build the E3 model: the onnx2torch-converted, partially
    fine-tuned ArcFace backbone (scripts/06_e3_finetune.py) plus its own
    head. Distinct embedding space from the frozen `rec` model used by
    gallery/e1/e2 -- never mix embeddings across modes."""
    import onnx
    import torch
    import torch.nn.functional as F
    from onnx2torch import convert

    # onnx2torch's convert(path) shape-infers via a NamedTemporaryFile in
    # the model's own directory, which onnx.load() then can't re-open on
    # Windows (the handle is still exclusively locked) -- PermissionError.
    # Passing an already-loaded ModelProto instead routes through an
    # in-memory shape-inference path with no temp file at all.
    model_proto = onnx.load(default_rec_model_path())
    backbone = convert(model_proto)
    backbone.eval()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    backbone_sd = {k[len("backbone."):]: v for k, v in state_dict.items() if k.startswith("backbone.")}
    backbone.load_state_dict(backbone_sd)

    weight = state_dict["weight"].numpy().astype(np.float64)
    weight = weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-9)

    def embed_fn(crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5) / 127.5
        x = torch.tensor(np.transpose(img, (2, 0, 1))[None], dtype=torch.float32)
        with torch.no_grad():
            out = backbone(x)
            if isinstance(out, (tuple, list)):
                out = out[0]
            emb = F.normalize(out, dim=1).numpy()[0].astype(np.float64)
        return emb

    return {"embed_fn": embed_fn, "weight": weight, "persons": ckpt["persons"]}


def load_display_names(mapping_path):
    names = {}
    if not mapping_path or not Path(mapping_path).exists():
        return names
    import re
    for line in Path(mapping_path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(p\d+)\s*\|\s*([^|]+)\|\s*([^|]+)\|", line)
        if m:
            names[m.group(1).strip().lower()] = f"{m.group(2).strip()}/{m.group(3).strip()}"
    return names


def pick_best_face(faces):
    if not faces:
        return None
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return max(faces, key=area)


PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face ID Demo</title>
<style>
  body { margin:0; background:#111; color:#eee; font-family: system-ui, sans-serif; text-align:center; }
  #wrap { position:relative; display:inline-block; max-width:100%; }
  video { width:100%; max-width:480px; background:#000; }
  #overlay { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }
  #result { font-size:1.6em; margin:14px 0; min-height:1.4em; }
  .ok { color:#4caf50; }
  .unknown { color:#ff5252; }
  .none { color:#888; }
  #score { font-size:0.9em; color:#aaa; }
  #modelNote { font-size:0.8em; color:#888; margin-top:2px; }
  button { font-size:1.1em; padding:10px 24px; margin:8px; border-radius:8px; border:none; background:#2979ff; color:#fff; }
  button:disabled { background:#555; }
  select { font-size:1em; padding:6px; margin:6px; max-width:90%; }
  select:disabled { opacity:0.6; }
</style>
</head>
<body>
<h2>Occlusion-Robust Face ID</h2>
<div id="wrap">
  <video id="video" autoplay playsinline muted></video>
</div>
<div id="result" class="none">Press Start</div>
<div id="score"></div>
<div>
  <select id="modelSelect">{{ model_options|safe }}</select>
</div>
<div id="modelNote"></div>
<div>
  <select id="camSelect"></select>
</div>
<div>
  <button id="startBtn">Start</button>
  <button id="stopBtn" disabled>Stop</button>
  <button id="switchBtn">Switch Camera</button>
</div>
<canvas id="canvas" style="display:none;"></canvas>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultEl = document.getElementById('result');
const scoreEl = document.getElementById('score');
const camSelect = document.getElementById('camSelect');
const modelSelect = document.getElementById('modelSelect');
const modelNoteEl = document.getElementById('modelNote');
let running = false;
let stream = null;
let facingMode = 'environment'; // toggled by Switch Camera when no specific device is chosen
let frameIntervalMs = 400; // slower for E3 (CPU-heavy, converted PyTorch graph)

async function switchModel() {
  const mode = modelSelect.value;
  modelSelect.disabled = true;
  const wasRunning = running;
  running = false; // pause the capture loop while the model swaps
  modelNoteEl.textContent = mode === 'e3'
    ? 'Loading E3 (fine-tuned model)... first switch can take 10-30s on CPU.'
    : 'Switching model...';
  try {
    const res = await fetch('/set_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    const data = await res.json();
    if (!data.ok) {
      modelNoteEl.textContent = 'Error: ' + data.error;
    } else {
      frameIntervalMs = (mode === 'e3') ? 900 : 400;
      const opt = modelSelect.options[modelSelect.selectedIndex];
      modelNoteEl.textContent = opt.dataset.rejectsUnknown === '1'
        ? 'This mode rejects unenrolled faces as UNKNOWN.'
        : 'Closed-set: always names one of the 30 enrolled people (no UNKNOWN rejection).';
    }
  } catch (e) {
    modelNoteEl.textContent = 'Error switching model: ' + e;
  }
  modelSelect.disabled = false;
  if (wasRunning) { running = true; loop(); }
}

function currentVideoConstraints() {
  // Prefer an explicit device pick (lets you avoid a blurry ultra-wide lens);
  // otherwise fall back to a facingMode request.
  const deviceId = camSelect.value;
  const base = {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 }
  };
  if (deviceId) {
    base.deviceId = { exact: deviceId };
  } else {
    base.facingMode = { ideal: facingMode };
  }
  return base;
}

async function refreshDeviceList() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === 'videoinput');
    const prev = camSelect.value;
    camSelect.innerHTML = '';
    const autoOpt = document.createElement('option');
    autoOpt.value = '';
    autoOpt.textContent = 'Auto (facingMode)';
    camSelect.appendChild(autoOpt);
    cams.forEach((d, i) => {
      const opt = document.createElement('option');
      opt.value = d.deviceId;
      opt.textContent = d.label || ('Camera ' + (i + 1));
      camSelect.appendChild(opt);
    });
    if (prev && cams.some(d => d.deviceId === prev)) camSelect.value = prev;
  } catch (e) { /* labels/enumeration may be unavailable before permission; ignore */ }
}

async function openStream() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: currentVideoConstraints(),
    audio: false
  });
  video.srcObject = stream;
  await refreshDeviceList(); // labels become available only after permission is granted
}

async function start() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    resultEl.className = 'unknown';
    resultEl.textContent = 'No camera API available';
    scoreEl.textContent = 'This page must be opened as https:// (not http://) for the camera to work on a phone.';
    return;
  }
  try {
    await openStream();
  } catch (e) {
    resultEl.className = 'unknown';
    resultEl.textContent = 'Camera error';
    scoreEl.textContent = e.name + ': ' + e.message;
    return;
  }
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
  resultEl.className = 'none';
  resultEl.textContent = 'Stopped';
  scoreEl.textContent = '';
}

async function switchCamera() {
  facingMode = (facingMode === 'environment') ? 'user' : 'environment';
  camSelect.value = ''; // switching facing mode overrides an explicit device pick
  if (!running) return;
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    await openStream();
  } catch (e) {
    resultEl.className = 'unknown';
    resultEl.textContent = 'Camera error';
    scoreEl.textContent = e.name + ': ' + e.message;
  }
}

camSelect.onchange = async () => {
  if (!running) return;
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    await openStream();
  } catch (e) {
    resultEl.className = 'unknown';
    resultEl.textContent = 'Camera error';
    scoreEl.textContent = e.name + ': ' + e.message;
  }
};

async function loop() {
  if (!running) return;
  const w = video.videoWidth, h = video.videoHeight;
  if (w && h) {
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(video, 0, 0, w, h);
    canvas.toBlob(async (blob) => {
      try {
        const form = new FormData();
        form.append('frame', blob, 'frame.jpg');
        const res = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();
        render(data);
      } catch (e) {
        resultEl.className = 'unknown';
        resultEl.textContent = 'Error: ' + e;
      }
      if (running) setTimeout(loop, frameIntervalMs);
    }, 'image/jpeg', 0.85);
  } else {
    if (running) setTimeout(loop, 200);
  }
}

function render(data) {
  if (!data.face_found) {
    resultEl.className = 'none';
    resultEl.textContent = 'No face detected';
    scoreEl.textContent = '';
    return;
  }
  const thresholdTxt = (data.threshold !== null && data.threshold !== undefined)
    ? ' / threshold ' + data.threshold.toFixed(3) : '';
  if (data.label === 'UNKNOWN') {
    resultEl.className = 'unknown';
    resultEl.textContent = 'UNKNOWN';
    scoreEl.textContent = 'score ' + data.score.toFixed(3) + thresholdTxt;
  } else {
    resultEl.className = 'ok';
    resultEl.textContent = data.display_name || data.label;
    scoreEl.textContent = '[' + data.mode + ']  score ' + data.score.toFixed(3) + thresholdTxt;
  }
}

document.getElementById('startBtn').onclick = start;
document.getElementById('stopBtn').onclick = stop;
document.getElementById('switchBtn').onclick = switchCamera;
modelSelect.onchange = switchModel;
if (modelSelect.options.length) {
  const opt = modelSelect.options[modelSelect.selectedIndex];
  modelNoteEl.textContent = opt.dataset.rejectsUnknown === '1'
    ? 'This mode rejects unenrolled faces as UNKNOWN.'
    : 'Closed-set: always names one of the 30 enrolled people (no UNKNOWN rejection).';
}
if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
  refreshDeviceList();
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    options_html = "\n".join(
        f'<option value="{value}" data-rejects-unknown="{1 if rejects else 0}"'
        f'{" selected" if value == STATE["active_mode"] else ""}>{label}</option>'
        for value, label, rejects in STATE["available_modes"]
    )
    return render_template_string(PAGE, model_options=options_html)


@app.route("/set_model", methods=["POST"])
def set_model():
    payload = request.get_json(force=True, silent=True) or {}
    mode = payload.get("mode")
    valid = {v for v, _, _ in STATE["available_modes"]}
    if mode not in valid:
        return jsonify({"ok": False, "error": f"mode '{mode}' not available"}), 400

    load_ms = 0
    if mode == "e3" and STATE["e3"] is None:
        t0 = time.time()
        try:
            STATE["e3"] = build_e3(STATE["e3_checkpoint_path"])
        except Exception as e:  # noqa: BLE001 -- report any load failure to the page, not a 500 traceback
            return jsonify({"ok": False, "error": f"failed to load E3: {e}"}), 500
        load_ms = int((time.time() - t0) * 1000)

    STATE["active_mode"] = mode
    STATE["emb_buffer"].clear()  # different mode = a different (incompatible) embedding space
    return jsonify({"ok": True, "mode": mode, "load_ms": load_ms})


def classify(mode: str, avg_emb: np.ndarray):
    """Returns (label, score, threshold_or_None)."""
    if mode == "gallery":
        g = STATE["gallery"]
        sims = g["centroids"] @ avg_emb
        best = int(sims.argmax())
        score = float(sims[best])
        label = g["persons"][best] if score >= g["threshold"] else "UNKNOWN"
        return label, score, g["threshold"]

    h = STATE["heads"][mode] if mode in STATE["heads"] else STATE["e3"]
    sims = h["weight"] @ avg_emb
    best = int(sims.argmax())
    return h["persons"][best], float(sims[best]), None


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame"}), 400

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "decode failed"}), 400

    mode = STATE["active_mode"]

    faces = STATE["det"].get(img)
    face = pick_best_face(faces)
    if face is None:
        STATE["emb_buffer"].clear()
        return jsonify({"face_found": False, "mode": mode})

    crop = align_crop(img, face.kps, image_size=112)

    if mode == "e3":
        if STATE["e3"] is None:
            return jsonify({"error": "E3 not loaded yet -- select it from the dropdown first"}), 400
        emb = STATE["e3"]["embed_fn"](crop)
    else:
        emb = STATE["rec"].get_feat([crop])[0]
        emb = emb / (np.linalg.norm(emb) + 1e-9)

    STATE["emb_buffer"].append(emb)
    avg = np.mean(STATE["emb_buffer"], axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-9)

    label, score, threshold = classify(mode, avg)
    display_name = "UNKNOWN" if label == "UNKNOWN" else STATE["names"].get(label, label)

    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    return jsonify({
        "face_found": True,
        "mode": mode,
        "label": label,
        "display_name": display_name,
        "score": score,
        "threshold": threshold,
        "rejects_unknown": mode == "gallery",
        "bbox": [x1, y1, x2, y2],
    })


# Gallery (nearest-centroid on the frozen backbone) is suppressed from the
# demo for now -- flip back to True to re-enable it. The code, weights, and
# --gallery/--threshold flags are untouched; this only controls whether it's
# registered as a selectable mode at startup.
SHOW_GALLERY_MODE = False


def main():
    runs_dir = Path(__file__).resolve().parent.parent / "runs"
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=str(runs_dir / "gallery.npz"))
    ap.add_argument("--names", default=str(Path(__file__).resolve().parent.parent / "metadata" / "person_id_mapping.txt"))
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=None, help="override calibrated threshold")
    ap.add_argument("--show-names", action="store_true", help="show real member/person names instead of p## IDs")
    ap.add_argument("--e1-head", default=str(runs_dir / "e1_head.pt"), help="path to E1 head checkpoint (skip if missing)")
    ap.add_argument("--e2-head", default=str(runs_dir / "e2_head.pt"), help="path to E2 head checkpoint (skip if missing)")
    ap.add_argument("--e3-checkpoint", default=str(runs_dir / "e3_model.pt"), help="path to E3 checkpoint (skip if missing; loaded lazily on first selection)")
    args = ap.parse_args()

    STATE["available_modes"] = []

    if SHOW_GALLERY_MODE and Path(args.gallery).exists():
        centroids, persons, threshold = load_gallery(args.gallery)
        if args.threshold is not None:
            threshold = args.threshold
        STATE["gallery"] = {"centroids": centroids, "persons": persons, "threshold": threshold}
        STATE["available_modes"].append(("gallery", "Gallery -- nearest-centroid (UNKNOWN-aware)", True))
    STATE["names"] = load_display_names(args.names) if args.show_names else {}

    for mode, path, label in [
        ("e1", args.e1_head, "E1 -- head, clean-only (94.3%)"),
        ("e2", args.e2_head, "E2 -- head, mixed-condition (99.9%)"),
    ]:
        if Path(path).exists():
            print(f"Loading {mode} head: {path}")
            STATE["heads"][mode] = load_head(path)
            STATE["available_modes"].append((mode, label, False))
        else:
            print(f"Skipping {mode}: {path} not found")

    if Path(args.e3_checkpoint).exists():
        STATE["e3_checkpoint_path"] = args.e3_checkpoint
        STATE["available_modes"].append(("e3", "E3 -- fine-tuned backbone (100.0%, loads on first use)", False))
    else:
        print(f"Skipping e3: {args.e3_checkpoint} not found")

    if not STATE["available_modes"]:
        raise RuntimeError(
            "No experiment checkpoints found (gallery.npz / e1_head.pt / e2_head.pt / "
            "e3_model.pt) -- nothing for the demo to serve. Build at least one first."
        )
    # Default to e2 (best accuracy that doesn't need E3's lazy CPU-heavy load);
    # fall back to whatever else was found.
    available = {v for v, _, _ in STATE["available_modes"]}
    STATE["active_mode"] = "e2" if "e2" in available else STATE["available_modes"][0][0]

    STATE["det"], STATE["rec"] = build_models()

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
    except Exception:
        lan_ip = "<this machine's LAN IP>"
    finally:
        s.close()

    mode_names = ", ".join(v for v, _, _ in STATE["available_modes"])
    print(f"\nModes available in the dropdown: {mode_names} (default: {STATE['active_mode']})")
    print(f"\nOn your phone (same WiFi), open:\n\n    https://{lan_ip}:{args.port}\n")
    print("Your browser will warn about the self-signed certificate -- this is")
    print("expected. Tap through it (Advanced -> Proceed / Visit Website).\n")

    # Flask's built-in dev server (ssl_context='adhoc') doesn't reliably
    # complete real TLS handshakes from mobile browsers -- it can accept
    # the TCP connection and then hang indefinitely on the handshake
    # (ALPN/TLS1.3 negotiation), which looks exactly like "took too long
    # / timed out" with zero server-side error. gevent's WSGIServer with
    # a real (persistent, self-signed) certificate handles this properly.
    from gevent.pywsgi import WSGIServer
    cert_path, key_path = ensure_cert(Path(__file__).resolve().parent.parent / "runs" / "certs")
    http_server = WSGIServer(("0.0.0.0", args.port), app, certfile=cert_path, keyfile=key_path)
    print("Server ready (gevent). Press Ctrl+C to stop.\n")
    http_server.serve_forever()


if __name__ == "__main__":
    main()
