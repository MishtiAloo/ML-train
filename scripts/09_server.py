"""Local demo server: runs on the laptop, serves a web page a phone opens
over the LAN. The phone's camera captures frames in the browser and
POSTs them here; this runs detect -> align -> embed -> gallery match ->
UNKNOWN rejection and returns the result as JSON.

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
    "rec": None,
    "centroids": None,
    "persons": None,
    "threshold": None,
    "names": {},
    "emb_buffer": deque(maxlen=6),
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
  button { font-size:1.1em; padding:10px 24px; margin:8px; border-radius:8px; border:none; background:#2979ff; color:#fff; }
  button:disabled { background:#555; }
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
  <select id="camSelect" style="font-size:1em; padding:6px; margin:6px; max-width:90%;"></select>
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
let running = false;
let stream = null;
let facingMode = 'environment'; // toggled by Switch Camera when no specific device is chosen

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
      if (running) setTimeout(loop, 400);
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
  } else if (data.label === 'UNKNOWN') {
    resultEl.className = 'unknown';
    resultEl.textContent = 'UNKNOWN';
    scoreEl.textContent = 'score ' + data.score.toFixed(3) + ' / threshold ' + data.threshold.toFixed(3);
  } else {
    resultEl.className = 'ok';
    resultEl.textContent = data.display_name || data.label;
    scoreEl.textContent = 'score ' + data.score.toFixed(3) + ' / threshold ' + data.threshold.toFixed(3);
  }
}

document.getElementById('startBtn').onclick = start;
document.getElementById('stopBtn').onclick = stop;
document.getElementById('switchBtn').onclick = switchCamera;
if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
  refreshDeviceList();
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame"}), 400

    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "decode failed"}), 400

    faces = STATE["det"].get(img)
    face = pick_best_face(faces)
    if face is None:
        STATE["emb_buffer"].clear()
        return jsonify({"face_found": False})

    crop = align_crop(img, face.kps, image_size=112)
    feat = STATE["rec"].get_feat([crop])[0]
    feat = feat / (np.linalg.norm(feat) + 1e-9)

    STATE["emb_buffer"].append(feat)
    avg = np.mean(STATE["emb_buffer"], axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-9)

    sims = STATE["centroids"] @ avg
    best = int(sims.argmax())
    score = float(sims[best])
    persons = STATE["persons"]
    threshold = STATE["threshold"]

    if score < threshold:
        label = "UNKNOWN"
        display_name = "UNKNOWN"
    else:
        label = persons[best]
        display_name = STATE["names"].get(label, label)

    x1, y1, x2, y2 = [float(v) for v in face.bbox]
    return jsonify({
        "face_found": True,
        "label": label,
        "display_name": display_name,
        "score": score,
        "threshold": threshold,
        "bbox": [x1, y1, x2, y2],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=str(Path(__file__).resolve().parent.parent / "runs" / "gallery.npz"))
    ap.add_argument("--names", default=str(Path(__file__).resolve().parent.parent / "metadata" / "person_id_mapping.txt"))
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--threshold", type=float, default=None, help="override calibrated threshold")
    ap.add_argument("--show-names", action="store_true", help="show real member/person names instead of p## IDs")
    args = ap.parse_args()

    centroids, persons, threshold = load_gallery(args.gallery)
    if args.threshold is not None:
        threshold = args.threshold
    STATE["centroids"] = centroids
    STATE["persons"] = persons
    STATE["threshold"] = threshold
    STATE["names"] = load_display_names(args.names) if args.show_names else {}

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

    print(f"\nGallery: {len(persons)} people, threshold={threshold:.4f}")
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
