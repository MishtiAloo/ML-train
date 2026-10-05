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

The page can also register a new person. It asks for a name, accepts five
photos that pass face, landmark, blur, lighting, and upright-head checks,
then permanently saves the best photo's E0 embedding in a separate demo
gallery. Saved enrollments are merged into the lookup gallery at startup;
the fixed 30-person experiment gallery is never modified. The People button
lists the gallery and can permanently delete demo-registered people only.

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
import threading
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
    "enrollments_path": None,
    "enroll_dir": None,
    "enrolled_benchmarks": None,
    "enrolled_persons": [],
    "enrolled_names": [],
    "enrollment": None,
    "enroll_required": 5,
    "enroll_quality": {},
}
ENROLL_LOCK = threading.Lock()


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


def enrollment_quality(img: np.ndarray, bbox: np.ndarray, kps: np.ndarray, crop: np.ndarray):
    """Check that an enrollment frame is clear, well lit, and roughly frontal."""
    cfg = STATE["enroll_quality"]
    if kps is None or np.asarray(kps).shape != (5, 2) or not np.isfinite(kps).all():
        return False, "All five face landmarks must be visible.", {}

    h, w = img.shape[:2]
    points = np.asarray(kps, dtype=np.float64)
    if not ((points[:, 0] >= 0).all() and (points[:, 0] < w).all()
            and (points[:, 1] >= 0).all() and (points[:, 1] < h).all()):
        return False, "Keep the whole face inside the camera frame.", {}

    left_eye, right_eye, nose, left_mouth, right_mouth = points
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0
    geometry_ok = (left_eye[0] < right_eye[0] and left_mouth[0] < right_mouth[0]
                   and left_eye[0] < nose[0] < right_eye[0]
                   and eye_y < nose[1] < mouth_y)
    if not geometry_ok:
        return False, "Look straight at the camera with the full face visible.", {}

    tilt = abs(float(np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                           right_eye[0] - left_eye[0]))))
    if tilt > cfg["max_tilt"]:
        return False, f"Keep your head upright (tilt {tilt:.1f} degrees).", {"tilt": tilt}

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    light = float(gray.mean())
    det_score = float(bbox[4]) if len(bbox) > 4 else 0.0
    metrics = {"blur": blur, "light": light, "tilt": tilt, "det_score": det_score}
    if det_score < cfg["min_det_score"]:
        return False, "Face detection is uncertain. Move closer and look at the camera.", metrics
    if blur < cfg["min_blur"]:
        return False, "Image is blurry. Hold the camera and your head still.", metrics
    if light < cfg["min_light"]:
        return False, "Image is too dark. Move to better lighting.", metrics
    if light > cfg["max_light"]:
        return False, "Image is too bright. Avoid strong light behind or on the face.", metrics

    light_score = max(0.0, 1.0 - abs(light - 127.5) / 127.5)
    upright_score = max(0.0, 1.0 - tilt / max(cfg["max_tilt"], 1e-9))
    quality = det_score + 0.25 * min(blur / max(cfg["min_blur"], 1e-9), 3.0)
    quality += 0.25 * light_score + 0.25 * upright_score
    metrics["quality"] = quality
    return True, "Accepted.", metrics


def load_enrollments(path: Path):
    if not path.exists():
        return np.empty((0, 512), dtype=np.float64), [], []
    saved = np.load(path, allow_pickle=False)
    benchmarks = saved["benchmarks"].astype(np.float64)
    persons = [str(v) for v in saved["persons"]]
    names = [str(v) for v in saved["names"]]
    if benchmarks.ndim != 2 or benchmarks.shape[1] != 512 or len(benchmarks) != len(persons):
        raise ValueError(f"invalid enrollment gallery: {path}")
    if len(names) != len(persons):
        raise ValueError(f"enrollment names do not match persons: {path}")
    benchmarks /= np.linalg.norm(benchmarks, axis=1, keepdims=True) + 1e-9
    return benchmarks, persons, names


def write_enrollments(bench: np.ndarray, persons: list, names: list):
    """Atomically rewrite the demo gallery file; remove it when empty."""
    path = STATE["enrollments_path"]
    if not persons:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp_path, benchmarks=bench.astype(np.float32),
                        persons=np.array(persons), names=np.array(names))
    temp_path.replace(path)


def save_enrollment(name: str, emb: np.ndarray, crop: np.ndarray):
    """Persist one E0 benchmark, then add it to the live gallery."""
    with ENROLL_LOCK:
        used = set(STATE["persons"])
        number = 1
        while f"enrolled_{number:03d}" in used:
            number += 1
        person = f"enrolled_{number:03d}"

        crop_path = STATE["enroll_dir"] / f"{person}.jpg"
        STATE["enroll_dir"].mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"could not save enrollment crop: {crop_path}")

        old_bench = STATE["enrolled_benchmarks"]
        new_bench = np.vstack([old_bench, emb[None]])
        new_persons = STATE["enrolled_persons"] + [person]
        new_names = STATE["enrolled_names"] + [name]
        write_enrollments(new_bench, new_persons, new_names)

        STATE["enrolled_benchmarks"] = new_bench
        STATE["enrolled_persons"] = new_persons
        STATE["enrolled_names"] = new_names
        STATE["benchmarks"] = np.vstack([STATE["benchmarks"], emb[None]])
        STATE["persons"].append(person)
        STATE["names"][person] = name
        return person, crop_path


def delete_enrollment(person: str) -> str:
    """Permanently remove one demo-registered person. Fixed-gallery people
    are never in enrolled_persons, so they cannot be deleted."""
    with ENROLL_LOCK:
        i = STATE["enrolled_persons"].index(person)
        name = STATE["enrolled_names"][i]
        new_bench = np.delete(STATE["enrolled_benchmarks"], i, axis=0)
        new_persons = STATE["enrolled_persons"][:i] + STATE["enrolled_persons"][i + 1:]
        new_names = STATE["enrolled_names"][:i] + STATE["enrolled_names"][i + 1:]
        write_enrollments(new_bench, new_persons, new_names)
        (STATE["enroll_dir"] / f"{person}.jpg").unlink(missing_ok=True)

        g = STATE["persons"].index(person)
        STATE["benchmarks"] = np.delete(STATE["benchmarks"], g, axis=0)
        STATE["persons"] = STATE["persons"][:g] + STATE["persons"][g + 1:]
        STATE["enrolled_benchmarks"] = new_bench
        STATE["enrolled_persons"] = new_persons
        STATE["enrolled_names"] = new_names
        STATE["names"].pop(person, None)
        STATE["emb_buffer"].clear()
        return name


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
  #enrollStatus { font-size:.9em; color:#ffd54f; min-height:1.2em; padding:4px 10px; }
  #modelNote { font-size:.8em; color:#888; margin-top:4px; padding:0 10px; }
  button { font-size:1.05em; padding:9px 20px; margin:6px; border-radius:8px; border:none; background:#2979ff; color:#fff; }
  button:disabled { background:#555; }
  select { font-size:1em; padding:6px; margin:5px; max-width:92%; }
  #people { display:none; max-width:480px; margin:6px auto 16px; padding:0 10px; text-align:left; font-size:.9em; }
  #people h4 { margin:10px 0 4px; } #people .fixed { color:#aaa; line-height:1.5; }
  #people .row { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #333; }
  #people .row button { font-size:.85em; padding:5px 12px; background:#d32f2f; }
</style>
</head>
<body>
<h3>Occlusion-Robust Face Identification</h3>
<video id="video" autoplay playsinline muted></video>
<div id="result" class="none">Press Start</div>
<div id="score"></div>
<div id="role"></div>
<div id="enrollStatus"></div>
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
  <button id="registerBtn">Register New Person</button>
  <button id="cancelEnrollBtn" disabled>Cancel Registration</button>
  <button id="peopleBtn">People</button>
</div>
<div id="people"></div>
<canvas id="canvas" style="display:none;"></canvas>
<script>
const video=document.getElementById('video'), canvas=document.getElementById('canvas');
const resultEl=document.getElementById('result'), scoreEl=document.getElementById('score');
const roleEl=document.getElementById('role'), noteEl=document.getElementById('modelNote');
const enrollEl=document.getElementById('enrollStatus');
const camSelect=document.getElementById('camSelect'), modelSelect=document.getElementById('modelSelect');
const foldSelect=document.getElementById('foldSelect');
const registerBtn=document.getElementById('registerBtn'), cancelEnrollBtn=document.getElementById('cancelEnrollBtn');
let running=false, enrolling=false, stream=null, facingMode='user', frameIntervalMs=500;

async function setModel() {
  if (enrolling) return;
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
  if (enrolling) fetch('/enroll/cancel', {method:'POST'}).catch(() => {});
  enrolling = false;
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  registerBtn.disabled = false; cancelEnrollBtn.disabled = true;
  modelSelect.disabled = foldSelect.disabled = false;
  resultEl.className='none'; resultEl.textContent='Stopped'; scoreEl.textContent=''; roleEl.textContent='';
  enrollEl.textContent='';
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
  if (!running || enrolling) return;
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
async function startEnrollment() {
  if (!running) {
    enrollEl.textContent = 'Press Start and allow the camera before registering.';
    return;
  }
  const name = prompt('Enter the new person name:');
  if (!name || !name.trim()) return;
  try {
    const res = await fetch('/enroll/start', {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:name.trim()})});
    const data = await res.json();
    if (!res.ok || !data.ok) { enrollEl.textContent = 'Registration error: ' + (data.error || res.status); return; }
    enrolling = true;
    registerBtn.disabled = true; cancelEnrollBtn.disabled = false;
    modelSelect.disabled = foldSelect.disabled = true;
    resultEl.className='none'; resultEl.textContent='Registering ' + data.name;
    scoreEl.textContent=''; roleEl.textContent='';
    enrollEl.textContent='0 / ' + data.required + ' accepted. Look straight at the camera.';
    enrollmentLoop();
  } catch (e) { enrollEl.textContent = 'Registration error: ' + e; }
}
async function cancelEnrollment() {
  try { await fetch('/enroll/cancel', {method:'POST'}); } catch (e) {}
  enrolling = false;
  registerBtn.disabled = false; cancelEnrollBtn.disabled = true;
  modelSelect.disabled = foldSelect.disabled = false;
  enrollEl.textContent='Registration cancelled.';
  if (running) loop();
}
async function enrollmentLoop() {
  if (!running || !enrolling) return;
  const w=video.videoWidth, h=video.videoHeight;
  if (!w || !h) { setTimeout(enrollmentLoop, 250); return; }
  canvas.width=w; canvas.height=h; canvas.getContext('2d').drawImage(video, 0, 0, w, h);
  canvas.toBlob(async (blob) => {
    try {
      const form=new FormData(); form.append('frame', blob, 'enroll.jpg');
      const res=await fetch('/enroll/frame', {method:'POST', body:form});
      const data=await res.json();
      if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
      if (data.complete) {
        enrolling=false; registerBtn.disabled=false; cancelEnrollBtn.disabled=true;
        modelSelect.disabled=foldSelect.disabled=false;
        resultEl.className='ok'; resultEl.textContent=data.name + ' registered';
        enrollEl.textContent='Best of ' + data.required + ' accepted photos saved permanently.';
        if (peopleEl.style.display === 'block') loadPeople();
        if (running) loop();
        return;
      }
      enrollEl.textContent=data.accepted_count + ' / ' + data.required + ' accepted. ' + data.message;
    } catch (e) { enrollEl.textContent='Registration error: ' + e; }
    if (running && enrolling) setTimeout(enrollmentLoop, 700);
  }, 'image/jpeg', 0.92);
}
const peopleEl=document.getElementById('people');
function heading(text) { const h=document.createElement('h4'); h.textContent=text; return h; }
async function loadPeople() {
  try {
    const data = await (await fetch('/people')).json();
    peopleEl.replaceChildren(heading('Original gallery (' + data.fixed.length + ', cannot be deleted)'));
    const fixed=document.createElement('div'); fixed.className='fixed';
    fixed.textContent = data.fixed.map(p => p.name).join(', ');
    peopleEl.appendChild(fixed);
    peopleEl.appendChild(heading('Registered in the demo (' + data.registered.length + ')'));
    if (!data.registered.length) {
      const none=document.createElement('div'); none.className='fixed'; none.textContent='Nobody yet.';
      peopleEl.appendChild(none);
    }
    data.registered.forEach(p => {
      const row=document.createElement('div'); row.className='row';
      const label=document.createElement('span'); label.textContent=p.name;
      const del=document.createElement('button'); del.textContent='Delete';
      del.onclick = () => deletePerson(p.id, p.name);
      row.append(label, del); peopleEl.appendChild(row);
    });
  } catch (e) { peopleEl.textContent = 'Could not load the list: ' + e; }
}
async function togglePeople() {
  if (peopleEl.style.display === 'block') { peopleEl.style.display='none'; return; }
  await loadPeople(); peopleEl.style.display='block';
}
async function deletePerson(id, name) {
  if (!confirm('Permanently delete ' + name + '?')) return;
  try {
    const res = await fetch('/enroll/delete', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({person: id})});
    const data = await res.json();
    enrollEl.textContent = data.ok ? (data.name + ' deleted.') : ('Delete error: ' + data.error);
  } catch (e) { enrollEl.textContent = 'Delete error: ' + e; }
  loadPeople();
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
registerBtn.onclick = startEnrollment; cancelEnrollBtn.onclick = cancelEnrollment;
document.getElementById('peopleBtn').onclick = togglePeople;
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


@app.route("/enroll/start", methods=["POST"])
def enroll_start():
    payload = request.get_json(force=True, silent=True) or {}
    name = " ".join(str(payload.get("name", "")).strip().split())
    if not name:
        return jsonify({"ok": False, "error": "name is required"}), 400
    if len(name) > 80:
        return jsonify({"ok": False, "error": "name must be 80 characters or fewer"}), 400
    if STATE["enrollment"] is not None:
        return jsonify({"ok": False, "error": "another registration is already active"}), 409
    if any(name.casefold() == saved.casefold() for saved in STATE["enrolled_names"]):
        return jsonify({"ok": False, "error": "that name is already registered"}), 409
    STATE["emb_buffer"].clear()
    STATE["enrollment"] = {"name": name, "samples": []}
    return jsonify({"ok": True, "name": name, "required": STATE["enroll_required"]})


@app.route("/enroll/cancel", methods=["POST"])
def enroll_cancel():
    STATE["enrollment"] = None
    return jsonify({"ok": True})


@app.route("/people")
def people():
    enrolled = set(STATE["enrolled_persons"])
    return jsonify({
        "fixed": [{"id": p, "name": display_name(p)} for p in STATE["persons"] if p not in enrolled],
        "registered": [{"id": p, "name": n}
                       for p, n in zip(STATE["enrolled_persons"], STATE["enrolled_names"])],
    })


@app.route("/enroll/delete", methods=["POST"])
def enroll_delete():
    person = str((request.get_json(force=True, silent=True) or {}).get("person", ""))
    if person not in STATE["enrolled_persons"]:
        return jsonify({"ok": False, "error": "only people registered in the demo can be deleted"}), 400
    try:
        name = delete_enrollment(person)
    except Exception as exc:  # noqa: BLE001 -- report to the page instead of a 500 page
        return jsonify({"ok": False, "error": f"could not delete: {exc}"}), 500
    return jsonify({"ok": True, "person": person, "name": name})


@app.route("/enroll/frame", methods=["POST"])
def enroll_frame():
    session = STATE["enrollment"]
    if session is None:
        return jsonify({"error": "no active registration"}), 409
    file = request.files.get("frame")
    if file is None:
        return jsonify({"error": "no frame"}), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "decode failed"}), 400

    bboxes, kpss = STATE["det"].detect(img, max_num=0, metric="default")
    if bboxes is not None and len(bboxes) > 1:
        return jsonify({"complete": False, "accepted": False,
                        "accepted_count": len(session["samples"]),
                        "required": STATE["enroll_required"],
                        "message": "Only one person should be visible during registration."})
    bbox, kps = pick_largest(bboxes, kpss)
    if bbox is None or kps is None:
        return jsonify({"complete": False, "accepted": False,
                        "accepted_count": len(session["samples"]),
                        "required": STATE["enroll_required"],
                        "message": "No clear face detected."})

    crop = align_crop(img, kps, image_size=112)
    accepted, message, metrics = enrollment_quality(img, bbox, kps, crop)
    if not accepted:
        return jsonify({"complete": False, "accepted": False,
                        "accepted_count": len(session["samples"]),
                        "required": STATE["enroll_required"], "message": message,
                        "metrics": metrics})

    # The fixed gallery was built with E0. New benchmarks must use that same
    # embedding space even when E1 or E2 is selected for live recognition.
    emb = embed(crop, "e0", STATE["fold"])
    if STATE["enrollment"] is not session:
        return jsonify({"error": "registration was cancelled"}), 409
    session["samples"].append({"embedding": emb, "crop": crop.copy(),
                               "quality": metrics["quality"]})
    accepted_count = len(session["samples"])
    if accepted_count < STATE["enroll_required"]:
        return jsonify({"complete": False, "accepted": True,
                        "accepted_count": accepted_count,
                        "required": STATE["enroll_required"], "message": message,
                        "metrics": metrics})

    best = max(session["samples"], key=lambda sample: sample["quality"])
    try:
        person, crop_path = save_enrollment(session["name"], best["embedding"], best["crop"])
    except Exception as exc:  # noqa: BLE001 -- return a useful registration error
        return jsonify({"error": f"could not save registration: {exc}"}), 500
    name = session["name"]
    STATE["enrollment"] = None
    STATE["emb_buffer"].clear()
    return jsonify({"complete": True, "accepted": True, "accepted_count": accepted_count,
                    "required": STATE["enroll_required"], "person": person, "name": name,
                    "crop_path": str(crop_path)})


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--benchmarks", default=str(ROOT / "runs" / "e1e2" / "person_benchmarks.npz"))
    ap.add_argument("--folds-file", default=str(ROOT / "metadata" / "e1e2_person_folds.json"))
    ap.add_argument("--names", default=str(ROOT / "metadata" / "person_id_mapping.txt"))
    ap.add_argument("--show-names", action="store_true", help="show real names instead of p## ids")
    ap.add_argument("--enrollments", default=str(ROOT / "runs" / "e1e2" / "demo_enrollments.npz"),
                    help="persistent gallery for people registered through the demo")
    ap.add_argument("--enroll-dir", default=str(ROOT / "runs" / "e1e2" / "enrolled_people"),
                    help="directory that stores the selected benchmark crop for each new person")
    ap.add_argument("--enroll-samples", type=int, default=5,
                    help="number of quality-approved photos required for registration")
    ap.add_argument("--enroll-min-blur", type=float, default=80.0,
                    help="minimum Laplacian variance for an enrollment crop")
    ap.add_argument("--enroll-min-light", type=float, default=55.0,
                    help="minimum mean grayscale value for an enrollment crop")
    ap.add_argument("--enroll-max-light", type=float, default=205.0,
                    help="maximum mean grayscale value for an enrollment crop")
    ap.add_argument("--enroll-max-tilt", type=float, default=15.0,
                    help="maximum absolute eye-line angle in degrees")
    ap.add_argument("--enroll-min-det-score", type=float, default=0.65,
                    help="minimum detector confidence for an enrollment photo")
    ap.add_argument("--threshold", type=float, default=0.3, help="acceptance threshold; below it the answer is UNKNOWN")
    ap.add_argument("--det-size", type=int, default=640)
    ap.add_argument("--det-thresh", type=float, default=0.5)
    ap.add_argument("--buffer", type=int, default=6, help="frames averaged before matching")
    ap.add_argument("--exp", default="e0", choices=list(EXPERIMENTS))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--port", type=int, default=5000)
    return ap


def init_state(args):
    """Load models, gallery, folds and saved enrollments into STATE.
    Shared with 29_webcam_demo.py, which drives the same routes locally."""
    if args.enroll_samples < 2:
        raise SystemExit("--enroll-samples must be at least 2")
    if not (0 <= args.enroll_min_light < args.enroll_max_light <= 255):
        raise SystemExit("enrollment light limits must satisfy 0 <= min < max <= 255")
    if args.enroll_min_blur < 0 or args.enroll_max_tilt <= 0:
        raise SystemExit("enrollment blur must be non-negative and tilt must be positive")

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
    STATE["enrollments_path"] = Path(args.enrollments)
    STATE["enroll_dir"] = Path(args.enroll_dir)
    STATE["enroll_required"] = args.enroll_samples
    STATE["enroll_quality"] = {
        "min_blur": args.enroll_min_blur,
        "min_light": args.enroll_min_light,
        "max_light": args.enroll_max_light,
        "max_tilt": args.enroll_max_tilt,
        "min_det_score": args.enroll_min_det_score,
    }

    setup = json.loads(Path(args.folds_file).read_text(encoding="utf-8"))
    STATE["folds"] = {f["fold"]: f for f in setup["folds"]}
    if args.show_names:
        STATE["names"] = load_names(Path(args.names))

    enrolled_bench, enrolled_persons, enrolled_names = load_enrollments(STATE["enrollments_path"])
    overlap = set(STATE["persons"]) & set(enrolled_persons)
    if overlap:
        raise SystemExit(f"enrollment IDs overlap the fixed gallery: {sorted(overlap)}")
    STATE["enrolled_benchmarks"] = enrolled_bench
    STATE["enrolled_persons"] = enrolled_persons
    STATE["enrolled_names"] = enrolled_names
    if len(enrolled_bench):
        STATE["benchmarks"] = np.vstack([STATE["benchmarks"], enrolled_bench])
        STATE["persons"].extend(enrolled_persons)
    STATE["names"].update(dict(zip(enrolled_persons, enrolled_names)))

    print("Loading detector and pretrained recognizer (CPU)...", flush=True)
    STATE["det"] = load_detector(str(det_path), args.det_size, args.det_thresh)
    STATE["rec_onnx"] = load_recognizer(str(rec_path))

    available = {e: (e not in FOLD_EXPERIMENTS or all(fold_checkpoint(e, k).exists() for k in STATE["folds"]))
                 for e in EXPERIMENTS}
    print(f"Gallery: {len(STATE['persons'])} people, threshold {STATE['threshold']}, "
          f"averaging {args.buffer} frames ({len(enrolled_persons)} demo enrollment(s))")
    print(f"Models: " + ", ".join(f"{e}{'' if ok else ' (checkpoints missing)'}" for e, ok in available.items()))


def main():
    args = build_parser().parse_args()
    init_state(args)

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
