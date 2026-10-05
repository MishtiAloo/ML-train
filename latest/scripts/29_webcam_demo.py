"""Live demo on this laptop's own webcam, in a desktop window.

Same recognition and registration as 28_demo_server.py (the phone demo), with
no copied logic: this script loads 28's state with its own init_state() and
drives 28's /predict, /set_model and /enroll/* routes in-process through
Flask's test client. Same models, same 30-person gallery, same 0.3 threshold,
same quality gate, and the same persistent demo_enrollments.npz -- a person
registered here is recognized by the phone demo too, and vice versa. The
People... window lists the gallery and deletes demo-registered people only.

The window is drawn with tkinter because the installed OpenCV build has no
GUI support (cv2.imshow is unavailable). Inference runs on a background
thread so the video stays smooth while each frame is being matched.

Run (from the project root):
    python latest/scripts/29_webcam_demo.py --show-names
    python latest/scripts/29_webcam_demo.py --show-names --camera 1   # other camera

Every option of 28_demo_server.py (threshold, buffer, enrollment limits, ...)
is accepted here too; --port is ignored.
"""

from __future__ import annotations

import importlib.util
import io
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

_spec = importlib.util.spec_from_file_location(
    "demo_server", Path(__file__).resolve().parent / "28_demo_server.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

GREEN, RED, GREY, YELLOW = "#4caf50", "#ff5252", "#888888", "#ffd54f"


class Worker(threading.Thread):
    """Owns every call into the demo server, so models and STATE are only
    ever touched from this one thread."""

    def __init__(self, client):
        super().__init__(daemon=True)
        self.client = client
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.frame = None                 # newest camera frame, consumed once
        self.result = None                # last /predict response
        self.note = ""                    # model note or error
        self.enroll_msg = ""
        self.enrolling = False
        self.running = True
        self.people_version = 0           # bumped whenever the gallery changes

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def run(self):
        while self.running:
            try:
                self.handle(self.commands.get_nowait())
                continue
            except queue.Empty:
                pass
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.02)
                continue
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                continue
            url = "/enroll/frame" if self.enrolling else "/predict"
            data = self.client.post(url, data={"frame": (io.BytesIO(jpg.tobytes()), "f.jpg")},
                                    content_type="multipart/form-data").get_json() or {}
            if url == "/predict":
                self.result = data
            elif "error" in data:
                self.enroll_msg = "Registration error: " + data["error"]
            elif data.get("complete"):
                self.enrolling = False
                self.enroll_msg = (f"{data['name']} registered. Best of {data['required']} "
                                   "accepted photos saved permanently.")
                self.people_version += 1
            else:
                self.enroll_msg = (f"{data['accepted_count']} / {data['required']} accepted. "
                                   f"{data['message']}")

    def handle(self, cmd):
        kind = cmd[0]
        if kind == "model":
            data = self.client.post("/set_model", json={"exp": cmd[1], "fold": cmd[2]}).get_json()
            self.note = data["note"] if data.get("ok") else "Error: " + data.get("error", "?")
            self.result = None
        elif kind == "enroll":
            data = self.client.post("/enroll/start", json={"name": cmd[1]}).get_json()
            if data.get("ok"):
                self.enrolling, self.result = True, None
                self.enroll_msg = f"Registering {data['name']}: look straight at the camera."
            else:
                self.enroll_msg = "Registration error: " + data.get("error", "?")
        elif kind == "cancel":
            self.client.post("/enroll/cancel")
            self.enrolling = False
            self.enroll_msg = "Registration cancelled."
        elif kind == "delete":
            data = self.client.post("/enroll/delete", json={"person": cmd[1]}).get_json()
            self.enroll_msg = (f"{data['name']} deleted." if data.get("ok")
                               else "Delete error: " + data.get("error", "?"))
            self.result = None
            self.people_version += 1


class App:
    def __init__(self, root, cap, worker, args):
        self.root, self.cap, self.worker = root, cap, worker
        root.title("Occlusion-Robust Face ID - webcam")
        root.configure(bg="#111")

        self.video = tk.Label(root, bg="#000")
        self.video.pack()
        self.result = tk.Label(root, text="Starting...", font=("Segoe UI", 20), fg=GREY, bg="#111")
        self.result.pack(pady=(8, 0))
        self.detail = tk.Label(root, text="", fg="#aaa", bg="#111")
        self.detail.pack()
        self.enroll = tk.Label(root, text="", fg=YELLOW, bg="#111", wraplength=620)
        self.enroll.pack()

        row = tk.Frame(root, bg="#111")
        row.pack(pady=4)
        self.exp_keys = list(demo.EXPERIMENTS)
        self.model = ttk.Combobox(row, state="readonly", width=42,
                                  values=[demo.EXPERIMENTS[k] for k in self.exp_keys])
        self.model.current(self.exp_keys.index(args.exp))
        self.model.pack(side="left", padx=4)
        folds = sorted(demo.STATE["folds"])
        self.fold = ttk.Combobox(row, state="readonly", width=8, values=[f"fold {k}" for k in folds])
        self.fold.current(folds.index(args.fold) if args.fold in folds else 0)
        self.fold.pack(side="left", padx=4)
        self.model.bind("<<ComboboxSelected>>", self.set_model)
        self.fold.bind("<<ComboboxSelected>>", self.set_model)
        self.note = tk.Label(root, text="", fg=GREY, bg="#111", wraplength=620)
        self.note.pack()

        row = tk.Frame(root, bg="#111")
        row.pack(pady=(4, 10))
        tk.Label(row, text="New person name:", fg="#eee", bg="#111").pack(side="left")
        self.name = tk.Entry(row, width=24)
        self.name.pack(side="left", padx=4)
        self.name.bind("<Return>", lambda e: self.register())
        tk.Button(row, text="Register New Person", command=self.register).pack(side="left", padx=4)
        tk.Button(row, text="Cancel Registration",
                  command=lambda: worker.commands.put(("cancel",))).pack(side="left", padx=4)
        tk.Button(row, text="People...", command=self.open_people).pack(side="left", padx=4)
        self.people_win = None
        self.list_client = demo.app.test_client()   # read-only /people calls from this thread

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.set_model()
        self.tick()

    def set_model(self, _event=None):
        exp = self.exp_keys[self.model.current()]
        fold = int(self.fold.get().split()[-1])
        self.note.config(text="Loading model...")
        self.worker.commands.put(("model", exp, fold))

    def register(self):
        name = self.name.get().strip()
        if not name:
            self.worker.enroll_msg = "Type the new person's name first."
            return
        self.worker.commands.put(("enroll", name))
        self.name.delete(0, "end")

    def open_people(self):
        if self.people_win is not None and self.people_win.winfo_exists():
            self.people_win.lift()
            return
        win = self.people_win = tk.Toplevel(self.root)
        win.title("People")
        win.configure(bg="#111", padx=12, pady=10)
        self.fixed_hdr = tk.Label(win, fg="#eee", bg="#111", font=("Segoe UI", 10, "bold"))
        self.fixed_hdr.pack(anchor="w")
        self.fixed_lbl = tk.Label(win, fg="#aaa", bg="#111", wraplength=420, justify="left")
        self.fixed_lbl.pack(anchor="w", pady=(0, 8))
        self.reg_hdr = tk.Label(win, fg="#eee", bg="#111", font=("Segoe UI", 10, "bold"))
        self.reg_hdr.pack(anchor="w")
        self.reg_list = tk.Listbox(win, width=50, height=8)
        self.reg_list.pack(fill="x")
        tk.Button(win, text="Delete selected", command=self.delete_selected).pack(pady=(6, 0))
        self.refresh_people()

    def refresh_people(self):
        data = self.list_client.get("/people").get_json()
        self.registered = data["registered"]
        self.fixed_hdr.config(text=f"Original gallery ({len(data['fixed'])}, cannot be deleted)")
        self.fixed_lbl.config(text=", ".join(p["name"] for p in data["fixed"]))
        self.reg_hdr.config(text=f"Registered in the demo ({len(self.registered)})")
        self.reg_list.delete(0, "end")
        for p in self.registered:
            self.reg_list.insert("end", p["name"])
        self.seen_version = self.worker.people_version

    def delete_selected(self):
        sel = self.reg_list.curselection()
        if not sel:
            messagebox.showinfo("People", "Select a registered person first.", parent=self.people_win)
            return
        p = self.registered[sel[0]]
        if messagebox.askyesno("Delete", f"Permanently delete {p['name']}?", parent=self.people_win):
            self.worker.commands.put(("delete", p["id"]))

    def tick(self):
        ok, frame = self.cap.read()
        if ok:
            self.worker.submit(frame.copy())
            self.show(frame)
        self.note.config(text=self.worker.note)
        self.enroll.config(text=self.worker.enroll_msg)
        if (self.people_win is not None and self.people_win.winfo_exists()
                and self.worker.people_version != self.seen_version):
            self.refresh_people()
        self.root.after(30, self.tick)

    def show(self, frame):
        w = self.worker
        d = w.result
        if w.enrolling:
            self.result.config(text="Registering...", fg=YELLOW)
            self.detail.config(text="")
        elif not d:
            self.result.config(text="Waiting for a frame...", fg=GREY)
            self.detail.config(text="")
        elif not d.get("face_found"):
            self.result.config(text="No face detected", fg=GREY)
            self.detail.config(text="")
        else:
            accepted = d["label"] != "UNKNOWN"
            color = GREEN if accepted else RED
            x1, y1, x2, y2 = (int(v) for v in d["bbox"])
            bgr = (80, 175, 76) if accepted else (82, 82, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, d["display_name"], (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)
            self.result.config(text=d["display_name"], fg=color)
            tag = d["exp"].upper() + (f" fold {d['fold']}" if d["uses_fold"] else "")
            if accepted:
                text = (f"[{tag}]  score {d['score']:.3f}  (next {d['runner_up']:.3f}, "
                        f"threshold {d['threshold']:.2f})")
            else:
                text = (f"best match {d['display_top']}  score {d['score']:.3f} "
                        f"< threshold {d['threshold']:.2f}")
            self.detail.config(text=text + (f"\n{d['role']}" if d.get("role") else ""))
        h, wd = frame.shape[:2]
        scale = min(1.0, 640 / wd)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(wd * scale), int(h * scale)))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.video.config(image=img)
        self.video.image = img                     # keep a reference, or tkinter drops it

    def close(self):
        if self.worker.enrolling:
            self.worker.commands.put(("cancel",))
            time.sleep(0.2)
        self.worker.running = False
        self.cap.release()
        self.root.destroy()


def main():
    ap = demo.build_parser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 for a second camera)")
    args = ap.parse_args()
    demo.init_state(args)

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened() or not cap.read()[0]:
        raise SystemExit(f"could not open webcam {args.camera}; try --camera 1")

    worker = Worker(demo.app.test_client())
    worker.start()
    root = tk.Tk()
    App(root, cap, worker, args)
    root.mainloop()


if __name__ == "__main__":
    main()
