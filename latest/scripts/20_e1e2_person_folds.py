"""E1/E2 v3: 6-fold rotation of the person-disjoint split (generalises
18_e1e2_person_split.py's single 20/5/5 split so every one of the 30
people is tested exactly once, not just 5 of them).

Same random person order as 18_e1e2_person_split.py (seed 42): the 30
people are shuffled once, then cut into 6 consecutive blocks of 5.
Fold k: test = block[k], val = block[(k+1) mod 6], train = the other 20.
Fold 1 here is therefore IDENTICAL to 18_e1e2_person_split.py's single
split (same test/val/train people).

Benchmark ("one best" clean regular-light photo, highest detection
score) and query exclusions (benchmark photo + its burst group + md5
copies) are per-person and fold-independent -- computed once, shared by
all 6 folds, identical to 18_e1e2_person_split.py's values.

Usage:
    python 20_e1e2_person_folds.py --manifest /home/tahmid/metadata/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

import numpy as np

SEED = 42
PEOPLE_PER_BLOCK = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/tahmid/metadata/manifest.csv")
    ap.add_argument("--embeddings", default="/home/tahmid/metadata/embeddings.npz")
    ap.add_argument("--setup-out", default="/home/tahmid/metadata/e1e2_person_folds.json")
    ap.add_argument("--bench-out", default="/home/tahmid/runs/e1e2/person_benchmarks.npz")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    persons = sorted(set(r["person_id"] for r in rows))
    assert len(persons) == 30

    rng = random.Random(SEED)
    shuffled = persons.copy()
    rng.shuffle(shuffled)
    n_blocks = len(persons) // PEOPLE_PER_BLOCK
    blocks = [shuffled[i * PEOPLE_PER_BLOCK:(i + 1) * PEOPLE_PER_BLOCK] for i in range(n_blocks)]

    folds = []
    for k in range(n_blocks):
        test = sorted(blocks[k])
        val = sorted(blocks[(k + 1) % n_blocks])
        train = sorted(p for p in persons if p not in test and p not in val)
        folds.append({"fold": k + 1, "train": train, "val": val, "test": test})

    # ---- one-best (highest det_score, clean, regular-light) benchmark photo per person ----
    # fold-independent: identical to 18_e1e2_person_split.py
    bench_rows = {}
    for p in persons:
        candidates = [r for r in rows if r["person_id"] == p and r["occlusion"] == "none"
                      and r["lighting"] == "regular"]
        assert candidates, f"{p} has no regular-light clean photo"
        bench_rows[p] = max(candidates, key=lambda r: float(r["det_score"]))

    data = np.load(args.embeddings, allow_pickle=True)
    idx = {c: i for i, c in enumerate(data["crop_paths"])}
    emb = data["embeddings"].astype(np.float64)
    bench = np.stack([emb[idx[bench_rows[p]["crop_path"]]] for p in persons])
    bench = bench / (np.linalg.norm(bench, axis=1, keepdims=True) + 1e-9)

    excluded = {}
    for p in persons:
        b = bench_rows[p]
        excluded[p] = sorted(r["crop_path"] for r in rows if r["person_id"] == p and
                              (r["group_id"] == b["group_id"] or r["md5"] == b["md5"]))

    os.makedirs(os.path.dirname(args.bench_out), exist_ok=True)
    np.savez_compressed(args.bench_out, benchmarks=bench.astype(np.float32),
                        persons=np.array(persons), kind="best")
    setup = {
        "seed": SEED,
        "manifest": args.manifest,
        "benchmarks": args.bench_out,
        "folds": folds,
        "benchmark_photo": {p: {k: bench_rows[p][k] for k in ("crop_path", "group_id", "md5", "lighting")}
                            for p in persons},
        "excluded": excluded,
    }
    with open(args.setup_out, "w", encoding="utf-8") as fh:
        json.dump(setup, fh, indent=2)
    print(f"Written: {args.setup_out}\nWritten: {args.bench_out}")

    # ---- assertions ----
    print("\n=== ASSERTIONS ===")
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(f"[{'OK' if cond else 'FAIL'}] {msg}")
        ok &= bool(cond)

    test_count = Counter(p for f in folds for p in f["test"])
    val_count = Counter(p for f in folds for p in f["val"])
    check(all(test_count[p] == 1 for p in persons), "every person is a test person in exactly one fold")
    check(all(val_count[p] == 1 for p in persons), "every person is a val person in exactly one fold")
    check(all(not (set(f["train"]) & set(f["val"]) or set(f["train"]) & set(f["test"])
                   or set(f["val"]) & set(f["test"])) and len(f["train"]) == 20
              for f in folds), "train/val/test disjoint and 20/5/5 sized in every fold")

    leaks = [r["crop_path"] for r in rows if r["crop_path"] not in set(excluded[r["person_id"]])
             and (r["group_id"] == bench_rows[r["person_id"]]["group_id"]
                  or r["md5"] == bench_rows[r["person_id"]]["md5"])]
    check(not leaks, f"no query photo shares a group or md5 with its own benchmark photo ({len(leaks)} found)")

    print("\n=== SUMMARY ===")
    for f in folds:
        print(f"fold {f['fold']}: test {f['test']}  val {f['val']}  train ({len(f['train'])} people)")

    if not ok:
        print("\n*** Hard assertions FAILED. ***")
        sys.exit(1)
    print("\nAll hard assertions passed.")


if __name__ == "__main__":
    main()
