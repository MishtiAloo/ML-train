"""E1/E2 v2: person-disjoint split (20 train / 5 val / 5 test people),
replacing the image-level split (16_e1e2_split.py) which leaked near-
duplicate/burst photos across train and test (see docs/report_e1e2.md
Section 4). Splitting by person instead means no image can ever cross
splits -- the leak is structurally impossible.

Random 20/5/5 person partition, seed 42 (not the team-member blocks used
by protocol 3's 14_unseen_setup.py -- this is a fresh, single split).

Benchmark: for EVERY person, the clean ('none'), regular-light photo with
the highest detection score (the "one best" rule used throughout this
project), from ALL of that person's photos. The manifest has no image-level
split column; train/validation/test roles are decided here by person.

Query exclusions (val/test people only): the benchmark photo itself,
every photo in its near-duplicate/burst group, and every byte-identical
copy -- never used as a query, so a person is never tested against a
near-duplicate of their own enrollment photo.

Usage:
    python 18_e1e2_person_split.py --manifest /home/tahmid/metadata/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys

import numpy as np

SEED = 42
N_VAL = 5
N_TEST = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/tahmid/metadata/manifest.csv")
    ap.add_argument("--embeddings", default="/home/tahmid/metadata/embeddings.npz")
    ap.add_argument("--setup-out", default="/home/tahmid/metadata/e1e2_person_split.json")
    ap.add_argument("--bench-out", default="/home/tahmid/runs/e1e2/person_benchmarks.npz")
    args = ap.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    persons = sorted(set(r["person_id"] for r in rows))
    assert len(persons) == 30

    rng = random.Random(SEED)
    shuffled = persons.copy()
    rng.shuffle(shuffled)
    test_people = sorted(shuffled[:N_TEST])
    val_people = sorted(shuffled[N_TEST:N_TEST + N_VAL])
    train_people = sorted(shuffled[N_TEST + N_VAL:])
    assert len(train_people) == 20 and len(val_people) == 5 and len(test_people) == 5
    assert not (set(train_people) & set(val_people) & set(test_people))

    # ---- one-best (highest det_score, clean, regular-light) benchmark photo per person ----
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

    # ---- query exclusions (benchmark photo + its burst group + md5 copies) ----
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
        "train": train_people,
        "val": val_people,
        "test": test_people,
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

    check(len(train_people) == 20 and len(val_people) == 5 and len(test_people) == 5,
          "train/val/test sizes are 20/5/5")
    check(not (set(train_people) & set(val_people)) and not (set(train_people) & set(test_people))
          and not (set(val_people) & set(test_people)), "train/val/test people are disjoint")
    check(all(r["occlusion"] == "none" and r["lighting"] == "regular" for r in bench_rows.values()),
          "every benchmark photo is none + regular")

    leaks = [r["crop_path"] for r in rows if r["crop_path"] not in set(excluded[r["person_id"]])
             and (r["group_id"] == bench_rows[r["person_id"]]["group_id"]
                  or r["md5"] == bench_rows[r["person_id"]]["md5"])]
    check(not leaks, f"no query photo shares a group or md5 with its own benchmark photo ({len(leaks)} found)")

    held_out = set(val_people) | set(test_people)
    n_query = {p: sum(1 for r in rows if r["person_id"] == p and r["crop_path"] not in set(excluded[p]))
               for p in held_out}
    n_query_occ = {p: sum(1 for r in rows if r["person_id"] == p and r["occlusion"] != "none"
                          and r["crop_path"] not in set(excluded[p])) for p in held_out}
    check(min(n_query_occ.values()) > 0, "every val/test person has at least one occluded query photo")

    print("\n=== SUMMARY ===")
    print(f"train (20): {train_people}")
    print(f"val   (5) : {val_people}")
    print(f"test  (5) : {test_people}")
    n_excl = {p: len(v) for p, v in excluded.items()}
    print(f"excluded photos per val/test person: min={min(n_excl[p] for p in held_out)} "
          f"max={max(n_excl[p] for p in held_out)}")
    print(f"query photos per val/test person: min={min(n_query.values())} "
          f"({min(n_query, key=n_query.get)})  total={sum(n_query.values())} "
          f"(occluded {sum(n_query_occ.values())})")
    off = bench @ bench.T
    np.fill_diagonal(off, -1)
    i, j = np.unravel_index(off.argmax(), off.shape)
    print(f"closest pair of benchmarks: {persons[i]} / {persons[j]}  cos={off[i, j]:.3f}")

    if not ok:
        print("\n*** Hard assertions FAILED. ***")
        sys.exit(1)
    print("\nAll hard assertions passed.")


if __name__ == "__main__":
    main()
