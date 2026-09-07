"""Stage B3+B4 of the preprocessing plan: build the manifest and the
group-aware train/val/test split.

Reads:   preprocess_log.csv   (from 01_preprocess.py)
Writes:  manifest.csv         (adds group_id, split columns)

Grouping rules (transitive closure):
  1. same person + same "base" source stem (a brightness-derived image
     and the original it came from) -> same group. This can legitimately
     span the regular/low lighting boundary, since a low copy is derived
     from a regular original.
  2. same person + parseable capture timestamps within 10 seconds
     (burst neighbours) -> same group.

Exact byte-identical duplicates (md5) are, by project decision, NOT a
grouping rule -- each is treated as an independent image and may land
on either side of the split. See docs/preprocessing_plan.md B1.

Split is stratified at (person_id, occlusion) granularity (not also by
lighting, since rule 1 can span lighting) at group level, targeting
60/20/20 by image count, seed=42.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

SEED = 42
BURST_SECONDS = 10
# Caps runaway chaining through continuous shooting. Several people shot an
# entire occlusion (20-30 photos) within a single 30s window (rapid manual
# or actual burst-mode capture despite the protocol asking against it) --
# with a 30s cap that collapsed the whole occlusion into ONE group, leaving
# no way to split it. 8s keeps genuine near-duplicate/retake frames grouped
# while still breaking a long rapid-fire sequence into several groups.
MAX_BURST_SPAN_SECONDS = 8
TARGET = {"train": 0.6, "val": 0.2, "test": 0.2}
MIN_PER_PERSON_PER_SPLIT = 2  # hard floor enforced by the repair pass below


# ---------- union-find ----------
class UF:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


# ---------- timestamp extraction from original camera filenames ----------
TS_PATTERNS = [
    re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"),           # Samsung: 20260817_195045
    re.compile(r"^IMG_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"),        # Android: IMG_20260822_112221
    re.compile(r"^PXL_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})"),        # Pixel:   PXL_20260903_171804978
]


def extract_ts(stem: str):
    for pat in TS_PATTERNS:
        m = pat.match(stem)
        if m:
            y, mo, d, h, mi, s = map(int, m.groups())
            try:
                return datetime(y, mo, d, h, mi, s)
            except ValueError:
                return None
    return None


def base_stem(stem: str) -> str:
    """Strip a trailing _brightness_NNN and any (N) disambiguator, so a
    darkened copy and its source share the same base."""
    stem = re.sub(r"_brightness_\d+$", "", stem)
    stem = re.sub(r"\(\d+\)$", "", stem)
    return stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/home/tahmid/metadata/preprocess_log.csv")
    ap.add_argument("--out", default="/home/tahmid/metadata/manifest.csv")
    args = ap.parse_args()

    with open(args.log, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    ok_rows = [r for r in rows if r["det_ok"] == "True"]
    print(f"Total logged: {len(rows)}   detected+cropped: {len(ok_rows)}")

    # ---- build groups ----
    uf = UF()
    keys = []
    for r in ok_rows:
        stem = Path(r["source_stem"]).stem if r["source_stem"] else Path(r["image_path"]).stem
        key = (r["person_id"], stem)
        keys.append(key)
        uf.find(key)

    # rule 1: same person + same base stem
    by_base = defaultdict(list)
    for (pid, stem) in keys:
        by_base[(pid, base_stem(stem))].append((pid, stem))
    for group in by_base.values():
        for k in group[1:]:
            uf.union(group[0], k)

    # rule 2: same person AND same occlusion, timestamps within
    # BURST_SECONDS of the previous shot AND within
    # MAX_BURST_SPAN_SECONDS of the group's first shot.
    #
    # Two safeguards here, both necessary:
    #  - the span cap: continuous rapid-fire capture (shots a few seconds
    #    apart, for minutes at a time) would otherwise chain via
    #    transitive closure into one supergroup covering an entire
    #    occlusion or an entire person's session.
    #  - the occlusion match: splitting happens independently per
    #    (person, occlusion) cell, so a group that spans two occlusions
    #    (e.g. formed at the moment a person switched from mask to
    #    scarf mid-burst) could be assigned to different splits in each
    #    cell -- which is exactly the "group spans >1 split" failure.
    by_bucket = defaultdict(list)
    for r, (pid, stem) in zip(ok_rows, keys):
        ts = extract_ts(stem)
        if ts is not None:
            by_bucket[(pid, r["occlusion"])].append((ts, (pid, stem)))
    for bucket_key, items in by_bucket.items():
        items.sort(key=lambda x: x[0])
        group_start_ts = None
        for (t1, k1), (t2, k2) in zip(items, items[1:]):
            if group_start_ts is None:
                group_start_ts = t1
            gap_ok = (t2 - t1).total_seconds() <= BURST_SECONDS
            span_ok = (t2 - group_start_ts).total_seconds() <= MAX_BURST_SPAN_SECONDS
            if gap_ok and span_ok:
                uf.union(k1, k2)
            else:
                group_start_ts = t2

    group_ids = {}
    for k in keys:
        root = uf.find(k)
        group_ids.setdefault(root, len(group_ids))
    for r, k in zip(ok_rows, keys):
        r["group_id"] = f"g{group_ids[uf.find(k)]}"

    n_groups = len(group_ids)
    print(f"Groups formed: {n_groups} (from {len(ok_rows)} images)")

    gsize = Counter(r["group_id"] for r in ok_rows)
    sizes = sorted(gsize.values(), reverse=True)
    print(f"Group size distribution: max={sizes[0]}, top10={sizes[:10]}, "
          f"groups>10={sum(1 for s in sizes if s > 10)}, groups==1={sum(1 for s in sizes if s == 1)}")

    # ---- stratified group-level split, per (person, occlusion) ----
    # group_split holds the single source of truth (group_id -> split) so
    # the repair pass below can move whole groups between splits before
    # anything is written to rows.
    rnd = random.Random(SEED)
    cell = defaultdict(list)  # (pid, occ) -> list of rows
    for r in ok_rows:
        cell[(r["person_id"], r["occlusion"])].append(r)

    cell_groups = {}  # (pid, occ) -> {group_id: [rows]}
    group_split = {}  # group_id -> split

    for (pid, occ), cell_rows in cell.items():
        groups = defaultdict(list)
        for r in cell_rows:
            groups[r["group_id"]].append(r)
        cell_groups[(pid, occ)] = groups

        # largest groups first: a big group is the hardest to place without
        # overshooting a target, so give it first pick of the split with
        # the most room; small groups (mostly singletons) fill the gaps
        # afterwards and rarely cause imbalance.
        group_list = list(groups.items())
        rnd.shuffle(group_list)
        group_list.sort(key=lambda kv: len(kv[1]), reverse=True)

        n_total = len(cell_rows)
        targets = {k: TARGET[k] * n_total for k in TARGET}

        counts = {"train": 0, "val": 0, "test": 0}
        for gid, grows in group_list:
            # assign to whichever split is currently furthest below its
            # target share (largest deficit) -- balanced multi-way greedy.
            deficits = {k: targets[k] - counts[k] for k in TARGET}
            dest = max(deficits, key=lambda k: deficits[k])
            group_split[gid] = dest
            counts[dest] += len(grows)

    # ---- repair pass: guarantee every person appears in val AND test ----
    # Some people shot an entire occlusion (or several) as one indivisible
    # group -- there is nothing to redistribute *within* that cell. When
    # every one of a person's cells is like this, the deficit greedy above
    # (correctly, in isolation) puts all of them in train, since train's
    # absolute target is far larger and a lone group can only go to one
    # split. Fix it at the person level: reassign whole groups (never
    # splitting one) from train into val/test until every person has at
    # least MIN_PER_PERSON_PER_SPLIT images in both. Spread the moves
    # across different occlusions where possible, so a person's coverage
    # gap doesn't collapse onto a single occlusion.
    persons_all = sorted(set(r["person_id"] for r in ok_rows))
    forced_moves = []  # (person, occlusion, group_id, from, to) for the report

    def person_split_count(pid, split):
        return sum(
            len(grows) for (p, o), groups in cell_groups.items() if p == pid
            for gid, grows in groups.items() if group_split[gid] == split
        )

    for pid in persons_all:
        pid_cells = [(o, cell_groups[(p, o)]) for (p, o) in cell_groups if p == pid]
        for target_split in ("val", "test"):
            # occlusions with the most groups first: safest to pull a
            # small group out of without gutting that cell's own coverage
            pid_cells.sort(key=lambda oc: -len(oc[1]))
            for occ, groups in pid_cells:
                if person_split_count(pid, target_split) >= MIN_PER_PERSON_PER_SPLIT:
                    break
                # candidate groups currently in train for this cell,
                # smallest first (minimise train data given up)
                candidates = sorted(
                    (gid for gid in groups if group_split[gid] == "train"),
                    key=lambda gid: len(groups[gid]),
                )
                for gid in candidates:
                    if person_split_count(pid, target_split) >= MIN_PER_PERSON_PER_SPLIT:
                        break
                    group_split[gid] = target_split
                    forced_moves.append((pid, occ, gid, "train", target_split))

    if forced_moves:
        print(f"\nRepair pass: {len(forced_moves)} group(s) force-moved to guarantee "
              f"every person has >= {MIN_PER_PERSON_PER_SPLIT} images in val and test.")
        by_pid = defaultdict(int)
        for pid, occ, gid, frm, to in forced_moves:
            by_pid[pid] += 1
        for pid, n in sorted(by_pid.items()):
            print(f"   {pid}: {n} group(s) moved")

    # ---- materialise split onto rows ----
    for r in ok_rows:
        r["split"] = group_split[r["group_id"]]

    # ---- write manifest ----
    fieldnames = [
        "image_path", "crop_path", "person_id", "occlusion", "lighting",
        "source_stem", "md5", "group_id", "split",
        "det_ok", "det_score", "bbox_area_frac", "det_stage",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ok_rows)
    print(f"Manifest written: {args.out}  ({len(ok_rows)} rows)")

    # ---- assertions ----
    print("\n=== ASSERTIONS ===")
    ok = True

    group_splits = defaultdict(set)
    for r in ok_rows:
        group_splits[r["group_id"]].add(r["split"])
    bad = {g: s for g, s in group_splits.items() if len(s) > 1}
    print(f"[{'OK' if not bad else 'FAIL'}] groups spanning >1 split: {len(bad)}")
    ok &= not bad

    persons = sorted(set(r["person_id"] for r in ok_rows))
    missing_split = [p for p in persons if len({r["split"] for r in ok_rows if r["person_id"] == p}) < 3]
    print(f"[{'OK' if not missing_split else 'FAIL'}] persons missing a split: {missing_split}")
    ok &= not missing_split

    occs = sorted(set(r["occlusion"] for r in ok_rows))
    missing_cell = []
    for p in persons:
        for o in occs:
            if not any(r["person_id"] == p and r["occlusion"] == o and r["split"] == "test" for r in ok_rows):
                missing_cell.append((p, o))
    print(f"[{'OK' if not missing_cell else 'WARN'}] (person,occlusion) missing from test: {len(missing_cell)}")
    if missing_cell[:10]:
        print("   e.g.", missing_cell[:10])

    print("\n=== SPLIT SIZES ===")
    print(Counter(r["split"] for r in ok_rows))

    print("\n=== DUPLICATE LEAKAGE (informational only, md5 is NOT grouped) ===")
    md5_split = defaultdict(set)
    for r in ok_rows:
        md5_split[r["md5"]].add(r["split"])
    cross = [m for m, s in md5_split.items() if len(s) > 1]
    n_test_leaked = sum(
        1 for r in ok_rows
        if r["split"] == "test" and len(md5_split[r["md5"]]) > 1
    )
    print(f"md5 values appearing in >1 split: {len(cross)}")
    print(f"test images that share an md5 with a train image: {n_test_leaked} "
          f"({100*n_test_leaked/max(1,sum(1 for r in ok_rows if r['split']=='test')):.1f}% of test)")

    if not ok:
        print("\n*** Hard assertions FAILED. Fix before training. ***")
        sys.exit(1)
    print("\nAll hard assertions passed.")


if __name__ == "__main__":
    main()
