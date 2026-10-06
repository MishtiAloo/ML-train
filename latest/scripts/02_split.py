"""Stage B3 of the preprocessing plan: build the group-aware manifest.

Reads:   preprocess_log.csv   (from 01_preprocess.py)
Writes:  manifest.csv         (successful crops plus group_id)

Grouping rules (transitive closure):
  1. same person + same "base" source stem (a brightness-derived image
     and the original it came from) -> same group. This can legitimately
     span the regular/low lighting boundary, since a low copy is derived
     from a regular original.
  2. same person + parseable capture timestamps within 10 seconds
     (burst neighbours) -> same group.

Exact byte-identical duplicates (md5) are kept as separate rows. The
person-fold builder later uses both md5 and group_id to exclude unsafe
validation/test queries.

This current version intentionally does NOT assign train/val/test here.
20_e1e2_person_folds.py makes that decision by person. The old image-level
split column was misleading and is not part of the current manifest schema.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BURST_SECONDS = 10
# Caps runaway chaining through continuous shooting. Several people shot an
# entire occlusion (20-30 photos) within a single 30s window (rapid manual
# or actual burst-mode capture despite the protocol asking against it) --
# with a 30s cap that collapsed the whole occlusion into ONE group, leaving
# no way to split it. 8s keeps genuine near-duplicate/retake frames grouped
# while still breaking a long rapid-fire sequence into several groups.
MAX_BURST_SPAN_SECONDS = 8
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

    # ---- write manifest ----
    fieldnames = [
        "image_path", "crop_path", "person_id", "occlusion",
        "lighting", "md5", "group_id", "det_score",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ok_rows)
    print(f"Manifest written: {args.out}  ({len(ok_rows)} rows)")

    # ---- assertions ----
    print("\n=== ASSERTIONS ===")
    group_people = defaultdict(set)
    for r in ok_rows:
        group_people[r["group_id"]].add(r["person_id"])
    mixed = {g: people for g, people in group_people.items() if len(people) > 1}
    assert not mixed, f"groups span more than one person: {mixed}"
    required = ("image_path", "crop_path", "person_id", "occlusion", "lighting",
                "md5", "group_id", "det_score")
    incomplete = [i for i, r in enumerate(ok_rows) if any(not r.get(k) for k in required)]
    assert not incomplete, f"rows missing required values: {incomplete[:10]}"
    print("[OK] every group belongs to exactly one person")
    print("[OK] every output row has all required manifest values")
    print("\nManifest grouping checks passed. Person folds are created separately by 20_e1e2_person_folds.py.")


if __name__ == "__main__":
    main()
