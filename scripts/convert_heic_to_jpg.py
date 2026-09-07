"""Recursively convert HEIC/HEIF images to JPEG in place.

The JPEG is written and verified before the source HEIC/HEIF file is removed.
Existing JPEG destinations are never overwritten.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener


register_heif_opener()


def convert_tree(root: Path, quality: int) -> int:
    root = root.resolve()
    sources = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".heic", ".heif"}
    )

    plans = [(source, source.with_suffix(".jpg")) for source in sources]
    collisions = [destination for _, destination in plans if destination.exists()]
    if collisions:
        preview = "\n".join(str(path) for path in collisions[:10])
        raise FileExistsError(
            f"Refusing to overwrite {len(collisions)} existing JPEG(s):\n{preview}"
        )

    print(f"HEIC/HEIF images found: {len(plans)}", flush=True)

    for index, (source, destination) in enumerate(plans, start=1):
        temporary = destination.with_name(
            f".__heic_convert_{uuid4().hex}.jpg"
        )
        try:
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(temporary, format="JPEG", quality=quality)

            # Confirm that Pillow can fully decode the new JPEG before replacing
            # the source file.
            with Image.open(temporary) as check:
                check.load()
                if check.format != "JPEG":
                    raise ValueError(f"Output is not JPEG: {temporary}")

            os.replace(temporary, destination)
            source.unlink()
        finally:
            if temporary.exists():
                temporary.unlink()

        if index % 100 == 0 or index == len(plans):
            print(f"Converted {index}/{len(plans)}", flush=True)

    return len(plans)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively convert HEIC/HEIF images to JPEG in place."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset root (defaults to the script's folder)",
    )
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()

    if not 1 <= args.quality <= 100:
        parser.error("--quality must be between 1 and 100")
    if not args.root.is_dir():
        parser.error(f"Folder does not exist: {args.root}")

    count = convert_tree(args.root, args.quality)
    print(f"Done. Converted {count} image(s).", flush=True)


if __name__ == "__main__":
    main()
