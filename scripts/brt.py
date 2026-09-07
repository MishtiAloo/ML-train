from PIL import Image, ImageEnhance, ImageOps
from pillow_heif import register_heif_opener
from pathlib import Path
import random


register_heif_opener()

input_folder = Path(input("Input folder: ").strip().strip('"'))
output_folder = Path(input("Output folder: ").strip().strip('"'))

output_folder.mkdir(parents=True, exist_ok=True)


# Get all image formats used by this dataset.
images = sorted([
    p for p in input_folder.iterdir()
    if p.is_file() and p.suffix.lower() in [
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
        ".webp", ".heic", ".heif", ".jfif", ".avif"
    ]
])


a = len(images)
b = 200 - a

print(f"Original images : {a}")
print(f"Images needed   : {b}")


if a == 0:
    print("No images found in input folder.")
    exit()

if b <= 0:
    print("Already have 200 or more images. Nothing to generate.")
    exit()


# Select random source images without replacement.
if b > a:
    print("Not enough unique source images to generate the required copies.")
    exit()

selected_images = random.sample(images, b)


# Generate augmented copies
for i, path in enumerate(selected_images, start=1):

    img = Image.open(path)

    # Fix EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Random brightness between 0.4 and 0.7
    brightness = random.uniform(0.4, 0.7)

    img = ImageEnhance.Brightness(img).enhance(brightness)

    # JPEG output requires RGB or grayscale pixels.
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Example:
    # originalname_brightness_001.jpg
    output_name = f"{path.stem}_brightness_{i:03d}.jpg"
    output_path = output_folder / output_name

    img.save(output_path, quality=95)

    print(
        f"[{i}/{b}] {path.name} -> {output_name} "
        f"(brightness={brightness:.3f})"
    )


print(f"\nDone. Generated {b} images.")
print(f"Original: {a}")
print(f"Original + generated: {a + b}")
