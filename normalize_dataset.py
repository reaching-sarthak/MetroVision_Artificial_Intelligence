"""
normalize_dataset.py

Reads images from explored_data/{Pothole,Crack,Both,Normal}/ and writes
normalized .npz shards that data_loader.py can read directly.

Each shard npz has:
    images : (N, 720, 720, 3) float32, pixel values in [0, 1]
    labels : (N,) string array of class names ("Pothole"/"Crack"/"Both"/"Normal")

String labels means data_loader.py resolves them by NAME, so the
Normal/Crack/Pothole/Both index-order mismatch between datapipeline.py
and dense4.py never becomes a problem here — no flag needed later.

USAGE
-----
    python normalize_dataset.py --input "D:\\Sarthak's World\\NextGenHacks_2026\\Explored_data" --output normalized_npz

Then point train.py at the output folder:

    python train.py --data normalized_npz --ckpt checkpoints/run1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_FOLDERS = ("Pothole", "Crack", "Both", "Normal")  # must match datapipeline.py's output folder names
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_and_normalize(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # -> [0, 1]
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Folder containing Pothole/Crack/Both/Normal subfolders")
    p.add_argument("--output", required=True, help="Folder to write shard_*.npz files into")
    p.add_argument("--image_size", type=int, default=720)
    p.add_argument("--shard_size", type=int, default=32, help="Images per npz shard")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    input_root = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []  # (path, class_name)
    for class_name in CLASS_FOLDERS:
        class_dir = input_root / class_name
        if not class_dir.is_dir():
            print(f"WARNING: {class_dir} not found, skipping.")
            continue
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"{class_name}: {len(files)} images")
        samples.extend((f, class_name) for f in files)

    if not samples:
        raise FileNotFoundError(f"No images found under {input_root}")

    # Shuffle so each shard is a mix of classes, not one class per shard.
    random.Random(args.seed).shuffle(samples)

    total = len(samples)
    num_shards = (total + args.shard_size - 1) // args.shard_size
    print(f"\n{total} images total -> {num_shards} shards of up to {args.shard_size} images each\n")

    for shard_index in range(num_shards):
        chunk = samples[shard_index * args.shard_size: (shard_index + 1) * args.shard_size]

        images = np.zeros((len(chunk), args.image_size, args.image_size, 3), dtype=np.float32)
        labels = []

        for i, (path, class_name) in enumerate(chunk):
            try:
                images[i] = load_and_normalize(path, args.image_size)
                labels.append(class_name)
            except Exception as e:
                print(f"  SKIPPING unreadable image {path}: {e}")
                images[i] = 0.0
                labels.append(class_name)  # keep alignment; consider filtering these later

        out_path = output_dir / f"shard_{shard_index:05d}.npz"
        np.savez(out_path, images=images, labels=np.array(labels))
        print(f"wrote {out_path}  ({len(chunk)} images)")

    print("\nDone.")


if __name__ == "__main__":
    main()
