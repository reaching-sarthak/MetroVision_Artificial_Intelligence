"""
data_loader.py

Feeds pre-normalized image .npz files into TrainFlowFixed.train_step().

--------------------------------------------------------------------------
IMPORTANT — CLASS INDEX MISMATCH BETWEEN YOUR TWO SCRIPTS
--------------------------------------------------------------------------
datapipeline.py writes labels.csv using:

    CLASS_MAPPING = {"Normal": 0, "Crack": 1, "Pothole": 2, "Both": 3}

but dense4.py / trainflow.py's CLASS_NAMES (the order the model's 4
output logits actually mean) is:

    CLASS_NAMES = ("Pothole", "Crack", "Both", "Normal")   # 0,1,2,3

Those are DIFFERENT orderings. If you feed datapipeline.py's
`class_index` column straight into train_step() as `target_class`, every
"Normal" sample (index 0 in datapipeline) will silently train the model
to call it "Pothole" (index 0 in dense4), and so on for every class.
The model would still run and the loss would still go down — it would
just be confidently learning the wrong mapping, which is the kind of
bug you only discover after a full training run when real-world
predictions come out backwards.

This loader always converts from the CLASS NAME STRING to the
trainflow-correct index, never from datapipeline's numeric column, so
this is fixed at the source. See `CLASS_NAME_TO_INDEX` below.
--------------------------------------------------------------------------

Two npz layouts are supported, auto-detected per file:

  (A) One image per file:
        npz["image"]  (or "X" / "img") : (720, 720, 3) float32, already
                                          normalized (e.g. to [0,1] or
                                          standardized)
        npz["label"]  (or "y" / "class") : either an int already in
                                          trainflow order, OR a numpy
                                          string scalar with the class
                                          name ("Pothole"/"Crack"/"Both"/"Normal")

  (B) A batched shard, many images per file:
        npz["images"] : (N, 720, 720, 3) float32
        npz["labels"] : (N,) int array OR (N,) string array of class names

If your labels are already integers, they are assumed to already use
trainflow's (Pothole=0, Crack=1, Both=2, Normal=3) order — pass
`labels_are_datapipeline_order=True` to iter_dataset()/iter_npz_file()
if instead they were written using datapipeline.py's
(Normal=0, Crack=1, Pothole=2, Both=3) order, and this loader will remap
them for you.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from trainflow import CLASS_NAMES  # ("Pothole", "Crack", "Both", "Normal")

CLASS_NAME_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# datapipeline.py's CLASS_MAPPING, duplicated here (as *indices*, not names)
# only so we can remap old integer-labeled shards if you tell us to.
_DATAPIPELINE_INDEX_TO_NAME = {0: "Normal", 1: "Crack", 2: "Pothole", 3: "Both"}


def _resolve_label(raw_label, labels_are_datapipeline_order: bool) -> int:
    # String class name -> always resolved by name, unambiguous.
    if isinstance(raw_label, (str, np.str_, bytes, np.bytes_)):
        name = raw_label.decode() if isinstance(raw_label, (bytes, np.bytes_)) else str(raw_label)
        if name not in CLASS_NAME_TO_INDEX:
            raise ValueError(f"Unknown class name '{name}'. Expected one of {CLASS_NAMES}.")
        return CLASS_NAME_TO_INDEX[name]

    # Integer label.
    idx = int(raw_label)
    if not (0 <= idx < len(CLASS_NAMES)):
        raise ValueError(f"Label index {idx} out of range for {CLASS_NAMES}.")

    if labels_are_datapipeline_order:
        name = _DATAPIPELINE_INDEX_TO_NAME[idx]
        return CLASS_NAME_TO_INDEX[name]

    return idx  # already trainflow order


def iter_npz_file(
    path: str | Path,
    labels_are_datapipeline_order: bool = False,
    start_index: int = 0,
) -> Iterator[tuple[int, np.ndarray, int]]:
    """
    Yields (sample_index_within_file, image, target_class) for one .npz
    file, auto-detecting single-image vs batched-shard layout.

    start_index: skip samples before this index (used to resume mid-shard
    after a crash).
    """
    data = np.load(path, allow_pickle=False)
    keys = set(data.files)

    if {"images", "labels"} & keys or "images" in keys:
        images = data["images"]
        labels = data["labels"]
        if len(images) != len(labels):
            raise ValueError(f"{path}: {len(images)} images but {len(labels)} labels")
        for i in range(start_index, len(images)):
            target = _resolve_label(labels[i], labels_are_datapipeline_order)
            yield i, images[i].astype(np.float32, copy=False), target
        return

    image_key = next((k for k in ("image", "X", "img") if k in keys), None)
    label_key = next((k for k in ("label", "y", "class") if k in keys), None)
    if image_key is None or label_key is None:
        raise ValueError(
            f"{path}: couldn't find image/label arrays. Keys present: {sorted(keys)}. "
            f"Expected 'images'+'labels' (shard) or 'image'+'label' (single sample)."
        )
    if start_index > 0:
        return  # single-sample file, nothing left to resume
    target = _resolve_label(data[label_key], labels_are_datapipeline_order)
    yield 0, data[image_key].astype(np.float32, copy=False), target


def list_shards(npz_dir: str | Path) -> list[Path]:
    """Sorted list of .npz files in a directory — sorted so resume order is deterministic."""
    npz_dir = Path(npz_dir)
    return sorted(npz_dir.glob("*.npz"))


def iter_dataset(
    npz_dir: str | Path,
    labels_are_datapipeline_order: bool = False,
    start_shard_index: int = 0,
    start_sample_index: int = 0,
) -> Iterator[tuple[int, int, np.ndarray, int]]:
    """
    Walks every .npz file in npz_dir in sorted order, yielding
    (shard_index, sample_index, image, target_class).

    start_shard_index / start_sample_index let you resume a crashed run
    exactly where it left off (see train.py).
    """
    shards = list_shards(npz_dir)
    if not shards:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    for shard_index, shard_path in enumerate(shards):
        if shard_index < start_shard_index:
            continue
        this_start = start_sample_index if shard_index == start_shard_index else 0
        for sample_index, image, target in iter_npz_file(
            shard_path, labels_are_datapipeline_order, start_index=this_start
        ):
            yield shard_index, sample_index, image, target
