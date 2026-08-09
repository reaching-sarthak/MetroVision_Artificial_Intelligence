"""
checkpoint.py

Saves/restores EVERYTHING needed to resume training exactly where it
left off if the process dies mid-run:

    - every trainable array (conv1.filters, conv1.bias, ... dense4.b)
    - AdamW's per-parameter first/second moment estimates (m, v)
    - AdamW's global step counter t
    - your own training progress (epoch, shard file, sample index,
      running loss) so the data loader can resume from the right place
      instead of re-training from image 0 every time.

Two files are always written together, atomically:

    <name>.npz   -- all numpy arrays (weights + optimizer m/v)
    <name>.json  -- small scalars (t, epoch, shard, sample_index, loss)

"Atomically" here means: write to a temp file first, then os.replace()
onto the real filename. os.replace is atomic on POSIX and Windows, so
a crash mid-write can never leave you with a half-written, corrupt
checkpoint that silently loads garbage.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np


PARAM_KEYS = (
    "conv1.filters", "conv1.bias",
    "conv2.filters", "conv2.bias",
    "conv3.filters", "conv3.bias",
    "dense128.W", "dense128.b",
    "dense4.W", "dense4.b",
)


def save_checkpoint(
    path: str | Path,
    trainflow,
    progress: dict | None = None,
) -> None:
    """
    path: e.g. "checkpoints/ckpt_latest" (no extension — .npz/.json added).
    trainflow: a model.TrainFlowFixed instance.
    progress: whatever you want to remember about where you are in the
        dataset, e.g. {"epoch": 3, "shard": "shard_0042.npz", "sample_index": 118,
                        "running_loss": 0.42}. Stored as-is in the .json file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = trainflow.get_parameters()
    opt = trainflow.optimizer

    arrays = {}
    for key in PARAM_KEYS:
        arrays[f"param__{key}"] = params[key]
    for key, arr in opt.m.items():
        arrays[f"adamw_m__{key}"] = arr
    for key, arr in opt.v.items():
        arrays[f"adamw_v__{key}"] = arr

    meta = {
        "optimizer_t": opt.t,
        "dropout_rate": trainflow.dropout_rate,
        "learning_rate": float(opt.lr),
        "beta1": float(opt.beta1),
        "beta2": float(opt.beta2),
        "epsilon": float(opt.epsilon),
        "weight_decay": float(opt.weight_decay),
        "saved_at_unix": time.time(),
        "progress": progress or {},
    }

    # --- write npz atomically ---
    # NOTE: np.savez silently appends ".npz" to whatever filename you give it
    # if the name doesn't already end in ".npz" — so the temp file's suffix
    # MUST be exactly ".npz", or savez writes to a different path than the
    # one we then try to os.replace() from (leaving an empty/missing temp
    # file and a leftover "*.npz.tmp.npz" on disk).
    npz_path = path.with_suffix(".npz")
    fd, tmp_npz = tempfile.mkstemp(dir=str(path.parent), suffix=".npz")
    os.close(fd)
    np.savez(tmp_npz, **arrays)
    os.replace(tmp_npz, npz_path)  # atomic on POSIX + Windows

    # --- write json atomically ---
    json_path = path.with_suffix(".json")
    fd, tmp_json = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp_json, json_path)


def load_checkpoint(path: str | Path):
    """
    Returns (state_dict_for_build_trainflow, adamw_m, adamw_v, meta_dict)
    or None if no checkpoint exists at this path yet.

    Typical use:

        from model import build_trainflow
        from checkpoint import load_checkpoint

        loaded = load_checkpoint("checkpoints/ckpt_latest")
        if loaded is None:
            tf = build_trainflow(learning_rate=1e-3)   # fresh start
            progress = {"shard_index": 0, "sample_index": 0, "epoch": 0}
        else:
            state, m, v, meta = loaded
            tf = build_trainflow(
                state,
                dropout_rate=meta["dropout_rate"],
                learning_rate=meta["learning_rate"],
                beta1=meta["beta1"], beta2=meta["beta2"],
                epsilon=meta["epsilon"], weight_decay=meta["weight_decay"],
            )
            tf.optimizer.m = m
            tf.optimizer.v = v
            tf.optimizer.t = meta["optimizer_t"]
            progress = meta["progress"]
    """
    path = Path(path)
    npz_path = path.with_suffix(".npz")
    json_path = path.with_suffix(".json")

    if not npz_path.exists() or not json_path.exists():
        return None

    with open(json_path) as f:
        meta = json.load(f)

    data = np.load(npz_path)

    state = {key: data[f"param__{key}"] for key in PARAM_KEYS}

    adamw_m, adamw_v = {}, {}
    for full_key in data.files:
        if full_key.startswith("adamw_m__"):
            adamw_m[full_key[len("adamw_m__"):]] = data[full_key]
        elif full_key.startswith("adamw_v__"):
            adamw_v[full_key[len("adamw_v__"):]] = data[full_key]

    return state, adamw_m, adamw_v, meta


def load_trainflow(path: str | Path, **override_kwargs):
    """
    Convenience one-liner: returns (trainflow, progress_dict), building a
    fresh model if no checkpoint exists yet at `path`.
    """
    from model import build_trainflow  # local import to avoid a cycle at module load time

    loaded = load_checkpoint(path)
    if loaded is None:
        tf = build_trainflow(**override_kwargs)
        return tf, {"shard_index": 0, "sample_index": 0, "epoch": 0, "running_loss": None}

    state, m, v, meta = loaded
    kwargs = dict(
        dropout_rate=meta["dropout_rate"],
        learning_rate=meta["learning_rate"],
        beta1=meta["beta1"],
        beta2=meta["beta2"],
        epsilon=meta["epsilon"],
        weight_decay=meta["weight_decay"],
    )
    kwargs.update(override_kwargs)

    tf = build_trainflow(state, **kwargs)
    tf.optimizer.m = m
    tf.optimizer.v = v
    tf.optimizer.t = meta["optimizer_t"]

    progress = meta.get("progress") or {"shard_index": 0, "sample_index": 0, "epoch": 0, "running_loss": None}
    return tf, progress
