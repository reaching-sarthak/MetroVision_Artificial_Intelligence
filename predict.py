"""
predict.py

Loads a trained checkpoint ONCE and classifies images against it.
Used by app.py (the web server) — import it directly if you want to
call it from your own script instead:

    import predict
    predict.load_model("checkpoints/run1", image_size=256)
    result = predict.predict(PIL.Image.open("road.jpg"))
    print(result["predicted_class"], result["confidence"])
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from checkpoint import load_trainflow
from trainflow import CLASS_NAMES  # ("Pothole", "Crack", "Both", "Normal")

_model = None
_image_size = None
_ckpt_path = None


def load_model(ckpt_path: str, image_size: int):
    """Loads the checkpoint into memory. Call this once at server startup."""
    global _model, _image_size, _ckpt_path
    tf, progress = load_trainflow(ckpt_path)
    if tf.optimizer.t == 0:
        raise FileNotFoundError(
            f"No trained checkpoint found at '{ckpt_path}' — got a freshly "
            f"initialized (untrained) model instead. Check the --ckpt path."
        )
    _model = tf
    _image_size = image_size
    _ckpt_path = ckpt_path
    return tf


def _preprocess(image: Image.Image) -> np.ndarray:
    # Must match normalize_dataset.py exactly: RGB, bilinear resize, /255.
    img = image.convert("RGB").resize((_image_size, _image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def predict(image: Image.Image) -> dict:
    """Classifies one PIL image. Returns predicted class, confidence, and
    the full probability breakdown."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model(ckpt_path, image_size) first.")

    array = _preprocess(image)
    predicted_index, probabilities = _model.predict(array)

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            name: float(p) for name, p in zip(CLASS_NAMES, probabilities)
        },
        "checkpoint": _ckpt_path,
        "trained_steps": int(_model.optimizer.t),
    }
