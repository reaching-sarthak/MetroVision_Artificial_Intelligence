"""
app.py

Web server for municipal staff: upload a road photo, get a damage
classification back from your trained checkpoint. No login, no
accounts — anyone who can reach this page can use it.

USAGE
-----
    python app.py --ckpt ../checkpoints/run1 --image_size 256

Then open http://127.0.0.1:5000 in a browser.

--image_size MUST match whatever --image_size you used with
normalize_dataset.py when you trained this checkpoint (256 in our
examples so far) — the model was trained on images resized to that
exact resolution, and predicting at a different size will still run
but silently degrade accuracy.

SECURITY NOTE — READ BEFORE DEPLOYING BEYOND YOUR OWN MACHINE
--------------------------------------------------------------------------
This intentionally has NO authentication, per your request. That's a
reasonable choice for testing on localhost or an internal-only
network. If this ever gets exposed to the public internet, anyone
who finds the URL can upload images and see the (currently
unauthenticated) results — there's nothing stopping abuse or scraping.
Put it behind your municipality's existing internal network / VPN
rather than a public IP if that's a concern, or ask and I can add a
simple shared-secret header check later without a full login system.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import predict as predictor

# Resolve relative to THIS file's location, not the current working directory -
# this way `python app.py` works the same whether you run it from the project
# root or anywhere else. index.html / style.css / app.js live right next to
# app.py (no separate "static" subfolder), matching how this project is
# actually laid out on disk.
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

MAX_UPLOAD_MB = 15
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def style():
    return send_from_directory(BASE_DIR, "style.css")


@app.route("/app.js")
def script():
    return send_from_directory(BASE_DIR, "app.js")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No image was uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image was selected."}), 400

    try:
        image = Image.open(file.stream)
        image.load()  # force-read now so a truncated file fails here, not mid-inference
    except Exception:
        return jsonify({"error": "That file isn't a readable image."}), 400

    try:
        result = predictor.predict(image)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "checkpoint": predictor._ckpt_path})


def main():
    p = argparse.ArgumentParser(description="Road damage assessment web app.")
    p.add_argument("--ckpt", required=True, help="Checkpoint path prefix, e.g. checkpoints/run1")
    p.add_argument("--image_size", type=int, required=True,
                    help="MUST match --image_size used in normalize_dataset.py for this checkpoint.")
    p.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow other machines on your network to reach it.")
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()

    print(f"Loading checkpoint from {args.ckpt} ...")
    tf = predictor.load_model(args.ckpt, args.image_size)
    print(f"Loaded. Model has completed {tf.optimizer.t} optimizer steps of training.")
    print(f"Starting server on http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()