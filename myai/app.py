"""Flask web UI: chat with the AI and draw shapes for it to recognize."""
from __future__ import annotations

import base64
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from .chat import ChatRunner
from .paths import CHATS_JSONL, SHAPE_EXAMPLES_DIR, SHAPE_MODEL_PT, ensure_dirs
from .shape_model import SHAPE_CLASSES, ShapeBrain, ShapeConfig
from .shape_synth import IMG


app = Flask(__name__, template_folder="templates", static_folder="static")

_chat_runner: ChatRunner | None = None
_shape_model: ShapeBrain | None = None


def _get_chat() -> ChatRunner | None:
    global _chat_runner
    if _chat_runner is None:
        try:
            _chat_runner = ChatRunner()
        except FileNotFoundError:
            return None
    return _chat_runner


def _get_shape() -> ShapeBrain | None:
    global _shape_model
    if _shape_model is None and SHAPE_MODEL_PT.exists():
        m = ShapeBrain(ShapeConfig())
        ckpt = torch.load(SHAPE_MODEL_PT, map_location="cpu", weights_only=False)
        m.load_state_dict(ckpt["state"])
        m.eval()
        _shape_model = m
    return _shape_model


def _reset_models() -> None:
    """Call this to force the next request to re-load weights from disk."""
    global _chat_runner, _shape_model
    _chat_runner = None
    _shape_model = None


@app.route("/")
def index():
    return render_template("index.html", shape_classes=SHAPE_CLASSES)


@app.post("/api/chat")
def api_chat():
    data = request.get_json(force=True)
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"error": "empty message"}), 400

    runner = _get_chat()
    if runner is None:
        return jsonify({
            "reply": "(i haven't learned to talk yet. run: python -m myai.train_chat --seed --steps 2000)",
            "untrained": True,
        })

    reply = runner.reply(user_text)
    _append_chat(user_text, reply)
    return jsonify({"reply": reply})


@app.post("/api/chat/feedback")
def api_chat_feedback():
    """Let the user correct a bad reply — saved as a training example."""
    data = request.get_json(force=True)
    user_text = (data.get("user") or "").strip()
    correct = (data.get("correct") or "").strip()
    if not user_text or not correct:
        return jsonify({"error": "need user + correct"}), 400
    _append_chat(user_text, correct, corrected=True)
    return jsonify({"ok": True})


@app.post("/api/shape/predict")
def api_shape_predict():
    data = request.get_json(force=True)
    img = _decode_image(data.get("image", ""))
    if img is None:
        return jsonify({"error": "bad image"}), 400
    model = _get_shape()
    if model is None:
        return jsonify({
            "error": "(shape brain not trained yet. run: python -m myai.train_shapes --steps 1500)"
        }), 503
    arr = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    label, conf, scores = model.predict(x)
    return jsonify({"label": label, "confidence": conf, "scores": scores})


@app.post("/api/shape/teach")
def api_shape_teach():
    """User confirms or corrects the label — saved as a labeled example."""
    data = request.get_json(force=True)
    img = _decode_image(data.get("image", ""))
    label = (data.get("label") or "").strip().lower()
    if img is None or label not in SHAPE_CLASSES:
        return jsonify({"error": "need image + valid label"}), 400
    ensure_dirs()
    out_dir = SHAPE_EXAMPLES_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    img.save(out_dir / fname)
    return jsonify({"ok": True, "saved_to": str(out_dir / fname)})


@app.post("/api/reload")
def api_reload():
    """Re-read weights from disk (call after a training run)."""
    _reset_models()
    return jsonify({"ok": True})


@app.get("/api/status")
def api_status():
    return jsonify({
        "chat_trained": (CHATS_JSONL.parent / "chats.jsonl").parent.exists() and (CHATS_JSONL.parent / "chats.jsonl").exists() or _get_chat() is not None,
        "shape_trained": SHAPE_MODEL_PT.exists(),
        "chats_logged": _count_chats(),
        "shape_examples": _count_shape_examples(),
    })


def _append_chat(user: str, bot: str, corrected: bool = False) -> None:
    ensure_dirs()
    row = {
        "ts": int(time.time()),
        "user": user,
        "bot": bot,
        "corrected": corrected,
    }
    with CHATS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count_chats() -> int:
    if not CHATS_JSONL.exists():
        return 0
    with CHATS_JSONL.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _count_shape_examples() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not SHAPE_EXAMPLES_DIR.exists():
        return counts
    for label_dir in SHAPE_EXAMPLES_DIR.iterdir():
        if label_dir.is_dir():
            counts[label_dir.name] = sum(1 for _ in label_dir.glob("*.png"))
    return counts


def _decode_image(b64: str) -> Image.Image | None:
    if not b64:
        return None
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("L")
        # The canvas is white-on-black on the client, already inverted for us.
        return img.resize((IMG, IMG), Image.LANCZOS)
    except Exception:
        return None


def main() -> None:
    ensure_dirs()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
