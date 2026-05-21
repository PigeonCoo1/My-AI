"""Central place for filesystem paths used by the project."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WEIGHTS_DIR = ROOT / "weights"
SHAPE_EXAMPLES_DIR = DATA_DIR / "shape_examples"

CHATS_JSONL = DATA_DIR / "chats.jsonl"
TOKENIZER_JSON = WEIGHTS_DIR / "tokenizer.json"
CHAT_MODEL_PT = WEIGHTS_DIR / "chat_brain.pt"
SHAPE_MODEL_PT = WEIGHTS_DIR / "shape_brain.pt"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    SHAPE_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
