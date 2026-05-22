"""Continuous background trainer.

Run this in a terminal and leave it. Every N minutes it will:
  - retrain the chat brain on chats.jsonl + seed corpus (a few steps)
  - retrain the shape brain on synthetic + user-labeled shapes

This is how the AI "keeps learning when you're not on it."
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from .train_chat import train as train_chat
from .train_shapes import train as train_shapes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval-minutes", type=int, default=10)
    p.add_argument("--chat-steps", type=int, default=300)
    p.add_argument("--shape-steps", type=int, default=300)
    args = p.parse_args()

    print(f"[bg] background trainer started. interval={args.interval_minutes}min")
    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[bg {ts}] === training round ===")
        try:
            train_chat(steps=args.chat_steps, seed=True, batch_size=32, lr=3e-4)
        except Exception as e:
            print(f"[bg] chat training failed: {e}")
        try:
            train_shapes(steps=args.shape_steps, batch_size=64, lr=1e-3)
        except Exception as e:
            print(f"[bg] shape training failed: {e}")
        print(f"[bg] sleeping {args.interval_minutes}min...")
        time.sleep(args.interval_minutes * 60)


if __name__ == "__main__":
    main()
