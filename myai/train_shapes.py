"""Train the shape brain on synthetic shapes (and any user-labeled examples)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .paths import SHAPE_EXAMPLES_DIR, SHAPE_MODEL_PT, ensure_dirs
from .shape_model import SHAPE_CLASSES, ShapeBrain, ShapeConfig
from .shape_synth import IMG, synth_batch


def _load_user_examples() -> tuple[torch.Tensor, torch.Tensor] | None:
    examples: list[tuple[np.ndarray, int]] = []
    if not SHAPE_EXAMPLES_DIR.exists():
        return None
    for label_dir in SHAPE_EXAMPLES_DIR.iterdir():
        if not label_dir.is_dir() or label_dir.name not in SHAPE_CLASSES:
            continue
        label_idx = SHAPE_CLASSES.index(label_dir.name)
        for img_path in label_dir.glob("*.png"):
            try:
                img = Image.open(img_path).convert("L").resize((IMG, IMG))
                examples.append((np.array(img, dtype=np.float32) / 255.0, label_idx))
            except Exception:
                continue
    if not examples:
        return None
    xs = torch.stack([torch.from_numpy(a).unsqueeze(0) for a, _ in examples])
    ys = torch.tensor([y for _, y in examples], dtype=torch.long)
    return xs, ys


def _save(model: ShapeBrain) -> None:
    ensure_dirs()
    torch.save({"state": model.state_dict()}, SHAPE_MODEL_PT)


def train(steps: int, batch_size: int, lr: float) -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShapeBrain(ShapeConfig()).to(device)
    if SHAPE_MODEL_PT.exists():
        ckpt = torch.load(SHAPE_MODEL_PT, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["state"])
            print("[info] resumed from existing shape brain weights.")
        except Exception as e:
            print(f"[warn] could not load prior shape weights: {e}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    user = _load_user_examples()

    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = synth_batch(batch_size)
        if user is not None and step % 4 == 0:
            ux, uy = user
            # Random subsample of user examples each iter.
            idx = torch.randint(0, ux.size(0), (min(batch_size // 2, ux.size(0)),))
            x = torch.cat([x, ux[idx]], dim=0)
            y = torch.cat([y, uy[idx]], dim=0)

        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                acc = (logits.argmax(-1) == y).float().mean().item()
            print(f"step {step:>5}/{steps}  loss={loss.item():.4f}  acc={acc:.2f}  elapsed={time.time() - t0:.1f}s")
        if step % 500 == 0:
            _save(model)

    _save(model)
    print(f"[done] saved to {SHAPE_MODEL_PT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    torch.manual_seed(0)
    train(steps=args.steps, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
