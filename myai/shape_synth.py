"""Generate synthetic shape images on-the-fly for training.

We don't use any external dataset. We render circles, squares, triangles,
stars, hearts, and lines with random sizes, positions, and noise.
"""
from __future__ import annotations

import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

from .shape_model import SHAPE_CLASSES


IMG = 28


def _blank() -> Image.Image:
    return Image.new("L", (IMG, IMG), color=0)


def _jitter_bbox() -> tuple[int, int, int, int]:
    pad = random.randint(2, 6)
    return pad, pad, IMG - pad, IMG - pad


def render_circle() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    d.ellipse(_jitter_bbox(), outline=255, width=random.choice([1, 2]))
    return img


def render_square() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    d.rectangle(_jitter_bbox(), outline=255, width=random.choice([1, 2]))
    return img


def render_triangle() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = _jitter_bbox()
    pts = [((x0 + x1) // 2, y0), (x0, y1), (x1, y1)]
    d.polygon(pts, outline=255)
    return img


def render_star() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    cx, cy = IMG // 2, IMG // 2
    r_out = random.randint(8, 12)
    r_in = r_out // 2
    pts = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    d.polygon(pts, outline=255)
    return img


def render_heart() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    cx, cy = IMG // 2, IMG // 2 + 1
    pts = []
    for t in np.linspace(0, 2 * math.pi, 60):
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x * 0.45, cy + y * 0.45))
    d.line(pts + [pts[0]], fill=255, width=1)
    return img


def render_line() -> Image.Image:
    img = _blank()
    d = ImageDraw.Draw(img)
    x0 = random.randint(2, IMG - 3)
    y0 = random.randint(2, IMG - 3)
    x1 = random.randint(2, IMG - 3)
    y1 = random.randint(2, IMG - 3)
    d.line([(x0, y0), (x1, y1)], fill=255, width=random.choice([1, 2]))
    return img


RENDERERS = {
    "circle": render_circle,
    "square": render_square,
    "triangle": render_triangle,
    "star": render_star,
    "heart": render_heart,
    "line": render_line,
}


def _add_noise(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 12, arr.shape)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def synth_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.zeros(batch_size, 1, IMG, IMG, dtype=torch.float32)
    ys = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        label_idx = random.randrange(len(SHAPE_CLASSES))
        label = SHAPE_CLASSES[label_idx]
        img = RENDERERS[label]()
        img = _add_noise(img)
        if random.random() < 0.5:
            img = img.rotate(random.uniform(-20, 20), fillcolor=0)
        arr = np.array(img, dtype=np.float32) / 255.0
        xs[i, 0] = torch.from_numpy(arr)
        ys[i] = label_idx
    return xs, ys
