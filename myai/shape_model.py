"""Tiny CNN for shape recognition, from scratch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


SHAPE_CLASSES = ["circle", "square", "triangle", "star", "heart", "line"]


@dataclass
class ShapeConfig:
    img_size: int = 28
    n_classes: int = len(SHAPE_CLASSES)


class ShapeBrain(nn.Module):
    def __init__(self, cfg: ShapeConfig | None = None):
        super().__init__()
        cfg = cfg or ShapeConfig()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        flat = 64 * (cfg.img_size // 8) * (cfg.img_size // 8)
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, cfg.n_classes)
        self.drop = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[str, float, dict[str, float]]:
        self.eval()
        logits = self(x)
        probs = F.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        scores = {SHAPE_CLASSES[i]: float(probs[i].item()) for i in range(len(SHAPE_CLASSES))}
        return SHAPE_CLASSES[idx], float(probs[idx].item()), scores
