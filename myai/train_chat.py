"""Train the chat brain from scratch on the seed corpus + your real chats.

Usage:
    python -m myai.train_chat --seed --steps 2000
    python -m myai.train_chat --steps 500   # incremental: just train on chats
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from .model import ChatBrain, ModelConfig
from .paths import CHAT_MODEL_PT, CHATS_JSONL, TOKENIZER_JSON, ensure_dirs
from .seed_corpus import seed_text
from .tokenizer import CharTokenizer


def _format_dialog(user: str, bot: str) -> str:
    return f"<user>{user}</user><bot>{bot}</bot>"


def build_corpus(use_seed: bool) -> str:
    parts: list[str] = []
    if use_seed:
        parts.append(seed_text())
    if CHATS_JSONL.exists():
        for line in CHATS_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = row.get("user", "").strip()
            bot = row.get("bot", "").strip()
            if user and bot:
                parts.append(_format_dialog(user, bot))
    return "\n".join(parts) if parts else seed_text()


def _load_or_init(corpus: str, device: torch.device) -> tuple[ChatBrain, CharTokenizer, ModelConfig]:
    if TOKENIZER_JSON.exists():
        tok = CharTokenizer.load(TOKENIZER_JSON)
    else:
        tok = CharTokenizer(extra_chars=set(corpus))
        tok.save(TOKENIZER_JSON)

    if CHAT_MODEL_PT.exists():
        ckpt = torch.load(CHAT_MODEL_PT, map_location=device, weights_only=False)
        cfg = ModelConfig.from_dict(ckpt["config"])
        if cfg.vocab_size != tok.vocab_size:
            cfg.vocab_size = tok.vocab_size
        model = ChatBrain(cfg).to(device)
        try:
            model.load_state_dict(ckpt["state"], strict=False)
        except Exception as e:
            print(f"[warn] could not load prior weights, starting fresh: {e}")
        return model, tok, cfg

    cfg = ModelConfig(vocab_size=tok.vocab_size)
    model = ChatBrain(cfg).to(device)
    return model, tok, cfg


def _save(model: ChatBrain, cfg: ModelConfig) -> None:
    ensure_dirs()
    torch.save({"state": model.state_dict(), "config": cfg.to_dict()}, CHAT_MODEL_PT)


def _get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, max(1, data.size(0) - block_size - 1), (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def train(steps: int, seed: bool, batch_size: int, lr: float) -> None:
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = build_corpus(use_seed=seed)
    if len(corpus) < 32:
        print("[info] not enough data to train. Add chats first, or pass --seed.")
        return

    model, tok, cfg = _load_or_init(corpus, device)
    ids = torch.tensor(tok.encode(corpus), dtype=torch.long)
    if ids.numel() < cfg.block_size + 2:
        # Pad by repeating so we can form batches.
        reps = (cfg.block_size + 2) // max(1, ids.numel()) + 2
        ids = ids.repeat(reps)

    print(f"[info] device={device} | params={model.num_params():,} | tokens={ids.numel():,}")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = _get_batch(ids, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 50 == 0 or step == 1:
            print(f"step {step:>5}/{steps}  loss={loss.item():.4f}  elapsed={time.time() - t0:.1f}s")
        if step % 500 == 0:
            _save(model, cfg)

    _save(model, cfg)
    print(f"[done] saved to {CHAT_MODEL_PT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", action="store_true", help="include built-in seed corpus")
    args = parser.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    train(steps=args.steps, seed=args.seed, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
