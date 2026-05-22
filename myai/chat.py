"""Run inference on the chat brain."""
from __future__ import annotations

import torch

from .model import ChatBrain, ModelConfig
from .paths import CHAT_MODEL_PT, TOKENIZER_JSON
from .tokenizer import CharTokenizer


_USER_OPEN = "<user>"
_USER_CLOSE = "</user>"
_BOT_OPEN = "<bot>"
_BOT_CLOSE = "</bot>"


class ChatRunner:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not CHAT_MODEL_PT.exists() or not TOKENIZER_JSON.exists():
            raise FileNotFoundError(
                "No trained chat brain found. Run `python -m myai.train_chat --seed --steps 2000` first."
            )
        self.tok = CharTokenizer.load(TOKENIZER_JSON)
        ckpt = torch.load(CHAT_MODEL_PT, map_location=self.device, weights_only=False)
        cfg = ModelConfig.from_dict(ckpt["config"])
        self.model = ChatBrain(cfg).to(self.device)
        self.model.load_state_dict(ckpt["state"], strict=False)
        self.model.eval()
        self.cfg = cfg

    def reply(
        self,
        user_text: str,
        max_new_tokens: int = 160,
        temperature: float = 0.9,
        top_k: int = 40,
    ) -> str:
        prompt = f"{_USER_OPEN}{user_text}{_USER_CLOSE}{_BOT_OPEN}"
        ids = self.tok.encode(prompt)
        ids = ids[-self.cfg.block_size:]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        out = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated = out[0, len(ids):].tolist()
        text = self.tok.decode(generated)

        # Stop at end-of-bot tag if it appears.
        if _BOT_CLOSE in text:
            text = text.split(_BOT_CLOSE, 1)[0]
        # Also stop if the model starts a new user turn.
        if _USER_OPEN in text:
            text = text.split(_USER_OPEN, 1)[0]
        return text.strip() or "..."
