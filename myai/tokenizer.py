"""Character-level tokenizer. Built from scratch — no external vocab.

We use a fixed character set covering printable ASCII plus a few control
tokens. New characters fall back to a generic <unk> id, so the model never
crashes on unseen input.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"
UNK = "<unk>"

SPECIALS = [PAD, BOS, EOS, SEP, UNK]

# Printable ASCII range that covers normal conversation.
_BASE_CHARS = [chr(c) for c in range(32, 127)] + ["\n", "\t"]


class CharTokenizer:
    def __init__(self, extra_chars: Iterable[str] = ()):  # noqa: B008
        chars = list(_BASE_CHARS)
        seen = set(chars)
        for c in extra_chars:
            if c not in seen:
                chars.append(c)
                seen.add(c)
        self.itos: list[str] = list(SPECIALS) + chars
        self.stoi: dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def sep_id(self) -> int:
        return self.stoi[SEP]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        unk = self.unk_id
        for ch in text:
            ids.append(self.stoi.get(ch, unk))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out: list[str] = []
        special_ids = {self.stoi[s] for s in SPECIALS}
        for i in ids:
            if i in special_ids:
                continue
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return "".join(out)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls.__new__(cls)
        tok.itos = list(data["itos"])
        tok.stoi = {s: i for i, s in enumerate(tok.itos)}
        return tok
