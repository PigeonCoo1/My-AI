# My-AI

Your very own AI, built from scratch. No OpenAI, no Claude API, no Llama, no
pre-trained weights from anyone else. Just math, PyTorch, and your data.

It has two brains:

1. **ChatBrain** — a small character-level transformer (nanoGPT-style)
   that learns to chat from your conversations.
2. **ShapeBrain** — a small CNN that learns to recognize shapes you draw
   (circle, square, triangle, star, ...).

Both start knowing nothing. They learn from you. The more you use it, the
better it gets.

## Quick start

```bash
pip install -r requirements.txt

# Seed the chat brain with a tiny built-in corpus so it can say its first words
python -m myai.train_chat --seed --steps 2000

# Seed the shape brain with synthetic shapes
python -m myai.train_shapes --steps 1500

# Launch the web UI (chat + draw shapes)
python -m myai.app
# open http://localhost:5000
```

## "Works when I'm not on it"

Two ways to keep it learning while you're away:

- **Local background trainer**: `python -m myai.background_trainer`
  Runs forever, retraining on new chats every few minutes.
- **GitHub Action**: `.github/workflows/nightly-train.yml` retrains
  on a cron schedule and commits the new weights back.

## Files

- `myai/model.py`         — the tiny transformer (chat brain)
- `myai/shape_model.py`   — the CNN (shape brain)
- `myai/tokenizer.py`     — character tokenizer
- `myai/train_chat.py`    — chat brain trainer
- `myai/train_shapes.py`  — shape brain trainer
- `myai/chat.py`          — sampling / inference
- `myai/app.py`           — Flask web UI
- `myai/background_trainer.py` — continuous retrainer
- `data/chats.jsonl`      — every conversation, appended forever
- `data/shape_examples/`  — shapes you've drawn + labels
- `weights/`              — saved model weights

## Honest expectations

This model has ~1-3 million parameters. ChatGPT has ~hundreds of billions.
At first it will produce gibberish. After a few hundred real conversations
plus the seed corpus, it'll start forming words and short replies. It will
never be ChatGPT — but it is genuinely, mathematically, **yours**.
