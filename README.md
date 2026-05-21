---
title: My-AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

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

## Use it from your phone (free hosting)

### Option A — Hugging Face Spaces (recommended, fully free, always-on)
1. Go to https://huggingface.co/new-space
2. Name it `my-ai`, choose **Docker** as the SDK, visibility = whatever you want.
3. After creation, on the Space page click **"Files" → "Add → Upload files"**, OR run:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/my-ai
   git push hf claude/custom-ai-chatbot-shapes-RzImY:main
   ```
4. Wait ~5 min for the build. You'll get a public URL like
   `https://<you>-my-ai.hf.space` — open that on your phone.

### Option B — Fly.io (free tier, always-on small VM)
```bash
brew install flyctl   # or: curl -L https://fly.io/install.sh | sh
fly auth signup
fly launch --no-deploy --copy-config --name my-ai
fly volumes create myai_data --region iad --size 1
fly deploy
```
Public URL: `https://my-ai.fly.dev`.

### Option C — Render (free, sleeps after 15 min idle)
1. Push this repo to GitHub.
2. https://render.com → **New → Web Service** → connect repo.
3. Environment: **Docker**. Plan: **Free**. Click Create.

### Why not Netlify / Vercel?
They only run serverless functions (10-second limit, no persistent disk,
~250MB code cap). PyTorch is ~200MB on its own and the chat brain takes
longer than 10 seconds during training — so they don't fit.

## Honest expectations

This model has ~1-3 million parameters. ChatGPT has ~hundreds of billions.
At first it will produce gibberish. After a few hundred real conversations
plus the seed corpus, it'll start forming words and short replies. It will
never be ChatGPT — but it is genuinely, mathematically, **yours**.
