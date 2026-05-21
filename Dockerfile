FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

WORKDIR /app

# CPU-only torch is much smaller and fits free tiers.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 && \
    pip install --no-cache-dir \
        numpy==1.26.4 \
        Pillow==10.4.0 \
        Flask==3.0.3 \
        gunicorn==22.0.0

COPY myai/ /app/myai/
COPY weights/ /app/weights/
COPY data/ /app/data/

# Train a tiny seed model at build time so the app responds on first boot.
RUN python -m myai.train_chat --seed --steps 600 && \
    python -m myai.train_shapes --steps 800

EXPOSE 7860

# HF Spaces and most platforms expect the app on $PORT.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120 myai.app:app"]
