#!/bin/bash
set -e

# Reuse a persistent RunPod Network Volume for the HF cache if one is
# mounted, so the Qwen2.5-3B-Instruct weights survive across pod/worker
# restarts instead of being re-downloaded on every boot. Falls back to the
# default HF cache location when no volume is attached (e.g. local
# `docker run`, or a Pod without a volume configured).
if [ -d "/runpod-volume" ]; then
  export HF_HOME="/runpod-volume/hf-cache"
  echo "[entrypoint] Network volume detected — HF_HOME=$HF_HOME (cache size: $(du -sh "$HF_HOME" 2>/dev/null | cut -f1 || echo 'empty'))"
else
  echo "[entrypoint] No network volume mounted — using default HF cache (will re-download on every restart)"
fi

if [ "$WORKER_MODE" = "serverless" ]; then
  # RunPod Serverless: RunPod manages its own job queue and invokes
  # handler.py's handler() per job — no Redis/Celery involved here.
  exec uv run python handler.py
fi

# Default: Pod-based Celery worker, consuming the "qwen_chat" queue from
# the shared Redis broker (see aafactory_nsfw's backend/celery_worker.py).
redis-server --protected-mode no &

uv run celery -A celery_worker.app worker --loglevel=info -Q qwen_chat -P solo
