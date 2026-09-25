#!/bin/bash
set -e

# Reuse a persistent RunPod Network Volume for the HF cache if one is
# mounted, so the ~124GB of MiniMax-H3 weights (fl2va transformer + Qwen3-VL
# conditioner, both released in bfloat16) survive across pod/worker
# restarts instead of being re-downloaded on every boot - same HF_HOME
# redirect as qwen_chat/entrypoint.sh, just for a much bigger cache.
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

# Default: Pod-based Celery worker, consuming the "minimax_h3" queue from
# the shared Redis broker (see aafactory_nsfw's backend/celery_worker.py).
redis-server --protected-mode no &

uv run celery -A celery_worker.app worker --loglevel=info -Q minimax_h3 -P solo
