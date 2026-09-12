#!/bin/bash
set -e

uv run python kandinsky/download_models.py &
./install_sage_attention.sh &
wait

if [ "$WORKER_MODE" = "serverless" ]; then
  # RunPod Serverless: RunPod manages its own job queue and invokes
  # handler.py's handler() per job — no Redis/Celery involved here.
  exec uv run python handler.py
fi

# Default: Pod-based Celery worker, consuming the "kandinsky" queue from the
# shared Redis broker (see aafactory_nsfw's backend/celery_worker.py).
redis-server --protected-mode no &

uv run celery -A celery_worker.app worker --loglevel=info -Q kandinsky -P solo
