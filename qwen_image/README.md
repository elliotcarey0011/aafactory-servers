# Running localy
```bash
docker build -t qwen_image_server .
docker run -p 6379:6379 qwen_image_server
```

# Running as a RunPod Serverless worker

Same image, different entrypoint mode — set `WORKER_MODE=serverless` as an
environment variable on the RunPod Serverless endpoint (instead of the
default, which runs a Celery worker for a persistent Pod). The container
then runs `handler.py` directly; there's no Redis/Celery involved, RunPod's
own queue dispatches jobs to it. Unlike zonos, this server has two task
types, so the job input carries a `task_name` key telling the handler which
one to run — matches what `aafactory_nsfw`'s `runpod_serverless.py` sends:

```json
{
  "input": {
    "task_name": "text_to_image",
    "positive_prompt": "a cat wearing sunglasses",
    "negative_prompt": "",
    "image_ratio": "1:1",
    "image_quality": "high"
  }
}
```

or for `image_to_image_edit`, swap in `image_bytes` (base64) in place of
`image_ratio`. Each task type's pipeline is loaded lazily and cached
per-worker on first use — a worker only pays to load the ones it's actually
asked to run.

Attach a RunPod Network Volume to the endpoint so the model weights (cached
under `HF_HOME`, redirected to `/runpod-volume/hf-cache` automatically when
a volume is mounted — see `entrypoint.sh`) persist across worker restarts
instead of re-downloading from HuggingFace on every cold start. These are
sizeable checkpoints (base Qwen-Image + Qwen-Image-Edit DFloat11 weights) —
size the volume generously (100GB+) rather than risk running out mid-cache.

# Remote Hardware requirements

Tested on:
small updaste
![Qwen Image Remote Hardware requirements Screenshot](../assets/qwen-image-requirements.png)