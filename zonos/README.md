# Running localy
```bash
docker build -t zonos_server .
docker run -p 6379:6379 zonos_server
```

# Running as a RunPod Serverless worker

Same image, different entrypoint mode — set `WORKER_MODE=serverless` as an
environment variable on the RunPod Serverless endpoint (instead of the
default, which runs a Celery worker for a persistent Pod). The container
then runs `handler.py` directly; there's no Redis/Celery involved, RunPod's
own queue dispatches jobs to it. Job input shape matches the existing
`custom_voice_to_audio` Celery task's kwargs:

```json
{
  "input": {
    "prompt": "hello",
    "voice_bytes": "<base64-encoded reference audio>",
    "language": "en-us"
  }
}
```

Attach a RunPod Network Volume to the endpoint so the model weights (cached
under `HF_HOME`, redirected to `/runpod-volume/hf-cache` automatically when
a volume is mounted — see `entrypoint.sh`) persist across worker restarts
instead of re-downloading from HuggingFace on every cold start.

# Remote Hardware requirements

Tested on:

![Qwen Image Remote Hardware requirements Screenshot](../assets/zonos-requirements.png)