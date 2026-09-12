# Running localy
```bash
docker build -t zonos_server .
docker run -p 6379:6379 kandinsky_server
```

# Remote Hardware requirements

Tested on:

![Infinite Talk Remote Hardware requirements Screenshot](../assets/infinite-talk-requirements.png)

# Running as a RunPod Serverless worker

Same image, different entrypoint mode — set `WORKER_MODE=serverless` as an
environment variable on the RunPod Serverless endpoint (instead of the
default, which runs a Celery worker for a persistent Pod). The container
then runs `handler.py` directly; there's no Redis/Celery involved, RunPod's
own queue dispatches jobs to it:

```json
{
  "input": {
    "task_name": "kandinsky_text_to_video",
    "prompt": "a dancing clown",
    "video_aspect_ratio": "3:2"
  }
}
```

`task_name` must be `kandinsky_text_to_video` — it's the same value
aafactory_nsfw's `send_task_to_server` uses as the Celery task name for the
Pod-based path, so `handler.py` dispatches on it too. Only this one task
type exists today (the SFT 5s checkpoint, per
`processing/compute_text_to_video.py`).

Both modes download model weights on container start (`download_models.py`,
run from `entrypoint.sh`) into `./weights` — there's no RunPod Network
Volume caching for this server yet, unlike z_image/qwen_image, so expect
every cold start to re-download the full set of checkpoints (DiT, VAE, and
two text encoders).