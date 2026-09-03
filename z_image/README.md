# Running localy
```bash
docker build -t z_image_server .
docker run -p 6379:6379 z_image_server
```

# Running as a RunPod Serverless worker

Same image, different entrypoint mode — set `WORKER_MODE=serverless` as an
environment variable on the RunPod Serverless endpoint (instead of the
default, which runs a Celery worker for a persistent Pod). The container
then runs `handler.py` directly; there's no Redis/Celery involved, RunPod's
own queue dispatches jobs to it:

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

Only `text_to_image` exists today — [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
has no released edit variant yet (an Edit checkpoint is announced but not
out), unlike qwen_image which has both.

Notes on this model vs. qwen_image:
- No DFloat11 compression or CPU offload — the bf16 pipeline fits within
  16GB VRAM per the model card, so it loads straight onto the GPU.
- It's a distilled "turbo" checkpoint: `guidance_scale` is fixed at `0.0`
  and step counts stay low (4-20 across the quality tiers) per the model
  card's guidance, rather than qwen_image's 20-50 step range.
- Requires `diffusers` installed from the GitHub `main` branch — `ZImagePipeline`
  isn't in a tagged release yet (see `pyproject.toml`).

Attach a RunPod Network Volume to the endpoint so the model weights (cached
under `HF_HOME`, redirected to `/runpod-volume/hf-cache` automatically when
a volume is mounted — see `entrypoint.sh`) persist across worker restarts
instead of re-downloading from HuggingFace on every cold start.
