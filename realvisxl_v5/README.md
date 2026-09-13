# Running localy
```bash
docker build -t realvisxl_v5_server .
docker run -p 6379:6379 realvisxl_v5_server
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
    "negative_prompt": "bad hands, bad anatomy, ugly, deformed",
    "image_ratio": "1:1",
    "image_quality": "high"
  }
}
```

Only `text_to_image` exists today — [RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0)
has no img2img/edit variant wired up.

Notes on this model vs. z_image/qwen_image:
- It's a full (non-distilled) SDXL checkpoint — `diffusers`' standard
  `StableDiffusionXLPipeline` loads it straight from a tagged release, no
  need for the `git+https` main-branch install z_image requires.
- Real classifier-free guidance applies (`guidance_scale=7.0`), unlike
  z_image's turbo checkpoint which is distilled for `guidance_scale=0.0`.
- The model card recommends the "DPM++ SDE Karras" sampler over SDXL's
  default Euler scheduler, and a minimum of 30 steps — see
  `processing/generate_image.py` for the scheduler swap and step tiers.
- Fits comfortably in 16GB VRAM at fp16 — no CPU offload needed.
- Licensed under `openrail++`, which carries some use-based restrictions —
  worth a read before productizing: https://huggingface.co/SG161222/RealVisXL_V5.0

Attach a RunPod Network Volume to the endpoint so the model weights (cached
under `HF_HOME`, redirected to `/runpod-volume/hf-cache` automatically when
a volume is mounted — see `entrypoint.sh`) persist across worker restarts
instead of re-downloading from HuggingFace on every cold start.
