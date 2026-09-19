# Running localy
```bash
docker build -t cyberrealistic_pony_server .
docker run -p 6379:6379 cyberrealistic_pony_server
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
    "positive_prompt": "score_9, score_8_up, score_7_up, a cat wearing sunglasses",
    "negative_prompt": "score_4, score_5, score_6, bad hands, bad anatomy, ugly, deformed",
    "image_ratio": "1:1",
    "image_quality": "high"
  }
}
```

Only `text_to_image` exists today — [CyberRealisticPony](https://huggingface.co/cyberdelia/CyberRealisticPony)
has no img2img/edit variant wired up.

Notes on this model vs. realvisxl_v5/z_image/qwen_image:
- It's not published in diffusers format — the HF repo is a stack of
  single-file Civitai-style checkpoints (`CyberRealisticPony_V*.safetensors`),
  not a `model_index.json` + `unet`/`vae`/`text_encoder` layout. So
  `processing/generate_image.py` loads it with
  `StableDiffusionXLPipeline.from_single_file(...)` pointed at one pinned
  version (`CyberRealisticPony_V18.0_FP16.safetensors`) instead of
  `from_pretrained(repo_id)` — bump `CHECKPOINT_URL` there to move to a
  newer version deliberately, rather than silently following "latest".
- It's a Pony/SDXL checkpoint (same architecture as RealVisXL V5), so it
  reuses the same `StableDiffusionXLPipeline`, aspect-ratio buckets, and
  DPM++ SDE Karras scheduler swap as `realvisxl_v5`.
- Pony's own convention is to lead prompts with score tags, e.g.
  `score_9, score_8_up, score_7_up, <subject>` (and score_4/5/6 in the
  negative prompt to push away low-quality outputs) — see the model card's
  example prompt. This isn't injected in code; callers are expected to
  include it in `positive_prompt`/`negative_prompt` themselves.
- The model card recommends CFG scale **5** (lower than RealVisXL's 7.0)
  and a minimum of 30 steps — see `IMAGE_QUALITY_TO_STEPS` and the
  `guidance_scale` in `processing/generate_image.py`.
- Baked-in VAE, fits comfortably in 16GB VRAM at fp16 — no CPU offload
  needed.
- Licensed under `creativeml-openrail-m` (not RealVisXL's `openrail++`) —
  it's a Pony-derived checkpoint commonly used for mature/NSFW content, so
  read the license and civitai model page before productizing:
  https://huggingface.co/cyberdelia/CyberRealisticPony /
  https://civitai.com/models/443821/cyberrealistic-pony

Attach a RunPod Network Volume to the endpoint so the model weights (cached
under `HF_HOME`, redirected to `/runpod-volume/hf-cache` automatically when
a volume is mounted — see `entrypoint.sh`) persist across worker restarts
instead of re-downloading from HuggingFace on every cold start.
