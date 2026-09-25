# Running localy
```bash
docker build -t minimax_h3_server .
docker run -p 6379:6379 --gpus all -e HF_TOKEN=hf_xxx minimax_h3_server
```

`HF_TOKEN` must belong to a HuggingFace account that (a) has accepted
MiniMax-H3's community license (gated download) and (b) has read access to
the private [`elliotcareydev/minimax-h3-fingering-lora`](https://huggingface.co/elliotcareydev/minimax-h3-fingering-lora)
repo — see [Model](#model)/[LoRA](#lora) below.

## Hardware

Unlike this repo's other image/video servers, this one is not a small
diffusers checkpoint — MiniMax-H3's fl2va workflow alone is a ~61.7GB
transformer plus a ~62.1GB Qwen3-VL conditioner, ~124GB in bfloat16. The
pipeline uses `ComponentsManager` auto CPU offload so this fits on a single
**80GB GPU** (e.g. A100/H100 80GB), but expect:
- ~124GB+ of free system RAM (the weights live there; the manager streams
  onto the GPU only what each step needs)
- A large, slow first boot to pull ~124GB of weights from HuggingFace —
  attach a RunPod Network Volume (see `entrypoint.sh`) so this only happens
  once
- Smaller cards (24-32GB) are possible with int8 quantization + block-level
  offload instead, per the diffusers MiniMax-H3 docs' "Memory" section —
  not implemented here, since this server was set up against an 80GB card

# Running as a RunPod Serverless worker

Same image, different entrypoint mode — set `WORKER_MODE=serverless` as an
environment variable on the RunPod Serverless endpoint (instead of the
default, which runs a Celery worker for a persistent Pod). The container
then runs `handler.py` directly; there's no Redis/Celery involved, RunPod's
own queue dispatches jobs to it. This server has a single task type, so
(unlike `qwen_chat`/`qwen_image`) the job input carries no `task_name` key:

```json
{
  "input": {
    "image_bytes": "<base64-encoded first-frame image>",
    "prompt": "The subject starts to dance",
    "num_frames": 124,
    "seed": null
  }
}
```

`num_frames` is optional (defaults to 124, ≈5.2s at MiniMax-H3's fixed
24fps) and gets snapped up to the next `17 * n + 5` the video VAE can
decode; the resulting clip must land between 5 and 15 seconds. There's no
`negative_prompt` — MiniMax-H3's released checkpoints are
guidance-distilled, so there's no guider/CFG at all, unlike the other
image servers in this repo.

Returns `{"video_base64": "..."}` — a base64-encoded MP4, audio track
already muxed in (MiniMax-H3 generates video and its synchronized
soundtrack together, in one denoising pass).

## Model

[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) — an
omni-modal video generation foundation model (text/image/video/audio in,
4-15s video up to 2K/24fps out, with synchronized stereo audio). It's
integrated in `diffusers` as **Modular Diffusers**
(`diffusers.ModularPipeline`), not the classic `DiffusionPipeline` used by
`qwen_image`/`cyberrealistic_pony` — there is no `DiffusionPipeline` half
to this integration, the modular blocks and `MiniMaxH3ModularPipeline` are
the whole thing. This server only loads the `fl2va` workflow (first/last
keyframe → video), since that's the image-to-video task it exposes; it
never touches the `ref2va` checkpoint partition.

`processing/generate_video.py` builds one `ModularPipeline` per worker
process and reuses it across requests (same `_PIPE` caching pattern as the
`_MODEL`/`_PIPES` dicts in `qwen_chat`/`qwen_image`), with
`ComponentsManager.enable_auto_cpu_offload` handling the GPU/host-RAM
juggling described above.

### LoRA

[`MinimaxH3-Fingering_000002000.safetensors`](https://huggingface.co/elliotcareydev/minimax-h3-fingering-lora)
is a custom fine-tune, loaded onto the transformer on every request. It's
155MB — over GitHub's 100MB push limit and far bigger than anything else
in this repo's git history — so instead of committing it, it's hosted on
a private HF repo and pulled down at pipeline-load time via
`hf_hub_download` (cached through the normal HF cache, same as the base
model — see [Hardware](#hardware) above for where that cache lives).

This model has no pipe-level `load_lora_weights()` (`ModularPipeline`
doesn't have one the way `DiffusionPipeline` does — compare
`qwen_image/processing/generate_image.py`'s
`pipe.load_lora_weights('starsfriday/Qwen-Image-NSFW', ...)`); instead
`MiniMaxH3Transformer3DModel` is itself a PEFT adapter host, so the LoRA is
loaded straight onto `pipe.transformer` via `load_lora_adapter()` with the
downloaded state dict (see `_get_pipe()` in
`processing/generate_video.py`).

I did not confirm the exact key-prefix convention the trainer that
produced this checkpoint used against `load_lora_adapter`'s default
`prefix="transformer"` filtering — if loading fails with an "unexpected
keys" style error, that's the first thing to check.

### A note on freshness

MiniMax-H3's Modular Diffusers integration is very recent — as of writing
it's only on `diffusers`' `main` branch, not a tagged PyPI release (see the
git dependency in `pyproject.toml`). That means `uv sync` pulls whatever
`main` currently has, which can change or break under this server without
warning; pin to a specific commit once this has been build-tested, and
switch to a normal version bound once MiniMax-H3 support ships in a
release.
