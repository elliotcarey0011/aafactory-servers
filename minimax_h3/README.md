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
diffusers checkpoint. This server loads **both** of MiniMax-H3's
checkpoint partitions (fl2va's `transformer/` and ref2va's
`transformer_ref/`) into one pipeline, so it can serve both tasks below —
that's ~61.7GB + ~61.7GB of transformers plus a shared ~62.1GB Qwen3-VL
conditioner, **~185GB in bfloat16**. The pipeline uses `ComponentsManager`
auto CPU offload so this fits on a single **80GB GPU** (e.g. A100/H100
80GB), but expect:
- **~185GB+ of free system RAM** (the weights live there; the manager
  streams onto the GPU only what each step needs) — noticeably more than
  an fl2va-only setup would need, since ref2va's transformer_ref adds
  another ~62GB on top
- A large, slow first boot to pull ~185GB of weights from HuggingFace —
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
own queue dispatches jobs to it. This server has two task types, so (like
`qwen_chat`/`qwen_image`) the job input carries a `task_name` key telling
the handler which one to run.

## `image_to_video` (fl2va)

The uploaded image becomes the video's **literal starting frame** — the
prompt describes motion/continuation from it, not a reinterpretation of
its content onto a different subject or scene.

```json
{
  "input": {
    "task_name": "image_to_video",
    "image_bytes": "<base64-encoded first-frame image>",
    "prompt": "The subject starts to dance",
    "num_frames": 124,
    "seed": null
  }
}
```

## `reference_to_video` (ref2va)

The uploaded image(s) are **labeled references**, not a starting frame —
MiniMax-H3 labels them `<Picture 1>`, `<Picture 2>`, etc. (in the order
`reference_images` lists them), and the prompt addresses them by that tag
to tell the model what to pull from which reference. This is the mode for
requests like "use the face from this photo on a German soldier":

```json
{
  "input": {
    "task_name": "reference_to_video",
    "reference_images": ["<base64-encoded reference image>"],
    "prompt": "<Picture 1> as a German WWII soldier, standing at attention in a snowy trench, cinematic lighting",
    "num_frames": 124,
    "seed": null
  }
}
```

`reference_images` takes up to 9 images; the model doesn't infer which
reference a part of the prompt means without an explicit `<Picture N>` tag
— a prompt with no tag at all just gets the ordinary text-to-video
behavior with the references present but unaddressed. Unlike
`image_to_video`, references don't bind the generated canvas — it defaults
to MiniMax-H3's own 16:9 regardless of the reference images' own aspect
ratios.

## Both tasks

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
the whole thing.

`processing/generate_video.py` builds **one** `ModularPipeline` per worker
process, loaded with no `workflow=` restriction so it holds both the fl2va
and ref2va transformer partitions and can serve either task from the same
`pipe(...)` call (it picks the workflow per-call based on whether `image=`
or `references=` was passed — see the diffusers MiniMax-H3 docs' "A
generation as a reference" section for this exact pattern). This shares
the ~62GB Qwen3-VL conditioner between both tasks instead of loading it
twice, at the cost of always paying for both transformers even if a given
deployment only ever gets one task type — see [Hardware](#hardware) above.
The pipe is reused across requests (same `_PIPE` caching pattern as the
`_MODEL`/`_PIPES` dicts in `qwen_chat`/`qwen_image`), with
`ComponentsManager.enable_auto_cpu_offload` handling the GPU/host-RAM
juggling.

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

It's applied only to `pipe.transformer` (fl2va, the `image_to_video` task)
— the LoRA was trained/validated against that checkpoint partition
specifically. `pipe.transformer_ref` (ref2va, `reference_to_video`) is
left un-adapted; applying this same state dict there is untested and not
assumed to be safe, so `reference_to_video` currently runs the stock
model with no fine-tune.

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
