# Running localy
```bash
docker build -t minimax_h3_server .
docker run -p 6379:6379 -e HF_TOKEN=hf_xxx minimax_h3_server
```

`HF_TOKEN` must be a HuggingFace access token with billing enabled for
Inference Providers (routed here through `fal-ai`) — see
[Model](#model) below for how the request is routed.

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
    "image_bytes": "<base64-encoded source image>",
    "prompt": "The subject starts to dance",
    "negative_prompt": null,
    "num_frames": null,
    "seed": null
  }
}
```

Returns `{"video_base64": "..."}` — a base64-encoded MP4.

## Model

[`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) — an
omni-modal video generation foundation model (text/image/video/audio in,
4-15s video up to 2K/24fps out, with synchronized stereo audio). Unlike the
other image/video servers in this repo (`qwen_image`, `wan_animate`,
`infinite_talk`), it's far too large to load and run locally in this repo's
GPU containers, so this server doesn't ship any model weights or a local
`diffusers`/ComfyUI pipeline at all. Instead it proxies each request to the
hosted `fal-ai` endpoint through HuggingFace's Inference Providers routing
(`huggingface_hub.InferenceClient(provider="fal-ai")`), which is how
HuggingFace itself documents calling this model — see the "Inference
Providers" code snippet on the model page (fal-ai / Python /
huggingface_hub / Inference API).

`processing/generate_video.py` builds one `InferenceClient` per worker
process and reuses it across requests (same caching pattern as the
`_MODEL`/`_PIPES` dicts in `qwen_chat`/`qwen_image`), then calls
`client.image_to_video(...)` with the model pinned to
`MiniMaxAI/MiniMax-H3`. Billing for the actual generation happens on
whatever HuggingFace/fal-ai account owns `HF_TOKEN`, not on this
container's own compute.
