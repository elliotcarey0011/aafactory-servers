# Running localy
```bash
docker build -t qwen_chat_server .
docker run -p 6379:6379 qwen_chat_server
```

# Running as a RunPod Serverless worker

Same image, different entrypoint mode — set `WORKER_MODE=serverless` as an
environment variable on the RunPod Serverless endpoint (instead of the
default, which runs a Celery worker for a persistent Pod). The container
then runs `handler.py` directly; there's no Redis/Celery involved, RunPod's
own queue dispatches jobs to it. This server has two task types, so the job
input carries a `task_name` key telling the handler which one to run —
matches what `aafactory_nsfw`'s `runpod_serverless.py` sends:

```json
{
  "input": {
    "task_name": "chat",
    "messages": [
      { "role": "user", "content": "What's a good way to light a portrait for a moody, cinematic look?" }
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }
}
```

or for `generate_prompt` (writing a ready-to-use SDXL/Pony-style tag prompt
for the other image servers, e.g. `cyberrealistic_pony`, `qwen_image`,
`z_image`):

```json
{
  "input": {
    "task_name": "generate_prompt",
    "subject": "a woman sitting by a rain-streaked window at night",
    "style_hints": "cinematic lighting, shallow depth of field, photorealistic"
  }
}
```

`generate_prompt` has a fixed system prompt (see
`processing/generate_text.py`) that writes tag-style prompts leading with
`score_9, score_8_up, score_7_up`, matching the Pony/SDXL convention
documented in `cyberrealistic_pony/README.md`. Both tasks return
`{"text": "..."}`.

## Model

[`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) —
chosen over similarly-sized alternatives (e.g. Llama-3.2-3B-Instruct) for two
reasons:
- **License**: Apache-2.0, with no usage restrictions — Meta's Llama license
  carries an Acceptable Use Policy that restricts sexual content, which
  conflicts with this app (`aafactory_nsfw`) generating NSFW content.
- **Consistency**: Qwen is already used elsewhere in this repo
  (`qwen_image`, and `infinite_talk`'s `prompt_extend.py` for prompt
  expansion), and is small enough (~6GB in bf16) for a fast cold start with
  no quantization needed.

The model is loaded once per worker process and reused across every
request (see `_MODEL`/`_TOKENIZER` in `processing/generate_text.py`) rather
than reloaded per job.

Attach a RunPod Network Volume to the endpoint so the model weights (cached
under `HF_HOME`, redirected to `/runpod-volume/hf-cache` automatically when
a volume is mounted — see `entrypoint.sh`) persist across worker restarts
instead of re-downloading from HuggingFace on every cold start.
