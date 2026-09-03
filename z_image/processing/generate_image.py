import random
from io import BytesIO
from typing import Callable, Optional
from diffusers import ZImagePipeline
import torch
import base64


MODEL_NAME = "Tongyi-MAI/Z-Image-Turbo"

# Loaded once per worker process and reused across every request, instead of
# rebuilding the pipeline on every single task.
_PIPES: dict[str, ZImagePipeline] = {}


def _get_text_to_image_pipe() -> ZImagePipeline:
    if "text_to_image" not in _PIPES:
        pipe = ZImagePipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        )
        # No DFloat11/CPU-offload needed here (unlike qwen_image) — Z-Image-Turbo
        # fits within 16GB VRAM per its model card, so the full bf16 pipeline
        # loads straight onto the GPU.
        pipe.to("cuda")
        _PIPES["text_to_image"] = pipe
    return _PIPES["text_to_image"]


# Turbo models are distilled for a small, fixed number of steps (the model
# card recommends 9) — these tiers stay within the range turbo checkpoints
# were trained for rather than reusing qwen_image's 20-50 step range, which
# would just add latency without the quality gains a non-distilled model gets.
IMAGE_QUALITY_TO_STEPS = {
    "low": 4,
    "medium": 9,
    "high": 14,
    "ultra": 20,
}

ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


def run_text_to_image(
    positive_prompt: str,
    negative_prompt: str,
    image_ratio: str,
    image_quality: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """
    Run text-to-image processing.

    Args:
        positive_prompt (str): The positive text prompt.
        negative_prompt (str): The negative text prompt.
        image_ratio (str): The desired image aspect ratio.
        image_quality (str): The desired image quality.
        progress_callback: optional, called as (step, total_steps) after each
            denoising step — step is 1-indexed, reaching total_steps on the
            final call. Left as a plain callback (not a RunPod/Celery import)
            so this module stays transport-agnostic; handler.py and
            celery_worker.py each pass in their own reporting mechanism.

    Returns:
        bytes: a base64-encoded string.
    """
    pipe = _get_text_to_image_pipe()

    width, height = ASPECT_RATIOS[image_ratio]
    random_seed = random.randint(0, 999999)
    total_steps = IMAGE_QUALITY_TO_STEPS[image_quality]

    def _on_step_end(pipe, step, timestep, callback_kwargs):
        if progress_callback:
            progress_callback(step + 1, total_steps)  # diffusers' step is 0-indexed
        return callback_kwargs

    with torch.inference_mode():
        image = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=total_steps,
            # Turbo checkpoints are distilled for guidance_scale=0.0 — the
            # model card is explicit that CFG must stay off for this model.
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(random_seed),
            callback_on_step_end=_on_step_end,
        ).images[0]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_data = buffer.getvalue()

    return base64.b64encode(image_data)
