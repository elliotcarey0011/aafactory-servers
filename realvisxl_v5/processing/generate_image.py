import random
from io import BytesIO
from typing import Callable, Optional
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
import base64


MODEL_NAME = "SG161222/RealVisXL_V5.0"

# Loaded once per worker process and reused across every request, instead of
# rebuilding the pipeline on every single task.
_PIPES: dict[str, StableDiffusionXLPipeline] = {}


def _get_text_to_image_pipe() -> StableDiffusionXLPipeline:
    if "text_to_image" not in _PIPES:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        # RealVisXL's own model card recommends "DPM++ SDE Karras" over SDXL's
        # default Euler scheduler — swap it in rather than leaving the
        # pipeline's default, which would undersell what the checkpoint was
        # tuned for.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
        # It's a full (non-distilled) SDXL checkpoint — fits comfortably in
        # 16GB VRAM at fp16, so no CPU offload needed (unlike qwen_image).
        pipe.to("cuda")
        _PIPES["text_to_image"] = pipe
    return _PIPES["text_to_image"]


# The model card calls for a minimum of 30 steps with DPM++ SDE Karras, so
# tiers start there rather than reusing qwen_image/z_image's lower floors.
IMAGE_QUALITY_TO_STEPS = {
    "low": 30,
    "medium": 35,
    "high": 40,
    "ultra": 50,
}

# Standard SDXL trained aspect-ratio buckets (multiples of 64, ~1024^2 total
# pixels) — RealVisXL V5 is a full SDXL checkpoint, so it shares these native
# resolutions rather than Qwen-Image/Z-Image's larger buckets.
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
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
            guidance_scale=7.0,
            generator=torch.Generator(device="cuda").manual_seed(random_seed),
            callback_on_step_end=_on_step_end,
        ).images[0]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_data = buffer.getvalue()

    return base64.b64encode(image_data)
