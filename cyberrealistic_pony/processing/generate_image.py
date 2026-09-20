import random
from io import BytesIO
from typing import Callable, Optional
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch
import base64


# CyberRealisticPony isn't published in diffusers format (no model_index.json
# / unet / vae / text_encoder folders) — it's a stack of single-file Civitai
# checkpoints. Pin one specific version rather than tracking "latest" so a
# new upload to the repo can't silently change the model underneath us.
#
# Must be a "blob/main/" URL, not "resolve/main/" — diffusers' from_single_file
# does its own resolve-URL construction from a blob URL, so passing a
# resolve URL here gets "resolve/main/" appended twice and 404s.
#
# Filename is "_F16" (no "P"), not "_FP16" like every earlier version on
# this repo — a naming inconsistency in the V18.0 upload itself, confirmed
# against the repo's actual file listing.
CHECKPOINT_URL = (
    "https://huggingface.co/cyberdelia/CyberRealisticPony/blob/main/"
    "CyberRealisticPony_V18.0_F16.safetensors"
)

# Loaded once per worker process and reused across every request, instead of
# rebuilding the pipeline on every single task.
_PIPES: dict[str, StableDiffusionXLPipeline] = {}


def _get_text_to_image_pipe() -> StableDiffusionXLPipeline:
    if "text_to_image" not in _PIPES:
        # from_single_file downloads (and, via HF_HOME, caches) the
        # checkpoint the same way from_pretrained does for a diffusers-format
        # repo — see entrypoint.sh for the network-volume cache redirect.
        pipe = StableDiffusionXLPipeline.from_single_file(
            CHECKPOINT_URL,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        # The model card recommends "DPM++ SDE Karras" over SDXL's default
        # Euler scheduler — same swap as realvisxl_v5, whose model card
        # recommends the same sampler.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
        # It's a Pony/SDXL checkpoint with a baked-in VAE — fits comfortably
        # in 16GB VRAM at fp16, so no CPU offload needed.
        pipe.to("cuda")
        _PIPES["text_to_image"] = pipe
    return _PIPES["text_to_image"]


# The model card calls for 30+ steps with DPM++ SDE Karras — same floor as
# realvisxl_v5's tiers.
IMAGE_QUALITY_TO_STEPS = {
    "low": 30,
    "medium": 35,
    "high": 40,
    "ultra": 50,
}

# Standard SDXL trained aspect-ratio buckets (multiples of 64, ~1024^2 total
# pixels). The model card's own recommended resolutions (896x1152, 832x1216)
# are two of these buckets, so the full SDXL set is used rather than a
# narrower one.
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
        positive_prompt (str): The positive text prompt. The model card
            recommends leading with Pony's score tags, e.g.
            "score_9, score_8_up, score_7_up, <subject>" — passed through
            as-is rather than injected here, so callers stay in control of
            the exact prompt sent to the model.
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
            # The model card recommends CFG scale 5 — lower than
            # realvisxl_v5's 7.0.
            guidance_scale=5.0,
            generator=torch.Generator(device="cuda").manual_seed(random_seed),
            callback_on_step_end=_on_step_end,
        ).images[0]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_data = buffer.getvalue()

    return base64.b64encode(image_data)
