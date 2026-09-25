import base64
import os
from typing import Callable, List, Optional

from huggingface_hub import InferenceClient

MODEL_NAME = "MiniMaxAI/MiniMax-H3"

# MiniMax-H3 is a multi-billion-parameter omni-modal video model with no
# realistic path to running locally in this repo's GPU containers (unlike
# qwen_image/cyberrealistic_pony's local diffusers pipelines), so this
# server proxies to the hosted fal-ai endpoint via HuggingFace's Inference
# Providers routing instead of loading any weights itself.
#
# Built once per worker process and reused across every request, instead of
# reconstructing it per job (mirrors the _MODEL/_PIPES caching pattern in
# qwen_chat/processing/generate_text.py and qwen_image/processing/generate_image.py).
_CLIENT: Optional[InferenceClient] = None


def _get_client() -> InferenceClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = InferenceClient(
            provider="fal-ai",
            api_key=os.environ["HF_TOKEN"],
        )
    return _CLIENT


def run_image_to_video(
    image_bytes: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    num_frames: Optional[float] = None,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """
    Args:
        image_bytes (str): base64-encoded source/first-frame image.
        prompt (str): text prompt guiding the generated motion.
        negative_prompt: optional, what to steer the generation away from.
        num_frames: optional frame-count hint — provider-specific, fal-ai
            derives actual clip duration from this and the model's own
            limits (MiniMax-H3 supports 4-15s clips).
        seed: optional, for reproducible generations.
        progress_callback: optional, kept for interface parity with the
            other servers' per-step reporting — the fal-ai image_to_video
            call is a single blocking HTTP request with no per-step hook,
            so this only fires once on completion.

    Returns:
        bytes: base64-encoded MP4 video.
    """
    client = _get_client()
    decoded_image = base64.b64decode(image_bytes)

    negative_prompts: Optional[List[str]] = [negative_prompt] if negative_prompt else None

    video = client.image_to_video(
        decoded_image,
        model=MODEL_NAME,
        prompt=prompt,
        negative_prompt=negative_prompts,
        num_frames=num_frames,
        seed=seed,
    )

    if progress_callback:
        progress_callback(1, 1)

    return base64.b64encode(video)
