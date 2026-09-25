import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Optional

import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
from diffusers.utils.export_utils import encode_video
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file as load_safetensors

MODEL_NAME = "MiniMaxAI/MiniMax-H3"

# Custom fine-tune trained for this server specifically, applied on top of
# the stock MiniMax-H3 fl2va transformer on every request. Too large for
# git (155MB, over GitHub's 100MB push limit), so it's hosted on a private
# HF repo and pulled down through the normal HF cache (HF_TOKEN needs read
# access to it, same token used for the gated base-model download - see
# entrypoint.sh's HF_HOME redirect for where that cache lives).
LORA_REPO_ID = "elliotcareydev/minimax-h3-fingering-lora"
LORA_FILENAME = "MinimaxH3-Fingering_000002000.safetensors"
LORA_ADAPTER_NAME = "fingering"

DEFAULT_NUM_FRAMES = 124  # ~5.2s at MiniMax-H3's fixed 24fps - see run_image_to_video's docstring.
MAX_REFERENCE_IMAGES = 9  # MiniMax-H3's own ref2va limit.

# Built once per worker process and reused across every request, instead of
# reloading weights on every single job - mirrors the _MODEL/_PIPES caching
# pattern in qwen_chat/processing/generate_text.py and
# qwen_image/processing/generate_image.py, just for a ModularPipeline
# instead of a plain model/DiffusionPipeline.
_PIPE: Optional[ModularPipeline] = None


def _get_pipe() -> ModularPipeline:
    global _PIPE
    if _PIPE is None:
        # No workflow= restriction: this server supports both fl2va
        # (run_image_to_video, first-frame -> video) and ref2va
        # (run_reference_to_video, labeled-image-references -> video).
        # Loading one pipeline with both transformer partitions shares the
        # ~62GB Qwen3-VL conditioner + VAEs between them, instead of
        # duplicating that shared weight across two separate pipeline
        # instances - see the diffusers MiniMax-H3 docs' "A generation as a
        # reference" section, which shows exactly this single-pipe,
        # both-workflows pattern. Costs meaningfully more memory than an
        # fl2va-only load though: ~185GB bf16 total (61.7GB transformer +
        # 61.7GB transformer_ref + 62.1GB conditioner) - see README's
        # Hardware section.
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(MODEL_NAME, components_manager=manager)
        pipe.load_components(dtype=torch.bfloat16)
        # Weights live in host RAM; the manager streams onto the GPU only
        # what each denoising step needs. Required even on an 80GB card,
        # since the transformers + conditioner don't all fit resident at
        # once - see the "Memory" section of the diffusers MiniMax-H3 docs
        # for the quantized recipe smaller cards would need instead (not
        # implemented here).
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")

        # ModularPipeline has no pipe-level load_lora_weights() (unlike the
        # DiffusionPipeline used in qwen_image/cyberrealistic_pony, see
        # qwen_image/processing/generate_image.py) - the LoRA is loaded
        # directly onto the transformer component, which is a
        # PeftAdapterMixin. hf_hub_download uses the standard HF cache, so
        # this only re-downloads after a cache eviction, not on every
        # worker restart.
        #
        # Applied only to `pipe.transformer` (the fl2va partition this LoRA
        # was actually trained/validated against) - ref2va's separate
        # `pipe.transformer_ref` component is left un-adapted. Revisit if a
        # ref2va-specific LoRA is ever trained; applying this same state
        # dict to transformer_ref is untested and not assumed to be safe.
        lora_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)
        lora_state_dict = load_safetensors(lora_path)
        pipe.transformer.load_lora_adapter(lora_state_dict, adapter_name=LORA_ADAPTER_NAME)
        pipe.transformer.set_adapters([LORA_ADAPTER_NAME])

        _PIPE = pipe
    return _PIPE


def _results_to_base64_video(results: dict) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        encode_video(
            results["videos"][0],
            fps=24,
            output_path=tmp.name,
            audio=results["audio"][0],
            audio_sample_rate=results["sampling_rate"],
        )
        video_bytes = Path(tmp.name).read_bytes()
    return base64.b64encode(video_bytes)


def run_image_to_video(
    image_bytes: str,
    prompt: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """fl2va: the given image becomes the video's literal starting frame -
    the prompt describes motion/continuation from it, and can't reinterpret
    its content onto a different subject/scene (that's run_reference_to_video
    below, MiniMax-H3's other, unrelated image-input mode).

    Args:
        image_bytes (str): base64-encoded first-frame image.
        prompt (str): text prompt guiding the generated motion.
        num_frames: frame count at MiniMax-H3's fixed 24fps. Snapped up to
            the next `17 * n + 5` the video VAE can decode; the resulting
            clip duration must land between 5 and 15 seconds.
        seed: optional, for reproducible generations.
        progress_callback: optional, kept for interface parity with the
            other servers' per-step reporting. MiniMax-H3's checkpoints are
            guidance-distilled (no guider, no negative_prompt, no
            guidance_scale - baked into the weights) and the modular blocks
            expose no per-step hook, so this only fires once on completion.

    Returns:
        bytes: base64-encoded MP4, with its synchronized audio track muxed
            in.
    """
    pipe = _get_pipe()

    decoded_image = base64.b64decode(image_bytes)
    input_image = Image.open(BytesIO(decoded_image)).convert("RGB")

    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()

    results = pipe(
        prompt=prompt,
        image=input_image,
        num_frames=num_frames,
        generator=generator,
        output=["videos", "audio", "sampling_rate"],
    )

    if progress_callback:
        progress_callback(1, 1)

    return _results_to_base64_video(results)


def run_reference_to_video(
    reference_images: List[str],
    prompt: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """ref2va: unlike run_image_to_video's fl2va, these images are labeled
    references the prompt can explicitly address, not a literal starting
    frame - MiniMax-H3 labels them "<Picture 1>", "<Picture 2>", etc., in
    that order, and the caller writes those tags into `prompt` wherever it
    wants the model to pull from a given reference. E.g. with one reference
    image of a face:

        prompt = "<Picture 1> as a German WWII soldier, standing at "
                 "attention in a snowy trench, cinematic lighting"

    References don't bind the generated geometry - the canvas defaults to
    MiniMax-H3's own 16:9, unlike run_image_to_video where the canvas
    follows the input image's aspect ratio.

    Args:
        reference_images: base64-encoded images, in the order the prompt
            addresses them via "<Picture N>" tags (1-indexed). At most
            MAX_REFERENCE_IMAGES (9, MiniMax-H3's own limit).
        prompt: text prompt, referencing images by "<Picture N>" tags where
            wanted - the model doesn't infer which reference is meant
            without one.
        num_frames, seed, progress_callback: see run_image_to_video.

    Returns:
        bytes: base64-encoded MP4, with its synchronized audio track muxed
            in.
    """
    if not reference_images:
        raise ValueError("reference_images must contain at least one image")
    if len(reference_images) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"MiniMax-H3 accepts at most {MAX_REFERENCE_IMAGES} image references, "
            f"got {len(reference_images)}"
        )

    pipe = _get_pipe()

    references = [
        MiniMaxH3ImageReference(
            image=Image.open(BytesIO(base64.b64decode(image_bytes))).convert("RGB")
        )
        for image_bytes in reference_images
    ]

    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()

    results = pipe(
        prompt=prompt,
        references=references,
        num_frames=num_frames,
        generator=generator,
        output=["videos", "audio", "sampling_rate"],
    )

    if progress_callback:
        progress_callback(1, 1)

    return _results_to_base64_video(results)
