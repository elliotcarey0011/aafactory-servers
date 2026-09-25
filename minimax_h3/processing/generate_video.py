import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import torch
from diffusers import ComponentsManager, ModularPipeline
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

# Built once per worker process and reused across every request, instead of
# reloading ~124GB of weights (61.7GB transformer + 62.1GB Qwen3-VL
# conditioner) on every single job - mirrors the _MODEL/_PIPES caching
# pattern in qwen_chat/processing/generate_text.py and
# qwen_image/processing/generate_image.py, just for a ModularPipeline
# instead of a plain model/DiffusionPipeline.
_PIPE: Optional[ModularPipeline] = None


def _get_pipe() -> ModularPipeline:
    global _PIPE
    if _PIPE is None:
        # fl2va only touches the `transformer/` checkpoint partition, not
        # the ~62GB `transformer_ref/` partition the ref2va workflow uses -
        # this server only does image-to-video, so restricting the
        # workflow at load time keeps the cache to what it actually needs.
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            MODEL_NAME, workflow="fl2va", components_manager=manager
        )
        pipe.load_components(dtype=torch.bfloat16)
        # Weights live in host RAM; the manager streams onto the GPU only
        # what each denoising step needs. Required even on an 80GB card,
        # since the transformer + conditioner (~124GB bf16) don't both fit
        # resident at once - see the "Memory" section of the diffusers
        # MiniMax-H3 docs for the quantized recipe smaller cards would need
        # instead (not implemented here).
        manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")

        # ModularPipeline has no pipe-level load_lora_weights() (unlike the
        # DiffusionPipeline used in qwen_image/cyberrealistic_pony, see
        # qwen_image/processing/generate_image.py) - the LoRA is loaded
        # directly onto the transformer component, which is a
        # PeftAdapterMixin. hf_hub_download uses the standard HF cache, so
        # this only re-downloads after a cache eviction, not on every
        # worker restart.
        lora_path = hf_hub_download(repo_id=LORA_REPO_ID, filename=LORA_FILENAME)
        lora_state_dict = load_safetensors(lora_path)
        pipe.transformer.load_lora_adapter(lora_state_dict, adapter_name=LORA_ADAPTER_NAME)
        pipe.transformer.set_adapters([LORA_ADAPTER_NAME])

        _PIPE = pipe
    return _PIPE


def run_image_to_video(
    image_bytes: str,
    prompt: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """
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
