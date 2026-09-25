import base64
import re
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
# diffusers' own default for every Modular Diffusers pipeline's
# num_inference_steps (modular_pipeline_utils.py's generic InputParam
# template) - MiniMax-H3's before_denoise.py doesn't override it. Both
# transformer partitions are guidance-distilled (no guider/CFG), so this is
# purely a quality/speed knob, not a correctness one.
DEFAULT_NUM_INFERENCE_STEPS = 50

# Built once per worker process and reused across every request, instead of
# reloading weights on every single job - mirrors the _MODEL/_PIPES caching
# pattern in qwen_chat/processing/generate_text.py and
# qwen_image/processing/generate_image.py, just for a ModularPipeline
# instead of a plain model/DiffusionPipeline.
_PIPE: Optional[ModularPipeline] = None

_QKV_KEY_RE = re.compile(r"^(?P<block>.+)\.attn\.qkv_proj\.(?P<part>lora_[AB])\.weight$")
_OUT_PROJ_KEY_RE = re.compile(r"^(?P<block>.+)\.attn\.out_proj\.(?P<part>lora_[AB])\.weight$")
_FC1_KEY_RE = re.compile(r"^(?P<block>.+)\.mlp\.fc1\.(?P<part>lora_[AB])\.weight$")
_FC2_KEY_RE = re.compile(r"^(?P<block>.+)\.mlp\.fc2\.(?P<part>lora_[AB])\.weight$")


def _convert_ai_toolkit_lora_state_dict(state_dict: dict) -> dict:
    """Converts an ai-toolkit-trained MiniMax-H3 LoRA (the "native",
    ComfyUI-style fused architecture ai-toolkit and several community
    checkpoints target) into the module names and tensor layout diffusers'
    MiniMaxH3Transformer3DModel (separate to_q/to_k/to_v, diffusers'
    SwiGLU feedforward) actually expects.

    Confirmed necessary from a real worker log: a plain rename of the
    "diffusion_model." key prefix to "transformer." got load_lora_adapter
    to actually attempt injection, but PEFT then failed with "Target
    modules {'qkv_proj', 'fc1', 'fc2', 'out_proj'} not found in the base
    model" - diffusers' attention has no fused qkv_proj (it's separate
    to_q/to_k/to_v Linears) and its FeedForward's SwiGLU up-projection
    orders its two halves differently than ai-toolkit's fc1 does. The
    recipe below matches the one published for
    huggingface.co/InstantX/MiniMax-H3-Turbo-Lora-Diffusers, which needed
    the identical conversion for the same model family:

    - attn.qkv_proj -> attn.to_q / attn.to_k / attn.to_v: lora_A is
      shared across all three (it's the fused projection's single input
      side, unchanged by the split); lora_B's output rows are split into
      three equal [q; k; v] chunks. This is exact, not approximate - the
      fused Linear's output *is* those three chunks concatenated, so the
      same input projection times each output slice reconstructs each
      per-projection delta exactly. Verified against this LoRA's actual
      shapes: qkv_proj.lora_B is [21504, 16] = 3 x [7168, 16].
    - attn.out_proj -> attn.to_out.0: straight rename, no split.
    - mlp.fc1 -> ff.net.0.proj: lora_B's two halves (each producing half
      of the SwiGLU inner dimension) are swapped from ai-toolkit's
      [gate; value] row order to diffusers' [value; gate] order. Verified
      against this LoRA's shapes: fc1.lora_B is [28672, 16] = 2 x
      [14336, 16], and fc2.lora_A's input dim (14336) matches that half
      size, confirming the SwiGLU gate/value split point.
    - mlp.fc2 -> ff.net.2: straight rename, no split.
    - blocks.N -> transformer_blocks.N, and token_refiner.blocks.N ->
      token_refiner.refiner_blocks.N: the checkpoint's block-container
      names, unchanged by the diffusers port itself - the same renames
      diffusers' own scripts/convert_minimax_h3_to_diffusers.py applies
      when converting the original (ai-toolkit-compatible) checkpoint
      layout to diffusers' MiniMaxH3Transformer3DModel, which defines
      the stacks as `self.transformer_blocks` and
      `self.token_refiner.refiner_blocks`. Confirmed necessary from a
      real worker log: without this, the four renames above still leave
      keys like "blocks.43.attn.to_q...", which PEFT can't find on the
      model and rejects with "Target modules {...} not found in the
      base model" - the bare `to_q`/`to_out.0`/etc. names alone aren't
      enough, they need to resolve under the right container path too.

    Any key not matching one of these four attention/FFN patterns passes
    through unchanged, aside from the blocks-path rename above (none
    exist in the one LoRA this was built against - its 416 tensors are
    100% covered by exactly these four patterns across 50 `blocks` + 2
    `token_refiner.blocks` - but a future differently-trained LoRA might
    have others, e.g. AdaLN, and should fail through load_lora_adapter's
    own prefix filtering rather than being silently dropped here).

    This has NOT been validated against this LoRA's actual generation
    output (no local GPU available to test against) - only against the
    "target modules not found" failures this fixes and the InstantX
    precedent for the identical to_q/to_k/to_v/to_out/ff conversion on
    the same model family. Confirm the LoRA is visibly having an effect
    on a real generation before trusting this blindly.
    """

    def _rename_block_path(block: str) -> str:
        if ".token_refiner.blocks." in block:
            return block.replace(".token_refiner.blocks.", ".token_refiner.refiner_blocks.", 1)
        return block.replace(".blocks.", ".transformer_blocks.", 1)

    converted: dict = {}
    for key, tensor in state_dict.items():
        match = _QKV_KEY_RE.match(key)
        if match:
            block, part = _rename_block_path(match["block"]), match["part"]
            if part == "lora_A":
                for target in ("to_q", "to_k", "to_v"):
                    converted[f"{block}.attn.{target}.lora_A.weight"] = tensor
            else:
                q, k, v = tensor.chunk(3, dim=0)
                converted[f"{block}.attn.to_q.lora_B.weight"] = q
                converted[f"{block}.attn.to_k.lora_B.weight"] = k
                converted[f"{block}.attn.to_v.lora_B.weight"] = v
            continue

        match = _OUT_PROJ_KEY_RE.match(key)
        if match:
            block, part = _rename_block_path(match["block"]), match["part"]
            converted[f"{block}.attn.to_out.0.{part}.weight"] = tensor
            continue

        match = _FC1_KEY_RE.match(key)
        if match:
            block, part = _rename_block_path(match["block"]), match["part"]
            if part == "lora_A":
                converted[f"{block}.ff.net.0.proj.lora_A.weight"] = tensor
            else:
                gate, value = tensor.chunk(2, dim=0)
                converted[f"{block}.ff.net.0.proj.lora_B.weight"] = torch.cat([value, gate], dim=0)
            continue

        match = _FC2_KEY_RE.match(key)
        if match:
            block, part = _rename_block_path(match["block"]), match["part"]
            converted[f"{block}.ff.net.2.{part}.weight"] = tensor
            continue

        converted[key] = tensor

    return converted


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
        # Without a workflow= restriction, diffusers treats a failed
        # component load as soft (logs "if this component is not required
        # for your workflow you can safely ignore this message" and leaves
        # the attribute None) rather than raising - it can't tell what's
        # actually required until a call comes in. That turned a disk-full
        # download failure (see README's Hardware/Troubleshooting section)
        # into a confusing `NoneType has no attribute 'load_lora_adapter'`
        # a few lines down instead of a clear error here, so check
        # explicitly for the two components this server actually needs.
        if pipe.transformer is None or pipe.transformer_ref is None:
            raise RuntimeError(
                "MiniMax-H3 pipeline loaded with transformer or transformer_ref "
                "missing - almost always means a component download failed "
                "partway (commonly disk space: this needs ~185GB and a RunPod "
                "Network Volume must be attached, or it downloads to the much "
                "smaller ephemeral container disk instead - see entrypoint.sh's "
                "HF_HOME redirect and README's Hardware section). Check the "
                "worker's earlier logs for 'Failed to create component' lines."
            )
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
        # This checkpoint's own __metadata__ (software: ai-toolkit,
        # https://github.com/ostris/ai-toolkit) confirms it was trained
        # with ai-toolkit, which exports denoiser LoRA keys prefixed
        # "diffusion_model." (e.g. "diffusion_model.blocks.0.attn...") -
        # not diffusers' own "transformer." convention that
        # load_lora_adapter's default prefix filtering expects. Without
        # this rename it silently matches zero keys (logged as "No LoRA
        # keys associated to MiniMaxH3Transformer3DModel found with the
        # prefix='transformer'" rather than raising), so the LoRA quietly
        # never applies at all - confirmed from a real worker log, not
        # speculation.
        lora_state_dict = {
            key.replace("diffusion_model.", "transformer.", 1): value
            for key, value in lora_state_dict.items()
        }
        # Renaming the prefix alone isn't enough - see
        # _convert_ai_toolkit_lora_state_dict's docstring for why (fused
        # qkv_proj vs. diffusers' separate to_q/to_k/to_v, and an
        # ai-toolkit-vs-diffusers SwiGLU half-ordering mismatch), also
        # confirmed from a real worker log.
        lora_state_dict = _convert_ai_toolkit_lora_state_dict(lora_state_dict)
        pipe.transformer.load_lora_adapter(lora_state_dict, adapter_name=LORA_ADAPTER_NAME)
        pipe.transformer.set_adapters([LORA_ADAPTER_NAME])

        _PIPE = pipe
    return _PIPE


# `ComponentsManager`'s auto-offload hooks intermittently hit a known
# PyTorch/NVML bug when moving a component onto the GPU mid-`pipe(...)` call
# - not anything wrong with this server's code, see
# https://github.com/pytorch/pytorch/issues/112950 and
# https://github.com/pytorch/pytorch/issues/123834. Confirmed from three
# separate real worker logs (three different RunPod workers/hosts, at three
# different times), always from the exact same `module.to(execution_device)`
# call inside `pipe(...)`, never from `_get_pipe()`'s own one-time load - so
# a bare retry of just the generation call (reusing the already-loaded
# `_PIPE`, no reload) is enough when the underlying host isn't persistently
# broken, which those GitHub issues suggest is the common case.
_GPU_ALLOCATOR_RETRY_ATTEMPTS = 2


def _is_nvml_allocator_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "NVML_SUCCESS" in str(exc)


def _call_pipe_with_retry(
    pipe: ModularPipeline,
    status_callback: Optional[Callable[[str], None]],
    **pipe_kwargs,
) -> dict:
    for attempt in range(1, _GPU_ALLOCATOR_RETRY_ATTEMPTS + 1):
        try:
            return pipe(**pipe_kwargs)
        except RuntimeError as exc:
            if attempt == _GPU_ALLOCATOR_RETRY_ATTEMPTS or not _is_nvml_allocator_error(exc):
                raise
            if status_callback:
                status_callback(
                    "GPU allocator error (NVML) during generation, retrying "
                    f"(attempt {attempt + 1}/{_GPU_ALLOCATOR_RETRY_ATTEMPTS})"
                )


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
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
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
        num_inference_steps: denoising step count (sigma grid points,
            terminal 0 included - see the diffusers MiniMax-H3 docs'
            "Generation constraints" section - so this drives one fewer
            model evaluation than the number given). Both transformer
            partitions are guidance-distilled (no guider/CFG), so this is
            purely a quality/speed knob: fewer steps trades quality for a
            faster generation, more steps the reverse.
        seed: optional, for reproducible generations.
        progress_callback: optional, kept for interface parity with the
            other servers' per-step reporting. MiniMax-H3's checkpoints are
            guidance-distilled (no guider, no negative_prompt, no
            guidance_scale - baked into the weights) and the modular blocks
            expose no per-step hook, so this only fires once on completion.
        status_callback: optional, fired with a human-readable message if
            generation hits the known NVML allocator error (see
            _call_pipe_with_retry) and is being retried - lets the caller
            surface that back through its own status channel (Celery task
            state, RunPod progress updates) instead of it only showing up
            in worker logs.

    Returns:
        bytes: base64-encoded MP4, with its synchronized audio track muxed
            in.
    """
    pipe = _get_pipe()

    decoded_image = base64.b64decode(image_bytes)
    input_image = Image.open(BytesIO(decoded_image)).convert("RGB")

    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()

    results = _call_pipe_with_retry(
        pipe,
        status_callback,
        prompt=prompt,
        image=input_image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
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
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
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
        num_frames, num_inference_steps, seed, progress_callback,
            status_callback: see run_image_to_video.

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

    results = _call_pipe_with_retry(
        pipe,
        status_callback,
        prompt=prompt,
        references=references,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output=["videos", "audio", "sampling_rate"],
    )

    if progress_callback:
        progress_callback(1, 1)

    return _results_to_base64_video(results)
