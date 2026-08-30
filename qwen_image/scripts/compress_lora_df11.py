"""One-off offline step: fuse the NSFW LoRA into the base Qwen-Image
transformer, then DFloat11-compress the result.

Why this exists: PEFT-based LoRA injection (pipe.load_lora_weights) is
incompatible with DFloat11's CPU-offloaded compressed Linear layers — PEFT
needs a real `.weight` tensor on each layer to build the adapter, and
DFloat11's offloaded layers don't expose one (weights are decompressed
on-the-fly during the forward pass instead). Fusing the LoRA into the base
weights *before* compression sidesteps this entirely: DFloat11 never needs
to know a LoRA was involved, and the runtime code (generate_image.py) goes
back to loading a single DFloat11 checkpoint with no PEFT/LoRA calls at all.

Run this once on a machine with enough VRAM/RAM to hold the full ~40GB
bf16 transformer (NOT the serverless worker — spin up a temporary Pod with
a large GPU, e.g. an 80GB A100). Re-run it any time the LoRA or its weight
changes.

Usage:
    uv run python scripts/compress_lora_df11.py \
        --save_path ./Qwen-Image-NSFW-DF11 \
        --save_single_file \
        --check_correctness

Then upload save_path's contents to your own HF repo (e.g.
<your-hf-username>/Qwen-Image-NSFW-DF11) and point generate_image.py's
_get_text_to_image_pipe() at that repo instead of "DFloat11/Qwen-Image-DF11",
removing the now-unnecessary pipe.load_lora_weights(...) call.
"""

from argparse import ArgumentParser

import torch
from diffusers import DiffusionPipeline, QwenImageTransformer2DModel
from dfloat11 import compress_model

MODEL_NAME = "Qwen/Qwen-Image"
LORA_REPO = "starsfriday/Qwen-Image-NSFW"
LORA_WEIGHT_NAME = "qwen_image_nsfw.safetensors"

# Every Linear submodule inside one QwenImageTransformerBlock, derived from
# diffusers' transformer_qwenimage.py (QwenImageTransformerBlock.__init__ +
# its _tp_plan). Unlike FLUX.1, Qwen-Image has only one block type — no
# separate "single_transformer_blocks".
PATTERN_DICT = {
    r"transformer_blocks\.\d+": (
        "img_mod.1",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_out.0",
        "attn.to_add_out",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mod.1",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    ),
}


def parse_args():
    parser = ArgumentParser("Fuse the NSFW LoRA into Qwen-Image, then DFloat11-compress it")
    parser.add_argument("--save_path", type=str, default="./Qwen-Image-NSFW-DF11")
    parser.add_argument("--save_single_file", action="store_true")
    parser.add_argument("--check_correctness", action="store_true")
    parser.add_argument("--block_range", type=int, nargs=2, default=(0, 100))
    return parser.parse_args()


def main():
    args = parse_args()

    # Full bf16 transformer this time — no no_init_weights()/DFloat11
    # shortcut, since we need real weight values to fuse the LoRA into and
    # then compress from scratch.
    transformer = QwenImageTransformer2DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=torch.bfloat16,
    )

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT_NAME, adapter_name="lora")
    pipe.fuse_lora()
    pipe.unload_lora_weights()  # back to plain nn.Linear layers, weights now include the LoRA delta

    compress_model(
        model=pipe.transformer,
        pattern_dict=PATTERN_DICT,
        save_path=args.save_path,
        save_single_file=args.save_single_file,
        check_correctness=args.check_correctness,
        block_range=tuple(args.block_range),
    )


if __name__ == "__main__":
    main()
