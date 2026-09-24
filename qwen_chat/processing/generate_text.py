from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Abliterated (refusal-removed) fine-tune of Qwen2.5-3B-Instruct - the base
# instruct model can refuse NSFW prompt-writing requests, which this server
# exists specifically to handle. Inherits Qwen2.5-3B-Instruct's Apache-2.0
# obligations (see the model card), so the licensing rationale for choosing
# Qwen over Llama still holds.
MODEL_NAME = "knoveleng/Qwen2.5-3B-Instruct-Uncensored"

# aafactory_nsfw expects image prompts written in the SDXL/Pony tag
# convention documented in cyberrealistic_pony/README.md and
# qwen_image/README.md - leading with quality tags and comma-separated
# descriptive tags rather than prose.
PROMPT_SYSTEM_PROMPT = (
    "You are a prompt-writing assistant for a Stable Diffusion XL / Pony "
    "based image generator. Given a subject and optional style hints, "
    "write ONE comma-separated, tag-style positive prompt - no prose, no "
    "explanation, just the tags. Lead with quality tags in this exact "
    "order: score_9, score_8_up, score_7_up, then the subject and style "
    "tags. Keep it under 100 tags."
)

# Loaded once per worker process and reused across every request, instead
# of reloading the model on every single job (mirrors the _PIPES pattern in
# qwen_image/processing/generate_image.py).
_MODEL: Optional[AutoModelForCausalLM] = None
_TOKENIZER: Optional[AutoTokenizer] = None


def _get_model_and_tokenizer():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
    return _MODEL, _TOKENIZER


def _generate(messages: list[dict], max_tokens: int, temperature: float) -> str:
    model, tokenizer = _get_model_and_tokenizer()

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    # Slice off the input tokens so we only decode the newly generated reply.
    new_tokens = generated_ids[0][model_inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_chat(
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """messages: OpenAI-style [{"role": "user"/"assistant"/"system", "content": "..."}].

    progress_callback: optional, kept for interface parity with the image
    servers' per-step reporting - plain `.generate()` has no equivalent
    per-step hook, so this is only invoked once on completion. A
    TextIteratorStreamer could be added later for token-level progress if
    that turns out to matter.
    """
    result = _generate(messages, max_tokens, temperature)
    if progress_callback:
        progress_callback(1, 1)
    return result


def run_generate_prompt(
    subject: str,
    style_hints: str = "",
    max_tokens: int = 256,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    user_content = subject if not style_hints else f"{subject}\nStyle hints: {style_hints}"
    messages = [
        {"role": "system", "content": PROMPT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    result = _generate(messages, max_tokens, temperature=0.8)
    if progress_callback:
        progress_callback(1, 1)
    return result
