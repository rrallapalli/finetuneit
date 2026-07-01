import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from app.training.prompt_templates import build_inference_prompt
from app.core.secrets import apply_secrets_to_environment, get_hf_token

# Module-level cache so repeated calls (FastAPI local mode) don't reload the
# base model + adapter every request. Keyed by (base_model, adapter_repo, 4bit).
_MODEL_CACHE: dict = {}


def _load(base_model: str, adapter_repo: str, load_in_4bit: bool):
    cache_key = (base_model, adapter_repo, load_in_4bit)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    token = get_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)

    kwargs = dict(device_map="auto", token=token)
    if load_in_4bit:
        # Serve in the same precision as training (QLoRA) to preserve accuracy
        # and avoid loading the full-precision base model into VRAM.
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    if adapter_repo:
        model = PeftModel.from_pretrained(model, adapter_repo, token=token)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, tokenizer)
    return model, tokenizer


def generate_response(
    base_model: str,
    adapter_repo: str,
    prompt: str,
    input_text: str = "",
    prompt_template: str = "alpaca",
    max_new_tokens: int = 128,
    load_in_4bit: bool = True,
) -> str:
    apply_secrets_to_environment()

    try:
        model, tokenizer = _load(base_model, adapter_repo, load_in_4bit)

        formatted_prompt = build_inference_prompt(prompt, input_text, prompt_template)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("### Response:")[-1].strip()
    except Exception as exc:
        return f"[inference error] {type(exc).__name__}: {exc}"
