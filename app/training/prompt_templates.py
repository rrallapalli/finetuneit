def build_prompt(example: dict, template_name: str, eos_token: str = "") -> str:
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if template_name in ["alpaca", "reasoning_alpaca"]:
        text = f"""
### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
        return text + eos_token

    if template_name == "instruction_only":
        text = f"""
### Instruction:
{instruction}

### Response:
{output}"""
        return text + eos_token

    raise ValueError(f"Unsupported prompt template: {template_name}")


def build_inference_prompt(prompt: str, input_text: str = "", template_name: str = "alpaca") -> str:
    if template_name in ["alpaca", "reasoning_alpaca"]:
        return f"""
### Instruction:
{prompt}

### Input:
{input_text}

### Response:"""

    if template_name == "instruction_only":
        return f"""
### Instruction:
{prompt}

### Response:"""

    raise ValueError(f"Unsupported prompt template: {template_name}")


# ---------------------------------------------------------------------------
# Chat / conversational support (added for configurable template types).
# These use the tokenizer's own chat template (apply_chat_template) so instruct
# models get correct ChatML/Qwen/Llama formatting without hardcoded markers.
# ---------------------------------------------------------------------------

_ROLE_MAP = {
    "human": "user", "user": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant",
    "system": "system",
}


def normalize_conversation(conversation) -> list:
    """Accept a list of turns in either OpenAI ({role, content}) or ShareGPT
    ({from, value}) form and return [{role, content}, ...]."""
    messages = []
    for turn in conversation or []:
        if isinstance(turn, dict) and "role" in turn and "content" in turn:
            messages.append({"role": turn["role"], "content": turn["content"]})
        elif isinstance(turn, dict) and "from" in turn and "value" in turn:
            messages.append({
                "role": _ROLE_MAP.get(str(turn["from"]).lower(), "user"),
                "content": turn["value"],
            })
    return messages


def instruction_to_messages(instruction: str, input_text: str = "", output=None) -> list:
    """Wrap an instruction/input/output row into a 2-turn chat so instruction
    data can train an instruct model through its chat template."""
    user = instruction if not input_text else f"{instruction}\n\n{input_text}"
    messages = [{"role": "user", "content": user}]
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    return messages


def build_chat_training_text(messages: list, tokenizer) -> str:
    """Full conversation (including the assistant turn) as a training string."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def build_chat_generation_prompt(messages: list, tokenizer) -> str:
    """Conversation up to (and including) the generation prompt for inference."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
