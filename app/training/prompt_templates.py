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
