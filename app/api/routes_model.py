from fastapi import APIRouter
from transformers import AutoTokenizer

from app.core.secrets import apply_secrets_to_environment, get_hf_token

router = APIRouter(prefix="/model", tags=["model"])

_CHAT_NAME_HINTS = ("instruct", "-chat", "chat-", "-it", "_it", "-rl", "sft")


def _name_looks_chat(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(hint in name for hint in _CHAT_NAME_HINTS)


@router.get("/detect")
def detect(model_name: str):
    """Detect whether a model is a chat/instruct model or a base model.

    Ground truth is whether its tokenizer has a chat template (loading only the
    tokenizer is cheap). If that can't be loaded (offline / not cached), fall
    back to a name-based heuristic. Either way the backend still enforces
    correctness at train/eval/infer time via the chat-template guard."""
    apply_secrets_to_environment()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=get_hf_token())
        has_chat = getattr(tokenizer, "chat_template", None) is not None
        return {
            "model": model_name,
            "has_chat_template": has_chat,
            "template_type": "chat" if has_chat else "instruction",
            "source": "tokenizer",
        }
    except Exception as exc:
        looks_chat = _name_looks_chat(model_name)
        return {
            "model": model_name,
            "has_chat_template": None,
            "template_type": "chat" if looks_chat else "instruction",
            "source": "name_heuristic",
            "detail": f"{type(exc).__name__}: {exc}",
        }
