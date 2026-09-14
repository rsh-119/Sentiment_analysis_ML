"""Chat-template prompt construction and label-token utilities for the Llama
fine-tuning pipeline. The model is fine-tuned via its own instruction chat
template to answer with a single label word; at inference we compare
next-token logits for the two label tokens rather than parsing generated text
(see train/train_llama.py and eval/llama_predictor.py for the rationale).
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a sentiment classification assistant. Read the movie review and "
    "respond with exactly one word: Positive or Negative."
)


def build_user_message(text: str) -> str:
    return f"Classify the sentiment of this movie review as Positive or Negative.\n\nReview:\n{text}"


def build_messages(text: str, label_token: str | None = None) -> list[dict[str, str]]:
    """Build the chat message list. Pass `label_token` (e.g. "Positive") to
    include the assistant's answer for SFT training examples; omit it to build
    an inference-time prompt to be completed by the model.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(text)},
    ]
    if label_token is not None:
        messages.append({"role": "assistant", "content": label_token})
    return messages


def label_to_token(label: int, positive_token: str, negative_token: str) -> str:
    return positive_token if label == 1 else negative_token


def check_single_token_labels(tokenizer, positive_token: str, negative_token: str) -> bool:
    """Verify the label strings each encode to exactly one token — required
    for the label-logit-comparison inference trick to be unambiguous.
    """
    pos_ids = tokenizer.encode(positive_token, add_special_tokens=False)
    neg_ids = tokenizer.encode(negative_token, add_special_tokens=False)
    return len(pos_ids) == 1 and len(neg_ids) == 1


def get_label_token_ids(tokenizer, positive_token: str, negative_token: str) -> tuple[int, int]:
    """Return (positive_token_id, negative_token_id). Caller should have
    already validated with `check_single_token_labels`.
    """
    pos_id = tokenizer.encode(positive_token, add_special_tokens=False)[0]
    neg_id = tokenizer.encode(negative_token, add_special_tokens=False)[0]
    return pos_id, neg_id
