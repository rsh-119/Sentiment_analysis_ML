"""Inference wrapper for the fine-tuned Llama model: loads the 4-bit base
model + LoRA adapter (plain transformers/peft, no Unsloth), and predicts via
label-logit comparison rather than open-ended generation (see
train/train_llama.py for the rationale). Produces predictions in the same
(pred_labels, pred_probs) shape the shared evaluator expects, directly
comparable to the BiLSTM's sigmoid output.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from imdb_sentiment.config import LlamaConfig
from imdb_sentiment.data.llama_prompt import build_messages, get_label_token_ids


class LlamaPredictor:
    def __init__(self, adapter_path: str, llama_cfg: LlamaConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # so the last non-pad token is always at position -1

        quant_config = BitsAndBytesConfig(
            load_in_4bit=llama_cfg.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            llama_cfg.base_model,
            quantization_config=quant_config if llama_cfg.load_in_4bit else None,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

        self.pos_id, self.neg_id = get_label_token_ids(
            self.tokenizer, llama_cfg.positive_token, llama_cfg.negative_token
        )
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def predict(self, texts: list[str]) -> tuple[list[int], list[float]]:
        prompts = [
            self.tokenizer.apply_chat_template(
                build_messages(text), tokenize=False, add_generation_prompt=True
            )
            for text in texts
        ]
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

        # logits_to_keep=1: only materialize logits for the last position, not the
        # full sequence — the full-sequence logits tensor (batch x seq_len x vocab)
        # is what pushed this into OOM on the shared GPU at larger batch sizes.
        # use_cache=False: this is a single forward pass, not incremental
        # generation, so building a KV cache only wastes memory.
        outputs = self.model(**encoded, logits_to_keep=1, use_cache=False)
        last_token_logits = outputs.logits[:, -1, :]  # left-padded, so -1 is always the real last token

        two_class_logits = last_token_logits[:, [self.neg_id, self.pos_id]]
        probs = F.softmax(two_class_logits, dim=-1)[:, 1]  # P(positive)

        pred_labels = (probs >= 0.5).long().cpu().tolist()
        pred_probs = probs.float().cpu().tolist()
        return pred_labels, pred_probs
