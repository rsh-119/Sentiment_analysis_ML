"""Train a Llama model on the IMDB sentiment dataset using PEFT/LoRA."""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from imdb_sentiment.config import AppConfig
from imdb_sentiment.data.llama_prompt import build_messages, check_single_token_labels, label_to_token
from imdb_sentiment.data.splits import load_splits
from imdb_sentiment.seeding import set_seed
from imdb_sentiment.utils.logging import get_logger

logger = get_logger(__name__)


def train_llama(cfg: AppConfig) -> dict:
    set_seed(cfg.seed.seed, cfg.seed.deterministic)

    tokenizer = AutoTokenizer.from_pretrained(cfg.llama.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not check_single_token_labels(tokenizer, cfg.llama.positive_token, cfg.llama.negative_token):
        raise ValueError(
            f"Label tokens '{cfg.llama.positive_token}'/'{cfg.llama.negative_token}' do not "
            "each encode to exactly one token with this tokenizer — pick a different pair "
            "(e.g. 'positive'/'negative' or '1'/'0') and update LlamaConfig."
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg.llama.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llama.base_model,
        quantization_config=quant_config if cfg.llama.load_in_4bit else None,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=cfg.llama.lora_r,
        lora_alpha=cfg.llama.lora_alpha,
        lora_dropout=cfg.llama.lora_dropout,
        target_modules=list(cfg.llama.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dsd = load_splits(cfg)
    train_ds, val_ds = dsd["train"], dsd["val"]
    if cfg.llama.train_subset_size is not None:
        train_ds = train_ds.select(range(min(cfg.llama.train_subset_size, len(train_ds))))
        logger.info(f"Using train_subset_size={cfg.llama.train_subset_size} for a fast dev run")

    def format_example(example: dict) -> dict:
        label_token = label_to_token(example["label"], cfg.llama.positive_token, cfg.llama.negative_token)
        messages = build_messages(example["text"], label_token=label_token)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names)

    checkpoint_dir = cfg.paths.checkpoints_dir / "llama_lora"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=cfg.llama.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.llama.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.llama.gradient_accumulation_steps,
        num_train_epochs=cfg.llama.epochs,
        learning_rate=cfg.llama.lr,
        optim=cfg.llama.optim,
        warmup_steps=cfg.llama.warmup_steps,
        weight_decay=cfg.llama.weight_decay,
        packing=cfg.llama.packing,
        seed=cfg.seed.seed,
        logging_steps=cfg.llama.logging_steps,
        eval_strategy="epoch",  # val is large (thousands of examples); only eval at epoch boundaries
        save_strategy="no",  # we save the final adapter ourselves below; no need for periodic checkpoints
        max_length=cfg.llama.max_seq_len,
        dataset_text_field="text",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))

    return {"train_result": train_result.metrics, "checkpoint_dir": str(checkpoint_dir)}
