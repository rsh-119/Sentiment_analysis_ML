"""Configuration dataclasses for the IMDB sentiment analysis project."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    project_root: Path = PROJECT_ROOT
    raw_csv: Path = PROJECT_ROOT / "dataset" / "IMDB Dataset.csv"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    full_dataset_dir: Path = PROJECT_ROOT / "data" / "processed" / "imdb_full"
    splits_dir: Path = PROJECT_ROOT / "data" / "processed" / "imdb_splits"
    split_manifest: Path = PROJECT_ROOT / "data" / "splits" / "split_manifest.json"
    vocab_path: Path = PROJECT_ROOT / "data" / "processed" / "vocab.json"
    checkpoints_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    metrics_dir: Path = PROJECT_ROOT / "outputs" / "metrics"
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    reports_dir: Path = PROJECT_ROOT / "reports"


@dataclass
class SeedConfig:
    seed: int = 42
    deterministic: bool = True


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


@dataclass
class BiLSTMConfig:
    vocab_size: int = 30_000
    min_freq: int = 2
    max_seq_len: int = 256
    embedding_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 10
    early_stopping_patience: int = 2
    num_workers: int = 2


@dataclass
class LlamaConfig:
    base_model: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_len: int = 512
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    positive_token: str = "Positive"
    negative_token: str = "Negative"
    lr: float = 2e-4
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: float = 1.0
    warmup_steps: int = 50
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    packing: bool = False
    logging_steps: int = 10
    per_device_eval_batch_size: int = 16
    train_subset_size: int | None = None  # dev-speed knob; None = use full train split


@dataclass
class EvalConfig:
    batch_size: int = 64
    latency_num_warmup: int = 10
    latency_num_runs: int = 100


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    bilstm: BiLSTMConfig = field(default_factory=BiLSTMConfig)
    llama: LlamaConfig = field(default_factory=LlamaConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _merge_dataclass(obj: Any, overrides: dict) -> Any:
    """Recursively apply a nested dict of overrides onto a dataclass instance."""
    for key, value in overrides.items():
        if not hasattr(obj, key):
            raise ValueError(f"Unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if isinstance(value, dict) and is_dataclass(current):
            _merge_dataclass(current, value)
        else:
            field_type = {f.name: f.type for f in fields(obj)}[key]
            if field_type is Path or field_type == "Path":
                value = Path(value)
            setattr(obj, key, value)
    return obj


def load_config(yaml_path: str | Path | None = None, overrides: dict | None = None) -> AppConfig:
    """Build the default AppConfig, then apply a YAML file and/or an overrides dict on top."""
    cfg = AppConfig()
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            with open(yaml_path) as f:
                yaml_overrides = yaml.safe_load(f) or {}
            _merge_dataclass(cfg, yaml_overrides)
    if overrides:
        _merge_dataclass(cfg, overrides)
    return cfg


def apply_cli_overrides(cfg: AppConfig, override_strings: list[str]) -> AppConfig:
    """Apply `section.key=value` style CLI overrides (e.g. 'bilstm.epochs=3')."""
    nested: dict[str, Any] = {}
    for item in override_strings:
        key_path, _, raw_value = item.partition("=")
        section, _, key = key_path.partition(".")
        value = yaml.safe_load(raw_value)  # cheap type coercion (int/float/bool/str)
        nested.setdefault(section, {})[key] = value
    return _merge_dataclass(cfg, nested)
