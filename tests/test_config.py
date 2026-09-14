from pathlib import Path

from imdb_sentiment.config import apply_cli_overrides, load_config


def test_default_config_has_expected_values():
    cfg = load_config()
    assert cfg.seed.seed == 42
    assert cfg.split.train_ratio == 0.70
    assert cfg.bilstm.epochs == 10
    assert cfg.llama.base_model == "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


def test_yaml_override_merges_onto_defaults(tmp_path: Path):
    yaml_path = tmp_path / "override.yaml"
    yaml_path.write_text("bilstm:\n  epochs: 3\n  lr: 0.0005\n")

    cfg = load_config(yaml_path)

    assert cfg.bilstm.epochs == 3
    assert cfg.bilstm.lr == 0.0005
    assert cfg.bilstm.hidden_size == 256  # untouched default


def test_cli_overrides_apply_on_top():
    cfg = load_config()
    cfg = apply_cli_overrides(cfg, ["bilstm.epochs=1", "seed.seed=7"])

    assert cfg.bilstm.epochs == 1
    assert cfg.seed.seed == 7
