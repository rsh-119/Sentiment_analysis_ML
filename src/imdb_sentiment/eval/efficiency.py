"""Computational-cost measurements shared by both models: inference latency,
throughput, parameter count, and on-disk model size.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch

from imdb_sentiment.utils.io import dir_size_bytes


def measure_inference_efficiency(
    predict_fn: Callable[[list[str]], object],
    texts: list[str],
    batch_size: int,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Time `predict_fn` over `num_runs` batches of `batch_size` texts (cycling
    through `texts`), after `num_warmup` untimed warmup calls.
    """
    n = len(texts)

    def get_batch(i: int) -> list[str]:
        start = (i * batch_size) % n
        end = start + batch_size
        if end <= n:
            return texts[start:end]
        return texts[start:] + texts[: end - n]

    for i in range(num_warmup):
        predict_fn(get_batch(i))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for i in range(num_runs):
        predict_fn(get_batch(num_warmup + i))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    total_examples = num_runs * batch_size
    return {
        "latency_ms_per_example": (elapsed / total_examples) * 1000,
        "throughput_examples_per_sec": total_examples / elapsed,
        "batch_size": batch_size,
        "num_runs": num_runs,
    }


def count_parameters(model) -> dict:
    """Total and trainable parameter counts. Handles two PEFT/bitsandbytes
    quirks that would otherwise silently misreport these numbers:
    - A `bitsandbytes` 4-bit-quantized parameter's `.numel()` reports its
      packed storage size (~half the logical element count), not the
      original parameter count. `quant_state.shape` holds the original shape.
    - A `PeftModel` loaded via `from_pretrained` for inference freezes its
      LoRA adapter weights (`requires_grad=False`) by default, so a
      requires_grad-based "trainable" count reports 0 even though those are
      exactly the parameters that were trained. Detect them by name instead.
    """
    total = 0
    lora_params = 0
    requires_grad_params = 0
    for name, p in model.named_parameters():
        quant_state = getattr(p, "quant_state", None)
        if quant_state is not None and hasattr(quant_state, "shape"):
            n = 1
            for d in quant_state.shape:
                n *= d
        else:
            n = p.numel()
        total += n
        if "lora_" in name:
            lora_params += n
        if p.requires_grad:
            requires_grad_params += n

    trainable = lora_params if lora_params > 0 else requires_grad_params
    return {"total": total, "trainable": trainable}


def model_disk_size_bytes(path: Path) -> int:
    return dir_size_bytes(path)
