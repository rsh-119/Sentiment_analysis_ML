# IMDB Sentiment Analysis

Three approaches to binary sentiment classification on the [IMDB 50k reviews
dataset](dataset/IMDB%20Dataset.csv), trained and compared side by side. All
three notebooks split the data **70% train / 15% validation / 15% test**:
validation drives early stopping / best-checkpoint selection, and test is
touched exactly once, at the end, for the reported numbers below — so these
are genuine held-out results, not validation numbers relabeled as test.

| Notebook | Approach | Saved model | Test accuracy |
|---|---|---|---|
| `notebooks/sentiment_analysis_part1.ipynb` | LSTM + trainable embeddings | `models/sentiment_classifier_1.keras` | 0.88 |
| `notebooks/sentiment_analysis_2.ipynb` | Bidirectional LSTM + Word2Vec embeddings + negation tagging | `models/sentiment_classifier_2.keras` | **0.90** |
| `notebooks/Sentiment_analysis_transformer.ipynb` | Fine-tuned DistilBERT | `notebooks/distilbert_imdb_finetuned/` | 0.88 |

The Word2Vec + negation-tagged BiLSTM (notebook 2) edges out both the plain
LSTM and DistilBERT on this run — plausibly because explicitly tagging words
following a negation (`NOT_worth`, `NOT_recommend`, ...) combined with a
bidirectional encoder captures signal the plain LSTM's simpler preprocessing
doesn't. Take the ranking as a single-run comparison, not a rigorous
ablation — none of these were tuned to their full potential (e.g. DistilBERT
was only fine-tuned for 3 epochs at `max_length=128`), so the gap could
narrow or flip with more tuning on any of the three.

Full precision/recall/F1 breakdowns are in each notebook's "Evaluate" /
"Model Evaluation" section.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name sentiment-imdb --display-name "Python (sentiment-imdb)"
```

Open a notebook and select the **Python (sentiment-imdb)** kernel. Each
notebook loads the dataset via a path relative to its own location
(`../dataset/IMDB Dataset.csv`), so run notebooks from the `notebooks/`
directory (Jupyter's default) rather than the repo root.

`requirements.txt` pins `torch` to a `cu129` build via `--extra-index-url`.
On a machine without an NVIDIA GPU this still installs and runs fine on CPU,
just as a larger download than a CPU-only wheel would be.

### GPU notes (NVIDIA, this machine's driver = CUDA 12.9)

Getting both frameworks onto the GPU cleanly needed two fixes beyond a plain
`pip install`, both wired into the `sentiment-imdb` kernelspec
(`~/.local/share/jupyter/kernels/sentiment-imdb/kernel.json`) so they apply
automatically — no manual `export` needed:

- **PyTorch**: the default PyPI `torch` wheel is built for a newer CUDA than
  this driver supports (`cu130` vs. driver's `12.9`), so `torch.cuda.is_available()`
  silently returns `False`. Fixed by installing from
  `https://download.pytorch.org/whl/cu129` instead (already reflected in
  `requirements.txt`).
- **TensorFlow / XLA**: JIT-compiling ops on GPU needs `libdevice.10.bc`
  (not bundled by a plain `pip install tensorflow`) and a `ptxas` new enough
  for this GPU's compute capability (8.6) — the system's `/usr/bin/ptxas` is
  CUDA 11.5 and too old, causing XLA to silently fall back to slower
  driver-side compilation. Fixed via `XLA_FLAGS=--xla_gpu_cuda_data_dir=...`
  pointing at `.venv/xla_cuda_data_dir` (a `libdevice.10.bc` symlinked in from
  the system CUDA toolkit) and prepending the newer `ptxas` bundled with the
  `triton` package (a `torch` dependency) onto `PATH`.

If you re-create the kernelspec from scratch, re-add these two `env` entries
or GPU ops will either error out (PyTorch) or silently run slower via
driver-fallback JIT (TensorFlow).

## Layout

```
dataset/    IMDB Dataset.csv (50k labeled reviews)
models/     saved .keras models from the two LSTM notebooks
notebooks/  the three training notebooks + the fine-tuned DistilBERT output
            (notebooks/distilbert_imdb_finetuned/) — intermediate training
            checkpoints are not kept around; only the final model is
```
