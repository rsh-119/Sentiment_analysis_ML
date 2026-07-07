# 🎬 IMDB Sentiment Analysis

A binary sentiment classification project (positive/negative) on 50,000 IMDB movie reviews, built as three progressive iterations — from an LSTM baseline, to a negation-aware BiLSTM + Word2Vec model, to a fine-tuned DistilBERT transformer.

Each iteration was driven by error analysis on the previous one: the LSTM baseline was blind to negation (`"not worth it"` embedded almost identically to `"worth it"`), which motivated the negation-aware preprocessing in v2, and DistilBERT's pretrained language understanding removed the need for manual negation handling altogether in v3.

---

## 📁 Project Structure

```
Sentiment_analysis_ML/
├── README.md
└── sentiment_analysis_IMDB/
    ├── dataset/
    │   └── IMDB Dataset.csv                     # Raw dataset (50,000 reviews)
    ├── models/
    │   ├── sentiment_classifier_1.keras         # v1: LSTM
    │   └── sentiment_classifier_2.keras         # v2: BiLSTM + Word2Vec
    └── notebooks/
        ├── sentiment_analysis_part1.ipynb       # v1: LSTM baseline
        ├── sentiment_analysis_2.ipynb           # v2: BiLSTM + Word2Vec + negation handling
        ├── Sentiment_analysis_transformer.ipynb # v3: DistilBERT fine-tuning
        └── distilbert_imdb_finetuned/           # v3: saved fine-tuned model + tokenizer
```

---

## 📊 Results at a Glance

| Version | Model | Accuracy | Precision | Recall | F1 | Test set |
|---------|-------|----------|-----------|--------|----|----|
| v1 | LSTM (own embeddings) | 89% | 0.89 | 0.89 | 0.89 | 12,500 reviews |
| v2 | BiLSTM + Word2Vec + negation-aware preprocessing | 90% | 0.90 | 0.90 | 0.90 | 10,000 reviews |
| v3 | Fine-tuned DistilBERT | **92%** | 0.92 | 0.92 | 0.92 | 10,000 reviews |

> Metrics are from `classification_report` on each notebook's held-out test split (weighted/macro avg — all classes were balanced).

---

## v1 — LSTM Baseline

```
Input (raw string)
    ↓
TextVectorization    — vocab size: 20,000 | max sequence length: 200
    ↓
Embedding            — 100-dim, trainable from scratch
    ↓
LSTM                 — 100 units
    ↓
Dense                — 64 units, relu
    ↓
Dropout               — rate: 0.5
    ↓
Dense                — 1 unit, sigmoid
```

**Preprocessing:** HTML stripping (BeautifulSoup) → lowercasing + punctuation removal → tokenization (NLTK) → lemmatization → stopword removal.

**Limitation found:** standard stopword removal strips negation words, so `"not worth it"` and `"worth it"` end up with nearly identical representations — the model predicted the same (positive) sentiment for both.

---

## v2 — BiLSTM + Word2Vec + Negation-Aware Preprocessing

```
Input (raw string)
    ↓
TextVectorization    — vocab size: 20,000 | max sequence length: 200
    ↓
Embedding            — 200-dim, pretrained Word2Vec weights (trainable)
    ↓
SpatialDropout1D     — rate: 0.3
    ↓
Bidirectional LSTM   — 100 units | dropout: 0.3 | activation: tanh
    ↓
Dropout              — rate: 0.5
    ↓
Dense                — 64 units, relu
    ↓
Dropout              — rate: 0.5
    ↓
Dense                — 1 unit, sigmoid
```

### Negation Handling

Fixed the v1 blind spot with two techniques:

**1. Preserve negation words from stopword removal:**
```python
NEGATION_WORDS = {"not", "no", "never", "won't", "don't", "doesn't", ...}
stopwords_filtered = [w for w in stopwords if w not in NEGATION_WORDS]
```

**2. Window-based negation tagging** (tags the next 4 tokens after a negation word with a `NOT_` prefix):
```python
"not worth it"  →  "not NOT_worth NOT_it"
"worth it"      →  "worth it"
```

This makes `NOT_worth` and `worth` distinct tokens, so Word2Vec (200-dim, CBOW, window=5, min_count=2, 8 epochs) learns separate embeddings for each.

**Preprocessing pipeline:** HTML tag removal → lowercasing/punctuation removal → tokenization → lemmatization → negation-preserving stopword removal → negation tagging → Word2Vec → BiLSTM.

---

## v3 — Fine-Tuned DistilBERT

Fine-tuned `distilbert-base-uncased` (Hugging Face `transformers`, PyTorch) with a classification head on top, using the `Trainer` API for training-loop management.

- **Tokenizer:** DistilBERT WordPiece tokenizer, `max_length=256` (covers 95%+ of IMDB reviews), `[CLS]`/`[SEP]` special tokens.
- **Fine-tuning config:** 3 epochs, learning rate `2e-5`, warmup steps, automatic LR scheduling and gradient clipping via `Trainer`.
- **Preprocessing:** minimal — HTML stripping and whitespace normalization only. No manual negation handling needed; the pretrained model already encodes negation and context.

Outperformed both from-scratch models while requiring the least manual feature engineering.

### Generalization check

The saved model was evaluated with `pipeline("text-classification")` on 20 hand-written reviews that were **not** part of training or testing — including negation-heavy and mixed/nuanced sentiment:

| Review | Prediction | Confidence |
|---|---|---|
| "An absolute cinematic triumph from start to finish" | ✅ Positive | 99.33% |
| "A painfully predictable and boring experience" | ❌ Negative | 99.59% |
| "Not the kind of film I would ever recommend" | ❌ Negative | 98.27% |
| "The movie never manages to find its footing" | ❌ Negative | 97.12% |
| "Strong opening that falls apart in the second half" | ✅ Positive | 93.80% |
| "Visually stunning but emotionally hollow" | ✅ Positive | 83.98% |

Correctly handled negation without any explicit negation-tagging logic, and produced sensible (if imperfect) calls on genuinely mixed-sentiment reviews.

---

## 📦 Dependencies

```bash
# v1 / v2 — LSTM & BiLSTM
pip install tensorflow numpy pandas matplotlib scikit-learn
pip install beautifulsoup4 nltk gensim

# v3 — DistilBERT
pip install torch transformers
```

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

---

## ⚠️ Known Limitations

| Limitation | Description |
|------------|--------------|
| Domain-specific | Trained on movie reviews — may not generalize well to other domains |
| Short text | Lower confidence on very short sentences (< 5 words) |
| Informal text | Slang (`u`, `r`, `ur`) may map to unknown tokens in v1/v2 |
| Complex negation (v1/v2 only) | Long-range negation beyond the 4-token window may be missed — not an issue for v3 |
| Mixed sentiment | All three models are binary classifiers; genuinely mixed reviews (e.g. "great acting, weak plot") get forced into one class |

---

## 🔮 Roadmap

- [x] **Phase 2 — Transformer model**: fine-tune DistilBERT for improved accuracy and native negation handling — done (92% accuracy)
- [ ] Contraction expansion (`won't` → `will not`) before tokenization in v1/v2
- [ ] Informal text normalization (`u` → `you`, `r` → `are`)
- [ ] Multi-class sentiment (positive / neutral / negative)
- [ ] Streamlit demo app
- [ ] Serve the DistilBERT model behind a lightweight inference API

---

## 📚 Dataset

**IMDB Movie Reviews Dataset**
- 50,000 reviews (25,000 positive / 25,000 negative)
- Balanced binary classification
- Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep learning (v1/v2) | TensorFlow / Keras |
| Deep learning (v3) | PyTorch, Hugging Face Transformers |
| Word embeddings | Gensim Word2Vec |
| Text processing | NLTK, BeautifulSoup |
| Evaluation | scikit-learn |
| Language | Python 3.x |
