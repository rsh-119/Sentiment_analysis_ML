# 🎬 IMDB Sentiment Analysis — BiLSTM Model

A deep learning pipeline for binary sentiment classification (positive/negative) on IMDB movie reviews, using a Bidirectional LSTM with Word2Vec embeddings and negation-aware text preprocessing.

---

## 📁 Project Structure

```
sentiment-analysis/
├── IMDB Dataset.csv        # Raw dataset (50,000 reviews)
├── bilstm_sentiment.py     # Main training script
├── README.md               # This file
└── models/
    └── sentiment_bilstm/   # Saved model (after training)
```

---

## 🧠 Model Architecture

```
Input (raw string)
    ↓
TextVectorization       — vocab size: 20,000 | max sequence length: 200
    ↓
Embedding Layer         — 200-dim Word2Vec pretrained weights (trainable)
    ↓
SpatialDropout1D        — rate: 0.3
    ↓
Bidirectional LSTM      — 100 units | dropout: 0.3 | activation: tanh
    ↓
Dropout                 — rate: 0.5
    ↓
Dense                   — 64 units | activation: relu
    ↓
Dropout                 — rate: 0.5
    ↓
Dense                   — 1 unit | activation: sigmoid
    ↓
Output (0 = Negative, 1 = Positive)
```

---

## ⚙️ Text Preprocessing Pipeline

Raw text goes through the following steps in order:

```
Raw Review
    ↓
1. HTML tag removal          (BeautifulSoup)
2. Lowercasing + punctuation removal
3. Contraction expansion     (won't → will not)
4. Tokenization              (NLTK word_tokenize)
5. Lemmatization             (WordNetLemmatizer)
6. Stopword removal          (negation words preserved)
7. Negation tagging          (NOT_ prefix, window=4)
    ↓
Clean Text → Word2Vec → BiLSTM
```

### Negation Handling

A key feature of this pipeline is **negation-aware preprocessing**. Standard stopword removal destroys the meaning of negations — `"not worth it"` becomes identical to `"worth it"` after stripping `"not"`.

This is fixed with two techniques:

**1. Preserve negation words from stopword removal:**
```python
NEGATION_WORDS = {"not", "no", "never", "won't", "don't", "doesn't", ...}
stopwords_filtered = [w for w in stopwords if w not in NEGATION_WORDS]
```

**2. Window-based negation tagging:**
```python
# Tags the next 4 words after a negation with NOT_ prefix
"not worth it"  →  "not NOT_worth NOT_it"
"worth it"      →  "worth it"
```

This ensures Word2Vec learns `NOT_worth` and `worth` as **distinct tokens** with different embedding vectors.

---

## 📦 Dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
pip install beautifulsoup4 nltk gensim
```

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

---

## 🚀 Training

```python
# Word2Vec config
Word2Vec(sentences=tokenized, vector_size=200, window=5, min_count=2, sg=0, epochs=8)

# Model training
model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    batch_size=32,
    epochs=20,
    callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
)
```

---

## 📊 Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~88%   |
| Precision | ~88%   |
| Recall    | ~88%   |
| F1 Score  | ~88%   |

> Evaluated on 20% held-out test split (10,000 reviews).

---

## 🔍 Inference

```python
def predict_sentiment(text, model, clean_text_fn):
    pred = model.predict(tf.constant([clean_text_fn(text)], dtype=tf.string))
    score = pred[0][0]
    label = "Positive" if score > 0.5 else "Negative"
    return label, float(score)

predict_sentiment("This movie is absolutely worth it", model, clean_text)
# → ('Positive', 0.91)

predict_sentiment("This movie is not worth it", model, clean_text)
# → ('Negative', 0.12)
```

---

## ⚠️ Known Limitations

| Limitation | Description |
|------------|-------------|
| Domain-specific | Trained on movie reviews — may not generalize well to other domains |
| Short text | Low confidence on very short sentences (< 5 words) |
| Informal text | Slang (`u`, `r`, `ur`) may map to unknown tokens |
| Complex negation | Long-range negation beyond window=4 may be missed |

---

## 🔮 Roadmap

- [ ] **Phase 2 — Transformer Model**: Fine-tune DistilBERT for improved accuracy (~93%) and native negation handling
- [ ] Contraction expansion (`won't` → `will not`) before tokenization
- [ ] Informal text normalization (`u` → `you`, `r` → `are`)
- [ ] Multi-class sentiment (positive / neutral / negative)
- [ ] Streamlit demo app

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
| Deep Learning | TensorFlow / Keras |
| Word Embeddings | Gensim Word2Vec |
| Text Processing | NLTK, BeautifulSoup |
| Evaluation | Scikit-learn |
| Language | Python 3.x |