# IMDB Sentiment Analysis

Binary sentiment classification on the IMDB 50k movie review dataset.

## Models

| Model | Approach | Test Accuracy | Notebook |
|---|---|:---:|---|
| BiLSTM + Word2Vec | Word2Vec embeddings + negation tagging → BiLSTM(100) | **0.90** | [`notebooks/sentiment_analysis_2.ipynb`](notebooks/sentiment_analysis_2.ipynb) |
| DistilBERT | Fine-tuned `distilbert-base-uncased` | 0.88 | [`notebooks/Sentiment_analysis_transformer.ipynb`](notebooks/Sentiment_analysis_transformer.ipynb) |

`notebooks/sentiment_analysis_part1.ipynb` (a plain LSTM) was an earlier
learning exercise and isn't a polished model — the two above are the real
comparison.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name sentiment-imdb --display-name "Python (sentiment-imdb)"
```

Open a notebook, select the **Python (sentiment-imdb)** kernel, and run it
from the `notebooks/` directory (each notebook loads the dataset via
`../dataset/IMDB Dataset.csv`).
