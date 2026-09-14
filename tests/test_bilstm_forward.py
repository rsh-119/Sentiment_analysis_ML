import torch

from imdb_sentiment.models.bilstm import BiLSTMClassifier


def test_forward_pass_produces_correct_shape():
    model = BiLSTMClassifier(vocab_size=50, embedding_dim=8, hidden_size=16, num_layers=2, dropout=0.1)
    input_ids = torch.randint(0, 50, (4, 10))
    lengths = torch.tensor([10, 8, 5, 3])

    logits = model(input_ids, lengths)

    assert logits.shape == (4,)
