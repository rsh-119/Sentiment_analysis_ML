from imdb_sentiment.data.vocab import PAD_TOKEN, UNK_TOKEN, build_vocab, encode, tokenize


def test_tokenize_strips_html_and_lowercases():
    tokens = tokenize("Great movie!<br />Loved it.")
    assert "br" not in tokens
    assert "great" in tokens
    assert "movie" in tokens


def test_build_vocab_respects_min_freq_and_specials():
    texts = ["good good good", "bad", "good bad bad"]
    vocab = build_vocab(texts, min_freq=2, max_size=100)

    assert vocab[PAD_TOKEN] == 0
    assert vocab[UNK_TOKEN] == 1
    assert "good" in vocab
    assert "bad" in vocab


def test_build_vocab_is_deterministic():
    texts = ["good good good", "bad", "good bad bad"]
    assert build_vocab(texts, min_freq=1, max_size=100) == build_vocab(texts, min_freq=1, max_size=100)


def test_encode_uses_unk_for_oov_and_truncates():
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, "good": 2}
    ids = encode("good bad ugly", vocab, max_len=2)
    assert len(ids) == 2
    assert ids[0] == 2
    assert ids[1] == vocab[UNK_TOKEN]
