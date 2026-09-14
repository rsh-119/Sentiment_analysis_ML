import torch

from imdb_sentiment.seeding import make_generator, set_seed


def test_set_seed_makes_torch_rand_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_make_generator_is_deterministic():
    g1 = make_generator(42)
    g2 = make_generator(42)
    a = torch.rand(5, generator=g1)
    b = torch.rand(5, generator=g2)
    assert torch.equal(a, b)
