import torch

from inference.sampling import sample_next


def test_sampling():
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])
    assert sample_next(logits, temperature=0) == 1
    assert sample_next(logits, temperature=0.8, top_k=1) == 1
    assert 0 <= sample_next(logits, temperature=1.0, top_p=0.9) < 4


if __name__ == "__main__":
    test_sampling()
    print("sampling OK")
