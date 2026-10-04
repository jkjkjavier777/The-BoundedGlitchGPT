import torch

from model.config import GPTConfig
from model.gpt import GPT


def test_forward_shapes_and_loss():
    cfg = GPTConfig(vocab_size=83, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(cfg)
    x = torch.randint(0, 83, (4, 16))
    y = torch.randint(0, 83, (4, 16))
    logits, loss = model(x, y)
    assert logits.shape == (4, 16, 83)
    assert loss.item() > 0


if __name__ == "__main__":
    test_forward_shapes_and_loss()
    print("model OK")
