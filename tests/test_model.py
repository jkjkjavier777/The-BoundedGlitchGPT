"""
Tests for the numpy GPT. Run from the repo root:

    python tests/test_model.py        (or: python -m pytest tests/)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.model import GPT, GPTConfig, CharTokenizer


def _tiny_model(dtype=np.float64):
    cfg = GPTConfig(vocab_size=11, block_size=8, d_model=16, n_layers=2, n_heads=2)
    model = GPT(cfg, seed=1, dtype=dtype)
    rng = np.random.default_rng(2)
    for k, v in model.params.items():  # perturb so biases / LN params matter
        model.params[k] = v + 0.05 * rng.standard_normal(v.shape)
    return model


def test_gradients_match_numerical():
    model = _tiny_model()
    rng = np.random.default_rng(3)
    x = rng.integers(0, 11, size=(2, 8))
    y = rng.integers(0, 11, size=(2, 8))

    _, grads = model.loss_and_grads(x, y)
    eps = 1e-5
    worst = 0.0
    for name, p in model.params.items():
        for _ in range(3):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + eps
            lp = model.loss(x, y)
            p[i] = old - eps
            lm = model.loss(x, y)
            p[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[name][i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
            assert rel < 1e-4, f"{name}{i}: analytic={ana:.3e} numeric={num:.3e}"
    print(f"  gradient check ok (worst relative error {worst:.2e})")


def test_causality():
    model = _tiny_model()
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    b = a.copy()
    b[0, 5:] = [0, 0, 0]  # change only the future
    la, _ = model.forward(a)
    lb, _ = model.forward(b)
    assert np.allclose(la[0, :5], lb[0, :5]), "future tokens leaked into the past"
    print("  causality ok")


def test_initial_loss_near_uniform():
    cfg = GPTConfig(vocab_size=50, block_size=16, d_model=32, n_layers=2, n_heads=4)
    model = GPT(cfg, seed=0)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 50, size=(4, 16))
    y = rng.integers(0, 50, size=(4, 16))
    loss = model.loss(x, y)
    assert abs(loss - np.log(50)) < 0.3, loss
    print(f"  initial loss {loss:.3f} ~ ln(V) {np.log(50):.3f}")


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    tok = CharTokenizer.from_text("hello world")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=8, d_model=16, n_layers=1, n_heads=2)
    model = GPT(cfg, seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "m.npz")
        model.save(path, tok)
        model2, tok2 = GPT.load(path)
    assert tok2.chars == tok.chars
    x = np.array([tok.encode("hello")])
    assert np.allclose(model.forward(x)[0], model2.forward(x)[0])
    print("  save/load ok")


def test_generate_length():
    model = _tiny_model(np.float32)
    out = model.generate([1, 2, 3], max_new_tokens=20, top_k=5, rng=np.random.default_rng(0))
    assert len(out) == 23 and all(0 <= t < 11 for t in out)
    print("  generate ok")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all tests passed")
