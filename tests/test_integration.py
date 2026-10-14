"""
Tests that the model works with both tokenizers and old checkpoints still load.

    python tests/test_integration.py
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.model import GPT, GPTConfig, CharTokenizer
from tokenizer.bpe import BPETokenizer

TEXT = (
    "USER: What is tau?\nGPT: Tau is the divergence threshold, fixed at 0.15.\n\n"
    "USER: How many raters?\nGPT: At least three independent, blinded raters.\n\n"
) * 15


def _roundtrip(tok):
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, d_model=16, n_layers=1, n_heads=2)
    model = GPT(cfg, seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "m.npz")
        model.save(path, tok)
        model2, tok2 = GPT.load(path)
    x = np.array([tok.encode("USER: What is tau?")[:16]])
    assert np.allclose(model.forward(x)[0], model2.forward(x)[0])
    assert tok2.encode(TEXT) == tok.encode(TEXT)
    assert tok2.decode(tok2.encode("USER: hi")) == "USER: hi"
    return model2, tok2


def test_char_checkpoint_roundtrip():
    tok = CharTokenizer.from_text(TEXT)
    _, tok2 = _roundtrip(tok)
    assert isinstance(tok2, CharTokenizer)
    print("  char-level checkpoint ok")


def test_bpe_checkpoint_roundtrip():
    tok = BPETokenizer.train(TEXT, vocab_size=320)
    _, tok2 = _roundtrip(tok)
    assert isinstance(tok2, BPETokenizer)
    print("  BPE checkpoint ok")


def test_old_format_checkpoint_still_loads():
    tok = CharTokenizer.from_text(TEXT)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, d_model=16, n_layers=1, n_heads=2)
    model = GPT(cfg, seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "old.npz")
        from dataclasses import asdict
        meta = json.dumps({"config": asdict(cfg), "chars": tok.chars})  # pre-BPE format
        np.savez(path, __meta__=np.array(meta), **model.params)
        model2, tok2 = GPT.load(path)
    assert isinstance(tok2, CharTokenizer) and tok2.chars == tok.chars
    print("  old-format checkpoint ok")


def test_bpe_generate_decodes():
    tok = BPETokenizer.train(TEXT, vocab_size=320)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=16, d_model=16, n_layers=1, n_heads=2)
    model = GPT(cfg, seed=0)
    ids = model.generate(tok.encode("USER:"), max_new_tokens=12, top_k=10,
                         rng=np.random.default_rng(0))
    assert len(ids) == len(tok.encode("USER:")) + 12
    assert isinstance(tok.decode(ids), str)
    print("  BPE generation ok")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all tests passed")
