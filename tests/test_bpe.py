"""
Tests for the BPE tokenizer. Run from the repo root:

    python tests/test_bpe.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tokenizer.bpe import BPETokenizer

CORPUS = (
    "USER: What is the five-perspective framework?\n"
    "GPT: Practical, conceptual, historical, systems and adversarial.\n\n"
    "USER: What does the validation layer do?\n"
    "GPT: It checks identity, grounding and safety before a reply goes out.\n\n"
) * 20


def _tok():
    return BPETokenizer.train(CORPUS, vocab_size=300)


def test_roundtrip_many_strings():
    tok = _tok()
    samples = [
        "", " ", "hello", "Hello, world!", "multiple   spaces   here",
        "line1\nline2\n\n\nline3", "tabs\tand\r\nwindows", "don't won't they're I'll",
        "snake_case_name and __dunder__", "numbers 123 4567 and 3.14",
        "unicode: caf\u00e9 na\u00efve \u65e5\u672c\u8a9e \u2603", "emoji \U0001F600\U0001F680 ok",
        "trailing space ", " leading space", CORPUS,
    ]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s, repr(s)
    print("  round-trip ok on", len(samples), "strings")


def test_compresses_better_than_bytes():
    tok = BPETokenizer.train(CORPUS, vocab_size=500)
    ids = tok.encode(CORPUS)
    ratio = len(CORPUS) / len(ids)
    assert ratio > 2.0, ratio
    print(f"  {ratio:.2f} chars per token")


def test_special_token():
    tok = _tok()
    ids = tok.encode("hi<|endoftext|>there")
    assert tok.eos_id in ids
    assert tok.decode(ids) == "hi<|endoftext|>there"
    plain = tok.encode("<|endoftext|>", allow_special=False)
    assert tok.eos_id not in plain
    assert tok.decode(plain) == "<|endoftext|>"
    print("  special tokens ok")


def test_vocab_ids_in_range():
    tok = _tok()
    ids = tok.encode(CORPUS + "\u65e5\u672c\u8a9e\U0001F600")
    assert all(0 <= i < tok.vocab_size for i in ids)
    print(f"  vocab size {tok.vocab_size}")


def test_deterministic_training():
    a, b = _tok(), _tok()
    assert a.to_dict() == b.to_dict()
    print("  training is deterministic")


def test_save_load():
    tok = _tok()
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "t.json")
        tok.save(path)
        tok2 = BPETokenizer.load(path)
    assert tok.encode(CORPUS) == tok2.encode(CORPUS)
    assert tok.vocab_size == tok2.vocab_size
    print("  save/load ok")


def test_small_vocab_rejected():
    try:
        BPETokenizer.train("abc", vocab_size=100)
    except ValueError:
        print("  small vocab rejected")
        return
    raise AssertionError("should have raised")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all tests passed")
