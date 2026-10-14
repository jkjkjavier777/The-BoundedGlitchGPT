"""
The-BoundedGlitchGPT -- tokenizer/bpe.py
========================================

Byte-level BPE tokenizer in pure Python (no dependencies).

- Starts from the 256 raw bytes, so ANY text can be encoded (no unknown chars).
- Learns merges from your corpus: frequent byte pairs become single tokens.
- Special tokens (default: <|endoftext|>) get ids after the merges.

Same interface as CharTokenizer: .vocab_size, .encode(text), .decode(ids).
"""

import json
import re
from collections import Counter

# Splits text into word-like chunks. Every character matches exactly one
# alternative, so joining the chunks always reproduces the original text.
_PAT = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+"
)

DEFAULT_SPECIAL = ["<|endoftext|>"]


def _merge(ids, pair, new_id):
    out = []
    i, n = 0, len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None, special_tokens=None):
        self.merges = {}  # (id_a, id_b) -> new id, in the order they were learned
        for i, pair in enumerate(merges or []):
            self.merges[(int(pair[0]), int(pair[1]))] = 256 + i
        self.special_tokens = list(DEFAULT_SPECIAL if special_tokens is None else special_tokens)
        self._build()

    def _build(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special_ids = {tok: base + i for i, tok in enumerate(self.special_tokens)}
        for tok, idx in self.special_ids.items():
            self.vocab[idx] = tok.encode("utf-8")
        self._special_re = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens else None
        )
        self._cache = {}

    # ---- properties --------------------------------------------------------

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eos_id(self):
        return self.special_ids.get("<|endoftext|>")

    # ---- training ----------------------------------------------------------

    @classmethod
    def train(cls, text, vocab_size=1000, special_tokens=None, verbose=False):
        specials = list(DEFAULT_SPECIAL if special_tokens is None else special_tokens)
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError("vocab_size is too small (needs at least 256 + special tokens)")
        for s in specials:  # special strings in the corpus are not learned as text
            text = text.replace(s, "\n")

        freqs = Counter(_PAT.findall(text))
        words = [[list(w.encode("utf-8")), c] for w, c in freqs.items()]

        merges = []
        for step in range(n_merges):
            counts = Counter()
            for ids, c in words:
                for pair in zip(ids, ids[1:]):
                    counts[pair] += c
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], p))
            if counts[best] < 2:
                break  # nothing left worth merging
            new_id = 256 + len(merges)
            merges.append(best)
            for w in words:
                if len(w[0]) > 1:
                    w[0] = _merge(w[0], best, new_id)
            if verbose and (step + 1) % 100 == 0:
                print(f"  merge {step + 1}/{n_merges}")
        return cls(merges, specials)

    # ---- encode / decode ---------------------------------------------------

    def _encode_word(self, w):
        cached = self._cache.get(w)
        if cached is not None:
            return cached
        ids = list(w.encode("utf-8"))
        while len(ids) > 1:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = self.merges.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            ids = _merge(ids, best, best_rank)
        self._cache[w] = ids
        return ids

    def encode(self, text, allow_special=True):
        ids = []
        if allow_special and self._special_re:
            parts = self._special_re.split(text)
        else:
            parts = [text]
        for part in parts:
            if allow_special and part in self.special_ids:
                ids.append(self.special_ids[part])
                continue
            for w in _PAT.findall(part):
                ids.extend(self._encode_word(w))
        return ids

    def decode(self, ids):
        data = b"".join(self.vocab.get(int(i), b"") for i in ids)
        return data.decode("utf-8", errors="replace")

    # ---- save / load -------------------------------------------------------

    def to_dict(self):
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        return {
            "type": "bpe",
            "merges": [list(p) for p, _ in ordered],
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["merges"], d.get("special_tokens"))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
