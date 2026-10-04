import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer: text <-> token IDs."""

    UNK = "<unk>"

    def __init__(self, chars=None):
        self.chars = [self.UNK] + sorted(set(chars or []) - {self.UNK})
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    @classmethod
    def from_text(cls, text):
        return cls(chars=set(text))

    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps({"chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls()
        tok.chars = data["chars"]
        tok.stoi = {c: i for i, c in enumerate(tok.chars)}
        tok.itos = {i: c for c, i in tok.stoi.items()}
        return tok
