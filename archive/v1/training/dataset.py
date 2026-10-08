from pathlib import Path

import torch


def load_corpus(data_dir, own_repeat=1):
    """Join all .txt files in data_dir.

    Files starting with 'general_' are loaded once. Everything else
    (your own corpus) is repeated own_repeat times so it isn't drowned out.
    """
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}/")
    parts = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore")
        times = 1 if p.name.startswith("general_") else own_repeat
        parts.extend([text] * times)
    return "\n\n".join(parts)


class TextData:
    """Holds the encoded corpus and serves random training batches."""

    def __init__(self, text, tokenizer, block_size, val_fraction=0.1):
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        split = int(len(ids) * (1 - val_fraction))
        self.train = ids[:split]
        self.val = ids[split:]
        self.block_size = block_size

        if len(self.val) <= block_size + 1:
            raise ValueError(
                "Corpus too small for this block_size. "
                "Add more text or lower --block_size."
            )

    def get_batch(self, split, batch_size, device):
        data = self.train if split == "train" else self.val
        starts = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in starts])
        return x.to(device), y.to(device)
