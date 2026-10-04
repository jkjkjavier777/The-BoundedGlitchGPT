from pathlib import Path

import torch


def load_corpus(data_dir):
    """Read every .txt file in data_dir and join them into one string."""
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}/")
    return "\n\n".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in files
    )


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
        # random starting points for each sequence in the batch
        starts = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in starts])
        # target is the same window shifted one character to the right
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in starts])
        return x.to(device), y.to(device)
