import torch
import torch.nn as nn


class Embeddings(nn.Module):
    """Token embedding + learned position embedding."""

    def __init__(self, vocab_size, block_size, n_embd, dropout):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, idx):
        # idx: (batch, seq_len) of token IDs
        seq_len = idx.size(1)
        positions = torch.arange(seq_len, device=idx.device)
        x = self.tok(idx) + self.pos(positions)
        return self.drop(x)
