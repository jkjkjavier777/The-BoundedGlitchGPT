import torch.nn as nn

from model.attention import CausalSelfAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """Attention + feed-forward, each with a residual connection."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual: add input back to output
        x = x + self.ff(self.ln2(x))
        return x
