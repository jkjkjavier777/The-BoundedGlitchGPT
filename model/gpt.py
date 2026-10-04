import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig
from model.embeddings import Embeddings
from model.transformer import TransformerBlock


class GPT(nn.Module):
    """token IDs -> embeddings -> transformer blocks -> logits."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(
            config.vocab_size, config.block_size, config.n_embd, config.dropout
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.n_embd, config.n_head, config.block_size, config.dropout
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        assert idx.size(1) <= self.config.block_size, "sequence longer than block_size"
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))           # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
