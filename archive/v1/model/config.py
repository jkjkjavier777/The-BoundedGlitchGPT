from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 0          # set from the tokenizer
    block_size: int = 128        # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
