import numpy as np


class TokenEmbedding:
    """Token embedding layer."""
    
    def __init__(self, vocab_size, embedding_dim, seed=42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # Xavier initialization
        self.weight = np.random.randn(vocab_size, embedding_dim) * np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.grad = np.zeros_like(self.weight)
    
    def forward(self, token_ids):
        """
        Forward pass.
        Args:
            token_ids: shape (seq_len,) or (batch_size, seq_len)
        Returns:
            embeddings: shape (..., embedding_dim)
        """
        return self.weight[token_ids]
    
    def backward(self, grad_output):
        """
        Backward pass.
        Args:
            grad_output: gradient of loss w.r.t. embeddings
        """
        self.grad = np.zeros_like(self.weight)
        # This is simplified; real backprop would accumulate gradients
        return grad_output


class PositionalEmbedding:
    """Positional encoding (non-learned)."""
    
    def __init__(self, max_seq_len, embedding_dim):
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.pe = self._create_positional_encoding(max_seq_len, embedding_dim)
    
    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding matrix."""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def forward(self, seq_len):
        """Get positional embeddings for sequence length."""
        return self.pe[:seq_len, :]
