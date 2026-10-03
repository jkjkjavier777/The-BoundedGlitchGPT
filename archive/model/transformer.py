import numpy as np
from attention import MultiHeadAttention


class FeedForward:
    """Position-wise feedforward network."""
    
    def __init__(self, embedding_dim, ff_dim, seed=42):
        np.random.seed(seed)
        self.embedding_dim = embedding_dim
        self.ff_dim = ff_dim
        
        # Two linear layers with ReLU in between
        self.W1 = np.random.randn(embedding_dim, ff_dim) * np.sqrt(2.0 / (embedding_dim + ff_dim))
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, embedding_dim) * np.sqrt(2.0 / (ff_dim + embedding_dim))
        self.b2 = np.zeros(embedding_dim)
    
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: shape (seq_len, embedding_dim)
        Returns:
            output: shape (seq_len, embedding_dim)
        """
        # First linear + ReLU
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        # Second linear
        output = hidden @ self.W2 + self.b2
        return output


class LayerNorm:
    """Layer normalization."""
    
    def __init__(self, embedding_dim, eps=1e-6):
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.gamma = np.ones(embedding_dim)
        self.beta = np.zeros(embedding_dim)
    
    def forward(self, x):
        """
        Args:
            x: shape (..., embedding_dim)
        Returns:
            normalized: same shape as x
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class TransformerBlock:
    """Single transformer block."""
    
    def __init__(self, embedding_dim, num_heads, ff_dim, seed=42):
        np.random.seed(seed)
        self.embedding_dim = embedding_dim
        
        self.attn = MultiHeadAttention(embedding_dim, num_heads, seed=seed)
        self.ff = FeedForward(embedding_dim, ff_dim, seed=seed)
        
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)
    
    def forward(self, x):
        """
        Forward pass with residual connections.
        Args:
            x: shape (seq_len, embedding_dim)
        Returns:
            output: shape (seq_len, embedding_dim)
        """
        # Self-attention with residual
        attn_out = self.attn.forward(x)
        x = self.norm1.forward(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.ff.forward(x)
        x = self.norm2.forward(x + ff_out)
        
        return x
