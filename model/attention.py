import numpy as np


class MultiHeadAttention:
    """Multi-head self-attention layer."""
    
    def __init__(self, embedding_dim, num_heads, seed=42):
        np.random.seed(seed)
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2.0 / (embedding_dim + embedding_dim))
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2.0 / (embedding_dim + embedding_dim))
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2.0 / (embedding_dim + embedding_dim))
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2.0 / (embedding_dim + embedding_dim))
        
        self.bias_q = np.zeros(embedding_dim)
        self.bias_k = np.zeros(embedding_dim)
        self.bias_v = np.zeros(embedding_dim)
        self.bias_o = np.zeros(embedding_dim)
        
        # Gradients
        self.grad_W_q = np.zeros_like(self.W_q)
        self.grad_W_k = np.zeros_like(self.W_k)
        self.grad_W_v = np.zeros_like(self.W_v)
        self.grad_W_o = np.zeros_like(self.W_o)
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        Args:
            x: input shape (seq_len, embedding_dim)
            mask: attention mask (optional)
        Returns:
            output: shape (seq_len, embedding_dim)
        """
        seq_len = x.shape[0]
        
        # Linear projections
        Q = x @ self.W_q + self.bias_q  # (seq_len, embedding_dim)
        K = x @ self.W_k + self.bias_k
        V = x @ self.W_v + self.bias_v
        
        # Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        K = K.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        V = V.reshape(seq_len, self.num_heads, self.head_dim).transpose(1, 0, 2)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.head_dim)  # (num_heads, seq_len, seq_len)
        
        # Causal mask (prevent attention to future tokens)
        if mask is None:
            mask = np.tril(np.ones((seq_len, seq_len)))
        scores = scores * mask - 1e9 * (1 - mask)
        
        # Softmax
        attn_weights = self._softmax(scores, axis=-1)  # (num_heads, seq_len, seq_len)
        
        # Apply attention to values
        context = attn_weights @ V  # (num_heads, seq_len, head_dim)
        
        # Concatenate heads
        context = context.transpose(1, 0, 2).reshape(seq_len, self.embedding_dim)
        
        # Final linear projection
        output = context @ self.W_o + self.bias_o
        
        return output
    
    def _softmax(self, x, axis=-1):
        """Numerical stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
