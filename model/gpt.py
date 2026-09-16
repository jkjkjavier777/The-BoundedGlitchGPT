import numpy as np
from embeddings import TokenEmbedding, PositionalEmbedding
from transformer import TransformerBlock


class GPT:
    """Generative Pre-trained Transformer model."""
    
    def __init__(self, vocab_size, embedding_dim=128, num_layers=4, num_heads=4, 
                 ff_dim=256, max_seq_len=512, seed=42):
        np.random.seed(seed)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim, seed=seed)
        self.pos_embedding = PositionalEmbedding(max_seq_len, embedding_dim)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(embedding_dim, num_heads, ff_dim, seed=seed+i)
            for i in range(num_layers)
        ]
        
        # Output projection to vocabulary
        self.W_out = np.random.randn(embedding_dim, vocab_size) * np.sqrt(2.0 / (embedding_dim + vocab_size))
        self.b_out = np.zeros(vocab_size)
    
    def forward(self, token_ids):
        """
        Forward pass.
        Args:
            token_ids: shape (seq_len,)
        Returns:
            logits: shape (seq_len, vocab_size)
        """
        seq_len = len(token_ids)
        
        # Token embedding
        x = self.token_embedding.forward(token_ids)  # (seq_len, embedding_dim)
        
        # Add positional embedding
        pos_emb = self.pos_embedding.forward(seq_len)  # (seq_len, embedding_dim)
        x = x + pos_emb
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block.forward(x)
        
        # Project to vocabulary size
        logits = x @ self.W_out + self.b_out  # (seq_len, vocab_size)
        
        return logits
    
    def generate(self, tokenizer, prompt, max_new_tokens=100, temperature=1.0):
        """
        Generate text autoregressively.
        Args:
            tokenizer: tokenizer instance
            prompt: string prompt
            max_new_tokens: number of tokens to generate
            temperature: controls randomness (higher = more random)
        Returns:
            generated text
        """
        token_ids = tokenizer.encode(prompt).tolist()
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            context = np.array(token_ids[-self.max_seq_len:], dtype=np.int32)
            
            # Forward pass
            logits = self.forward(context)  # (seq_len, vocab_size)
            
            # Get logits for last token
            next_logits = logits[-1, :] / temperature
            
            # Softmax and sample
            probs = self._softmax(next_logits)
            next_token = np.random.choice(self.vocab_size, p=probs)
            
            token_ids.append(next_token)
            
            # Stop if we generate end-of-sequence
            if next_token == 0:  # Assuming 0 is EOS
                break
        
        return tokenizer.decode(np.array(token_ids, dtype=np.int32))
    
    def _softmax(self, x):
        """Numerical stable softmax."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def save(self, path):
        """Save model weights."""
        weights = {
            'token_embedding': self.token_embedding.weight,
            'W_out': self.W_out,
            'b_out': self.b_out,
            'config': {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'num_layers': self.num_layers,
                'num_heads': self.num_heads,
                'max_seq_len': self.max_seq_len
            }
        }
        np.save(path, weights, allow_pickle=True)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model weights."""
        weights = np.load(path, allow_pickle=True).item()
        self.token_embedding.weight = weights['token_embedding']
        self.W_out = weights['W_out']
        self.b_out = weights['b_out']
        print(f"Model loaded from {path}")
