"""
BoundedGlitchGPT: Minimal GPT Implementation
==============================================

A single-file GPT implementation for learning purposes.
Everything is here: tokenizer, embeddings, attention, transformer blocks, and the full model.

Once you understand how this works, we'll refactor into separate modules.

Key Concept: A GPT is a stack of Transformer blocks that predict the next token
given a sequence of previous tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

class GPTConfig:
    """Hyperparameters for the GPT model. Adjust these to change model size."""
    
    # Vocabulary size: how many unique tokens can the model understand?
    vocab_size: int = 50_257  # Standard GPT-2 vocabulary size
    
    # Context length: how many tokens can the model see at once?
    # (e.g., max_context_length=512 means the model can see up to 512 tokens of history)
    max_context_length: int = 512
    
    # Embedding dimension: how many numbers represent each token?
    # Higher = more expressive, but slower and more memory
    embed_dim: int = 768  # ~medium size
    
    # Number of attention heads: parallel attention mechanisms
    # (must divide embed_dim evenly)
    num_heads: int = 12
    
    # Number of transformer blocks: how many layers of processing?
    # Deeper models are more capable but slower to train
    num_layers: int = 12
    
    # Dropout: regularization to prevent overfitting (0.1 = 10%)
    dropout: float = 0.1
    
    # Feed-forward inner dimension: intermediate layer in each block
    # Typically 4x the embedding dimension
    ffn_hidden_dim: int = 3072
    
    # Learning rate for training
    learning_rate: float = 0.0001
    
    # Batch size during training
    batch_size: int = 32
    
    # Number of training epochs
    num_epochs: int = 3


# ============================================================================
# SIMPLE TOKENIZER
# ============================================================================

class SimpleTokenizer:
    """
    A minimal tokenizer for this example.
    In production, you'd use a pre-trained tokenizer (e.g., from Hugging Face).
    """
    
    def __init__(self, vocab_size: int = 50_257):
        """Initialize with a given vocabulary size."""
        self.vocab_size = vocab_size
        self.stoi = {}  # string-to-integer: maps text to token IDs
        self.itos = {}  # integer-to-string: maps token IDs to text
        
        # For this demo, we'll use simple character-level encoding
        # In reality, you'd load a pre-trained BPE or WordPiece vocab
        characters = set()
        
        # Build character vocabulary
        common_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-\n"
        for char in common_chars:
            characters.add(char)
        
        # Add special tokens
        special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        
        # Build the vocabulary
        idx = 0
        for token in special_tokens:
            self.stoi[token] = idx
            self.itos[idx] = token
            idx += 1
        
        for char in sorted(characters):
            if char not in self.stoi:
                self.stoi[char] = idx
                self.itos[idx] = char
                idx += 1
        
        # Pad remaining vocab with dummy tokens
        while idx < vocab_size:
            self.stoi[f"<TOKEN_{idx}>"] = idx
            self.itos[idx] = f"<TOKEN_{idx}>"
            idx += 1
    
    def encode(self, text: str) -> list:
        """Convert text to list of token IDs."""
        tokens = []
        for char in text:
            if char in self.stoi:
                tokens.append(self.stoi[char])
            else:
                # Unknown character → use <UNK> token
                tokens.append(self.stoi["<UNK>"])
        return tokens
    
    def decode(self, tokens: list) -> str:
        """Convert list of token IDs back to text."""
        return "".join([self.itos.get(t, "<UNK>") for t in tokens])


# ============================================================================
# POSITIONAL EMBEDDINGS
# ============================================================================

class PositionalEmbedding(nn.Module):
    """
    Positional encoding tells the model WHERE in the sequence each token is.
    
    Why? Because the transformer's self-attention mechanism doesn't inherently
    know the order of tokens. We need to inject position information.
    
    Uses the sinusoidal positional encoding from "Attention is All You Need".
    """
    
    def __init__(self, max_seq_length: int, embed_dim: int):
        super().__init__()
        
        # Create a matrix of shape (max_seq_length, embed_dim)
        # that encodes position information
        pe = torch.zeros(max_seq_length, embed_dim)
        
        # Position indices: 0, 1, 2, ..., max_seq_length-1
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Dimension indices: 0, 1, 2, ..., embed_dim-1
        # The formula uses powers of 10000 to create different frequency scales
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            -(math.log(10000.0) / embed_dim))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if embed_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of the module state)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_length, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to token embeddings.
        
        Args:
            x: token embeddings of shape (batch_size, seq_length, embed_dim)
        
        Returns:
            embeddings + positional encoding
        """
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length, :]


# ============================================================================
# MULTI-HEAD SELF-ATTENTION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    This is the core innovation of the Transformer. It allows the model to
    attend to different parts of the sequence simultaneously.
    
    How it works:
    1. Split the embedding into multiple "heads"
    2. Each head computes attention independently
    3. Concatenate the results
    
    This lets different heads learn different ways to attend to the sequence.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head
        
        # Linear projections for Query, Key, Value
        # Each takes embed_dim input and produces embed_dim output
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection: concatenate all heads and project back
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor: we divide by sqrt(head_dim) to prevent attention weights
        # from growing too large (which would make gradients unstable)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head self-attention.
        
        Args:
            query: (batch_size, seq_len, embed_dim) - what to look for
            key: (batch_size, seq_len, embed_dim) - what to look at
            value: (batch_size, seq_len, embed_dim) - what to extract
            mask: (batch_size, seq_len, seq_len) - which positions to attend to
        
        Returns:
            output: (batch_size, seq_len, embed_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = query.size()
        
        # Project inputs
        Q = self.query_proj(query)  # (batch_size, seq_len, embed_dim)
        K = self.key_proj(key)      # (batch_size, seq_len, embed_dim)
        V = self.value_proj(value)  # (batch_size, seq_len, embed_dim)
        
        # Reshape for multi-head attention
        # From: (batch_size, seq_len, embed_dim)
        # To:   (batch_size, seq_len, num_heads, head_dim)
        # Then: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores shape: (batch_size, num_heads, seq_len, seq_len)
        
        # Apply mask if provided (e.g., causal mask to prevent attending to future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Convert scores to attention weights via softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, V)
        # context shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Concatenate heads: (batch_size, num_heads, seq_len, head_dim) → (batch_size, seq_len, embed_dim)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, embed_dim)
        
        # Final output projection
        output = self.output_proj(context)
        
        return output, attention_weights


# ============================================================================
# FEED-FORWARD NETWORK
# ============================================================================

class FeedForward(nn.Module):
    """
    Feed-forward network: Linear → Activation → Linear
    
    After attention, each position goes through a fully-connected network.
    This is called a "position-wise feed-forward network" because it's
    applied to each position independently.
    
    Architecture:
        input (embed_dim)
        → Linear (hidden_dim) 
        → ReLU (non-linearity)
        → Linear (embed_dim)
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        x = self.fc1(x)  # Project up
        x = F.relu(x)    # Non-linearity
        x = self.dropout(x)
        x = self.fc2(x)  # Project down
        return x


# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer block.
    
    Combines attention and feed-forward in a residual structure:
    1. Input → Self-Attention → Residual Connection → Layer Norm
    2. Output → Feed-Forward → Residual Connection → Layer Norm
    
    Residual connections allow gradients to flow through deep networks.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ffn_hidden_dim: int, 
                 dropout: float = 0.0):
        super().__init__()
        
        # Self-attention sub-layer
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward sub-layer
        self.ffn = FeedForward(embed_dim, ffn_hidden_dim, dropout)
        
        # Layer normalization (applied before each sub-layer)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through a transformer block.
        
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: causal mask to prevent attending to future tokens
        
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        # Attention with residual connection
        # Pre-norm: normalize before applying attention
        attention_out, _ = self.attention(
            self.norm1(x), 
            self.norm1(x), 
            self.norm1(x),
            mask=mask
        )
        x = x + self.dropout(attention_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        
        return x


# ============================================================================
# MAIN GPT MODEL
# ============================================================================

class GPT(nn.Module):
    """
    A minimal GPT language model.
    
    Architecture:
    1. Token embedding: convert token IDs to vectors
    2. Positional embedding: add position information
    3. Stack of transformer blocks: the core of the model
    4. Output head: convert last hidden state to logits over vocabulary
    
    The model predicts the next token given a sequence of previous tokens.
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token embedding layer: maps token IDs to embed_dim vectors
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        
        # Positional embedding: adds position information to each token
        self.positional_embedding = PositionalEmbedding(
            config.max_context_length, 
            config.embed_dim
        )
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.ffn_hidden_dim,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(config.embed_dim)
        
        # Output projection: convert hidden states to vocabulary logits
        # This tells us the probability of each token being next
        self.output_head = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Initialize weights (important for training stability)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with reasonable defaults."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, token_ids: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the GPT model.
        
        Args:
            token_ids: (batch_size, seq_len) - token IDs
            targets: (batch_size, seq_len) - target token IDs for loss computation (optional)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - next token probabilities
            loss: scalar loss (only if targets provided)
        """
        batch_size, seq_len = token_ids.size()
        
        # Check that sequence doesn't exceed max context length
        assert seq_len <= self.config.max_context_length, \
            f"Sequence length {seq_len} exceeds max context {self.config.max_context_length}"
        
        # Token embedding: token IDs → vectors
        x = self.token_embedding(token_ids)  # (batch_size, seq_len, embed_dim)
        
        # Add positional encoding
        x = self.positional_embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # Create causal mask: each position can only attend to itself and earlier positions
        # This prevents the model from "cheating" by looking at future tokens during training
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=token_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=causal_mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Project to vocabulary: predict next token
        logits = self.output_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for loss computation
            # Loss computes cross-entropy between predicted logits and target tokens
            logits_flat = logits.view(-1, self.config.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, prompt_ids: list, max_new_tokens: int, temperature: float = 1.0) -> list:
        """
        Generate new tokens given a prompt.
        
        Args:
            prompt_ids: list of token IDs to start with
            max_new_tokens: how many tokens to generate
            temperature: controls randomness (lower = more deterministic, higher = more random)
        
        Returns:
            list of token IDs (prompt + generated tokens)
        """
        tokens = prompt_ids.copy()
        
        for _ in range(max_new_tokens):
            # Keep only the last max_context_length tokens to fit in context
            context_tokens = tokens[-self.config.max_context_length:]
            
            # Convert to tensor
            input_tensor = torch.tensor([context_tokens], dtype=torch.long, device=next(self.parameters()).device)
            
            # Forward pass
            logits, _ = self(input_tensor)
            
            # Get logits for the last token
            next_logits = logits[0, -1, :] / temperature
            
            # Convert logits to probabilities
            probs = F.softmax(next_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(next_token)
        
        return tokens


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BoundedGlitchGPT: Minimal GPT Implementation")
    print("=" * 70)
    
    # Initialize configuration
    config = GPTConfig()
    print(f"\nModel Configuration:")
    print(f"  Vocab Size: {config.vocab_size}")
    print(f"  Max Context Length: {config.max_context_length}")
    print(f"  Embedding Dimension: {config.embed_dim}")
    print(f"  Number of Heads: {config.num_heads}")
    print(f"  Number of Layers: {config.num_layers}")
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(config.vocab_size)
    print(f"\nTokenizer initialized with {config.vocab_size} tokens")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(config).to(device)
    print(f"\nModel initialized on {device}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Example forward pass
    print("\n" + "=" * 70)
    print("EXAMPLE: Forward Pass")
    print("=" * 70)
    
    prompt = "Hello, world!"
    token_ids = tokenizer.encode(prompt)
    print(f"\nPrompt: '{prompt}'")
    print(f"Token IDs: {token_ids}")
    
    # Create batch (add batch dimension)
    batch = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(batch)
    
    print(f"\nOutput shape: {logits.shape}")
    print(f"(batch_size={logits.size(0)}, seq_len={logits.size(1)}, vocab_size={logits.size(2)})")
    
    # Get the predicted next token
    next_token_logits = logits[0, -1, :]  # Last token of first sequence
    next_token_id = next_token_logits.argmax().item()
    next_char = tokenizer.itos.get(next_token_id, "?")
    
    print(f"\nNext predicted token ID: {next_token_id}")
    print(f"Next predicted character: '{next_char}'")
    
    print("\n" + "=" * 70)
    print("SUCCESS: Model is working!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run training/train.py to train the model")
    print("2. Run inference/generate.py to generate text")
    print("3. Read the comments above to understand each component")
