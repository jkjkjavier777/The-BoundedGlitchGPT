import numpy as np
import json
from pathlib import Path


class CharTokenizer:
    """Simple character-level tokenizer."""
    
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        self.built = False
    
    def build_vocab(self, text):
        """Build vocabulary from text."""
        unique_chars = sorted(set(text))
        self.char_to_id = {char: idx for idx, char in enumerate(unique_chars)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
        self.built = True
        print(f"Vocab built: {self.vocab_size} unique characters")
    
    def encode(self, text):
        """Convert text to token IDs."""
        if not self.built:
            raise ValueError("Tokenizer vocab not built. Call build_vocab first.")
        return np.array([self.char_to_id.get(char, 0) for char in text], dtype=np.int32)
    
    def decode(self, token_ids):
        """Convert token IDs back to text."""
        if not self.built:
            raise ValueError("Tokenizer vocab not built.")
        return ''.join([self.id_to_char.get(int(tid), '?') for tid in token_ids])
    
    def save(self, path):
        """Save tokenizer vocab."""
        data = {
            'char_to_id': self.char_to_id,
            'id_to_char': {str(k): v for k, v in self.id_to_char.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path):
        """Load tokenizer vocab."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.char_to_id = data['char_to_id']
        self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        self.vocab_size = data['vocab_size']
        self.built = True
