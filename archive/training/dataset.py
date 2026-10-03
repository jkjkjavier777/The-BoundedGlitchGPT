import numpy as np
from pathlib import Path


class TextDataset:
    """Dataset for text data with sliding window sampling."""
    
    def __init__(self, text, tokenizer, seq_len=256, stride=1):
        """
        Args:
            text: raw text string
            tokenizer: tokenizer instance
            seq_len: sequence length (context + 1 for prediction)
            stride: step size when creating sequences
        """
        self.text = text
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        
        # Tokenize entire text
        self.tokens = tokenizer.encode(text)
        self.num_tokens = len(self.tokens)
        
        # Create sequence indices
        self.sequences = []
        for i in range(0, self.num_tokens - seq_len, stride):
            self.sequences.append(i)
        
        self.num_sequences = len(self.sequences)
        print(f"Dataset: {self.num_tokens} tokens, {self.num_sequences} sequences of length {seq_len}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """Get a sequence and its target."""
        start = self.sequences[idx]
        end = start + self.seq_len
        
        # Input: all tokens except last
        input_tokens = self.tokens[start:end-1]
        # Target: all tokens except first (shifted by 1)
        target_tokens = self.tokens[start+1:end]
        
        return input_tokens, target_tokens
    
    def get_batch(self, batch_indices):
        """Get a batch of sequences."""
        batch_inputs = []
        batch_targets = []
        
        for idx in batch_indices:
            inp, tgt = self[idx]
            batch_inputs.append(inp)
            batch_targets.append(tgt)
        
        # Stack into arrays
        # Note: in a real implementation, you'd pad sequences to same length
        return batch_inputs, batch_targets


class CorpusLoader:
    """Load text data from files."""
    
    @staticmethod
    def load_from_directory(directory):
        """Load and concatenate all text files from a directory."""
        directory = Path(directory)
        texts = []
        
        for txt_file in directory.glob("*.txt"):
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
        
        return '\n\n'.join(texts)
    
    @staticmethod
    def load_from_file(filepath):
        """Load text from a single file."""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
