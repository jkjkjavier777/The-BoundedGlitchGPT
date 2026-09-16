#!/usr/bin/env python3
"""
The-BoundedGlitchGPT Bot
Trains on startup and provides an interactive conversation interface.
"""

import numpy as np
import sys
from pathlib import Path

from tokenizer import CharTokenizer
from gpt import GPT
from dataset import TextDataset, CorpusLoader
from train import Trainer


class BoundedGlitchBot:
    """Main bot class - handles training and conversation."""
    
    def __init__(self, data_dir="data", model_path="model.npy", tokenizer_path="tokenizer.json"):
        self.data_dir = data_dir
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        self.model = None
        self.tokenizer = None
        self.dataset = None
    
    def load_corpus(self):
        """Load text corpus from data directory."""
        print("[*] Loading corpus...")
        corpus_text = CorpusLoader.load_from_directory(self.data_dir)
        
        if not corpus_text:
            print("[!] Error: No text files found in data directory")
            return None
        
        print(f"[✓] Corpus loaded: {len(corpus_text):,} characters")
        return corpus_text
    
    def build_tokenizer(self, corpus_text):
        """Build or load tokenizer."""
        if Path(self.tokenizer_path).exists():
            print(f"[*] Loading tokenizer from {self.tokenizer_path}...")
            tokenizer = CharTokenizer()
            tokenizer.load(self.tokenizer_path)
            print(f"[✓] Tokenizer loaded: {tokenizer.vocab_size} tokens")
        else:
            print("[*] Building tokenizer...")
            tokenizer = CharTokenizer()
            tokenizer.build_vocab(corpus_text)
            tokenizer.save(self.tokenizer_path)
            print(f"[✓] Tokenizer built and saved: {tokenizer.vocab_size} tokens")
        
        return tokenizer
    
    def build_model(self):
        """Create GPT model."""
        print("[*] Building model...")
        model = GPT(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=128,
            num_layers=2,
            num_heads=4,
            ff_dim=256,
            max_seq_len=256
        )
        print("[✓] Model created")
        return model
    
    def train(self, corpus_text, epochs=3, batch_size=8):
        """Train the model."""
        print("\n" + "="*50)
        print("TRAINING")
        print("="*50)
        
        # Create dataset
        print("[*] Creating dataset...")
        dataset = TextDataset(corpus_text, self.tokenizer, seq_len=128, stride=50)
        
        # Train
        trainer = Trainer(self.model, self.tokenizer)
        trainer.fit(dataset, epochs=epochs, batch_size=batch_size, checkpoint_path=self.model_path)
        
        # Save model
        self.model.save(self.model_path)
        print(f"[✓] Model saved to {self.model_path}")
    
    def startup(self, force_retrain=False):
        """Initialize bot - load or train model."""
        print("\n" + "="*50)
        print("THE-BOUNDEDGLITCHGPT")
        print("="*50 + "\n")
        
        # Load corpus
        corpus_text = self.load_corpus()
        if not corpus_text:
            sys.exit(1)
        
        # Build tokenizer
        self.tokenizer = self.build_tokenizer(corpus_text)
        
        # Check if model exists
        model_exists = Path(self.model_path).exists()
        
        if model_exists and not force_retrain:
            print(f"\n[*] Loading model from {self.model_path}...")
            self.model = self.build_model()
            self.model.load(self.model_path)
            print("[✓] Model loaded")
        else:
            print("\n[*] Training new model...")
            self.model = self.build_model()
            self.train(corpus_text, epochs=3, batch_size=8)
        
        print("\n[✓] Bot ready for conversation!")
    
    def chat(self, user_input, max_length=100, temperature=0.8):
        """Generate response to user input."""
        if self.model is None:
            return "[Error] Model not initialized"
        
        try:
            response = self.model.generate(
                self.tokenizer,
                user_input,
                max_new_tokens=max_length,
                temperature=temperature
            )
            return response
        except Exception as e:
            return f"[Error] {str(e)}"
    
    def interactive_mode(self):
        """Run interactive conversation loop."""
        print("\n" + "="*50)
        print("INTERACTIVE MODE")
        print("="*50)
        print("Type your prompts below. Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n[*] Goodbye!")
                    break
                
                print("\nBot: ", end="", flush=True)
                response = self.chat(user_input, max_length=150, temperature=0.8)
                print(response)
                print()
            
            except KeyboardInterrupt:
                print("\n\n[*] Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[Error] {str(e)}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="The-BoundedGlitchGPT Bot")
    parser.add_argument("--retrain", action="store_true", help="Force retrain the model")
    parser.add_argument("--data-dir", default="data", help="Directory containing training data")
    
    args = parser.parse_args()
    
    # Create and initialize bot
    bot = BoundedGlitchBot(data_dir=args.data_dir)
    bot.startup(force_retrain=args.retrain)
    
    # Start interactive conversation
    bot.interactive_mode()


if __name__ == "__main__":
    main()
