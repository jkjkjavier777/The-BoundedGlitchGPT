"""
BoundedGlitchGPT: Text Generation Script
=========================================

This script loads a trained model and generates text.

Features:
- Load from checkpoint
- Interactive mode (generate text on-demand)
- Batch mode (generate multiple samples)
- Temperature control (randomness)
- Top-k sampling (filter unlikely tokens)

Usage:
    # Interactive mode
    python inference/generate.py --checkpoint checkpoints/best_model.pt
    
    # Batch generation
    python inference/generate.py --checkpoint checkpoints/best_model.pt \\
        --prompt "The quick brown" --num_samples 5 --max_tokens 100
"""

import torch
import argparse
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import GPT, GPTConfig, SimpleTokenizer


# ============================================================================
# GENERATOR
# ============================================================================

class TextGenerator:
    """Handles text generation with a trained GPT model."""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Load a trained model from checkpoint.
        
        Args:
            model_path: Path to checkpoint file
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        
        print(f"Loading checkpoint from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config from checkpoint
        config_dict = checkpoint['config']
        self.config = GPTConfig()
        
        # Update config with saved values
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Initialize model
        self.model = GPT(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded")
        print(f"  Vocab size: {self.config.vocab_size}")
        print(f"  Embedding dim: {self.config.embed_dim}")
        print(f"  Num layers: {self.config.num_layers}")
        print(f"  Device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer(self.config.vocab_size)
    
    def generate(self, prompt: str, num_tokens: int = 100, 
                 temperature: float = 1.0, top_k: int = None) -> str:
        """
        Generate text given a prompt.
        
        Args:
            prompt: Starting text
            num_tokens: How many tokens to generate
            temperature: Randomness (0.1 = deterministic, 1.0 = normal, 2.0 = very random)
            top_k: If set, only sample from top-k most likely tokens (reduces nonsense)
        
        Returns:
            Generated text (prompt + generated continuation)
        """
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt)
        print(f"\nPrompt ({len(prompt_ids)} tokens): {prompt}")
        print("-" * 70)
        
        # Generate tokens
        with torch.no_grad():
            generated_ids = self._generate_tokens(
                prompt_ids,
                num_tokens,
                temperature,
                top_k
            )
        
        # Decode to text
        generated_text = self.tokenizer.decode(generated_ids)
        
        return generated_text
    
    @torch.no_grad()
    def _generate_tokens(self, prompt_ids: list, num_tokens: int,
                        temperature: float = 1.0, top_k: int = None) -> list:
        """
        Generate token IDs using the model.
        
        Args:
            prompt_ids: Starting token IDs
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k tokens
        
        Returns:
            List of token IDs (prompt + generated)
        """
        tokens = prompt_ids.copy()
        
        for i in range(num_tokens):
            # Keep only the last max_context_length tokens
            context_tokens = tokens[-self.config.max_context_length:]
            
            # Convert to tensor
            input_tensor = torch.tensor(
                [context_tokens],
                dtype=torch.long,
                device=self.device
            )
            
            # Forward pass
            logits, _ = self.model(input_tensor)
            
            # Get logits for the last token
            next_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering: only sample from top-k most likely tokens
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                # Zero out all non-top-k logits
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits[top_k_indices] = top_k_logits
            
            # Convert logits to probabilities
            probs = torch.softmax(next_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)
            
            # Show progress
            if (i + 1) % 20 == 0 or (i + 1) == num_tokens:
                partial_text = self.tokenizer.decode(tokens[len(prompt_ids):])
                print(f"Generated {i + 1}/{num_tokens} tokens: {partial_text[:50]}...")
        
        return tokens
    
    def interactive_mode(self, temperature: float = 1.0, top_k: int = None):
        """
        Interactive mode: generate text on-demand in a loop.
        
        Args:
            temperature: Default sampling temperature
            top_k: Default top-k value
        """
        print("\n" + "=" * 70)
        print("INTERACTIVE GENERATION MODE")
        print("=" * 70)
        print("Commands:")
        print("  Type a prompt and press Enter to generate text")
        print("  'exit' or 'quit' to stop")
        print("  'temp X' to set temperature (e.g., 'temp 0.5')")
        print("  'topk X' to set top-k (e.g., 'topk 50')")
        print("=" * 70 + "\n")
        
        while True:
            try:
                user_input = input("Prompt (or 'exit'): ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                # Handle commands
                if user_input.lower().startswith('temp '):
                    try:
                        temperature = float(user_input.split()[1])
                        print(f"Temperature set to {temperature}")
                    except:
                        print("Invalid temperature. Usage: temp 0.5")
                    continue
                
                if user_input.lower().startswith('topk '):
                    try:
                        top_k = int(user_input.split()[1])
                        print(f"Top-k set to {top_k}")
                    except:
                        print("Invalid top-k. Usage: topk 50")
                    continue
                
                if not user_input:
                    continue
                
                # Generate
                generated = self.generate(
                    user_input,
                    num_tokens=100,
                    temperature=temperature,
                    top_k=top_k
                )
                
                print(f"\n{'Generated text:':>20}")
                print(generated)
                print()
            
            except KeyboardInterrupt:
                print("\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point for generation."""
    
    parser = argparse.ArgumentParser(description="Generate text with BoundedGlitchGPT")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt text (if not set, uses interactive mode)')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0.1=deterministic, 1.0=normal, 2.0=random)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling (only sample from top-k most likely tokens)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate (with --prompt)')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        print("\nFirst, train a model:")
        print("  python training/train.py --data_path data/train.txt")
        sys.exit(1)
    
    # Initialize generator
    generator = TextGenerator(args.checkpoint, device=args.device)
    
    # Generate or interactive mode
    if args.prompt:
        # Batch generation mode
        print(f"\n{'='*70}")
        print(f"BATCH GENERATION ({args.num_samples} samples)")
        print(f"{'='*70}\n")
        
        for sample_idx in range(args.num_samples):
            print(f"\n--- Sample {sample_idx + 1}/{args.num_samples} ---")
            generated = generator.generate(
                args.prompt,
                num_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            print(generated)
            print()
    else:
        # Interactive mode
        generator.interactive_mode(
            temperature=args.temperature,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()
