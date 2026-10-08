"""
The-BoundedGlitchGPT -- generate.py
===================================

    python inference/generate.py --prompt "The glitch" --max_tokens 150
    python inference/generate.py            # interactive mode
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.model import GPT


def complete(model, tok, prompt, max_tokens, temperature, top_k, rng):
    ids = model.generate(tok.encode(prompt), max_new_tokens=max_tokens,
                         temperature=temperature, top_k=top_k, rng=rng)
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser(description="Generate text with The-BoundedGlitchGPT")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model.npz"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--max_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}\nTrain first: python training/train.py")

    model, tok = GPT.load(args.checkpoint)
    rng = np.random.default_rng(args.seed)
    print(f"Loaded {model.num_params():,} params | vocab {tok.vocab_size} | "
          f"context {model.config.block_size}")

    if args.prompt is not None:
        print(complete(model, tok, args.prompt, args.max_tokens, args.temperature, args.top_k, rng))
        return

    print("Interactive mode. Type a prompt, or 'quit'.\n")
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip().lower() in ("quit", "exit"):
            break
        if prompt:
            print(complete(model, tok, prompt, args.max_tokens, args.temperature, args.top_k, rng), "\n")


if __name__ == "__main__":
    main()
