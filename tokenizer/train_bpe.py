"""
Train a BPE tokenizer on every .txt file in data/.

    python tokenizer/train_bpe.py --vocab_size 1000
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tokenizer.bpe import BPETokenizer


def main():
    ap = argparse.ArgumentParser(description="Train a BPE tokenizer")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "tokenizer" / "bge_bpe.json"))
    ap.add_argument("--vocab_size", type=int, default=1000)
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {args.data_dir}")
    text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
    print(f"Corpus: {len(files)} files, {len(text):,} characters")

    tok = BPETokenizer.train(text, vocab_size=args.vocab_size, verbose=True)
    ids = tok.encode(text)
    assert tok.decode(ids) == text, "round-trip failed"
    tok.save(args.out)

    print(f"\nVocab size:      {tok.vocab_size}")
    print(f"Tokens in corpus: {len(ids):,}")
    print(f"Chars per token:  {len(text) / len(ids):.2f}  (char-level = 1.00)")
    print(f"Saved:            {args.out}")

    demo = "USER: What is the five-perspective framework?\nGPT:"
    pieces = [tok.decode([i]) for i in tok.encode(demo)]
    print("\nExample:", demo.replace("\n", "\\n"))
    print("Tokens: ", pieces)


if __name__ == "__main__":
    main()
