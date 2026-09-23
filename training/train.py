#!/usr/bin/env python3
"""
Train BoundedGlitchGPT (model/model.py) on a text corpus.

Usage:
    python training/train.py --data path/to/corpus.txt --steps 3000
    python training/train.py --data path/to/corpus_dir/ --steps 3000

Saves a checkpoint compatible with inference/generate.py:
    {'config': {...}, 'model_state_dict': ..., 'training_history': [...]}
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.model import GPT, GPTConfig, SimpleTokenizer


def load_corpus(data_path: Path) -> str:
    if data_path.is_dir():
        files = sorted(data_path.glob("*.txt"))
        if not files:
            raise SystemExit(f"No .txt files found in {data_path}")
        text = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
        print(f"Loaded {len(files)} files from {data_path}")
    else:
        text = data_path.read_text(encoding="utf-8", errors="ignore")
        print(f"Loaded {data_path}")
    return text


def make_batches(data: torch.Tensor, ctx_len: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - ctx_len - 1, (batch_size,))
    x = torch.stack([data[i : i + ctx_len] for i in ix])
    y = torch.stack([data[i + 1 : i + ctx_len + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    p = argparse.ArgumentParser(description="Train BoundedGlitchGPT")
    p.add_argument("--data", type=str, required=True, help="Text file or directory of .txt files")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vocab_size", type=int, default=77, help="77 = real chars only, no padding waste")
    p.add_argument("--max_context_length", type=int, default=128)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--ffn_hidden_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    args = p.parse_args()

    cfg = GPTConfig()
    cfg.vocab_size = args.vocab_size
    cfg.max_context_length = args.max_context_length
    cfg.embed_dim = args.embed_dim
    cfg.num_heads = args.num_heads
    cfg.num_layers = args.num_layers
    cfg.ffn_hidden_dim = args.ffn_hidden_dim
    cfg.dropout = args.dropout

    tokenizer = SimpleTokenizer(cfg.vocab_size)
    text = load_corpus(Path(args.data))
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    unk_frac = (ids == tokenizer.stoi["<UNK>"]).float().mean().item()
    print(f"{len(ids):,} tokens, {unk_frac:.1%} unknown characters")

    split = int(0.9 * len(ids))
    train_data, val_data = ids[:split], ids[split:]
    if len(val_data) <= cfg.max_context_length + 2:
        raise SystemExit(
            f"Corpus too small ({len(ids)} tokens) for context length "
            f"{cfg.max_context_length}. Use more text or a smaller --max_context_length."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    model = GPT(cfg).to(device)
    print(f"{sum(p_.numel() for p_ in model.parameters()):,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for step in range(1, args.steps + 1):
        model.train()
        x, y = make_batches(train_data, cfg.max_context_length, args.batch_size, device)
        _, loss = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = make_batches(val_data, cfg.max_context_length, args.batch_size, device)
                _, vloss = model(vx, targets=vy)
            print(f"step {step}/{args.steps}: train {loss.item():.3f}  val {vloss.item():.3f}")
            history.append({"step": step, "train_loss": loss.item(), "val_loss": vloss.item()})

    model.eval()
    print("\n--- sample ---")
    sample_ids = model.generate(tokenizer.encode("hello"), max_new_tokens=150, temperature=0.8)
    print(tokenizer.decode(sample_ids))

    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": cfg.vocab_size,
                "max_context_length": cfg.max_context_length,
                "embed_dim": cfg.embed_dim,
                "num_heads": cfg.num_heads,
                "num_layers": cfg.num_layers,
                "dropout": cfg.dropout,
                "ffn_hidden_dim": cfg.ffn_hidden_dim,
            },
            "training_history": history,
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint to {ckpt_path}")
    print(f"Generate text with:\n  python inference/generate.py --checkpoint {ckpt_path} --prompt \"hello\"")


if __name__ == "__main__":
    main()
