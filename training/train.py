import argparse

import torch

from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.tokenizer import CharTokenizer
from training.dataset import TextData, load_corpus
from training.trainer import Trainer


def main():
    p = argparse.ArgumentParser(description="Train BoundedGlitchGPT")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    text = load_corpus(args.data_dir)
    tokenizer = CharTokenizer.from_text(text)
    print(f"Corpus: {len(text):,} characters | vocab: {tokenizer.vocab_size}")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    data = TextData(text, tokenizer, args.block_size)

    model = GPT(config).to(device)
    print(f"Parameters: {model.num_parameters():,}")

    trainer = Trainer(model, data, tokenizer, config, device,
                      lr=args.lr, ckpt_path=args.out)
    trainer.fit(args.steps, args.batch_size, args.eval_every)


if __name__ == "__main__":
    main()
