"""
The-BoundedGlitchGPT -- train.py
================================

Trains the numpy GPT on every .txt file in data/.

    python training/train.py
    python training/train.py --steps 3000 --d_model 96 --n_layers 3
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.model import GPT, GPTConfig, CharTokenizer


class AdamW:
    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k, p in params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            update = (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)
            if p.ndim == 2:  # decay weight matrices only
                update = update + self.wd * p
            p -= lr * update


def clip_grads(grads, max_norm):
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for k in grads:
            grads[k] = grads[k] * scale
    return total


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def load_corpus(data_dir):
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found in {data_dir}")
    text = "\n\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files)
    print(f"Corpus: {len(files)} files, {len(text):,} characters")
    return text


def get_batch(data, batch_size, block_size, rng):
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size] for i in ix])
    y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


def estimate_loss(model, data, batch_size, block_size, rng, batches=10):
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, block_size, rng)
        losses.append(model.loss(x, y))
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser(description="Train The-BoundedGlitchGPT (numpy)")
    ap.add_argument("--data_dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "model.npz"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=64)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    text = load_corpus(args.data_dir)
    tok = CharTokenizer.from_text(text)
    data = np.array(tok.encode(text), dtype=np.int64)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    if len(val_data) <= args.block_size + 1:
        sys.exit("Corpus too small for the chosen block_size.")
    print(f"Vocab: {tok.vocab_size} chars | train {len(train_data):,} / val {len(val_data):,} tokens")

    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads)
    model = GPT(cfg, seed=args.seed)
    opt = AdamW(model.params)
    print(f"Model: {model.num_params():,} parameters")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        x, y = get_batch(train_data, args.batch_size, args.block_size, rng)
        loss, grads = model.loss_and_grads(x, y)
        clip_grads(grads, args.grad_clip)
        opt.step(model.params, grads, lr_at(step - 1, args.steps, args.lr, args.warmup))

        if step % args.eval_every == 0 or step == args.steps:
            val = estimate_loss(model, val_data, args.batch_size, args.block_size, rng)
            mark = ""
            if val < best_val:
                best_val = val
                model.save(args.out, tok)
                mark = "  (saved)"
            print(f"step {step:5d} | train {loss:.3f} | val {val:.3f} | {time.time() - t0:6.1f}s{mark}")

    print(f"\nBest val loss {best_val:.3f}. Checkpoint: {args.out}")
    best, _ = GPT.load(args.out)
    sample = best.generate(tok.encode("\n"), max_new_tokens=200, temperature=0.8, top_k=20, rng=rng)
    print("\n--- sample ---\n" + tok.decode(sample))


if __name__ == "__main__":
    main()
