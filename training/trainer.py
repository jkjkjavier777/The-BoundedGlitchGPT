import time
from pathlib import Path

import torch


class Trainer:
    """Runs the training loop and saves the best checkpoint."""

    def __init__(self, model, data, tokenizer, config, device,
                 lr=3e-4, ckpt_path="checkpoints/model.pt"):
        self.model = model
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.ckpt_path = Path(ckpt_path)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.best_val = float("inf")

    @torch.no_grad()
    def estimate_loss(self, batch_size, eval_batches=20):
        """Average loss over several random batches (less noisy than one)."""
        self.model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(eval_batches)
            for k in range(eval_batches):
                x, y = self.data.get_batch(split, batch_size, self.device)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save(self, step, val_loss):
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.config.to_dict(),
                "chars": self.tokenizer.chars,   # vocab travels with weights
                "step": step,
                "val_loss": val_loss,
            },
            self.ckpt_path,
        )

    def fit(self, steps, batch_size, eval_every=250):
        self.model.train()
        start = time.time()

        for step in range(steps + 1):
            if step % eval_every == 0 or step == steps:
                losses = self.estimate_loss(batch_size)
                note = ""
                if losses["val"] < self.best_val:
                    self.best_val = losses["val"]
                    self.save(step, losses["val"])
                    note = "  <- saved"
                print(
                    f"step {step:5d} | train {losses['train']:.4f} | "
                    f"val {losses['val']:.4f} | {time.time() - start:.0f}s{note}"
                )
            if step == steps:
                break

            x, y = self.data.get_batch("train", batch_size, self.device)
            _, loss = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # stop rare huge gradients from wrecking training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        print(f"Done. Best val loss: {self.best_val:.4f} -> {self.ckpt_path}")
