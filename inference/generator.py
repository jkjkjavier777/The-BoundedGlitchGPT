import torch

from inference.sampling import sample_next
from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.tokenizer import CharTokenizer


class Generator:
    """Loads a checkpoint and turns a prompt into generated text."""

    def __init__(self, checkpoint_path="checkpoints/model.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=True)

        self.config = GPTConfig.from_dict(ckpt["config"])
        self.tokenizer = CharTokenizer(chars=ckpt["chars"][1:])  # skip <unk>
        # make sure IDs match exactly what training used
        self.tokenizer.chars = ckpt["chars"]
        self.tokenizer.stoi = {c: i for i, c in enumerate(self.tokenizer.chars)}
        self.tokenizer.itos = {i: c for c, i in self.tokenizer.stoi.items()}

        self.model = GPT(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_tokens=200, temperature=0.8,
                 top_k=None, top_p=None):
        ids = self.tokenizer.encode(prompt) or [0]
        n_prompt = len(ids)

        for _ in range(max_tokens):
            # model can only see the last block_size tokens
            context = ids[-self.config.block_size:]
            x = torch.tensor([context], dtype=torch.long, device=self.device)
            logits, _ = self.model(x)
            next_id = sample_next(logits[0, -1], temperature, top_k, top_p)
            ids.append(next_id)

        return self.tokenizer.decode(ids[n_prompt:])
