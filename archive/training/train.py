import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for sub in ("tokenizer", "model", "training"):
    sys.path.insert(0, str(_ROOT / sub))

import numpy as np

from tokenizer import CharTokenizer
from gpt import GPT
from dataset import TextDataset, CorpusLoader
from loss import CrossEntropyLoss, compute_accuracy


class Adam:
    def __init__(self, learning_rate=0.003, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, parameters):
        self.t += 1
        for i, (param, grad) in enumerate(parameters):
            if i not in self.m:
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Trainer:
    def __init__(self, model, tokenizer, learning_rate=0.003):
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(learning_rate=learning_rate)
        self.losses = []
        self.accuracies = []

    def train_epoch(self, dataset, batch_size=32):
        num_sequences = len(dataset)
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        for i in range(0, num_sequences, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_inputs, batch_targets = dataset.get_batch(batch_indices.tolist())
            if not batch_inputs:
                continue

            self.model.zero_grad()

            batch_loss = 0.0
            batch_accuracy = 0.0

            for inp, tgt in zip(batch_inputs, batch_targets):
                input_tokens = inp.astype(np.int32)
                target_tokens = tgt.astype(np.int32)

                logits = self.model.forward(input_tokens)
                loss = self.loss_fn.forward(logits, target_tokens)
                accuracy = compute_accuracy(logits, target_tokens)

                dlogits = self.loss_fn.backward()
                self.model.backward(dlogits)

                batch_loss += loss
                batch_accuracy += accuracy

            n = len(batch_inputs)
            batch_loss /= n
            batch_accuracy /= n

            params = self.model.parameters()
            for _, grad in params:
                grad /= n
            self.optimizer.step(params)

            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_accuracy = epoch_accuracy / max(num_batches, 1)

        self.losses.append(avg_loss)
        self.accuracies.append(avg_accuracy)

        return avg_loss, avg_accuracy

    def sample(self, prompt="The", max_new_tokens=80, temperature=0.8):
        try:
            return self.model.generate(self.tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        except Exception as e:
            return f"[sample error: {e}]"

    def fit(self, dataset, epochs=100, batch_size=32, checkpoint_path="model_checkpoint.npy", sample_every=10):
        print(f"Training for {epochs} epochs...")

        for epoch in range(epochs):
            loss, accuracy = self.train_epoch(dataset, batch_size)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

            if (epoch + 1) % sample_every == 0 or (epoch + 1) == epochs:
                sample_text = self.sample()
                print(f"  Sample: {sample_text!r}")

            if (epoch + 1) % max(1, epochs // 5) == 0:
                self.model.save(checkpoint_path)

        print("Training complete!")


def main():
    DATA_DIR = str(_ROOT / "data")
    SEQ_LEN = 128
    EMBEDDING_DIM = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    FF_DIM = 256
    EPOCHS = 150
    BATCH_SIZE = 8
    LEARNING_RATE = 0.003

    print("Loading corpus...")
    corpus_text = CorpusLoader.load_from_directory(DATA_DIR)

    if not corpus_text:
        print("Error: No text files found in data directory")
        sys.exit(1)

    print(f"Corpus size: {len(corpus_text)} characters")

    print("Building tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(corpus_text)
    tokenizer.save(str(_ROOT / "tokenizer.json"))

    print("Creating dataset...")
    dataset = TextDataset(corpus_text, tokenizer, seq_len=SEQ_LEN, stride=50)

    print("Creating model...")
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
    )

    trainer = Trainer(model, tokenizer, learning_rate=LEARNING_RATE)
    trainer.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, checkpoint_path=str(_ROOT / "model.npy"), sample_every=10)

    model.save(str(_ROOT / "model.npy"))
    print("Training script complete!")


if __name__ == "__main__":
    main()
