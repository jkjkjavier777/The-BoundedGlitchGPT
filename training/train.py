import numpy as np
import sys
from pathlib import Path

from tokenizer import CharTokenizer
from gpt import GPT
from dataset import TextDataset, CorpusLoader
from loss import CrossEntropyLoss, compute_accuracy


class Trainer:
    """Trainer class for GPT model."""
    
    def __init__(self, model, tokenizer, learning_rate=0.001):
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.loss_fn = CrossEntropyLoss()
        self.losses = []
        self.accuracies = []
    
    def train_epoch(self, dataset, batch_size=32):
        """Train for one epoch."""
        num_sequences = len(dataset)
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)
        
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        for i in range(0, num_sequences, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_inputs, batch_targets = dataset.get_batch(batch_indices.tolist())
            
            # For simplicity, train on first sequence in batch
            if len(batch_inputs) > 0:
                input_tokens = batch_inputs[0].astype(np.int32)
                target_tokens = batch_targets[0].astype(np.int32)
                
                # Forward pass
                logits = self.model.forward(input_tokens)
                
                # Compute loss
                loss = self.loss_fn.forward(logits, target_tokens)
                accuracy = compute_accuracy(logits, target_tokens)
                
                epoch_loss += loss
                epoch_accuracy += accuracy
                num_batches += 1
                
                # Simple gradient descent (real implementation would use proper backprop)
                # For now, just apply small random updates to weights
                self.model.W_out += np.random.randn(*self.model.W_out.shape) * 0.0001
        
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_accuracy = epoch_accuracy / max(num_batches, 1)
        
        self.losses.append(avg_loss)
        self.accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def fit(self, dataset, epochs=3, batch_size=32, checkpoint_path="model_checkpoint.npy"):
        """Train the model."""
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            loss, accuracy = self.train_epoch(dataset, batch_size)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % max(1, epochs // 3) == 0:
                self.model.save(checkpoint_path)
        
        print("Training complete!")


def main():
    """Main training script."""
    
    # Configuration
    DATA_DIR = "data"
    SEQ_LEN = 128
    EMBEDDING_DIM = 128
    NUM_LAYERS = 2
    NUM_HEADS = 4
    FF_DIM = 256
    EPOCHS = 5
    BATCH_SIZE = 8
    
    # Load corpus
    print("Loading corpus...")
    corpus_text = CorpusLoader.load_from_directory(DATA_DIR)
    
    if not corpus_text:
        print("Error: No text files found in data directory")
        sys.exit(1)
    
    print(f"Corpus size: {len(corpus_text)} characters")
    
    # Build tokenizer
    print("Building tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(corpus_text)
    tokenizer.save("tokenizer.json")
    
    # Create dataset
    print("Creating dataset...")
    dataset = TextDataset(corpus_text, tokenizer, seq_len=SEQ_LEN, stride=50)
    
    # Create model
    print("Creating model...")
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM
    )
    
    # Create trainer
    trainer = Trainer(model, tokenizer, learning_rate=0.001)
    
    # Train
    trainer.fit(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE, checkpoint_path="model.npy")
    
    # Save final model
    model.save("model.npy")
    
    print("Training script complete!")


if __name__ == "__main__":
    main()
