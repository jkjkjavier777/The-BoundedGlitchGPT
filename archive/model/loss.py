import numpy as np


class CrossEntropyLoss:
    """Cross-entropy loss for language modeling."""
    
    def __init__(self):
        self.loss_value = 0
        self.grad = None
    
    def forward(self, logits, target_ids):
        """
        Compute cross-entropy loss.
        Args:
            logits: shape (seq_len, vocab_size) - model output
            target_ids: shape (seq_len,) - ground truth token ids
        Returns:
            loss: scalar
        """
        seq_len = logits.shape[0]
        vocab_size = logits.shape[1]
        
        # Compute softmax probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Compute cross-entropy
        loss = 0
        for i in range(seq_len):
            # Avoid log(0)
            target_prob = np.clip(probs[i, target_ids[i]], 1e-10, 1.0)
            loss += -np.log(target_prob)
        
        loss = loss / seq_len
        self.loss_value = loss
        self.probs = probs
        self.target_ids = target_ids
        
        return loss
    
    def backward(self, logits):
        """
        Compute gradient of loss w.r.t. logits.
        Args:
            logits: shape (seq_len, vocab_size)
        Returns:
            grad: same shape as logits
        """
        seq_len = logits.shape[0]
        grad = self.probs.copy()
        
        # Subtract 1 from the target class
        for i in range(seq_len):
            grad[i, self.target_ids[i]] -= 1
        
        grad = grad / seq_len
        return grad


def compute_accuracy(logits, target_ids):
    """Compute token accuracy."""
    predictions = np.argmax(logits, axis=-1)
    correct = np.sum(predictions == target_ids)
    return correct / len(target_ids)
