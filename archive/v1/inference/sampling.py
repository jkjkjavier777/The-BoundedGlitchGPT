import torch
import torch.nn.functional as F


def sample_next(logits, temperature=0.8, top_k=None, top_p=None):
    """Pick the next token ID from a vector of logits (one score per token)."""
    # temperature 0 = greedy: always take the highest-scoring token
    if temperature <= 0:
        return int(torch.argmax(logits))

    # low temperature = safer/repetitive, high = more random
    logits = logits / temperature

    # top-k: keep only the k best tokens
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        cutoff = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))

    # top-p: keep the smallest set of tokens whose probabilities add up to p
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > top_p
        remove[1:] = remove[:-1].clone()   # always keep the top token
        remove[0] = False
        logits[sorted_idx[remove]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))
