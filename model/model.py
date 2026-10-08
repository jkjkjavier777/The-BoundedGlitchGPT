"""
The-BoundedGlitchGPT -- model.py
================================

A GPT written from scratch in pure numpy. No torch, no autograd: every layer
has a hand-written forward AND backward pass.

Everything lives in this one file for now:

    1. GPTConfig        hyperparameters
    2. CharTokenizer    character-level tokenizer
    3. Primitives       softmax, GELU, LayerNorm, Linear, attention, MLP
    4. GPT              parameters, forward, backward, sampling, save/load

Architecture (GPT-2 style, pre-LayerNorm, tied input/output embeddings):

    tokens -> wte[tokens] + wpe[positions]
           -> N x [ x + Attn(LN(x)) ;  x + MLP(LN(x)) ]
           -> LN -> logits = x @ wte.T
"""

import json
import math
from dataclasses import dataclass, asdict

import numpy as np


# ============================================================================
# 1. CONFIG
# ============================================================================

@dataclass
class GPTConfig:
    vocab_size: int = 0
    block_size: int = 64      # max context length
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 0             # 0 -> 4 * d_model

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads"


# ============================================================================
# 2. TOKENIZER (character level)
# ============================================================================

class CharTokenizer:
    """Maps each distinct character in the corpus to an integer id."""

    def __init__(self, chars=None):
        self.chars = list(chars) if chars else []
        self._rebuild()

    def _rebuild(self):
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """Unknown characters are skipped."""
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), "") for i in ids)


# ============================================================================
# 3. PRIMITIVES  (each *_fwd returns (output, cache); *_bwd uses the cache)
# ============================================================================

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


# ---- GELU (tanh approximation) --------------------------------------------

_GELU_C = math.sqrt(2.0 / math.pi)


def gelu_fwd(x):
    u = _GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * x * (1.0 + t), (x, t)


def gelu_bwd(dy, cache):
    x, t = cache
    du_dx = _GELU_C * (1.0 + 3.0 * 0.044715 * x ** 2)
    dydx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du_dx
    return dy * dydx


# ---- LayerNorm -------------------------------------------------------------

def layernorm_fwd(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return g * xhat + b, (xhat, inv, g)


def layernorm_bwd(dy, cache):
    xhat, inv, g = cache
    d = xhat.shape[-1]
    dxhat = dy * g
    dx = inv * (dxhat
                - dxhat.mean(axis=-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    dg = (dy * xhat).reshape(-1, d).sum(axis=0)
    db = dy.reshape(-1, d).sum(axis=0)
    return dx, dg, db


# ---- Linear ----------------------------------------------------------------

def linear_fwd(x, W, b):
    return x @ W + b, (x, W)


def linear_bwd(dy, cache):
    x, W = cache
    x2 = x.reshape(-1, x.shape[-1])
    dy2 = dy.reshape(-1, dy.shape[-1])
    dx = dy @ W.T
    dW = x2.T @ dy2
    db = dy2.sum(axis=0)
    return dx, dW, db


# ---- Causal multi-head self-attention -------------------------------------

def attention_fwd(x, Wqkv, bqkv, Wo, bo, n_heads):
    B, T, D = x.shape
    hd = D // n_heads

    qkv, c_qkv = linear_fwd(x, Wqkv, bqkv)
    q, k, v = np.split(qkv, 3, axis=-1)

    def split_heads(t):  # (B,T,D) -> (B,H,T,hd)
        return t.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal_mask, -1e9, scores)
    att = softmax(scores)

    out = att @ v                                        # (B,H,T,hd)
    merged = out.transpose(0, 2, 1, 3).reshape(B, T, D)  # (B,T,D)
    y, c_o = linear_fwd(merged, Wo, bo)
    return y, (c_qkv, c_o, q, k, v, att, hd, n_heads)


def attention_bwd(dy, cache):
    c_qkv, c_o, q, k, v, att, hd, n_heads = cache

    dmerged, dWo, dbo = linear_bwd(dy, c_o)
    B, T, D = dmerged.shape
    dout = dmerged.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)

    datt = dout @ v.transpose(0, 1, 3, 2)
    dv = att.transpose(0, 1, 3, 2) @ dout
    # softmax backward (masked entries have att == 0, so their grad is 0)
    dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
    dscores = dscores / math.sqrt(hd)
    dq = dscores @ k
    dk = dscores.transpose(0, 1, 3, 2) @ q

    def merge_heads(t):  # (B,H,T,hd) -> (B,T,D)
        return t.transpose(0, 2, 1, 3).reshape(B, T, D)

    dqkv = np.concatenate([merge_heads(dq), merge_heads(dk), merge_heads(dv)], axis=-1)
    dx, dWqkv, dbqkv = linear_bwd(dqkv, c_qkv)
    return dx, dWqkv, dbqkv, dWo, dbo


# ---- MLP -------------------------------------------------------------------

def mlp_fwd(x, W1, b1, W2, b2):
    u, c1 = linear_fwd(x, W1, b1)
    a, cg = gelu_fwd(u)
    y, c2 = linear_fwd(a, W2, b2)
    return y, (c1, cg, c2)


def mlp_bwd(dy, cache):
    c1, cg, c2 = cache
    da, dW2, db2 = linear_bwd(dy, c2)
    du = gelu_bwd(da, cg)
    dx, dW1, db1 = linear_bwd(du, c1)
    return dx, dW1, db1, dW2, db2


# ============================================================================
# 4. GPT
# ============================================================================

class GPT:
    def __init__(self, config: GPTConfig, seed=0, dtype=np.float32):
        assert config.vocab_size > 0, "set config.vocab_size first"
        self.config = config
        self.dtype = dtype
        self.params = self._init_params(np.random.default_rng(seed))

    # ---- parameters --------------------------------------------------------

    def _init_params(self, rng):
        c, dt = self.config, self.dtype
        D, F, V, T = c.d_model, c.d_ff, c.vocab_size, c.block_size

        def normal(*shape, std=0.02):
            return (rng.standard_normal(shape) * std).astype(dt)

        res_std = 0.02 / math.sqrt(2 * c.n_layers)  # scale residual projections
        P = {
            "wte": normal(V, D),
            "wpe": normal(T, D),
            "lnf_g": np.ones(D, dt),
            "lnf_b": np.zeros(D, dt),
        }
        for l in range(c.n_layers):
            P[f"h{l}.ln1_g"] = np.ones(D, dt)
            P[f"h{l}.ln1_b"] = np.zeros(D, dt)
            P[f"h{l}.Wqkv"] = normal(D, 3 * D)
            P[f"h{l}.bqkv"] = np.zeros(3 * D, dt)
            P[f"h{l}.Wo"] = normal(D, D, std=res_std)
            P[f"h{l}.bo"] = np.zeros(D, dt)
            P[f"h{l}.ln2_g"] = np.ones(D, dt)
            P[f"h{l}.ln2_b"] = np.zeros(D, dt)
            P[f"h{l}.W1"] = normal(D, F)
            P[f"h{l}.b1"] = np.zeros(F, dt)
            P[f"h{l}.W2"] = normal(F, D, std=res_std)
            P[f"h{l}.b2"] = np.zeros(D, dt)
        return P

    def num_params(self):
        return sum(p.size for p in self.params.values())

    # ---- forward -----------------------------------------------------------

    def forward(self, idx):
        """idx: int array (B, T). Returns (logits (B,T,V), cache)."""
        c, P = self.config, self.params
        idx = np.asarray(idx)
        B, T = idx.shape
        assert T <= c.block_size, f"sequence length {T} > block_size {c.block_size}"

        h = P["wte"][idx] + P["wpe"][:T]
        layer_caches = []
        for l in range(c.n_layers):
            g = lambda n: P[f"h{l}.{n}"]
            a, c_ln1 = layernorm_fwd(h, g("ln1_g"), g("ln1_b"))
            at, c_att = attention_fwd(a, g("Wqkv"), g("bqkv"), g("Wo"), g("bo"), c.n_heads)
            h = h + at
            a2, c_ln2 = layernorm_fwd(h, g("ln2_g"), g("ln2_b"))
            m, c_mlp = mlp_fwd(a2, g("W1"), g("b1"), g("W2"), g("b2"))
            h = h + m
            layer_caches.append((c_ln1, c_att, c_ln2, c_mlp))

        hf, c_lnf = layernorm_fwd(h, P["lnf_g"], P["lnf_b"])
        logits = hf @ P["wte"].T  # tied embeddings
        return logits, (idx, layer_caches, c_lnf, hf)

    # ---- loss + backward ---------------------------------------------------

    def loss_and_grads(self, idx, targets):
        """Cross-entropy loss and gradients for every parameter."""
        c, P = self.config, self.params
        logits, (idx, layer_caches, c_lnf, hf) = self.forward(idx)
        B, T, V = logits.shape
        targets = np.asarray(targets)

        probs = softmax(logits)
        n = B * T
        flat_t = targets.reshape(-1)
        loss = -np.log(probs.reshape(-1, V)[np.arange(n), flat_t] + 1e-12).mean()

        dlogits = probs.reshape(-1, V).copy()
        dlogits[np.arange(n), flat_t] -= 1.0
        dlogits = (dlogits / n).reshape(B, T, V)

        grads = {}
        # tied head: logits = hf @ wte.T
        dhf = dlogits @ P["wte"]
        grads["wte"] = dlogits.reshape(-1, V).T @ hf.reshape(-1, c.d_model)

        dh, grads["lnf_g"], grads["lnf_b"] = layernorm_bwd(dhf, c_lnf)

        for l in reversed(range(c.n_layers)):
            c_ln1, c_att, c_ln2, c_mlp = layer_caches[l]
            # h = h_mid + mlp(ln2(h_mid))
            dx, dW1, db1, dW2, db2 = mlp_bwd(dh, c_mlp)
            dx, dg2, db_ln2 = layernorm_bwd(dx, c_ln2)
            dh = dh + dx
            # h_mid = h_in + attn(ln1(h_in))
            dx, dWqkv, dbqkv, dWo, dbo = attention_bwd(dh, c_att)
            dx, dg1, db_ln1 = layernorm_bwd(dx, c_ln1)
            dh = dh + dx

            grads.update({
                f"h{l}.ln1_g": dg1, f"h{l}.ln1_b": db_ln1,
                f"h{l}.Wqkv": dWqkv, f"h{l}.bqkv": dbqkv,
                f"h{l}.Wo": dWo, f"h{l}.bo": dbo,
                f"h{l}.ln2_g": dg2, f"h{l}.ln2_b": db_ln2,
                f"h{l}.W1": dW1, f"h{l}.b1": db1,
                f"h{l}.W2": dW2, f"h{l}.b2": db2,
            })

        # embeddings
        np.add.at(grads["wte"], idx, dh)
        dwpe = np.zeros_like(P["wpe"])
        dwpe[:T] = dh.sum(axis=0)
        grads["wpe"] = dwpe
        return float(loss), grads

    def loss(self, idx, targets):
        """Loss only (no gradients) -- for evaluation."""
        logits, _ = self.forward(idx)
        B, T, V = logits.shape
        probs = softmax(logits).reshape(-1, V)
        t = np.asarray(targets).reshape(-1)
        return float(-np.log(probs[np.arange(B * T), t] + 1e-12).mean())

    # ---- sampling ----------------------------------------------------------

    def generate(self, prompt_ids, max_new_tokens=100, temperature=1.0,
                 top_k=None, rng=None):
        """Autoregressively extend prompt_ids. Returns prompt + new ids."""
        rng = rng or np.random.default_rng()
        ids = list(prompt_ids) or [0]
        bs = self.config.block_size

        for _ in range(max_new_tokens):
            ctx = np.array([ids[-bs:]])
            logits, _ = self.forward(ctx)
            l = logits[0, -1].astype(np.float64)

            if temperature <= 0:
                nxt = int(np.argmax(l))
            else:
                l = l / temperature
                if top_k is not None and 0 < top_k < len(l):
                    kth = np.partition(l, -top_k)[-top_k]
                    l = np.where(l < kth, -np.inf, l)
                p = softmax(l)
                nxt = int(rng.choice(len(p), p=p))
            ids.append(nxt)
        return ids

    # ---- save / load -------------------------------------------------------

    def save(self, path, tokenizer):
        meta = json.dumps({"config": asdict(self.config), "chars": tokenizer.chars})
        np.savez(path, __meta__=np.array(meta), **self.params)

    @classmethod
    def load(cls, path, dtype=np.float32):
        with np.load(path, allow_pickle=False) as f:
            meta = json.loads(str(f["__meta__"]))
            config = GPTConfig(**meta["config"])
            tokenizer = CharTokenizer(meta["chars"])
            model = cls(config, dtype=dtype)
            for k in model.params:
                model.params[k] = f[k].astype(dtype)
        return model, tokenizer
