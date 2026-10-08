# The-BoundedGlitchGPT

A GPT built from scratch in **pure numpy**. No torch, no autograd: every layer has a
hand-written forward and backward pass. The only dependency is `numpy`.

## Layout

    data/               your .txt corpus (all files are used)
    tokenizer/          (reserved) tokenizer lives in model/model.py for now
    model/model.py      config, char tokenizer, layers + backprop, GPT, save/load, sampling
    training/train.py   AdamW, cosine LR, grad clipping, checkpointing
    inference/generate.py   prompt or interactive generation
    tests/              gradient check, causality, save/load
    server/             (reserved) HTTP API, coming next

## Use

    pip install -r requirements.txt
    python tests/test_model.py
    python training/train.py
    python inference/generate.py --prompt "The glitch"

## Architecture

GPT-2 style: token + position embeddings, pre-LayerNorm blocks (causal multi-head
attention + GELU MLP, residual connections), final LayerNorm, output head tied to the
token embedding. Character-level tokenizer.
