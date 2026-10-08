"""
The-BoundedGlitchGPT -- server.py
Endpoints the BoSK bridge expects:
    GET  /health
    POST /generate   {"prompt": "...", "max_tokens": 150} -> {"text": "..."}

    python server/server.py --port 8000
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model.model import GPT

LOCK = threading.Lock()
STATE = {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            m = STATE["model"]
            self._send(200, {"status": "ok", "params": m.num_params(),
                             "context": m.config.block_size})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            prompt = data.get("prompt")
            max_tokens = max(1, min(int(data.get("max_tokens", 150)), 1000))
            temperature = float(data.get("temperature", 0.8))
            top_k = data.get("top_k", 20)
        except (ValueError, TypeError):
            return self._send(400, {"error": "bad request"})
        if not isinstance(prompt, str) or not prompt.strip():
            return self._send(400, {"error": "prompt must be a non-empty string"})

        model, tok = STATE["model"], STATE["tok"]
        ids = tok.encode(prompt)
        with LOCK:
            out = model.generate(ids, max_new_tokens=max_tokens, temperature=temperature,
                                 top_k=top_k, rng=np.random.default_rng())
        self._send(200, {"text": tok.decode(out[max(len(ids), 1):])})

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model.npz"))
    args = ap.parse_args()

    if not Path(args.checkpoint).exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    STATE["model"], STATE["tok"] = GPT.load(args.checkpoint)
    print(f"Serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
