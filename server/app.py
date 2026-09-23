#!/usr/bin/env python3
"""
Minimal HTTP server wrapping BoundedGlitchGPT for engine/boundedGlitchGPT.js.

Exposes exactly the contract the JS bridge expects:
    GET  /health            -> 200 {"status": "ok"}
    POST /generate           body: {"prompt": str, "max_tokens": int}
                              -> 200 {"text": str}

Usage:
    python server/app.py --checkpoint checkpoints/model.pt --port 8000

Stdlib only (no Flask/FastAPI dependency) so it needs nothing beyond
what inference/generate.py already needs (torch).
"""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inference.generate import TextGenerator

GENERATOR: TextGenerator = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/generate":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._send_json(400, {"error": "prompt (non-empty string) is required"})
            return
        max_tokens = int(body.get("max_tokens", 100))
        temperature = float(body.get("temperature", 1.0))
        top_k = body.get("top_k")

        try:
            full_text = GENERATOR.generate(
                prompt, num_tokens=max_tokens, temperature=temperature, top_k=top_k
            )
            # generate() returns prompt + continuation decoded together; since this
            # is a character-level tokenizer (1 token == 1 char), slicing by
            # len(prompt) exactly recovers just the new continuation.
            continuation = full_text[len(prompt):]
            self._send_json(200, {"text": continuation})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, fmt, *args):
        print(f"[server] {self.address_string()} - {fmt % args}")


def main():
    p = argparse.ArgumentParser(description="Serve BoundedGlitchGPT over HTTP")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--device", type=str, default="cuda" if _cuda_available() else "cpu")
    args = p.parse_args()

    global GENERATOR
    GENERATOR = TextGenerator(args.checkpoint, device=args.device)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server] Listening on http://{args.host}:{args.port}")
    print(f"[server] GET  /health")
    print(f"[server] POST /generate  {{'prompt': str, 'max_tokens': int}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down")
        server.shutdown()


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
