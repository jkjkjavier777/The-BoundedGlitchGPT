#!/usr/bin/env python3

"""
BoundedGlitchGPT CLI Bot

This bot is the client/runtime layer.
The actual GPT model runs in server/app.py.

Architecture:

    bot.py
       ↓
    HTTP POST /generate
       ↓
    server/app.py
       ↓
    inference/generate.py
       ↓
    model/model.py
       ↓
    trained checkpoint
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_SERVER = os.environ.get(
    "BOUNDED_GLITCHGPT_URL",
    "http://127.0.0.1:8000"
)

DEFAULT_MAX_TOKENS = int(
    os.environ.get("BOUNDED_GLITCHGPT_MAX_TOKENS", "150")
)

DEFAULT_TEMPERATURE = float(
    os.environ.get("BOUNDED_GLITCHGPT_TEMPERATURE", "0.8")
)


class BoundedGlitchBot:
    """CLI client for the BoundedGlitchGPT inference server."""

    def __init__(
        self,
        server_url=DEFAULT_SERVER,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    ):
        self.server_url = server_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

    def health(self):
        """Check whether the GPT inference server is running."""

        url = f"{self.server_url}/health"

        try:
            request = urllib.request.Request(
                url,
                method="GET",
                headers={"Accept": "application/json"},
            )

            with urllib.request.urlopen(request, timeout=5) as response:
                data = response.read().decode("utf-8")

            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return {"status": "ok", "raw": data}

        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
            }

    def generate(
        self,
        prompt,
        max_tokens=None,
        temperature=None,
    ):
        """Send a prompt to BoundedGlitchGPT."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if temperature is not None:
            payload["temperature"] = temperature
        else:
            payload["temperature"] = self.temperature

        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self.server_url}/generate",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                response_body = response.read().decode("utf-8")

        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")

            raise RuntimeError(
                f"GPT server returned HTTP {exc.code}: {error_body}"
            ) from exc

        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not connect to BoundedGlitchGPT at "
                f"{self.server_url}. "
                f"Make sure server/app.py is running."
            ) from exc

        except TimeoutError as exc:
            raise RuntimeError(
                "Request to BoundedGlitchGPT timed out."
            ) from exc

        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"GPT server returned invalid JSON:\n{response_body}"
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                "GPT server returned an invalid response."
            )

        if "error" in data:
            raise RuntimeError(str(data["error"]))

        text = data.get("text")

        if not isinstance(text, str):
            raise RuntimeError(
                f"GPT server response did not contain text:\n{data}"
            )

        return text.strip()

    def interactive(self):
        """Run the terminal chat interface."""

        print()
        print("=" * 60)
        print("THE-BOUNDEDGLITCHGPT")
        print("=" * 60)
        print(f"Server: {self.server_url}")
        print()
        print("Commands:")
        print("  /health       Check GPT server")
        print("  /tokens N     Set maximum generated tokens")
        print("  /temp X       Set generation temperature")
        print("  /quit         Exit")
        print()

        status = self.health()

        if status.get("status") == "error":
            print("[!] GPT server is not available.")
            print(f"    {status.get('error')}")
            print()
            print("Start it with:")
            print("    python server/app.py")
            print()

        else:
            print("[✓] BoundedGlitchGPT server connected.")
            print()

        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                command = user_input.lower()

                if command in {"/quit", "/exit", "quit", "exit"}:
                    print("[*] Goodbye.")
                    break

                if command == "/health":
                    print(json.dumps(self.health(), indent=2))
                    print()
                    continue

                if command.startswith("/tokens "):
                    try:
                        value = int(user_input.split(maxsplit=1)[1])

                        if value <= 0:
                            raise ValueError

                        self.max_tokens = value
                        print(f"[*] max_tokens = {value}")
                    except ValueError:
                        print("[!] Usage: /tokens 150")

                    continue

                if command.startswith("/temp "):
                    try:
                        value = float(user_input.split(maxsplit=1)[1])

                        if value <= 0:
                            raise ValueError

                        self.temperature = value
                        print(f"[*] temperature = {value}")
                    except ValueError:
                        print("[!] Usage: /temp 0.8")

                    continue

                print()
                print("GPT: ", end="", flush=True)

                response = self.generate(
                    user_input,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                print(response)
                print()

            except KeyboardInterrupt:
                print("\n[*] Goodbye.")
                break

            except EOFError:
                print("\n[*] Goodbye.")
                break

            except Exception as exc:
                print(f"\n[ERROR] {exc}\n")


def main():
    parser = argparse.ArgumentParser(
        description="BoundedGlitchGPT terminal client"
    )

    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help="BoundedGlitchGPT server URL",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum generated tokens",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Generation temperature",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Generate one response and exit",
    )

    args = parser.parse_args()

    bot = BoundedGlitchBot(
        server_url=args.server,
        max_tokens=args.tokens,
        temperature=args.temperature,
    )

    if args.prompt:
        try:
            print(
                bot.generate(
                    args.prompt,
                    max_tokens=args.tokens,
                    temperature=args.temperature,
                )
            )
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            sys.exit(1)

        return

    bot.interactive()


if __name__ == "__main__":
    main()
