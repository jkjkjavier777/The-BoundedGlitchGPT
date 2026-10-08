"""Clean raw text files from data/raw/ into data/general_*.txt.

- strips Project Gutenberg headers/footers
- normalizes fancy punctuation to plain ASCII
- collapses extra blank lines
"""
import re
import sys
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data")

REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": " - ", "\u2026": "...", "\ufeff": "",
    "\u00a0": " ",
}


def strip_gutenberg(text):
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    if start:
        text = text[start.end():]
    if end:
        text = text[: end.start() - (start.end() if start else 0)] if False else text.split(end.group(0))[0]
    return text


def clean(text):
    for a, b in REPLACEMENTS.items():
        text = text.replace(a, b)
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # keep only printable ASCII + newline so the vocab stays small
    text = "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)
    return text.strip()


def main():
    files = sorted(RAW.glob("*.txt"))
    if not files:
        sys.exit("No .txt files in data/raw/")
    total = 0
    for p in files:
        text = clean(strip_gutenberg(p.read_text(encoding="utf-8", errors="ignore")))
        out = OUT / f"general_{p.stem}.txt"
        out.write_text(text, encoding="utf-8")
        total += len(text)
        print(f"{p.name}: {len(text):,} chars -> {out}")
    print(f"Total general text: {total:,} characters")


if __name__ == "__main__":
    main()
