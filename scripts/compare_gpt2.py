"""Check this tokenizer against OpenAI's tiktoken, using GPT-2's own vocabulary on both sides.

If the algorithm here is right, the ids must be identical on any input. The corpus is a mix of English
prose, source code, several languages, emoji, odd whitespace and random unicode, because the places
tokenizers disagree are the unusual ones.

    python scripts/compare_gpt2.py [--time]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bpe.gpt2 import load_gpt2  # noqa: E402


def corpus() -> list[str]:
    texts: list[str] = []
    root = Path(sys.prefix) / "Lib" if sys.platform == "win32" else Path(sys.prefix) / "lib"
    for pattern in ("**/LICENSE*", "**/README*", "**/*.rst", "**/*.md", "**/*.py"):
        for path in sorted(root.glob(pattern))[:120]:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if 200 < len(text) < 200_000:
                texts.append(text)
    texts += [
        "Hello, world!  Two  spaces,\ttab,\r\nCRLF, trailing space \n\n\n",
        "I'm sure we'll be fine; they've said it's what you'd expect. DON'T SHOUT.",
        "日本語のテキスト、中文文本、한국어 텍스트, Ελληνικά, русский текст, العربية, עברית",
        "emoji 🙂🔥👍🏽 flags 🇯🇵 family 👨‍👩‍👧‍👦 combining é é zero​width",
        "numbers 1234567890 3.14159 1,000,000 0xDEADBEEF 2026-10-09T12:00:00Z",
        "<|endoftext|> in the middle of text <|endoftext|>",
        "",
        " ",
        "\n",
        "a" * 5000,
        "ab" * 3000,
    ]
    rng = random.Random(1)
    alphabet = [chr(c) for c in list(range(32, 127)) + list(range(0xA0, 0x250)) + list(range(0x3040, 0x3100)) + list(range(0x1F300, 0x1F400))]
    alphabet += ["\n", "\t", " ", "  ", "​", "́"]
    for _ in range(400):
        texts.append("".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 300))))
    return texts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", action="store_true", help="also compare speed")
    args = parser.parse_args()

    mine = load_gpt2()
    reference = tiktoken.get_encoding("gpt2")
    texts = corpus()

    mismatches = 0
    total_tokens = 0
    for text in texts:
        expected = reference.encode(text, allowed_special="all")
        got = mine.encode(text)
        total_tokens += len(expected)
        if got != expected:
            mismatches += 1
            if mismatches <= 3:
                print(f"MISMATCH on {text[:60]!r}: {got[:12]} vs {expected[:12]}")
        elif mine.decode(got) != text:
            mismatches += 1
            print(f"ROUND TRIP FAILED on {text[:60]!r}")
    chars = sum(len(t) for t in texts)
    print(f"{len(texts)} texts, {chars:,} characters, {total_tokens:,} tokens: {mismatches} differences")

    if args.time:
        sample = [t for t in texts if len(t) > 2000][:60]
        size = sum(len(t) for t in sample)
        for name, fn in (("this tokenizer", mine.encode), ("tiktoken", reference.encode)):
            start = time.perf_counter()
            for t in sample:
                fn(t)
            seconds = time.perf_counter() - start
            print(f"{name:15s} {size / seconds / 1e6:6.2f} million characters per second (cold cache)")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
