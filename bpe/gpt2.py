"""Load OpenAI's published GPT-2 vocabulary into this tokenizer, to check it against the real thing.

The two files are the ones GPT-2 shipped: encoder.json (token string to id) and vocab.bpe (the 50,000
merges, one per line, in the order they were learned). They are downloaded once and kept in a cache folder.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from .tokenizer import BPETokenizer

BASE_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/"
END_OF_TEXT = "<|endoftext|>"


def _fetch(name: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def load_gpt2(cache: str | Path = ".cache/gpt2") -> BPETokenizer:
    cache = Path(cache)
    vocab = json.loads(_fetch("encoder.json", cache).read_text(encoding="utf-8"))
    special_id = vocab.pop(END_OF_TEXT)
    merges = []
    for line in _fetch("vocab.bpe", cache).read_text(encoding="utf-8").splitlines()[1:]:
        if line:
            a, b = line.split(" ")
            merges.append((a, b))
    return BPETokenizer(merges, vocab, {END_OF_TEXT: special_id})
