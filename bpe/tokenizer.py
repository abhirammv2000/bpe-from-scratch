"""A byte-level BPE tokenizer in the GPT-2 style: encode, decode, save and load.

The tokenizer works on bytes, so any input (any language, emoji, broken UTF-8) can be encoded and
decoded back exactly. Text is first cut into pieces by a regular expression, and merges are applied
inside each piece only, so a token never spans a space-then-word boundary the way the pattern forbids.

Token strings use GPT-2's trick: each of the 256 byte values is shown as one printable unicode
character (`bytes_to_unicode`), so vocab and merge files are plain text and the same files that GPT-2
ships can be loaded here. `scripts/compare_gpt2.py` does that and checks the output against tiktoken.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import regex

# GPT-2's pre-tokenization pattern: contractions, then runs of letters, digits, other symbols (each
# optionally with one leading space), then whitespace.
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a printable character, the way GPT-2 does.

    Bytes that are already printable (and not a space or control character) map to themselves. The
    rest are shifted to unused code points starting at 256, so no token string contains whitespace
    or a control character.
    """
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = keep[:]
    shift = 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + shift)
            shift += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def base_alphabet() -> list[str]:
    """The 256 single-byte tokens in GPT-2's id order (which is not plain byte order)."""
    table = bytes_to_unicode()
    return [table[b] for b in sorted(table, key=lambda b: ord(table[b]))]


class BPETokenizer:
    def __init__(
        self,
        merges: list[tuple[str, str]],
        vocab: dict[str, int] | None = None,
        special_tokens: dict[str, int] | None = None,
        pattern: str = GPT2_PATTERN,
    ):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        if vocab is None:
            vocab = {tok: i for i, tok in enumerate(base_alphabet())}
            for a, b in self.merges:
                vocab[a + b] = len(vocab)
        self.vocab = vocab
        self.id_to_token = {i: t for t, i in vocab.items()}
        self.special_tokens = dict(special_tokens or {})
        self.id_to_special = {i: t for t, i in self.special_tokens.items()}
        self._pattern_text = pattern
        self._pattern = regex.compile(pattern)
        self._special_pattern = (
            regex.compile("(" + "|".join(regex.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")")
            if self.special_tokens
            else None
        )
        self._cache: dict[str, tuple[str, ...]] = {}
        self._to_unicode = bytes_to_unicode()
        self._to_byte = unicode_to_bytes()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    # ---- encoding ----------------------------------------------------------------------------

    def _merge_piece(self, piece: str) -> tuple[str, ...]:
        """Apply the merges to one pre-tokenized piece (already in the byte-unicode alphabet)."""
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        symbols = list(piece)
        while len(symbols) > 1:
            # the pair with the lowest rank (earliest learned merge) goes first
            best = min(zip(symbols, symbols[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            first, second = best
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        result = tuple(symbols)
        if len(self._cache) < 100_000:
            self._cache[piece] = result
        return result

    def _encode_ordinary(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._pattern.findall(text):
            mapped = "".join(self._to_unicode[b] for b in piece.encode("utf-8"))
            ids.extend(self.vocab[token] for token in self._merge_piece(mapped))
        return ids

    def encode(self, text: str, allowed_special: set[str] | str = "all") -> list[int]:
        """Text to token ids. Special tokens in the text become their single id when allowed."""
        if not self._special_pattern:
            return self._encode_ordinary(text)
        allowed = set(self.special_tokens) if allowed_special == "all" else set(allowed_special)
        ids: list[int] = []
        for part in self._special_pattern.split(text):
            if part in allowed and part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self._encode_ordinary(part))
        return ids

    # ---- decoding ----------------------------------------------------------------------------

    def decode_bytes(self, ids: list[int]) -> bytes:
        out = bytearray()
        for i in ids:
            if i in self.id_to_special:
                out += self.id_to_special[i].encode("utf-8")
            else:
                out += bytes(self._to_byte[c] for c in self.id_to_token[i])
        return bytes(out)

    def decode(self, ids: list[int]) -> str:
        """Ids back to text. Bytes that are not valid UTF-8 (a cut in the middle of a character) become U+FFFD."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # ---- files -------------------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Write vocab.json and merges.txt, in the layout GPT-2 uses."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8")
        lines = ["#version: 0.2"] + [f"{a} {b}" for a, b in self.merges]
        (directory / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self.special_tokens:
            (directory / "special_tokens.json").write_text(json.dumps(self.special_tokens), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, pattern: str = GPT2_PATTERN) -> "BPETokenizer":
        directory = Path(directory)
        vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        merges = []
        for line in (directory / "merges.txt").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#version"):
                continue
            a, b = line.split(" ")
            merges.append((a, b))
        special_path = directory / "special_tokens.json"
        special = json.loads(special_path.read_text(encoding="utf-8")) if special_path.exists() else {}
        return cls(merges, vocab, special, pattern)
