"""Learn BPE merges from text.

Start with the 256 byte tokens. Count how often each adjacent pair of tokens occurs, merge the most
frequent pair into a new token, and repeat until the vocabulary has the wanted size.

Counting every pair again after every merge would be far too slow, so the corpus is kept as a table of
distinct pre-tokenized pieces with their counts (a text has few distinct words), and only the pieces that
contain the merged pair are updated. A heap with lazy deletion finds the best pair without scanning.

Ties are broken by the smaller pair of token strings, so the same corpus always gives the same merges.
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict

import regex

from .tokenizer import GPT2_PATTERN, BPETokenizer, bytes_to_unicode


def train(
    text: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    pattern: str = GPT2_PATTERN,
) -> BPETokenizer:
    """Train a tokenizer with `vocab_size` tokens in total, special tokens included."""
    special_tokens = special_tokens or []
    n_merges = vocab_size - 256 - len(special_tokens)
    if n_merges < 0:
        raise ValueError("vocab_size must be at least 256 plus the number of special tokens")

    to_unicode = bytes_to_unicode()
    compiled = regex.compile(pattern)
    for special in special_tokens:  # special tokens are never learned from, and never split
        text = text.replace(special, "\n")

    piece_counts = Counter(
        "".join(to_unicode[b] for b in piece.encode("utf-8")) for piece in compiled.findall(text)
    )
    pieces = [list(p) for p in piece_counts]
    freqs = [piece_counts[p] for p in piece_counts]

    pair_counts: Counter = Counter()
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    for i, symbols in enumerate(pieces):
        for pair in zip(symbols, symbols[1:]):
            pair_counts[pair] += freqs[i]
            where[pair].add(i)

    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[str, str]] = []
    while len(merges) < n_merges and heap:
        negative, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -negative:  # stale entry: the count changed after it was pushed
            continue
        if -negative < 1:
            break
        merges.append(pair)
        first, second = pair
        new_token = first + second

        touched: set[tuple[str, str]] = set()
        for i in list(where[pair]):
            symbols = pieces[i]
            # take this piece's pairs out of the counts, merge, and put the new pairs back
            for old in zip(symbols, symbols[1:]):
                pair_counts[old] -= freqs[i]
                touched.add(old)
            merged: list[str] = []
            j = 0
            while j < len(symbols):
                if j < len(symbols) - 1 and symbols[j] == first and symbols[j + 1] == second:
                    merged.append(new_token)
                    j += 2
                else:
                    merged.append(symbols[j])
                    j += 1
            pieces[i] = merged
            for new in zip(merged, merged[1:]):
                pair_counts[new] += freqs[i]
                where[new].add(i)
                touched.add(new)
        where.pop(pair, None)
        pair_counts.pop(pair, None)
        for t in touched:
            count = pair_counts.get(t, 0)
            if count > 0:
                heapq.heappush(heap, (-count, t))

    first_special_id = 256 + len(merges)
    specials = {s: first_special_id + k for k, s in enumerate(special_tokens)}
    return BPETokenizer(merges, special_tokens=specials, pattern=pattern)


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on a text file.")
    parser.add_argument("text_file")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--out", default="tokenizer")
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = parser.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")
    tokenizer = train(text, args.vocab_size, args.special)
    tokenizer.save(args.out)
    ids = tokenizer.encode(text)
    n_bytes = len(text.encode("utf-8"))
    print(f"vocab {tokenizer.vocab_size}, {len(tokenizer.merges)} merges, {n_bytes} bytes -> {len(ids)} tokens "
          f"({n_bytes / len(ids):.2f} bytes per token)")


if __name__ == "__main__":
    main()
