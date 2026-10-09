# bpe-from-scratch

A byte-level BPE tokenizer written from scratch in Python: training, encoding, decoding, and loading
GPT-2's published vocabulary. About 330 lines with comments, no tokenizer library.

It is checked against the real thing: loaded with GPT-2's 50,000 merges, it produces the same ids as
OpenAI's `tiktoken` on 534 texts (1.8 million characters, 917,000 tokens: prose, source code, several
scripts, emoji, odd whitespace, `<|endoftext|>`, random unicode). `scripts/compare_gpt2.py` runs that check.

```python
from bpe import train

tokenizer = train(open("data.txt", encoding="utf-8").read(), vocab_size=2048, special_tokens=["<|endoftext|>"])
ids = tokenizer.encode("Hello, world! 🙂")
tokenizer.decode(ids)          # 'Hello, world! 🙂'
tokenizer.save("my-tokenizer")  # vocab.json and merges.txt, the same layout GPT-2 uses
```

## How it works

1. Text is cut into pieces with GPT-2's regular expression (words with one leading space, numbers, symbols, whitespace).
2. Each piece is turned into bytes, and each byte is shown as one printable character (GPT-2's table), so
   any text can be encoded and decoded exactly, and ids cut in the middle of a character decode without crashing.
3. **Training:** count adjacent token pairs over the distinct pieces, merge the most frequent pair into a
   new token, repeat. Only the pieces that contain the merged pair are updated, and a heap with lazy
   deletion finds the next pair, so training on 1 MB of text took 1 to 3 seconds here. Ties go to the smaller
   pair, so the same text always gives the same merges.
4. **Encoding:** inside a piece, repeatedly merge the adjacent pair that was learned earliest.

## Measured

Trained on the first 90% of Tiny Shakespeare (1.1 MB) and measured on the last 10%:

| vocabulary | bytes per token on held-out text |
|---|---|
| 512 | 1.88 |
| 1,024 | 2.26 |
| 2,048 | 2.56 |
| 4,096 | 2.90 |
| GPT-2 (50,257, trained on far more text) | 3.09 |

A small tokenizer trained on in-domain text gets close to GPT-2's compression with a vocabulary 12 times
smaller. That says nothing about which is better for a language model; a bigger vocabulary also has more
embeddings to learn.

Speed is the weak point. This is plain Python and encodes about 1.1 million characters per second, against
4.4 million for tiktoken in the same cold-cache run on my laptop. tiktoken is Rust.

## Limits

- Training stops early if the text runs out of pairs to merge, so a tiny corpus gives a smaller vocabulary than requested.
- The pre-tokenizer is GPT-2's. Other models use different patterns (the pattern is a parameter, but only GPT-2's is tested against a reference).
- Special tokens are exact string matches; there is no chat-template handling.

## Run it

```
pip install regex tiktoken pytest
python -m pytest            # 17 tests (one needs the network, to fetch the GPT-2 vocabulary)
python scripts/compare_gpt2.py --time
python -m bpe.train data.txt --vocab-size 2048 --out my-tokenizer
```

Tiny Shakespeare is from Karpathy's char-rnn repository. MIT licence.
