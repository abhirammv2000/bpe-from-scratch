import random

import pytest

from bpe import BPETokenizer, bytes_to_unicode, train
from bpe.tokenizer import base_alphabet

CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 40
    + "Tokenizers turn text into numbers; a tokenizer that merges frequent pairs saves tokens. " * 30
    + "naïve café 日本語のテキスト emoji 🙂🙂 numbers 12345 and 67890.\n" * 20
)


def test_every_byte_has_its_own_printable_character():
    table = bytes_to_unicode()
    assert sorted(table) == list(range(256))
    assert len(set(table.values())) == 256
    assert all(c.isprintable() and not c.isspace() for c in table.values())


def test_base_ids_follow_gpt2s_order():
    alphabet = base_alphabet()
    assert alphabet[0] == "!" and alphabet.index("~") == 93
    assert alphabet.index(bytes_to_unicode()[ord(" ")]) == 220  # the id GPT-2 gives to a lone space
    assert len(set(alphabet)) == 256


def test_the_classic_example_learns_the_classic_merges():
    """'aaabdaaabac' is the standard worked example: aa first, then ab, then the two together."""
    tokenizer = train("aaabdaaabac", 256 + 3, pattern=r"\w+")
    assert tokenizer.merges == [("a", "a"), ("a", "b"), ("aa", "ab")]
    assert tokenizer.encode("aaabdaaabac") == [tokenizer.vocab["aaab"], tokenizer.vocab["d"], tokenizer.vocab["aaab"],
                                                tokenizer.vocab["a"], tokenizer.vocab["c"]]


def test_training_is_deterministic():
    a = train(CORPUS, 400)
    b = train(CORPUS, 400)
    assert a.merges == b.merges


def rich_corpus():
    rng = random.Random(5)
    syllables = ["ka", "to", "mi", "ra", "ne", "so", "lu", "vi", "pe", "dan", "gor", "thy", "qua", "zel", "bri"]
    words = ["".join(rng.choice(syllables) for _ in range(rng.randrange(1, 4))) for _ in range(400)]
    return " ".join(rng.choice(words) for _ in range(6000))


def test_vocabulary_has_exactly_the_requested_size():
    tokenizer = train(rich_corpus(), 420, special_tokens=["<|endoftext|>"])
    assert tokenizer.vocab_size == 420
    ids = sorted(tokenizer.vocab.values()) + sorted(tokenizer.special_tokens.values())
    assert ids == list(range(420))


def test_a_corpus_that_runs_out_of_pairs_gives_a_smaller_vocabulary():
    """Once every piece is a single token there is nothing left to merge, so the vocabulary stops growing."""
    tokenizer = train("ab ab ab", 1000)
    assert tokenizer.vocab_size < 1000
    assert len(tokenizer.encode("ab ab ab")) == 3  # "ab", " ab", " ab"


def test_vocab_too_small_is_refused():
    with pytest.raises(ValueError):
        train(CORPUS, 200)


def test_more_merges_never_need_more_tokens_on_the_training_text():
    lengths = [len(train(CORPUS, size).encode(CORPUS)) for size in (256, 300, 400, 600)]
    assert lengths == sorted(lengths, reverse=True)
    assert lengths[0] == len(CORPUS.encode("utf-8"))  # no merges: one token per byte
    assert lengths[-1] < lengths[0] / 2


def random_text(rng, length):
    alphabet = list("abcdefghij KLMNO 0123456789 .,;!?'\"\n\t  ") + list("éüñßçø日本語中文한국어Ωδπ🙂🔥👍🏽é") + ["́", "​", "\x00"]
    return "".join(rng.choice(alphabet) for _ in range(length))


def test_decode_inverts_encode_for_arbitrary_text():
    tokenizer = train(CORPUS, 500)
    rng = random.Random(7)
    for _ in range(300):
        text = random_text(rng, rng.randrange(0, 80))
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_empty_and_whitespace_only_text():
    tokenizer = train(CORPUS, 300)
    assert tokenizer.encode("") == []
    for text in (" ", "   ", "\n\n", "\t \n"):
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_a_token_cut_in_the_middle_of_a_character_decodes_to_the_replacement_character():
    tokenizer = train("x", 256)  # only byte tokens
    ids = tokenizer.encode("é")  # two bytes, so two tokens
    assert len(ids) == 2
    assert tokenizer.decode(ids[:1]) == "�"
    assert tokenizer.decode_bytes(ids[:1]) == "é".encode()[:1]


def test_special_tokens_become_one_id_and_can_be_turned_off():
    tokenizer = train(CORPUS, 300, special_tokens=["<|endoftext|>"])
    end = tokenizer.special_tokens["<|endoftext|>"]
    ids = tokenizer.encode("the end<|endoftext|>the start")
    assert ids.count(end) == 1
    assert tokenizer.decode(ids) == "the end<|endoftext|>the start"
    as_text = tokenizer.encode("the end<|endoftext|>", allowed_special=set())
    assert end not in as_text
    assert tokenizer.decode(as_text) == "the end<|endoftext|>"


def test_special_tokens_are_not_learned_from_the_training_text():
    tokenizer = train("<|endoftext|>" * 500 + "hello world " * 20, 300, special_tokens=["<|endoftext|>"])
    assert not any("<|" in a + b for a, b in tokenizer.merges)


def test_save_and_load_give_the_same_tokens(tmp_path):
    tokenizer = train(CORPUS, 450, special_tokens=["<|endoftext|>"])
    tokenizer.save(tmp_path)
    loaded = BPETokenizer.load(tmp_path)
    rng = random.Random(3)
    for text in [CORPUS[:500], "<|endoftext|>", ""] + [random_text(rng, 60) for _ in range(40)]:
        assert loaded.encode(text) == tokenizer.encode(text)


def test_merges_apply_by_rank_not_by_position():
    # "ab" outranks "bc", so "abc" is [ab, c], not [a, bc]
    tokenizer = BPETokenizer([("a", "b"), ("b", "c")])
    assert tokenizer.decode(tokenizer.encode("abc")) == "abc"
    assert [tokenizer.id_to_token[i] for i in tokenizer.encode("abc")] == ["ab", "c"]
    assert [tokenizer.id_to_token[i] for i in tokenizer.encode("bcab")] == ["bc", "ab"]


def test_overlapping_pairs_merge_left_to_right():
    tokenizer = BPETokenizer([("a", "a")])
    assert [tokenizer.id_to_token[i] for i in tokenizer.encode("aaa")] == ["aa", "a"]
    assert [tokenizer.id_to_token[i] for i in tokenizer.encode("aaaa")] == ["aa", "aa"]
