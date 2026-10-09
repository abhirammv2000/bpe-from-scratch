"""Same ids as tiktoken on GPT-2's vocabulary. Needs network (to fetch the vocabulary) and tiktoken, so it skips without them."""
import sys
from pathlib import Path

import pytest

tiktoken = pytest.importorskip("tiktoken")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_gpt2 import corpus  # noqa: E402

from bpe.gpt2 import load_gpt2  # noqa: E402


@pytest.fixture(scope="module")
def tokenizers():
    try:
        mine = load_gpt2(Path(__file__).resolve().parents[1] / ".cache" / "gpt2")
        reference = tiktoken.get_encoding("gpt2")
    except Exception as error:  # no network, or the files moved
        pytest.skip(f"cannot fetch the GPT-2 vocabulary: {error}")
    return mine, reference


def test_ids_match_tiktoken_on_a_wide_mix_of_text(tokenizers):
    mine, reference = tokenizers
    for text in corpus():
        assert mine.encode(text) == reference.encode(text, allowed_special="all"), text[:80]
        assert mine.decode(mine.encode(text)) == text
