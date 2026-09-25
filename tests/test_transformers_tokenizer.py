"""Tests for the transformers tokenizer adapter (ROADMAP M15.3; transformers#48967): what it reads from a built
tokenizer and how it decides on a folder, on fakes - no transformers. The rules are tested in
test_vocab_contract.py; the case end to end in testbed/m10_e3/tf48967.py.
Run: python tests/test_transformers_tokenizer.py"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load  # noqa: E402
from entail.adapters import transformers_tokenizer  # noqa: E402
from entail.contracts import Verdict  # noqa: E402


class FakeTokenizer:
    def __init__(self, size, added=0):
        self.vocab_size = size
        self._n = size + added

    def __len__(self):
        return self._n


def folder(tokenizer_json, vocab_txt=None, config_vocab=None):
    d = tempfile.mkdtemp()
    json.dump({"model": {"type": "WordPiece", "vocab": {f"t{i}": i for i in range(tokenizer_json)}}},
              open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8"))
    if vocab_txt is not None:
        open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8").write("".join(f"w{i}\n" for i in range(vocab_txt)))
    if config_vocab is not None:
        json.dump({"vocab_size": config_vocab}, open(os.path.join(d, "config.json"), "w", encoding="utf-8"))
    return d


def decide(name, tok):
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            transformers_tokenizer._decide(name, {}, tok)
    finally:
        core.set_mode("off")
    return load.LEDGER.decisions[n:]


def test_read_choice_gives_the_base_size_and_the_length():
    assert transformers_tokenizer.read_choice(FakeTokenizer(100, 3)) == (100, 103)
    assert transformers_tokenizer.read_choice(object()) == (None, None)


def test_a_local_folder_with_the_wrong_tokenizer_is_reported_before_the_first_id():
    d = decide(folder(5, vocab_txt=8, config_vocab=8), FakeTokenizer(5))
    assert len(d) == 1 and d[0].verdict is Verdict.BROKEN and "vocab.txt" in d[0].note, d
    transformers_tokenizer.reset()


def test_a_name_that_is_not_local_is_not_checked_and_says_so():
    d = decide("no-such-org/no-such-model-entail-test", FakeTokenizer(5))
    assert len(d) == 1 and d[0].verdict is Verdict.UNKNOWN and "no local folder" in d[0].note, d
    transformers_tokenizer.reset()


def test_an_ordinary_folder_is_quiet():
    d = decide(folder(10, config_vocab=12), FakeTokenizer(10))
    assert all(x.verdict is Verdict.PASS for x in d), d
    transformers_tokenizer.reset()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
