"""Tests for the vocabulary contract in the core (ROADMAP M15.3; transformers#48967): what a folder declares about
its vocabulary, the two rules, and what ordinary folders look like. Pure Python: temp folders, a hand-written
safetensors header, no torch, no engine. Run: python tests/test_vocab_contract.py"""
import io
import json
import os
import struct
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, vocab_contract  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.facts import Vocab  # noqa: E402

B, C = "load:test.tokenizer", "test.tokenizer"


def folder(tokenizer_json=None, vocab_txt=None, vocab_json=None, config_vocab=None, rows=None, spm_pieces=None):
    """A model folder: tokenizer.json with n base tokens, vocab.txt with n lines, vocab.json with n entries,
    config.json vocab_size, and a safetensors embedding with `rows` rows (header only, zero data)."""
    d = tempfile.mkdtemp()
    if tokenizer_json is not None:
        json.dump({"model": {"type": "WordPiece", "vocab": {f"t{i}": i for i in range(tokenizer_json)}},
                   "added_tokens": []}, open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8"))
    if vocab_txt is not None:
        open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8").write("".join(f"w{i}\n" for i in range(vocab_txt)))
    if vocab_json is not None:
        json.dump({f"b{i}": i for i in range(vocab_json)}, open(os.path.join(d, "vocab.json"), "w", encoding="utf-8"))
    if config_vocab is not None:
        json.dump({"model_type": "bert", "vocab_size": config_vocab}, open(os.path.join(d, "config.json"), "w",
                                                                              encoding="utf-8"))
    if rows is not None:
        header = json.dumps({"bert.embeddings.word_embeddings.weight": {"dtype": "F32", "shape": [rows, 4],
                                                                        "data_offsets": [0, rows * 16]}}).encode()
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(header)) + header + b"\0" * (rows * 16))
    if spm_pieces is not None:
        # a ModelProto with `spm_pieces` pieces (field 1, length-delimited) and a trainer_spec (field 2)
        piece = b"\x0a\x02\x0a\x00"            # field 1, len 2: {field 1 (piece) len 0}
        open(os.path.join(d, "tokenizer.model"), "wb").write(piece * spm_pieces + b"\x12\x00")
    return d


def one(decisions):
    assert len(decisions) == 1, decisions
    return decisions[0]


def test_the_fact_needs_a_positive_size():
    Vocab(size=3)
    try:
        Vocab(size=0)
    except ValueError as e:
        assert "Vocab.size" in str(e)
    else:
        raise AssertionError("size 0 must be refused")


def test_what_a_folder_declares_is_read_from_every_source():
    s = vocab_contract.sources(folder(tokenizer_json=5, vocab_txt=8, vocab_json=6, config_vocab=8, rows=8,
                                      spm_pieces=3))
    assert sorted(s.candidates) == [(3, "tokenizer.model (sentencepiece pieces)"), (5, "tokenizer.json (model.vocab)"),
                                    (6, "vocab.json"), (8, "vocab.txt")], s.candidates
    assert (s.rows, s.rows_verified) == (8, True) and "word_embeddings" in s.rows_where


def test_without_a_checkpoint_the_config_stands_in_for_the_rows():
    s = vocab_contract.sources(folder(tokenizer_json=5, config_vocab=9))
    assert (s.rows, s.rows_verified, s.rows_where) == (9, False, "config.json vocab_size")


def test_an_ordinary_folder_passes_even_with_a_padded_embedding():
    """Qwen-like: 12 rows for 10 tokens; one tokenizer source; the engine holds it."""
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=10, config_vocab=12, rows=12), 10, 10, "t", record=False))
    assert d.verdict is Verdict.PASS and d.rule == RULES["match"], d


def test_the_model_case_two_vocabularies_and_the_engine_built_the_wrong_one():
    """transformers#48967: vocab.txt 8 (the model's, = config and rows), tokenizer.json 5 (another model's)."""
    f = folder(tokenizer_json=5, vocab_txt=8, config_vocab=8, rows=8)
    d = one(vocab_contract.check(B, C, f, 5, 5, "BertTokenizer", record=False))
    assert d.verdict is Verdict.BROKEN and d.rule == RULES["vocab_not_the_models"], d
    assert "vocab.txt is the model's tokenizer" in d.note and d.declared.value == Vocab(size=8), d.note
    ok = one(vocab_contract.check(B, C, f, 8, 8, "BertTokenizer", record=False))
    assert ok.verdict is Verdict.PASS and "another vocabulary" in ok.note, ok


def test_ids_past_the_embedding_are_broken():
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=10, config_vocab=8, rows=8), 10, 10, "t", record=False))
    assert d.verdict is Verdict.BROKEN and d.rule == RULES["vocab_out_of_range"], d


def test_a_tokenizer_from_nowhere_in_the_folder_is_unknown_not_broken():
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=10, config_vocab=12, rows=12), 7, 7, "t", record=False))
    assert d.verdict is Verdict.UNKNOWN and "cannot be told" in d.note, d


def test_nothing_to_compare_with_is_unknown():
    d = one(vocab_contract.check(B, C, tempfile.mkdtemp(), 10, 10, "t", record=False))
    assert d.verdict is Verdict.UNKNOWN, d


def test_recording_goes_through_the_ledger_and_stops_only_where_the_policy_says():
    from entail.core import RoleError
    f = folder(tokenizer_json=5, vocab_txt=8, config_vocab=8, rows=8)
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            vocab_contract.check(B, C, f, 5, 5, "t")
    finally:
        core.set_mode("off")
    d = load.LEDGER.decisions[n:]
    assert len(d) == 1 and d[0].verdict is Verdict.BROKEN, d
    os.environ["ENTAIL_ON_BROKEN"] = "stop"
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            vocab_contract.check(B, C, f, 5, 5, "t")
    except RoleError as e:
        assert "vocab.txt" in str(e)
    else:
        raise AssertionError("the strict policy refuses the tokenizer that is not the model's")
    finally:
        os.environ.pop("ENTAIL_ON_BROKEN", None)
        core.set_mode("off")
    vocab_contract.reset(B)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
