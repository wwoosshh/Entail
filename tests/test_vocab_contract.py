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
    # two tokenizer sources of one size, and the engine's tokenizer holds a third: nothing here names its source
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=10, vocab_txt=10, config_vocab=12, rows=12), 7, 7, "t",
                                 record=False))
    assert d.verdict is Verdict.UNKNOWN and "cannot be told" in d.note, d
    # with tokenizer.json as the only tokenizer source it is not counted (the engine built from it; M15.6 cost), so
    # the engine's count stands and the folder's rows are the only thing to compare with
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=10, config_vocab=12, rows=12), 7, 7, "t", record=False))
    assert d.verdict is Verdict.PASS and "the only tokenizer source" in d.declared.source.where, d


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



def test_an_added_token_past_the_rows_is_noted_not_broken():
    """gemma-3-1b-it: base 262,144 = rows, but an added <image_soft_token> sits at id 262,144 (M15.3 review)."""
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=10, config_vocab=10, rows=10), 10, 11, "t", record=False))
    assert d.verdict is Verdict.PASS and "added tokens reach id 10" in d.note and "noted, not broken" in d.note, d


def test_a_base_vocabulary_past_the_rows_is_broken():
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=12, config_vocab=10, rows=10), 12, 12, "t", record=False))
    assert d.verdict is Verdict.BROKEN and d.rule == RULES["vocab_out_of_range"], d


def test_a_position_embedding_under_a_similar_name_does_not_stand_in_for_the_rows():
    """A checkpoint whose first embedding-like tensor is a position embedding (77 rows, CLIP style): the token
    embedding is the one with at least the config's vocabulary of rows (M15.3 review)."""
    d = folder(tokenizer_json=10, config_vocab=10)
    header = json.dumps({"text.position_embedding.weight": {"dtype": "F32", "shape": [7, 4], "data_offsets": [0, 112]},
                         "text.token_embedding.weight": {"dtype": "F32", "shape": [10, 4],
                                                          "data_offsets": [112, 272]}}).encode()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + bytes(272))
    s = vocab_contract.sources(d)
    assert (s.rows, s.rows_verified) == (10, True) and "token_embedding" in s.rows_where, (s.rows, s.rows_where)


def test_two_sources_neither_equal_to_the_model_are_unknown_not_judged():
    """A stale, larger vocab.json beside the model's tokenizer.json, rows equal to neither: no source IS the model's
    vocabulary, so which one the engine should hold is not decided (M15.3 review: no 'largest below the rows')."""
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=9, vocab_json=11, config_vocab=10, rows=10), 9, 9, "t",
                                 record=False))
    assert d.verdict is Verdict.UNKNOWN and "picks none" in d.note, d



def test_rows_read_elsewhere_stand_in_for_the_checkpoint():
    """The M15.7 sweep reads the embedding's rows from a safetensors header fetched without the weights."""
    s = vocab_contract.sources(folder(tokenizer_json=10, config_vocab=10), rows=(12, "embed_tokens [12, 4] header"))
    assert (s.rows, s.rows_verified, s.rows_where) == (12, True, "embed_tokens [12, 4] header")
    d = one(vocab_contract.check(B, C, folder(tokenizer_json=10, config_vocab=10), 10, 10, "t", record=False,
                                 rows=(12, "header")))
    assert d.verdict is Verdict.PASS, d



def test_a_tokenizer_built_from_a_folder_without_tokenizer_files_is_unknown_not_pass():
    """transformers 5.17 builds a Qwen2Tokenizer of one base token from a folder holding only tokenizer_config.json
    (seen in the E1 static run); with no source to name it and a size that is not the model's, it is unknown."""
    d = one(vocab_contract.check(B, C, folder(config_vocab=151936), 1, 26, "Qwen2Tokenizer", record=False))
    assert d.verdict is Verdict.UNKNOWN and "no tokenizer file" in d.note and "not the model's tokenizer" in d.note, d
    # a size that IS the model's passes even without a source (nothing contradicts it)
    d = one(vocab_contract.check(B, C, folder(config_vocab=50257), 50257, 50257, "GPT2Tokenizer", record=False))
    assert d.verdict is Verdict.PASS, d



def test_a_folder_with_only_tokenizer_json_is_not_parsed_and_the_engines_count_stands():
    """S4 (M15.6): parsing 2-33 MB of tokenizer.json three or four times a run took the load share to 6%; with no
    second source there is nothing to compare its count with, so it is not counted."""
    f = folder(tokenizer_json=10, config_vocab=12, rows=12)
    s = vocab_contract.sources(f)
    assert s.candidates == [(None, "tokenizer.json (the only tokenizer source; the engine's own count stands)")], s
    d = one(vocab_contract.check(B, C, f, 10, 10, "t", record=False))
    assert d.verdict is Verdict.PASS and d.declared.value == Vocab(size=10) and "tokenizer.json" in d.declared.source.where
    # with a second source it IS counted
    s = vocab_contract.sources(folder(tokenizer_json=5, vocab_txt=8, config_vocab=8, rows=8))
    assert sorted(s.candidates) == [(5, "tokenizer.json (model.vocab)"), (8, "vocab.txt")], s.candidates


def test_a_folder_is_read_once_per_process_and_again_when_a_file_changes():
    f = folder(tokenizer_json=5, vocab_txt=8, config_vocab=8, rows=8)
    a = vocab_contract.sources(f)
    assert vocab_contract.sources(f) is a                       # the in-process cache
    import time
    time.sleep(1.1)                                              # mtime resolution
    open(os.path.join(f, "vocab.txt"), "a", encoding="utf-8").write("w8\nw9\n")
    b = vocab_contract.sources(f)
    assert b is not a and (10, "vocab.txt") in b.candidates, b.candidates


def test_the_cross_process_cache_is_written_and_read_back():
    import tempfile as _tf
    logs = _tf.mkdtemp()
    os.environ["ENTAIL_LOG_DIR"] = logs
    try:
        vocab_contract._FILE_CACHE = None
        f = folder(tokenizer_json=5, vocab_txt=8, config_vocab=8, rows=8)
        a = vocab_contract.sources(f)
        assert os.path.isfile(os.path.join(logs, vocab_contract.CACHE_NAME))
        vocab_contract._CACHE.clear()
        vocab_contract._FILE_CACHE = None                        # another process: the file, not the memory
        b = vocab_contract.sources(f)
        assert b.candidates == a.candidates and b.rows == a.rows and b is not a
    finally:
        os.environ.pop("ENTAIL_LOG_DIR", None)
        vocab_contract._FILE_CACHE = None
        vocab_contract._CACHE.clear()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
