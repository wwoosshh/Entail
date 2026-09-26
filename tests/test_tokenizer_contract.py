"""Tests for the tokenizer contract in the core (ROADMAP M18.1; LIBRARY_DESIGN.md 11 M18): the folder's declared
tokenizer, run on the fixed probe texts, against the tokenizer the engine built. Pure Python with the tokenizers
library (a dependency of transformers); no torch, no engine. Run: python tests/test_tokenizer_contract.py"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
LOGS = tempfile.mkdtemp()
os.environ["ENTAIL_LOG_DIR"] = LOGS          # the machine cache (tokenizer_ids.json) goes to a temp folder
from entail import core, tokenizer_contract  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.core import RoleError  # noqa: E402
from entail.facts import Tokenization  # noqa: E402

try:
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, trainers
except ImportError:   # the CI venv installs transformers, which brings tokenizers; elsewhere the tests skip
    Tokenizer = None

B, C = "load:test.tokenizer", "test.tokenizer"
PROBES = tokenizer_contract.PROBES


def folder(added=("<think>", "</think>"), config=True, files=("tokenizer.json",)):
    """A model folder with a real tokenizer.json (word level, whitespace pre-tokenizer, trained on the probes) and
    tokenizer_config.json declaring its added tokens; or another set of files."""
    d = tempfile.mkdtemp()
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(list(PROBES) + ["extra words here"], trainers.WordLevelTrainer(special_tokens=["[UNK]"]))
    tok.add_tokens([AddedToken(t, normalized=False) for t in added])
    if "tokenizer.json" in files:
        tok.save(os.path.join(d, "tokenizer.json"))
    if "tiktoken.model" in files:
        open(os.path.join(d, "tiktoken.model"), "wb").write(b"IQ== 0\nIg== 1\n")
    if config:
        decoder = {str(tok.token_to_id(t)): {"content": t, "special": False} for t in added}
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "added_tokens_decoder": decoder},
                  open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8"))
    return d, tok


class Engine:
    """A tokenizer as the contract reads one: encode(text, add_special_tokens=False) and convert_tokens_to_ids.
    `shift` makes one probe's ids differ; `wrong` maps an added token's content to another id."""

    def __init__(self, tok, shift=None, wrong=None):
        self.tok, self.shift, self.wrong = tok, shift, wrong or {}
        self.unk_token_id = tok.token_to_id("[UNK]")

    def encode(self, text, add_special_tokens=False):
        ids = self.tok.encode(text, add_special_tokens=add_special_tokens).ids
        return [i + 1 for i in ids] if self.shift is not None and text == PROBES[self.shift] else ids

    def convert_tokens_to_ids(self, t):
        if t in self.wrong:
            return self.wrong[t]
        i = self.tok.token_to_id(t)
        return self.unk_token_id if i is None else i


def decide(path, engine, where="StandInTokenizer built from the folder"):
    core.set_mode("load")
    tokenizer_contract.reset(B)
    with redirect_stdout(io.StringIO()):
        return tokenizer_contract.check(B, C, path, engine, where)


def test_the_fact_is_a_hex_digest_and_a_probe_count():
    t = Tokenization(digest="0f3a", probes=10, added=3)
    assert t.digest == "0f3a" and t.probes == 10 and t.added == 3
    assert Tokenization(digest="0f3a", probes=0).probes == 0, "only the added tokens compared: no probe"
    for bad in (dict(digest="xyz", probes=1), dict(digest="", probes=1), dict(digest="0f", probes=-1)):
        try:
            Tokenization(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_a_tokenizer_that_does_what_the_folder_declares_passes():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    ds = decide(d, Engine(tok))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS and ds[0].rule == RULES["match"], ds
    assert ds[0].declared.value.probes == len(PROBES) and ds[0].declared.value.added == 2
    assert ds[0].declared.value.digest == ds[0].chosen.value.digest
    assert "tokenizers" in ds[0].declared.source.where and "probe texts encode alike" in ds[0].note
    assert "2 declared added tokens match" in ds[0].note
    assert tokenizer_contract.stats(B)["checks"] == 1 and tokenizer_contract.stats(B)["broken"] == 0


def test_ids_that_differ_on_a_probe_are_broken_and_name_the_probe():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    ds = decide(d, Engine(tok, shift=3))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["tokenizer_ids"], ds
    assert not ds[0].blocking, "the default policy reports and goes on (M5.4)"
    assert "1 of 10 probe texts encode differently" in ds[0].note and "leading spaces" in ds[0].note, ds[0].note
    assert "StandInTokenizer" in ds[0].note and "tokenizer.json" in ds[0].note
    assert ds[0].declared.value.digest != ds[0].chosen.value.digest
    assert tokenizer_contract.stats(B)["broken"] == 1


def test_a_declared_added_token_with_another_id_is_broken():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    other = tok.token_to_id("<think>")
    ds = decide(d, Engine(tok, wrong={"</think>": other}))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["added_token_id"], ds
    assert f"declares '</think>' as id {tok.token_to_id('</think>')}" in ds[0].note and f"gives it id {other}" in ds[0].note
    assert ds[0].declared.source.where == "tokenizer_config.json added_tokens_decoder"


def test_a_declared_added_token_the_engine_does_not_know_is_said_so():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    ds = decide(d, Engine(tok, wrong={"</think>": tok.token_to_id("[UNK]"), "<think>": None}))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["added_token_id"], ds
    assert "not a token of the engine's tokenizer" in ds[0].note and "2 of 2 declared added tokens differ" in ds[0].note


def test_a_folder_without_a_runnable_declaration_is_unknown():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d = tempfile.mkdtemp()
    _, tok = folder()
    ds = decide(d, Engine(tok))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "no tokenizer file" in ds[0].note, ds


def test_a_tiktoken_folder_compares_the_added_tokens_only():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder(files=("tiktoken.model",))
    ds = decide(d, Engine(tok))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN, ds
    assert "tiktoken.model declares the ranks" in ds[0].note and "2 declared added tokens compared: all match" in ds[0].note
    ds = decide(d, Engine(tok, wrong={"</think>": tok.token_to_id("<think>")}))
    assert [x.verdict for x in ds] == [Verdict.BROKEN, Verdict.UNKNOWN] and ds[0].rule == RULES["added_token_id"], ds
    assert "all match" not in ds[1].note


def test_a_folder_whose_declarations_disagree_is_unknown_not_broken():
    """A Llama-2-era folder: tokenizer.json exported in legacy mode (a Prepend normalizer) next to a
    tokenizer_config.json that declares legacy=false. The engine follows the flag and the ids differ from the file
    on a text that starts with whitespace; which one the model was trained with is not decidable here (M18.1 E2:
    TinyLlama). With legacy=true declared (the declarations agree) the same difference is broken (tiny-random-Llama)."""
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    p = os.path.join(d, "tokenizer.json")
    raw = json.load(open(p, encoding="utf-8"))
    raw["normalizer"] = {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"},
                                                              {"type": "Replace", "pattern": {"String": " "},
                                                               "content": "▁"}]}
    json.dump(raw, open(p, "w", encoding="utf-8"))
    cfg_path = os.path.join(d, "tokenizer_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["legacy"] = False
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"))
    assert "declarations disagree" in (tokenizer_contract.declarations_disagree(d) or "")
    ds = decide(d, Engine(Tokenizer.from_file(p), shift=3))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN, ds
    assert "legacy=false" in ds[0].note and "cannot be told here" in ds[0].note and "leading spaces" in ds[0].note
    cfg["legacy"] = True
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"))
    tokenizer_contract.reset()
    assert tokenizer_contract.declarations_disagree(d) is None
    ds = decide(d, Engine(Tokenizer.from_file(p), shift=3))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["tokenizer_ids"], ds


def test_the_declared_tokenizer_is_run_once_per_folder_and_kept_on_the_machine():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    tokenizer_contract.reset()
    r1 = tokenizer_contract.reference(d)
    assert r1.ids is not None and len(r1.ids) == len(PROBES) and r1.added and r1.where.startswith("tokenizer.json")
    assert tokenizer_contract.reference(d) is r1, "in-process: once per folder"
    cache = json.load(open(os.path.join(LOGS, tokenizer_contract.CACHE_NAME), encoding="utf-8"))
    assert os.path.abspath(d) in cache and cache[os.path.abspath(d)]["probes"] == tokenizer_contract.PROBES_DIGEST
    # a new process: the machine cache answers, the declared tokenizer is not built again
    tokenizer_contract._CACHE.clear()
    tokenizer_contract._FILE_CACHE = None
    orig = tokenizer_contract._encoder
    tokenizer_contract._encoder = lambda path: (_ for _ in ()).throw(AssertionError("built again"))
    try:
        r2 = tokenizer_contract.reference(d)
    finally:
        tokenizer_contract._encoder = orig
    assert r2.ids == r1.ids and r2.added == r1.added
    # the files change: the stamp differs and the declaration is run again
    tok.add_tokens(["<new>"])
    tok.save(os.path.join(d, "tokenizer.json"))
    os.utime(os.path.join(d, "tokenizer.json"), (0, 0))
    r3 = tokenizer_contract.reference(d)
    assert r3 is not r2 and r3.ids is not None


def test_where_the_policy_stops_the_mismatch_is_refused():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    os.environ["ENTAIL_ON_BROKEN"] = "stop"
    try:
        try:
            decide(d, Engine(tok, shift=0))
        except RoleError as e:
            assert "probe texts encode differently" in str(e), str(e)
        else:
            raise AssertionError("no stop")
        assert tokenizer_contract.stats(B)["refused"] == 1
    finally:
        os.environ.pop("ENTAIL_ON_BROKEN", None)
        core.set_mode("off")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
