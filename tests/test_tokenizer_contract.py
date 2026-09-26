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


def legacy_folder(legacy):
    """A Llama-2-era shape: tokenizer.json exported in legacy mode (a Prepend normalizer) next to a
    tokenizer_config.json that declares `legacy`; <s> and </s> declared as bos and eos."""
    d, tok = folder()
    p = os.path.join(d, "tokenizer.json")
    raw = json.load(open(p, encoding="utf-8"))
    raw["normalizer"] = {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"},
                                                              {"type": "Replace", "pattern": {"String": " "},
                                                               "content": "▁"}]}
    raw["pre_tokenizer"] = None      # a sentencepiece-style pipeline: no pre-tokenizer, spaces replaced
    json.dump(raw, open(p, "w", encoding="utf-8"))
    cfg_path = os.path.join(d, "tokenizer_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg.update(legacy=legacy, bos_token="<s>", eos_token="</s>")
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"))
    return d, Tokenizer.from_file(p), p      # the file's own pipeline, normalizer included, is the reference


def test_a_difference_the_folders_own_flag_explains_is_the_sources_disagreeing():
    """M18.1 review, findings 2 and 3. tokenizer_config.json declares legacy=false while tokenizer.json is a legacy
    export: a probe that differs only after a declared token, exactly as the flag's pipeline gives it, is the
    folder's own disagreement (unknown, sources_disagree, the flag recorded as the conflicting source). A probe
    that differs elsewhere (text that starts with whitespace, probe 3) stays broken, because every declaration the
    folder holds gives the same ids there (TinyLlama). With legacy=true nothing is excused."""
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok, p = legacy_folder(False)
    assert tokenizer_contract.declared_flags(d) == {"legacy": False}
    assert "</s>" in tokenizer_contract.special_strings(d) and "<think>" in tokenizer_contract.special_strings(d)
    file_ids = tokenizer_contract.reference(d).ids
    # the flag's pipeline (Prepend dropped, Metaspace first): a stand-in for what transformers 5 builds
    built = tokenizer_contract.flag_variant(d, {"legacy": False})
    assert built is not None and built[1] == "first"
    variant = built[0]
    assert tokenizer_contract.flag_variant(d, {"legacy": True}) is None, "the file prepends: legacy=true agrees"
    orig_variant = tokenizer_contract.flag_variant
    tokenizer_contract.flag_variant = lambda path, flags: ((lambda s: [9, 9, 9] if "<s>" in s else variant(s)), "first")

    class Follows(Engine):
        """An engine that follows the flag on the probe with declared tokens (probe 9) and agrees elsewhere."""

        def encode(self, text, add_special_tokens=False):
            return [9, 9, 9] if "<s>" in text else super().encode(text, add_special_tokens)

    try:
        ds = decide(d, Follows(tok))
        assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and ds[0].rule == RULES["sources_disagree"], ds
        assert "legacy=false" in ds[0].note and "cannot be told here" in ds[0].note and ds[0].conflict
        assert "tokenizer_config.json legacy=false" in ds[0].conflict[0].source.where
        assert ds[0].contract.meaning_changing == ("Tokenization",)
        # the same flag, but the difference is on text with no token in it: broken, with the excused probe noted
        class FollowsAndShifts(Follows):
            def encode(self, text, add_special_tokens=False):
                ids = super().encode(text, add_special_tokens)
                return [i + 1 for i in ids] if text == PROBES[3] else ids

        tokenizer_contract.reset()
        ds = decide(d, FollowsAndShifts(tok))
        assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["tokenizer_ids"], ds
        assert "1 of 10 probe texts encode differently" in ds[0].note and "leading spaces" in ds[0].note
        assert "1 further differ only after a declared token" in ds[0].note
    finally:
        tokenizer_contract.flag_variant = orig_variant
    assert file_ids is not None
    # legacy=true over a file that already prepends: the declarations agree, so the probe-9 difference is broken
    # like any other (the real flag_variant returns None: nothing to excuse)
    d2, tok2, p2 = legacy_folder(True)
    assert tokenizer_contract.declared_flags(d2) == {"legacy": True}
    assert tokenizer_contract.flag_variant(d2, {"legacy": True}) is None
    ds = decide(d2, Follows(tok2))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["tokenizer_ids"], ds
    # add_prefix_space=true over a file that prepends nothing (dolphin-yi in the static corpus): the flag's
    # "always" pipeline explains a difference at the start of any text, so it is the sources disagreeing
    d3, _ = folder()
    p3 = os.path.join(d3, "tokenizer.json")
    raw = json.load(open(p3, encoding="utf-8"))
    raw["normalizer"] = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}   # no Prepend
    raw["pre_tokenizer"] = None
    json.dump(raw, open(p3, "w", encoding="utf-8"))
    tok3 = Tokenizer.from_file(p3)
    cfg_path = os.path.join(d3, "tokenizer_config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg["add_prefix_space"] = True
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"))
    built = tokenizer_contract.flag_variant(d3, {"add_prefix_space": True})
    assert built is not None and built[1] == "always"
    tokenizer_contract.flag_variant = lambda path, flags: ((lambda s: [7, 7] if s == PROBES[0] else Engine(tok3).encode(s)), "always")

    class Prefixes(Engine):
        def encode(self, text, add_special_tokens=False):
            return [7, 7] if text == PROBES[0] else super().encode(text, add_special_tokens)

    try:
        ds = decide(d3, Prefixes(tok3))
        assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and ds[0].rule == RULES["sources_disagree"], ds
        assert "add_prefix_space=true" in ds[0].note
    finally:
        tokenizer_contract.flag_variant = orig_variant
    # a byte-level file (a Split or ByteLevel pre-tokenizer) next to legacy=true: the flag speaks of a
    # sentencepiece-style prefix, not of this pipeline, so nothing is excused (the static corpus:
    # DeepSeek-R1-0528-Qwen3-8B declares LlamaTokenizerFast over a byte-level tokenizer.json and 5.17 drops it whole)
    d4, tok4 = folder()
    p4 = os.path.join(d4, "tokenizer.json")
    raw = json.load(open(p4, encoding="utf-8"))
    raw["normalizer"] = {"type": "NFC"}
    raw["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
    json.dump(raw, open(p4, "w", encoding="utf-8"))
    assert tokenizer_contract.flag_variant(d4, {"legacy": True}) is None
    assert tokenizer_contract.flag_variant(d4, {"add_prefix_space": True}) is None
    assert tokenizer_contract.has_declaration(d4)


def test_a_users_own_build_setting_is_the_users_choice():
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    core.set_mode("load")
    tokenizer_contract.reset(B)
    with redirect_stdout(io.StringIO()):
        ds = tokenizer_contract.check(B, C, d, Engine(tok, shift=3), "t", user_kwargs={"legacy": False})
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and ds[0].rule == RULES["user_choice"], ds
    assert "legacy=False given by the user" in ds[0].note and "leading spaces" in ds[0].note


def test_the_files_padding_and_truncation_are_not_applied_by_the_reference():
    """M18.1 review, finding 5: tokenizer.json can carry padding and truncation, which an engine's encode()
    removes; the reference clears them too. BPE dropout (a training-time setting) is not compared."""
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()
    tok.enable_padding(length=64, pad_id=0, pad_token="[UNK]")
    tok.enable_truncation(max_length=3)
    tok.save(os.path.join(d, "tokenizer.json"))
    plain = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    plain.no_padding()
    plain.no_truncation()
    ds = decide(d, Engine(plain))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds
    p = os.path.join(d, "tokenizer.json")
    raw = json.load(open(p, encoding="utf-8"))
    raw["model"] = {"type": "BPE", "dropout": 0.5, "unk_token": "[UNK]", "continuing_subword_prefix": None,
                    "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                    "vocab": {"[UNK]": 0, "a": 1, "b": 2}, "merges": []}
    json.dump(raw, open(p, "w", encoding="utf-8"))
    tokenizer_contract.reset()
    r = tokenizer_contract.reference(d)
    assert r.ids is None and any("dropout" in x for x in r.problems), r.problems


def test_a_lookup_that_raises_counts_the_token_as_missing():
    """M18.1 review, finding 9: a class that raises for a token it lacks shows the strongest mismatch; every token
    is compared and the raised ones are counted, not dropped."""
    if Tokenizer is None:
        print("ok skipped (no tokenizers library)")
        return
    d, tok = folder()

    class Raises(Engine):
        def convert_tokens_to_ids(self, t):
            if t == "</think>":
                raise KeyError(t)
            return super().convert_tokens_to_ids(t)

    ds = decide(d, Raises(tok))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["added_token_id"], ds
    assert "not a token of the engine's tokenizer" in ds[0].note and "1 raised on lookup" in ds[0].note

    class NoLookup:
        unk_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return tok.encode(text, add_special_tokens=add_special_tokens).ids

    ds = decide(d, NoLookup())
    assert [x.verdict for x in ds] == [Verdict.UNKNOWN, Verdict.PASS] and "no convert_tokens_to_ids" in ds[0].note, ds


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
    entry = cache[os.path.abspath(d)]
    assert entry["reference"] == tokenizer_contract.REFERENCE_VERSION and entry["library"].startswith("tokenizers ")
    # ids built by another library version or reference algorithm are not reused (M18.1 review, finding 10)
    entry["library"] = "tokenizers 0.0.0"
    json.dump(cache, open(os.path.join(LOGS, tokenizer_contract.CACHE_NAME), "w", encoding="utf-8"))
    tokenizer_contract._CACHE.clear()
    tokenizer_contract._FILE_CACHE = None
    assert tokenizer_contract._from_file_cache(d, tokenizer_contract._stamp(d)) is None
    tokenizer_contract._FILE_CACHE = None
    tokenizer_contract.reference(d)
    cache = json.load(open(os.path.join(LOGS, tokenizer_contract.CACHE_NAME), encoding="utf-8"))
    assert cache[os.path.abspath(d)]["library"].startswith("tokenizers ") and "0.0.0" not in cache[os.path.abspath(d)]["library"]
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
