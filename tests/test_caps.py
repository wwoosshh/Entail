"""Tests for the capability table as data and the probe method (ROADMAP M3.1): the packaged table, every refusal
with its wording, what a consumer uses, routing, and the probe's verdicts with a stand-in runner (no GPU).
Run: python tests/test_caps.py"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import caps, probes  # noqa: E402
from entail.facts import Certainty, Fact, ModelProps, Source  # noqa: E402


def row(**kw):
    base = {"consumer": "e.attention.a", "fact": "ModelProps.softcap", "honours": True, "evidence": "measured",
            "ref": "r"}
    base.update(kw)
    return base


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        assert text in str(e), f"{text!r} not in {str(e)!r}"
        return
    raise AssertionError(f"expected ValueError with {text!r}")


def test_packaged_table():
    t = caps.load_table()
    assert len(t.rows) == 38 and len({(r.consumer, r.fact) for r in t.rows}) == 38   # 30 in M3, 8 parsers in M5.3
    assert {r.evidence for r in t.rows} == {"measured", "code"}
    assert sum(r.evidence == "measured" for r in t.rows) == 23   # 22 in M3; vLLM's hermes parser in M5.3
    assert t.preferred("sglang.attention") == ("triton",)
    assert t.preferred("transformers.attention") == ("eager", "flex_attention")
    assert caps.consumed(t, "sglang.attention") == ("ModelProps.softcap", "ModelProps.sliding_window")
    assert {caps.group_of(r.consumer) for r in t.rows} == {"transformers.attention", "transformers.paged_attention",
                                                          "sglang.attention", "vllm.attention", "vllm.tool_parser",
                                                          "sglang.tool_parser"}
    assert caps.consumed(t, "vllm.tool_parser") == ("Template.tool_call_format",)
    assert t.preferred("vllm.tool_parser") == ("hermes", "llama3_json", "pythonic")
    assert t.preferred("transformers.paged_attention") == ()   # no paged kernel is measured to honour softcap
    assert probes.full_name("transformers", "paged|sdpa") == "transformers.paged_attention.sdpa"
    assert all(r.ref.strip() and r.version for r in t.rows)


def test_table_matches_the_old_one_but_for_the_recorded_change():
    """The table written in preflight.py before M3.1, cell by cell. One cell changed on purpose (LIBRARY_DESIGN.md
    11, M3): sglang flashinfer's window, which a code audit found dropped on the ragged prefill path."""
    old = {"transformers": {"eager": (1, 1), "sdpa": (0, 1), "flex_attention": (1, 1), "paged|eager": (0, 1),
                            "paged|sdpa": (0, 1)},
           "sglang": {"triton": (1, 1), "flashinfer": (0, 1), "flex_attention": (0, 0), "torch_native": (0, 1),
                      "trtllm_mha": (0, 1)},
           "vllm": {"FLASH_ATTN": (1, 1), "TRITON_ATTN": (1, 1), "FLASHINFER": (1, 1), "FLEX_ATTENTION": (1, 1),
                    "ROCM_ATTN": (0, 1)}}
    t = caps.load_table()
    changed = []
    for engine, backends in old.items():
        for short, (sc, sw) in backends.items():
            for fact, want in (("ModelProps.softcap", sc), ("ModelProps.sliding_window", sw)):
                cell = caps.lookup(t, probes.full_name(engine, short), fact)
                if cell.honours != bool(want):
                    changed.append((engine, short, fact))
    assert changed == [("sglang", "flashinfer", "ModelProps.sliding_window")], changed
    from entail.preflight import CAPS
    assert CAPS["sglang"]["flashinfer"][0].sliding_window is False and CAPS["transformers"]["sdpa"][0].softcap is False


def test_every_refusal_and_its_wording():
    raises(lambda: caps.Capability(**row(consumer="sdpa")), "consumer must be named engine.role.name")
    raises(lambda: caps.Capability(**row(consumer="e..a")), "consumer must be named engine.role.name")
    raises(lambda: caps.Capability(**row(fact="softcap")), "fact must be written Name.field")
    raises(lambda: caps.Capability(**row(fact="Softcap.value")), "unknown fact name 'Softcap'")
    raises(lambda: caps.Capability(**row(fact="ModelProps.cap")), "ModelProps has no field 'cap'")
    raises(lambda: caps.Capability(**row(honours="yes")), "honours must be true or false")
    raises(lambda: caps.Capability(**row(evidence="believed")), "evidence 'believed' is not one of")
    raises(lambda: caps.Capability(**row(ref=" ")), "every row needs a ref")
    raises(lambda: caps.from_rows([row(), row()]), "e.attention.a ModelProps.softcap appears twice")
    raises(lambda: caps.from_rows([row(colour="red")]), "unknown row fields ['colour']")
    raises(lambda: caps.from_rows([row()], {"e.attention": ["b"]}), "prefer lists e.attention.b, which has no row")
    d = tempfile.mkdtemp()
    for data, text in (({"schema": 2, "vocab_version": 1, "rows": []}, "schema 2, this library reads 1"),
                       ({"schema": 1, "vocab_version": 9, "rows": []}, "written for vocabulary v9")):
        p = os.path.join(d, "t.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        raises(lambda: caps.load_table(p), text)
    shutil.rmtree(d)


def small():
    return caps.from_rows([
        row(consumer="e.attention.a", fact="ModelProps.softcap", honours=True),
        row(consumer="e.attention.a", fact="ModelProps.sliding_window", honours=True),
        row(consumer="e.attention.b", fact="ModelProps.softcap", honours=False),
        row(consumer="e.attention.b", fact="ModelProps.sliding_window", honours=True, evidence="code"),
        row(consumer="e.attention.c", fact="ModelProps.softcap", honours=True, evidence="code"),
        row(consumer="e.attention.c", fact="ModelProps.sliding_window", honours=True),
        row(consumer="e.attention.d", fact="ModelProps.sliding_window", honours=False),
    ], {"e.attention": ["c", "a"]})


def test_what_a_consumer_uses():
    t, both = small(), ModelProps(softcap=50.0, sliding_window=4096)
    assert caps.uses(t, "e.attention.a", both).value == both
    u = caps.uses(t, "e.attention.b", both)
    assert u.value == ModelProps(sliding_window=4096) and u.dropped == ("ModelProps.softcap",) and u.evidence == "code"
    u = caps.uses(t, "e.attention.x", both)                      # not in the table at all
    assert u.value is None and u.unknown == ("ModelProps.softcap", "ModelProps.sliding_window")
    u = caps.uses(t, "e.attention.d", both)                      # one field dropped, the other unknown
    assert u.value == ModelProps() and u.dropped == ("ModelProps.sliding_window",) and u.unknown == ("ModelProps.softcap",)
    assert caps.uses(t, "e.attention.b", ModelProps(sliding_window=16)).value == ModelProps(sliding_window=16)
    # the rows behind a mismatch, with their evidence (M11.4: a code-read mismatch is inferred, not acted on)
    assert caps.disagreements(t, "e.attention.a", both) == []
    assert caps.disagreements(t, "e.attention.b", both) == [("ModelProps.softcap", "measured", None)]
    assert caps.disagreements(t, "e.attention.x", both) == []                  # nothing known: nothing disagrees
    assert caps.disagreements(t, "e.attention.b", ModelProps(sliding_window=16)) == []
    guessed = caps.from_rows([row(consumer="e.attention.g", fact="ModelProps.softcap", honours=False, evidence="code"),
                              row(consumer="e.parser.p", fact="Template.tool_call_format", honours=False,
                                  evidence="code", reads="pythonic")], {})
    assert caps.disagreements(guessed, "e.attention.g", both) == [("ModelProps.softcap", "code", None)]
    from entail.facts import Template
    assert caps.disagreements(guessed, "e.parser.p", Template(tool_call_format="hermes")) == \
        [("Template.tool_call_format", "code", "pythonic")]


def test_the_chosen_fact_says_how_it_is_known():
    t = small()
    declared = Fact("ModelProps", ModelProps(softcap=50.0, sliding_window=4096), Source("config", "c.json"),
                    Certainty.DECLARED)
    f = caps.chosen_fact(t, "e.attention.a", declared)
    assert f.certainty is Certainty.VERIFIED and f.source.kind == "engine"
    assert f.source.where == "e.attention.a [softcap: honours (measured); sliding_window: honours (measured)]"
    f = caps.chosen_fact(t, "e.attention.b", declared)
    assert f.certainty is Certainty.INFERRED and f.value == ModelProps(sliding_window=4096)
    f = caps.chosen_fact(t, "e.attention.x", declared)
    assert f.certainty is Certainty.UNKNOWN and f.value is None and "not in the capability table" in f.source.where
    f = caps.chosen_fact(t, "e.attention.d", declared)
    assert f.certainty is Certainty.INFERRED and f.value == ModelProps()
    tie = Fact("ModelProps", ModelProps(tie_word_embeddings=True), Source("config", "c"), Certainty.DECLARED)
    assert caps.chosen_fact(t, "e.attention.a", tie).certainty is Certainty.UNKNOWN   # no row: the site projects first
    raises(lambda: caps.chosen_fact(t, "e.attention.a", Fact("ModelProps", ModelProps(), Source("config", "c"),
                                                             Certainty.DECLARED)), "fills no field; project it first")


def test_route_takes_measured_consumers_only():
    t = small()
    assert caps.route(t, "e.attention", ModelProps(softcap=50.0)) == "a"              # c's softcap is only code
    assert caps.route(t, "e.attention", ModelProps(sliding_window=16)) == "c"
    assert caps.route(t, "e.attention", ModelProps(softcap=50.0), measured_only=False) == "c"
    assert caps.route(t, "e.attention", ModelProps(softcap=50.0), exclude=("a",)) is None
    real = caps.load_table()
    gemma = ModelProps(softcap=50.0, sliding_window=4096)
    assert caps.route(real, "sglang.attention", gemma) == "triton"
    assert caps.route(real, "transformers.attention", gemma) == "eager"
    assert caps.route(real, "transformers.attention", ModelProps(softcap=50.0), exclude=("eager",)) == "flex_attention"
    assert caps.route(real, "transformers.attention", gemma, exclude=("eager",)) is None   # flex window: code only
    assert caps.route(real, "vllm.attention", gemma) == "FLASH_ATTN"


def test_project_keeps_what_a_group_acts_on():
    wanted = ("ModelProps.softcap", "ModelProps.sliding_window")
    p = caps.project(ModelProps(softcap=30.0, sliding_window=4096, tie_word_embeddings=True), wanted)
    assert p == ModelProps(softcap=30.0, sliding_window=4096)
    assert caps.project(ModelProps(tie_word_embeddings=True), wanted) is None
    assert caps.project(None, wanted) is None


def test_probe_configs_bind_and_remove():
    bound, removed = probes.configs({"attn_logit_softcapping": 50.0, "x": 1}, "ModelProps.softcap")
    assert bound == {"attn_logit_softcapping": 5.0, "x": 1} and removed == {"x": 1}
    bound, removed = probes.configs({"text_config": {"sliding_window": 4096}}, "ModelProps.sliding_window")
    assert bound == {"text_config": {"sliding_window": 16}} and removed == {"text_config": {}}
    raises(lambda: probes.configs({"x": 1}, "ModelProps.softcap"), "does not declare attn_logit_softcapping")
    raises(lambda: probes.configs({"x": 1}, "Rotary.theta"), "no way to bind Rotary.theta")


def test_probe_verdicts():
    assert probes.judge([[1]], [[1]], [[2]]) == ("honours", 1)
    assert probes.judge([[1], [2]], [[1], [2]], [[1], [2]]) == ("ignores", 0)
    assert probes.judge([[1]], [[3]], [[1]]) == ("inconclusive", None)


def test_probe_with_a_stand_in_runner():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"attn_logit_softcapping": 50.0}, f)
    made = []

    def copy(model_dir, cfg):                      # no symlinks: this test also runs on Windows
        c = tempfile.mkdtemp()
        with open(os.path.join(c, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        made.append(c)
        return c

    def runner_for(reads, flaky=()):
        def run(model_dir, consumer, runs):
            short = consumer.rsplit(".", 1)[1]
            cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
            seen = cfg.get("attn_logit_softcapping") if short in reads else None
            return [[f"{short}:{seen}:{i if short in flaky else 0}"] for i in range(runs)]
        return run

    saved, probes.model_copy = probes.model_copy, copy
    try:
        t = small()
        r = probes.probe("e", "a", "ModelProps.softcap", d, table=t, runner=runner_for({"a"}))
        assert r["measured"] == "honours" and r["agrees"] is True and r["consumer"] == "e.attention.a"
        r = probes.probe("e", "b", "ModelProps.softcap", d, table=t, runner=runner_for({"c"}))
        assert r["measured"] == "ignores" and r["gate"]["consumer"] == "e.attention.c" and r["agrees"] is True
        r = probes.probe("e", "b", "ModelProps.softcap", d, table=t, runner=runner_for(set()))
        assert r["measured"] == "inconclusive" and "does not bind" in r["why"] and r["agrees"] is None
        r = probes.probe("e", "a", "ModelProps.softcap", d, table=t, runner=runner_for({"a"}, flaky={"a"}))
        assert r["measured"] == "inconclusive" and r["agrees"] is None
        r = probes.probe("e", "a", "ModelProps.softcap", d, table=t, runner=runner_for(set()), gate=False)
        assert r["measured"] == "ignores" and r["agrees"] is False        # the table says a honours it
        r = probes.probe("e", "z", "ModelProps.softcap", d, table=t, runner=runner_for({"z"}))
        assert r["table"] is None and r["agrees"] is None

        def broken(model_dir, consumer, runs):
            raise RuntimeError("engine failed to start")
        r = probes.probe("e", "a", "ModelProps.softcap", d, table=t, runner=broken)
        assert r["measured"] == "inconclusive" and "RuntimeError: engine failed to start" in r["why"]
    finally:
        probes.model_copy = saved
        for p in made + [d]:
            shutil.rmtree(p, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
