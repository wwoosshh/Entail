"""Tests for the verdicts (ROADMAP M1.2, M1.3): every row of the verdict order, for every name in the vocabulary (all
ten kinds), plus the ledger's exact wording and the policy. Run: python tests/test_contracts.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import contracts, policies, record, sources  # noqa: E402
from entail.contracts import RULES, Contract, Resolution, Verdict, agrees, decide  # noqa: E402
from entail.coverage import Coverage  # noqa: E402
from entail.facts import (VOCABULARY, Assumed, Certainty, Epoch, Fact, Identity, LatentScale, Layout,  # noqa: E402
                          ModelProps, Origin, Positions, Prediction, Quantized, Reduction, Rotary, Source, Template,
                          Valid)
from entail.kv_contract import KvExtent  # noqa: E402

# (declared, a different value the consumer might use) for every vocabulary name
SAMPLES = {
    "Layout": (Layout("q8_0", packing="interleaved"), Layout("q8_0", packing="split")),
    "Quantized": (Quantized("float8_e4m3fn", 0.5), Quantized("bfloat16")),
    "Rotary": (Rotary("llama3", theta=500000.0, factor=32.0), Rotary("default", theta=10000.0)),
    "Positions": (Positions("absolute"), Positions("chunk_relative", 16)),
    "Valid": (Valid(length=5), Valid(length=4)),
    "KvExtent": (KvExtent(held=8, needed=8), KvExtent(held=7, needed=8)),
    "ModelProps": (ModelProps(softcap=50.0), ModelProps(sliding_window=4096)),
    "Prediction": (Prediction("v", True), Prediction("eps", False)),
    "LatentScale": (LatentScale(0.13025), LatentScale(0.18215)),
    "Template": (Template(reasoning_history="keep"), Template(reasoning_history="drop")),
    "Coverage": (Coverage(3, 3, ()), Coverage(3, 0, ("a", "b", "c"))),
    "Reduction": (Reduction("R"), Reduction("P")),
    "Epoch": (Epoch(3), Epoch(2)),
    "Identity": (Identity("kv_block", 0, "aa"), Identity("kv_block", 0, "bb")),
    "Assumed": (Assumed((("batch", 4),)), Assumed((("batch", 0),))),
    "Origin": (Origin("temperature", "user"), Origin("temperature", "default")),
}


def fact(name, value, kind="file", certainty=Certainty.DECLARED):
    return Fact(name, value, Source(kind, f"{kind}:{name}"), certainty)


def used(name, value, kind="engine"):
    return Fact(name, value, Source(kind, f"consumer:{name}"), Certainty.DECLARED)


def contract(name, meaning_changing=False):
    return Contract(f"load:{name}", f"consumer.{name}", (name,), (name,) if meaning_changing else ())


def one(name, declared=None, chosen=None, policy=None, observed=None, meaning_changing=False):
    [d] = decide(contract(name, meaning_changing), {name: declared} if declared is not None else {},
                 {name: chosen} if chosen is not None else {}, policy, {name: observed} if observed else None)
    return d


class resolutions:
    """Register resolutions for a test and put the registry back afterwards."""
    def __init__(self, *pairs):
        self.pairs = pairs

    def __enter__(self):
        self.saved = {k: list(v) for k, v in contracts.RESOLUTIONS.items()}
        for name, r in self.pairs:
            contracts.register(name, r)

    def __exit__(self, *exc):
        contracts.RESOLUTIONS.clear()
        contracts.RESOLUTIONS.update(self.saved)


def test_samples_cover_the_whole_vocabulary():
    assert set(SAMPLES) == set(VOCABULARY)
    for name, (a, b) in SAMPLES.items():
        assert agrees(a, a) and not agrees(a, b), name


def test_row_pass():
    for name, (a, _) in SAMPLES.items():
        d = one(name, fact(name, a), used(name, a))
        assert (d.verdict, d.rule, d.blocking) == (Verdict.PASS, RULES["match"], False), name


def test_row_resolved():
    for name, (a, b) in SAMPLES.items():
        with resolutions((name, Resolution("route to a consumer that honours it", "switch"))):
            d = one(name, fact(name, a), used(name, b))
        assert (d.verdict, d.rule, d.handle, d.blocking) == (Verdict.RESOLVED, RULES["resolved"], "switch", False), name
        assert d.resolution == f"route to a consumer that honours it ({b} -> {a})", d.resolution


STRICT = policies.Policy(on_broken="stop")


def test_row_broken_by_default_and_refused_where_the_policy_stops():
    """Nothing repairs it: reported as broken and the run goes on (M5.4), unless the policy stops - then refused."""
    for name, (a, b) in SAMPLES.items():
        d = one(name, fact(name, a), used(name, b))
        assert (d.verdict, d.rule, d.blocking) == (Verdict.BROKEN, RULES["no_resolution"], False), name
        for stop in (STRICT, policies.Policy(mode="debug"), policies.Policy(overrides=((name, "stop"),))):
            d = one(name, fact(name, a), used(name, b), stop)
            assert (d.verdict, d.rule, d.blocking) == (Verdict.REFUSED, RULES["no_resolution"], True), (name, stop)
        d = one(name, fact(name, a), used(name, b), policies.Policy(on_broken="stop", overrides=((name, "report"),)))
        assert (d.verdict, d.blocking) == (Verdict.BROKEN, False), name
        with resolutions((name, Resolution("fix", "h"))):
            d = one(name, fact(name, a), used(name, b), policies.Policy(on_mismatch="refuse"))
            assert (d.verdict, d.rule, d.blocking) == (Verdict.BROKEN, RULES["policy_refuses"], False), name
            d = one(name, fact(name, a), used(name, b), policies.Policy(on_mismatch="refuse", on_broken="stop"))
            assert (d.verdict, d.rule, d.blocking) == (Verdict.REFUSED, RULES["policy_refuses"], True), name
            d = one(name, fact(name, a), used(name, b, kind="user"))
            assert (d.verdict, d.rule, d.blocking) == (Verdict.BROKEN, RULES["user_choice"], False), name
            d = one(name, fact(name, a), used(name, b, kind="user"), STRICT)
            assert (d.verdict, d.rule, d.blocking) == (Verdict.REFUSED, RULES["user_choice"], True), name
        with resolutions((name, Resolution("never", "h", when=lambda d, c: False))):
            assert one(name, fact(name, a), used(name, b)).rule == RULES["no_resolution"], name


def test_row_unknown_consumer():
    for name, (a, _) in SAMPLES.items():
        d = one(name, fact(name, a))
        assert (d.verdict, d.rule, d.blocking) == (Verdict.UNKNOWN, RULES["consumer_unknown"], False), name
        assert one(name, fact(name, a), policy=policies.Policy(mode="debug")).blocking, name
        unknown_use = Fact(name, None, Source("engine", "consumer"), Certainty.UNKNOWN)
        assert one(name, fact(name, a), unknown_use).rule == RULES["consumer_unknown"], name


def test_row_unknown_declaration():
    for name, (a, _) in SAMPLES.items():
        d = one(name, None, used(name, a))
        assert (d.verdict, d.rule, d.blocking) == (Verdict.UNKNOWN, RULES["undeclared"], False), name
        d = one(name, None, used(name, a), meaning_changing=True)
        assert (d.verdict, d.blocking) == (Verdict.UNKNOWN, False), name    # reported, never a silent default (M5.4)
        require = policies.Policy(on_unknown_meaning_changing="require")
        assert one(name, None, used(name, a), require, meaning_changing=True).blocking, name
        debug = policies.Policy(mode="debug")   # an undeclared fact follows its setting in every mode
        assert not one(name, None, used(name, a), debug, meaning_changing=True).blocking, name
        assert one(name, None, used(name, a), policies.Policy(mode="debug", on_unknown_meaning_changing="require"),
                   meaning_changing=True).blocking, name
        d = one(name, fact(name, a, kind="probe", certainty=Certainty.INFERRED), used(name, a), meaning_changing=True)
        assert (d.verdict, d.rule, d.blocking) == (Verdict.UNKNOWN, RULES["inferred_only"], False), name
        d = one(name, fact(name, a, kind="probe", certainty=Certainty.INFERRED), used(name, a), require,
                meaning_changing=True)
        assert d.blocking, name
        d = one(name, fact(name, a, kind="default", certainty=Certainty.DEFAULTED), used(name, a))
        assert (d.rule, d.blocking) == (RULES["defaulted_only"], False), name
        p = policies.Policy(on_unknown_meaning_changing="require", overrides=((name, "report"),))
        assert not one(name, None, used(name, a), p, meaning_changing=True).blocking, name
        p = policies.Policy(on_unknown_other="stop")
        assert one(name, None, used(name, a), p).blocking, name


def test_row_sources_disagree():
    for name, (a, b) in SAMPLES.items():
        if sources.compatible(a, b):   # ModelProps: the two samples fill different fields, so they do not disagree
            b = ModelProps(softcap=30.0)
        pair = (fact(name, b, kind="config"), fact(name, a, kind="file"))   # file outranks config
        d = one(name, pair, used(name, a))
        assert (d.verdict, d.declared.source.kind, len(d.conflict)) == (Verdict.PASS, "file", 2), name
        d = one(name, pair, used(name, a), policies.Policy(on_source_conflict="stop"))
        assert (d.verdict, d.rule, d.blocking) == (Verdict.REFUSED, RULES["sources_disagree"], True), name
        same = (fact(name, a, kind="config"), fact(name, a, kind="file"))
        assert one(name, same, used(name, a)).conflict == (), name
    # a value that says less is not a contradiction
    best, conflict = sources.pick([fact("Prediction", Prediction("v")), fact("Prediction", Prediction("v", True),
                                                                              kind="config")])
    assert best.source.kind == "file" and conflict == ()


def test_sources_that_fill_different_fields_are_combined():
    """Nothing a source says may be dropped: the file gives softcap, the config the window; both reach the contract."""
    file_ = fact("ModelProps", ModelProps(softcap=50.0), kind="file")
    config = fact("ModelProps", ModelProps(sliding_window=4096), kind="config")
    best, conflict = sources.pick([config, file_])
    assert conflict == () and best.value == ModelProps(softcap=50.0, sliding_window=4096), best
    assert best.source.where == "file: file:ModelProps; config: config:ModelProps" and best.certainty is Certainty.DECLARED
    guess = fact("ModelProps", ModelProps(tie_word_embeddings=True), kind="probe", certainty=Certainty.INFERRED)
    best, _ = sources.pick([file_, guess])
    assert best.certainty is Certainty.INFERRED   # one inferred field makes the whole fact inferred
    d = one("ModelProps", (file_, config), used("ModelProps", ModelProps(softcap=50.0)))
    assert (d.verdict, d.rule) == (Verdict.BROKEN, RULES["no_resolution"])   # the consumer ignores the window
    chosen, conflicts = sources.merge([file_, config, fact("Epoch", Epoch(1))])
    assert set(chosen) == {"ModelProps", "Epoch"} and conflicts == []


def test_row_declaration_against_the_data():
    # The data contradicts a declaration only where both say something (M3.2); every other pair already overlaps.
    data = dict(SAMPLES, ModelProps=(ModelProps(softcap=50.0), ModelProps(softcap=30.0)))
    window = fact("ModelProps", ModelProps(sliding_window=4096), kind="data", certainty=Certainty.VERIFIED)
    both = ModelProps(softcap=50.0, sliding_window=4096)
    partial = one("ModelProps", fact("ModelProps", ModelProps(softcap=50.0)), used("ModelProps", both), observed=window)
    assert (partial.verdict, partial.declared.certainty) == (Verdict.PASS, Certainty.VERIFIED)
    assert partial.declared.value == both                           # what the data adds is kept, and then required
    d = one("ModelProps", fact("ModelProps", ModelProps(softcap=50.0)), used("ModelProps", ModelProps(softcap=50.0)),
            observed=window)
    assert d.verdict is Verdict.BROKEN                              # the consumer drops the window the data shows
    for name, (a, b) in data.items():
        seen_b = fact(name, b, kind="data", certainty=Certainty.VERIFIED)
        d = one(name, fact(name, a), used(name, b), observed=seen_b)
        assert (d.verdict, d.rule, d.blocking) == (Verdict.BROKEN, RULES["false_declaration"], False), name
        d = one(name, fact(name, a), used(name, b), STRICT, observed=seen_b)
        assert (d.verdict, d.rule, d.blocking) == (Verdict.REFUSED, RULES["false_declaration"], True), name
        d = one(name, fact(name, a), used(name, b), policies.Policy(on_false_declaration="use_data"), observed=seen_b)
        assert (d.verdict, d.rule, d.declared.source.kind) == (Verdict.PASS, RULES["data_used"], "data"), name
        seen_a = fact(name, a, kind="data", certainty=Certainty.VERIFIED)
        d = one(name, fact(name, a), used(name, a), observed=seen_a)
        assert (d.verdict, d.declared.certainty) == (Verdict.PASS, Certainty.VERIFIED), name
        d = one(name, None, used(name, a), observed=seen_a, meaning_changing=True)   # the data is known, not guessed
        assert (d.verdict, d.declared.source.kind) == (Verdict.PASS, "data"), name


def test_contracts_and_inputs_are_checked():
    for text, fn in [
        ("unknown fact name 'Colour'", lambda: Contract("b", "c", ("Colour",))),
        ("Contract.needs: expected a non-empty tuple", lambda: Contract("b", "c", ())),
        ("['Epoch'] are not in needs", lambda: Contract("b", "c", ("Layout",), ("Epoch",))),
        ("Contract.consumer: expected a name", lambda: Contract("b", "", ("Layout",))),
        ("must be a Fact named 'Layout'", lambda: decide(contract("Layout"), {"Layout": fact("Epoch", Epoch(1))}, {})),
        ("register: unknown fact name 'Colour'", lambda: contracts.register("Colour", Resolution("x", "y"))),
    ]:
        try:
            fn()
            raise AssertionError(f"expected ValueError: {text}")
        except ValueError as e:
            assert text in str(e), (text, str(e))


def test_ledger_says_everything_in_one_line():
    c = Contract("load:comfyui.sampler", "comfyui.sampler", ("Prediction",), ("Prediction",))
    declared = Fact("Prediction", Prediction("v"), Source("file", "m.safetensors#modelspec.prediction_type"),
                    Certainty.DECLARED)
    chosen = Fact("Prediction", Prediction("eps", False), Source("engine", "model_sampling"), Certainty.DECLARED)
    ledger = record.Ledger()
    with resolutions(("Prediction", Resolution("set the sampler", "set_sampling"))):
        ledger.extend(decide(c, {"Prediction": declared}, {"Prediction": chosen}))
    ledger.extend(decide(c, {}, {"Prediction": chosen}))
    ledger.extend(decide(c, {}, {"Prediction": chosen}, policies.Policy(on_unknown_meaning_changing="require")))
    ledger.extend(decide(c, {"Prediction": declared}, {"Prediction": chosen}))
    ledger.extend(decide(c, {"Prediction": declared}, {"Prediction": chosen}, STRICT))
    said = ("[entail] {v} at load:comfyui.sampler: Prediction declared v-prediction (file: "
            "m.safetensors#modelspec.prediction_type, declared); comfyui.sampler uses eps without zero terminal SNR "
            "(engine: model_sampling, declared); rule: " + RULES["no_resolution"] + "; {end}")
    unknown = ("[entail] unknown at load:comfyui.sampler: Prediction nothing declared; comfyui.sampler uses eps "
               "without zero terminal SNR (engine: model_sampling, declared); rule: " + RULES["undeclared"])
    assert ledger.lines() == [
        "[entail] resolved at load:comfyui.sampler: Prediction declared v-prediction "
        "(file: m.safetensors#modelspec.prediction_type, declared); comfyui.sampler uses eps without zero terminal SNR "
        "(engine: model_sampling, declared); rule: " + RULES["resolved"] + "; changed: set the sampler "
        "(eps without zero terminal SNR -> v-prediction)",
        unknown,
        unknown + "; stops here",
        said.format(v="broken", end="reported, not stopped"),
        said.format(v="refused", end="stops here"),
    ], ledger.lines()
    assert [d.verdict for d in ledger.blocking()] == [Verdict.UNKNOWN, Verdict.REFUSED]
    assert [d.verdict for d in ledger.broken()] == [Verdict.BROKEN, Verdict.REFUSED]
    j = ledger.to_json()["decisions"][0]
    assert (j["verdict"], j["handle"], j["declared"]["source"]["where"], j["chosen"]["value"]) == \
        ("resolved", "set_sampling", "m.safetensors#modelspec.prediction_type", "eps without zero terminal SNR")
    found = ledger.locate()   # M7.1: the first boundary that broke is the problem area
    assert (found.broken_at, found.all_intact) == (ledger.broken()[0].contract.boundary, False), found


def test_policy_from_the_environment():
    p = policies.from_env({"ENTAIL": "debug", "ENTAIL_POLICY": "refuse", "ENTAIL_UNKNOWN": "stop",
                           "ENTAIL_FACT_POLICY": "Prediction=resolve, Template=report"})
    assert (p.mode, p.on_mismatch, p.on_unknown_meaning_changing) == ("debug", "refuse", "stop")
    assert p.mismatch_setting("Prediction") == "resolve" and p.mismatch_setting("Layout") == "refuse"
    assert p.unknown_setting("Template", True) == "report" and p.unknown_setting("Layout", True) == "stop"
    assert policies.from_env({}) == policies.Policy()
    d = policies.Policy()   # the defaults stop nowhere (M5.4)
    assert (d.on_broken, d.on_unknown_meaning_changing, d.on_unknown_other) == ("report", "report", "report")
    assert not d.stops("Layout") and not d.stops_unknown("Layout", True)
    p = policies.from_env({"ENTAIL": "load", "ENTAIL_ON_BROKEN": "stop", "ENTAIL_FACT_POLICY": "Template=report"})
    assert p.stops("Layout") and not p.stops("Template")
    p = policies.from_env({"ENTAIL": "load", "ENTAIL_FACT_POLICY": "Layout=stop"})
    assert p.stops("Layout") and not p.stops("Template") and p.stops_unknown("Layout", False)
    assert policies.Policy(mode="debug").stops("Template")
    for env, text in [({"ENTAIL": "on"}, "policy mode: 'on' is not one of"),
                      ({"ENTAIL_ON_BROKEN": "warn"}, "policy on_broken: 'warn' is not one of"),
                      ({"ENTAIL_FACT_POLICY": "Prediction"}, "expected Name=setting"),
                      ({"ENTAIL_FACT_POLICY": "Colour=report"}, "unknown fact name 'Colour'"),
                      ({"ENTAIL_FACT_POLICY": "Layout=maybe"}, "override for Layout: 'maybe' is not one of")]:
        try:
            policies.from_env(env)
            raise AssertionError(text)
        except ValueError as e:
            assert text in str(e), (text, str(e))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
