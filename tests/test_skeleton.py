"""Tests for the M0.3 skeleton (LIBRARY_DESIGN.md 4): every planned module imports, every planned function says which
milestone builds it, and nothing already public changes. Run: python tests/test_skeleton.py"""
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import entail  # noqa: E402
from entail import boundaries, caps, contracts, core, facts, manifest, policies, record, sites, sources, testing  # noqa: E402
from entail.adapters import base  # noqa: E402

PLANNED = [
    (sources.read_all, ("x",), "M2.1"),
    (sources.merge, ([],), "M2.2"),
    (manifest.load, ("x",), "M2.3"),
    (manifest.find, ("x", []), "M2.3"),
    (manifest.infer, ("x",), "M2.3"),
    (caps.load_table, ("x",), "M3.1"),
    (caps.lookup, ([], "c", "f"), "M3.1"),
    (caps.probe, ("e", "c", "f"), "M3.1"),
    (contracts.decide, (None, {}, {}, None, None), "M1.2"),
    (policies.from_env, (), "M1.2"),
    (sites.check_static, ("m", "e", {}), "M3.4"),
    (sites.at_load, ("m", "e", {}, None), "M3.2"),
    (sites.at_container, (None, "w", None), "M5.1"),
    (sites.at_request, ({}, {}, None), "M5.3"),
    (sites.debug_propagation, (), "M7.1"),
    (testing.problems, (), "M7.2"),
]


def test_public_names_unchanged():
    """The package still exports the functions `boundary` and `policy` after the new submodules are imported.
    (A submodule named `boundary` or `policy` would have replaced them; that is why the modules are plural.)"""
    assert entail.boundary is core.boundary and callable(entail.boundary)
    assert entail.policy is core.policy and callable(entail.policy)


def test_planned_functions_name_their_milestone():
    for fn, args, milestone in PLANNED:
        try:
            fn(*args)
            raise AssertionError(f"{fn.__module__}.{fn.__name__} should not run yet")
        except NotImplementedError as e:
            assert str(e).startswith(milestone), (fn.__name__, str(e))


def test_ledger_methods_are_planned():
    ledger = record.Ledger()
    for call, milestone in ((lambda: ledger.add(None), "M1.2"), (ledger.lines, "M1.2"), (ledger.to_json, "M1.2"),
                            (ledger.locate, "M7.1")):
        try:
            call()
            raise AssertionError("should not run yet")
        except NotImplementedError as e:
            assert str(e).startswith(milestone), str(e)


def test_boundaries_is_the_same_objects():
    assert boundaries.boundary is core.boundary and boundaries.carry is core.carry and boundaries.tag is core.tag


def test_fact_envelope():
    f = facts.Fact("Prediction", facts.Prediction("v"), facts.Source("file", "m.safetensors#modelspec.prediction_type"),
                   facts.Certainty.DECLARED)
    assert f.kind == "PROPERTY" and f.vocab_version == facts.VOCAB_VERSION
    assert set(facts.VOCABULARY.values()) <= facts.FACT_KINDS
    assert [c.value for c in facts.Certainty] == ["declared", "verified", "inferred", "defaulted", "unknown"]


def test_verdicts_and_policy_defaults():
    assert [v.value for v in contracts.Verdict] == ["pass", "resolved", "refused", "unknown"]
    p = policies.Policy()
    assert (p.mode, p.on_mismatch, p.on_unknown_meaning_changing, p.on_unknown_other) == \
        ("off", "resolve", "require", "report")
    assert sources.DEFAULT_PRECEDENCE[:3] == ("user", "manifest", "file")
    assert set(sites.SITES) == {"load", "container", "request", "debug"} and set(sites.BUDGET) == set(sites.SITES)


def test_adapter_interface_has_three_parts():
    methods = {name for name, _ in inspect.getmembers(base.Adapter, inspect.isfunction) if not name.startswith("_")}
    assert methods == {"hooks", "read_choice", "handles"}, methods


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
