"""Tests for contract.py: a declaration against what the consumer uses. Run: python tests/test_contract.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import contract, core, declared  # noqa: E402
from entail.adapters import _shared  # noqa: E402
from entail.facts import Prediction  # noqa: E402

V_FILE = declared.from_header([], {"modelspec.prediction_type": "v"})
NOOB = declared.from_header(["v_pred", "ztsnr"], {})


def _raises(fn):
    try:
        fn()
    except core.RoleError as e:
        return str(e)
    raise AssertionError("expected RoleError")


def test_agreement_does_nothing():
    calls = []
    fact, out = contract.reconcile(Prediction, NOOB, Prediction("v", True), "E", resolve=calls.append)
    assert fact == Prediction("v", True) and out is None and not calls


def test_declaration_wins_and_is_said():
    """The M7 case: the file says v, the engine set eps up. The declaration's open zsnr keeps the engine's value."""
    calls, before = [], len(_shared.RESOLUTIONS)
    fact, out = contract.reconcile(Prediction, V_FILE, Prediction("eps", False), "E",
                                   resolve=lambda f: calls.append(f) or "switched", what="prediction type")
    assert fact == Prediction("v", False) and out == "switched" and calls == [Prediction("v", False)]
    note = _shared.RESOLUTIONS[before]
    assert note["source"] == "metadata modelspec.prediction_type" and note["to"] == str(Prediction("v", False))


def test_explicit_choice_is_not_overridden():
    msg = _raises(lambda: contract.reconcile(Prediction, NOOB, Prediction("eps", False), "E", resolve=lambda f: f,
                                             explicit=True))
    assert "set explicitly" in msg and "key 'v_pred'" in msg


def test_explicit_schedule_detail_is_the_users():
    """A zsnr=false node on a ztsnr checkpoint (the user's own workflow) is a choice, not a contradiction."""
    fact, out = contract.reconcile(Prediction, NOOB, Prediction("v", False), "E", resolve=lambda f: 1 / 0,
                                   explicit=True)
    assert fact == Prediction("v", False) and out is None


def test_refuse_and_no_resolver_stop():
    core.set_policy("refuse")
    try:
        assert "says v-prediction" in _raises(lambda: contract.reconcile(Prediction, V_FILE, Prediction("eps"), "E",
                                                                         resolve=lambda f: f))
    finally:
        core.set_policy("resolve")
    assert "Nothing here can switch it" in _raises(lambda: contract.reconcile(Prediction, V_FILE, Prediction("eps"),
                                                                              "E"))


def test_evidence_decides_only_when_nothing_is_declared():
    fact, _ = contract.reconcile(Prediction, None, Prediction("eps", False), "E", resolve=lambda f: f,
                                 evidence=(Prediction("v"), "first model call, cosine -0.00"))
    assert fact == Prediction("v", False)
    # A declaration outranks behaviour: the file says v, the behaviour looked eps (M7), the engine uses v.
    fact, out = contract.reconcile(Prediction, V_FILE, Prediction("v", False), "E", resolve=lambda f: 1 / 0,
                                   evidence=(Prediction("eps"), "first model call, cosine 0.94"))
    assert fact == Prediction("v", False) and out is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
