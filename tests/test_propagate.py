"""Tests for fact propagation (the diagnosis site, M7.1). Run: python tests/test_propagate.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import boundaries, core, load, sites  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.core import RoleError, boundary  # noqa: E402
from entail.facts import Layout, Positions, Reduction  # noqa: E402
from entail.propagate import RolePropagation  # noqa: E402


def setup():
    core.set_mode("debug")


def test_facts_survive_a_view():
    setup()
    with RolePropagation():
        t = core.tag(torch.zeros(4, 8), Positions("absolute"))
        v = t.reshape(8, 4)
        assert core.facts_of(v).get("Positions") == Positions("absolute"), core.facts_of(v)
        assert core.facts_of(v.contiguous().clone()).get("Positions") == Positions("absolute")


def test_transpose_invalidates_an_axis_fact_instead_of_dropping_it():
    setup()
    with RolePropagation():
        w = core.tag(torch.zeros(4, 8), Layout("dense"), Positions("absolute"))
        t = w.transpose(0, 1)
        facts = core.facts_of(t)
        assert "Layout" not in facts, facts
        assert facts["Invalidated"].kind == "Layout" and "transpose" in facts["Invalidated"].why, facts
        assert facts["Positions"] == Positions("absolute"), facts  # not axis-dependent, still true


def test_a_boundary_says_why_the_fact_is_gone():
    setup()

    @boundary("kernel", w=Layout("dense"))
    def kernel(*, w):
        return w

    with RolePropagation():
        w = core.tag(torch.zeros(4, 8), Layout("dense"))
        moved = w.transpose(0, 1)
    try:
        kernel(w=moved)
    except RoleError as e:
        assert "made untrue by transpose" in str(e), e
    else:
        raise AssertionError("the boundary accepted a value whose layout fact was invalidated")
    d = load.LEDGER.decisions[-1]
    assert (d.contract.boundary, d.rule, d.lost_by) == ("boundary:kernel", RULES["invalidated"], "transpose"), d


def test_mixing_two_frames_is_an_error():
    """A decision at the operation, decided with the policy: debug mode stops (refused), before the sum is used."""
    setup()
    with RolePropagation():
        a = core.tag(torch.zeros(4), Positions("absolute"))
        b = core.tag(torch.zeros(4), Positions("chunk_relative", offset=16))
        try:
            a + b
        except RoleError as e:
            assert "refused at op:add: Positions" in str(e) and RULES["disagree"] in str(e), e
        else:
            raise AssertionError("adding absolute and chunk-relative positions was accepted")


def test_a_disagreement_can_be_recorded_while_the_run_goes_on():
    """on_conflict="record" (measuring a run to its end): broken and reported once, not stopped."""
    setup()
    boundaries.REPEATS.clear()
    before = len(load.LEDGER.decisions)
    with RolePropagation(on_conflict="record"):
        a = core.tag(torch.zeros(4), Positions("absolute"))
        b = core.tag(torch.zeros(4), Positions("chunk_relative", offset=16))
        for _ in range(3):
            c = a + b
    new = load.LEDGER.decisions[before:]
    assert [(d.contract.boundary, d.verdict, d.blocking) for d in new] == [("op:add", Verdict.BROKEN, False)], new
    assert core.facts_of(c)["Positions"] == Positions("absolute"), "the first input's fact is carried on"


def test_matching_facts_combine_quietly():
    setup()
    with RolePropagation():
        a = core.tag(torch.zeros(4), Reduction("P"))
        b = core.tag(torch.zeros(4), Reduction("P"))
        assert core.facts_of(a + b).get("Reduction") == Reduction("P")


def test_an_invalidated_fact_is_carried_not_dropped():
    """Turning it back does not make the fact true again, and the reason must not be lost on the way."""
    setup()
    with RolePropagation():
        w = core.tag(torch.zeros(4, 8), Layout("dense"), Reduction("P"))
        back = w.transpose(0, 1).t().reshape(4, 8)
        mark = core.facts_of(back)["Invalidated"]
        assert mark.kind == "Layout,Reduction" and mark.why == "transpose", mark   # the operation that made them untrue
        both = back + w.flip(0)   # two markers merge; they are not a disagreement
        assert core.facts_of(both)["Invalidated"].kind == "Layout,Reduction", core.facts_of(both)


def test_writing_into_a_value_invalidates_its_facts():
    """A KV cache written in place keeps the same tensor, so a stale length would survive unnoticed."""
    setup()
    from entail.facts import Valid

    with RolePropagation():
        cache = core.tag(torch.zeros(1, 2, 8, 4), Valid(length=3))
        cache[:, :, 3:4, :] = torch.ones(1, 2, 1, 4)
        facts = core.facts_of(cache)
        assert "Valid" not in facts, facts
        assert facts["Invalidated"].kind == "Valid" and facts["Invalidated"].why.startswith("setitem, which wrote"), facts
        other = core.tag(torch.zeros(1, 2, 8, 4), Valid(length=3))
        other[:, :, 3:4, :].copy_(torch.ones(1, 2, 1, 4))   # through a view: the value it is a view of changes too
        assert core.facts_of(other)["Invalidated"].why.startswith("copy_"), core.facts_of(other)


def test_unknown_ops_stop_propagation():
    """A fact must not survive an operation whose meaning we have not declared."""
    setup()
    with RolePropagation():
        a = core.tag(torch.zeros(4, 4), Layout("dense"))
        out = torch.matmul(a, a)
        assert core.facts_of(out) == {}, core.facts_of(out)


def test_off_mode_is_inert():
    core.set_mode("off")
    with RolePropagation():
        t = core.tag(torch.zeros(4), Positions("absolute"))
        assert core.facts_of(t.reshape(2, 2)) == {}
    core.set_mode("debug")


def test_the_diagnosis_site_is_entered_only_in_debug_mode():
    for mode, entered in (("off", False), ("load", False), ("debug", True)):
        core.set_mode(mode)
        assert isinstance(sites.debug_propagation(), RolePropagation) is entered, mode
    core.set_mode("debug")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
