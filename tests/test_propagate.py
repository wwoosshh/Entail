"""Tests for fact propagation. Run: python tests/test_propagate.py"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core  # noqa: E402
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
        assert "made untrue by aten.transpose" in str(e), e
    else:
        raise AssertionError("the boundary accepted a value whose layout fact was invalidated")


def test_mixing_two_frames_is_an_error():
    setup()
    with RolePropagation():
        a = core.tag(torch.zeros(4), Positions("absolute"))
        b = core.tag(torch.zeros(4), Positions("chunk_relative", offset=16))
        try:
            a + b
        except RoleError as e:
            assert "disagree about Positions" in str(e), e
        else:
            raise AssertionError("adding absolute and chunk-relative positions was accepted")


def test_matching_facts_combine_quietly():
    setup()
    with RolePropagation():
        a = core.tag(torch.zeros(4), Reduction("P"))
        b = core.tag(torch.zeros(4), Reduction("P"))
        assert core.facts_of(a + b).get("Reduction") == Reduction("P")


def test_writing_into_a_value_invalidates_its_facts():
    """A KV cache written in place keeps the same tensor, so a stale length would survive unnoticed."""
    setup()
    from entail.facts import Valid

    with RolePropagation():
        cache = core.tag(torch.zeros(1, 2, 8, 4), Valid(length=3))
        cache[:, :, 3:4, :] = torch.ones(1, 2, 1, 4)
        facts = core.facts_of(cache)
        assert "Valid" not in facts, facts
        assert facts["Invalidated"].kind == "Valid" and "wrote into" in facts["Invalidated"].why, facts


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
