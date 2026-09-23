"""Tests for TIME and SPECIALIZATION on the host side (ROADMAP M5.2): buffer epochs, values that read a buffer later,
binding them at the hand-over, and artifacts reused under other conditions. Pure Python: no torch, no engine.
Run: python tests/test_epochs.py"""
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
from entail import core, epochs, load  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.core import RoleError  # noqa: E402
from entail.facts import Assumed, Epoch  # noqa: E402

B, C = "container:test.time", "test.reader"


class Owner:      # a cache that keeps a position counter
    pass


class Reader:     # a mask function that reads the counter when attention runs
    pass


def in_mode(mode, fn, *args, **kw):
    """Run fn in a mode; returns (result, the decisions it recorded, the RoleError text or None)."""
    n = len(load.LEDGER.decisions)
    core.set_mode(mode)
    try:
        with redirect_stdout(io.StringIO()):
            return fn(*args, **kw), load.LEDGER.decisions[n:], None
    except RoleError as e:
        return None, load.LEDGER.decisions[n:], str(e)
    finally:
        core.set_mode("off")


def test_a_buffer_counts_its_writes():
    epochs.reset()
    cache = Owner()
    assert epochs.epoch(cache, "layer 0 length") == 0
    assert epochs.advance(cache, "layer 0 length") == 1 and epochs.advance(cache, "layer 0 length") == 2
    assert epochs.epoch(cache, "layer 1 length") == 0 and epochs.epoch(Owner(), "layer 0 length") == 0


def test_a_value_read_before_the_buffer_is_written_again_passes():
    epochs.reset()
    cache, counter, mask = Owner(), Reader(), Reader()
    epochs.live(counter, cache, "length")
    epochs.carried(mask, counter)
    _, made, err = in_mode("load", epochs.read, B, C, "attention", mask)
    assert err is None and made == [] and epochs.stats(B)["passed"] == {"epoch_stale": 1}


def test_a_stale_read_is_refused():
    """rolebench 10: the mask was made from the counter, the update wrote the counter, attention reads it."""
    epochs.reset()
    cache, counter, mask = Owner(), Reader(), Reader()
    epochs.live(counter, cache, "length")
    epochs.carried(mask, counter)
    epochs.advance(cache, "length")                                  # the layer's update
    _, made, err = in_mode("load", epochs.read, B, C, "attention", mask)
    assert err is not None and "1 write(s) in between" in err, err
    (d,) = made
    assert d.verdict is Verdict.REFUSED and d.rule == RULES["epoch_stale"] and d.blocking, d
    assert d.declared.value == Epoch(0, owner="Owner.length") and d.chosen.value == Epoch(1, owner="Owner.length")


def test_binding_at_the_hand_over_is_the_resolution():
    epochs.reset()
    cache, counter = Owner(), Reader()
    epochs.live(counter, cache, "length")
    snap = Reader()                                                   # what snapshot() returns: reads nothing later
    given, made, err = in_mode("load", epochs.bind, B, C, "mask builder", counter, lambda: snap)
    assert given is snap and err is None, (given, err)
    (d,) = made
    assert d.verdict is Verdict.RESOLVED and d.rule == RULES["epoch_live"] and d.handle == "epochs.bind", d
    epochs.advance(cache, "length")
    mask = Reader()
    epochs.carried(mask, given)                                       # built from the snapshot: nothing to follow
    _, made, err = in_mode("load", epochs.read, B, C, "attention", mask)
    assert err is None and made == []
    # the same resolution again is counted, not recorded again
    _, made, _ = in_mode("load", epochs.bind, B, C, "mask builder", counter, lambda: Reader())
    assert made == [] and epochs.stats(B)["resolved"] == 2


def test_a_policy_that_refuses_binds_nothing_and_the_read_is_refused():
    epochs.reset()
    cache, counter, mask = Owner(), Reader(), Reader()
    epochs.live(counter, cache, "length")
    core.set_policy("refuse")
    try:
        given, made, _ = in_mode("load", epochs.bind, B, C, "mask builder", counter, lambda: Reader())
        assert given is counter and made == []
        epochs.carried(mask, given)
        epochs.advance(cache, "length")
        _, made, err = in_mode("load", epochs.read, B, C, "attention", mask)
        assert err is not None and made[0].verdict is Verdict.REFUSED
    finally:
        core.set_policy("resolve")


def test_an_artifact_made_for_other_conditions_is_remade_or_refused():
    """rolebench 09: a graph captured for valid length 100 replayed at 160; 11: compiled on an empty input."""
    epochs.reset()
    epochs.assume(B, ("graph", 1), valid_length=100)
    assert in_mode("load", epochs.reuse, B, C, "replay", ("graph", 1), valid_length=100)[0] == "as_is"
    dropped = []
    status, made, err = in_mode("load", epochs.reuse, B, C, "replay", ("graph", 1),
                                remake=lambda: dropped.append(1), valid_length=160)
    assert status == "remade" and dropped == [1] and err is None, (status, err)
    (d,) = made
    assert d.verdict is Verdict.RESOLVED and d.rule == RULES["assumed_changed"], d
    assert d.declared.value == Assumed((("valid_length", 100),)) and d.chosen.value == Assumed((("valid_length", 160),))
    assert "valid_length: 100 -> 160" in d.note
    # after a remake the caller records what it made the artifact for
    epochs.assume(B, ("graph", 1), valid_length=160)
    epochs.assume(B, ("graph", 2), extra_rows_zero=True)
    _, made, err = in_mode("load", epochs.reuse, B, C, "compiled call", ("graph", 2), extra_rows_zero=False)
    assert err is not None and made[0].verdict is Verdict.REFUSED and "nothing can remake it" in made[0].note


def test_an_artifact_nobody_recorded_is_not_checked():
    epochs.reset()
    assert in_mode("load", epochs.reuse, B, C, "replay", "never seen", valid_length=1)[0] == "as_is"
    assert epochs.stats(B)["skipped"] == 1 and epochs.stats(B)["checks"] == 0


def test_entail_failing_never_breaks_the_engine():
    epochs.reset()
    result, made, err = in_mode("load", epochs.guarded, B, C, lambda: {}["missing"])
    assert result is None and err is None and made[0].verdict is Verdict.UNKNOWN and "KeyError" in made[0].note


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
