"""Unit tests for the shared core. Run: python tests/test_core.py (no pytest needed)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
import entail as rc  # noqa: E402


def expect_error(fn, text):
    try:
        fn()
    except rc.RoleError as e:
        assert text in str(e), str(e)
        return str(e)
    raise AssertionError("expected RoleError containing " + text)


def test_off_mode_is_inert():
    rc.set_mode("off")
    t = rc.tag(torch.zeros(2), rc.Positions("absolute"))
    assert rc.facts_of(t) == {}

    @rc.boundary(q=rc.Positions("absolute"))
    def f(*, q):
        return q.sum()

    assert float(f(q=torch.ones(3))) == 3.0  # no declaration needed when off
    rc.check_props(rc.ModelProps(softcap=50.0), rc.KernelCaps(), "sdpa")  # no error when off


def test_boundary_checks_in_debug():
    rc.set_mode("debug")

    @rc.boundary(q=rc.Positions("absolute"))
    def f(*, q):
        return q

    ok = rc.tag(torch.zeros(2), rc.Positions("absolute"))
    f(q=ok)
    # resolution first (M4.1): chunk-relative positions with their offset are made absolute ...
    assert f(q=rc.tag(torch.zeros(2), rc.Positions("chunk_relative", 256))).tolist() == [256.0, 256.0]
    rc.set_policy("refuse")   # ... and under `refuse` the mismatch stops the call
    try:
        expect_error(lambda: f(q=rc.tag(torch.zeros(2), rc.Positions("chunk_relative", 256))),
                     "the policy repairs nothing")
    finally:
        rc.set_policy("resolve")
    expect_error(lambda: f(q=torch.zeros(2)), "nothing declares it")
    expect_error(lambda: f(torch.zeros(2)), "must be passed by keyword")


def test_closed_layout_set():
    rc.set_mode("debug")
    accepted = (rc.Layout("q8_0", packing="interleaved"),)

    @rc.boundary(w=accepted)
    def dequant(*, w):
        return w

    dequant(w=rc.tag(torch.zeros(4), rc.Layout("q8_0", packing="interleaved")))
    expect_error(lambda: dequant(w=rc.tag(torch.zeros(4), rc.Layout("q8_0", packing="split"))),
                 "no resolution is registered")
    try:
        rc.Layout("q9_9")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown layout kind must be rejected")


def test_load_checks():
    rc.set_mode("load")
    msg = expect_error(lambda: rc.check_props(rc.ModelProps(softcap=50.0), rc.KernelCaps(), "sdpa"), "softcap=50.0")
    rc.check_props(rc.ModelProps(softcap=50.0), rc.KernelCaps(softcap=True), "eager")
    expect_error(lambda: rc.check_config_keys({"rope_scale": 4.0, "hidden_size": 8}, {"hidden_size"}, "config"),
                 "rope_scale")
    e, h = torch.ones(3, 2), torch.zeros(3, 2)
    expect_error(lambda: rc.check_tied(True, e, h, "loader"), "different head")
    rc.check_tied(True, e, e.clone(), "loader")
    return msg


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    rc.set_mode("off")
