"""Tests for coverage.py: what a consumer took from what it was given. Run: python tests/test_coverage.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, coverage  # noqa: E402


def test_count():
    cov = coverage.count(["a", "b", "c"], ["a", "x"])
    assert (cov.total, cov.taken, cov.left) == (3, 1, ("b", "c")) and not cov.none_taken and not cov.all_taken
    assert coverage.count(["a"], []).none_taken and coverage.count([], []).all_taken


def test_check_outcomes():
    assert coverage.check(coverage.count(["a"], ["a"]), "E", "LoRA modules") == "all"
    assert coverage.check(coverage.count(["a", "b"], ["a"]), "E", "LoRA modules") == "partial"
    try:
        coverage.check(coverage.count(["a", "b"], []), "E", "LoRA modules")
        raise AssertionError("expected RoleError")
    except core.RoleError as e:
        assert "2 of 2 were not taken" in str(e)
    try:
        coverage.check(coverage.count(["a", "b"], ["a"]), "E", "config keys", stop_when="any")
        raise AssertionError("expected RoleError")
    except core.RoleError as e:
        assert "(b)" in str(e)


def test_config_keys_use_the_same_count():
    """rolebench case 15: an unknown key must not be kept silently. Same message as before the refactor."""
    core.set_mode("load")
    try:
        core.check_config_keys({"rope_theta": 1, "rope_scale": 2}, {"rope_theta"}, "LlamaConfig")
        raise AssertionError("expected RoleError")
    except core.RoleError as e:
        assert str(e) == "LlamaConfig: unrecognised config keys ['rope_scale']"
    finally:
        core.set_mode("off")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
