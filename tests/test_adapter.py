"""Tests for the transformers adapter's verdict and for install/uninstall. Run: python tests/test_adapter.py"""
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from entail import core  # noqa: E402
from entail.adapters import transformers_adapter as adapter  # noqa: E402


def cfg(**kw):
    return SimpleNamespace(**kw)


def test_verdict_violation():
    kind, msg = adapter.verdict(cfg(attn_logit_softcapping=50.0), "sdpa")
    assert kind == "violation" and "attn_logit_softcapping=50.0" in msg
    assert adapter.verdict(cfg(attn_logit_softcapping=50.0), "eager") is None
    assert adapter.verdict(cfg(attn_logit_softcapping=50.0), "paged|sdpa")[0] == "violation"


def test_verdict_nothing_declared():
    assert adapter.verdict(cfg(), "sdpa") is None
    assert adapter.verdict(cfg(), "flash_attention_2") is None  # nothing to lose, nothing to say


def test_verdict_uncovered():
    kind, msg = adapter.verdict(cfg(attn_logit_softcapping=50.0), "flash_attention_2")
    assert kind == "uncovered" and "not in the capability table" in msg


def test_nested_text_config():
    c = cfg(text_config=cfg(attn_logit_softcapping=30.0))
    assert adapter.verdict(c, "sdpa")[0] == "violation"


def test_install_is_reversible():
    from transformers import PreTrainedModel
    before = PreTrainedModel._check_and_adjust_attn_implementation
    assert adapter.install() == 1
    assert adapter.install() == 0  # already installed
    assert PreTrainedModel._check_and_adjust_attn_implementation is not before
    assert adapter.uninstall() == 1
    assert adapter.uninstall() == 0
    assert PreTrainedModel._check_and_adjust_attn_implementation is before


def _gemma_on_meta(impl):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    path = os.path.join(os.path.expanduser(os.environ.get("ENTAIL_TEST_MODELS", "~/models")), "gemma-2-2b-it")
    if not os.path.isdir(path):
        return None
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), attn_implementation=impl)


def test_resolve_switches_to_an_implementation_that_honours_the_model():
    """Under the default policy a dropped property is fixed, not refused: sdpa becomes eager, and it is said."""
    from entail.adapters import _shared

    adapter.install()
    core.set_mode("load")
    core.set_policy("resolve")
    before = len(_shared.RESOLUTIONS)
    try:
        model = _gemma_on_meta("sdpa")
        if model is None:
            print("skip (no local model)")
            return
        assert model.config._attn_implementation == "eager", model.config._attn_implementation
        last = _shared.RESOLUTIONS[before]
        assert last["from"] == "sdpa" and last["to"] == "eager", last
    finally:
        core.set_mode("off")
        adapter.uninstall()


def test_refuse_still_refuses():
    adapter.install()
    core.set_mode("load")
    core.set_policy("refuse")
    try:
        try:
            model = _gemma_on_meta("sdpa")
        except core.RoleError as e:
            assert "does not honour" in str(e), e
        else:
            if model is not None:
                raise AssertionError("refuse policy let a dropped property through")
    finally:
        core.set_policy("resolve")
        core.set_mode("off")
        adapter.uninstall()


def test_nothing_to_resolve_to_is_still_refused():
    """The paged kernels have no alternative that honours softcap, so resolve has to stop and say what works."""
    cfg = SimpleNamespace(attn_logit_softcapping=50.0)
    from entail.adapters import _shared

    assert _shared.choose(cfg, "transformers", ["paged|eager", "paged|sdpa"]) is None
    assert _shared.choose(cfg, "transformers", adapter.PREFERENCE) == "eager"


def test_mode_off_does_nothing():
    """With the mode off the wrapper must not raise even on a violating pair."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    path = os.path.join(os.path.expanduser(os.environ.get("ENTAIL_TEST_MODELS", "~/models")), "gemma-2-2b-it")
    if not os.path.isdir(path):
        print("skip test_mode_off_does_nothing (no local model)")
        return
    adapter.install()
    core.set_mode("off")
    try:
        with torch.device("meta"):
            AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), attn_implementation="sdpa")
    finally:
        adapter.uninstall()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
