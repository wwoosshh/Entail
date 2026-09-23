"""Tests for the legacy RoPE spelling resolver. Run: python tests/test_rope_alias.py  (CPU, local configs only)

The property that matters: whatever route the old name takes after the config is built, the result is the
rope_parameters that the same key would have produced in config.json.
"""
import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load  # noqa: E402
from entail.adapters import rope_alias  # noqa: E402
from entail.contracts import Verdict  # noqa: E402

MODELS = {name: os.path.join(os.path.expanduser(os.environ.get("ENTAIL_TEST_MODELS", "~/models")), name)
          for name in ("Qwen3-4B", "Llama-3.2-3B-Instruct", "gemma-3-1b-it", "gemma-2-2b-it")}
YARN = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
LINEAR = {"rope_type": "linear", "factor": 2.0}


def _have(name):
    return os.path.isfile(os.path.join(MODELS[name], "config.json"))


def _via_config_json(name, edit):
    """rope_parameters as transformers builds them when `edit` is written into config.json (the reference)."""
    from transformers import AutoConfig

    d = tempfile.mkdtemp(prefix="rope_alias_test_")
    try:
        with open(os.path.join(MODELS[name], "config.json"), encoding="utf-8") as f:
            raw = json.load(f)
        raw.update(copy.deepcopy(edit))
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(raw, f)
        was = core.mode()
        core.set_mode("off")  # the reference is transformers alone
        try:
            return AutoConfig.from_pretrained(d).rope_parameters
        finally:
            core.set_mode(was)
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _On:
    def __init__(self, policy="resolve"):
        self.policy = policy

    def __enter__(self):
        rope_alias.install()
        core.set_mode("load")
        core.set_policy(self.policy)
        self.n = len(load.LEDGER.decisions)
        return self

    def new(self):
        """The repairs made since entering: resolved decisions in the ledger."""
        return [d for d in load.LEDGER.decisions[self.n:] if d.verdict is Verdict.RESOLVED]

    def __exit__(self, *exc):
        core.set_policy("resolve")
        core.set_mode("off")
        rope_alias.uninstall()


def test_every_route_equals_config_json():
    """kwarg to from_pretrained and attribute set afterwards, for both old names, on four model families."""
    from transformers import AutoConfig

    checked = 0
    for name in MODELS:
        if not _have(name):
            continue
        for key, val in (("rope_theta", 12345.0), ("rope_scaling", YARN), ("rope_scaling", LINEAR)):
            ref = _via_config_json(name, {key: val})
            with _On() as on:
                by_kwarg = AutoConfig.from_pretrained(MODELS[name], **{key: copy.deepcopy(val)}).rope_parameters
                cfg = AutoConfig.from_pretrained(MODELS[name])
                setattr(cfg, key, copy.deepcopy(val))
                by_attr = cfg.rope_parameters
                assert on.new(), f"{name} {key}: nothing was said"
            assert by_kwarg == ref, f"{name} {key}={val} by kwarg:\n  {by_kwarg}\n  config.json gives {ref}"
            assert by_attr == ref, f"{name} {key}={val} by attribute:\n  {by_attr}\n  config.json gives {ref}"
            checked += 1
    if not checked:
        print("skip (no local models)")
    return checked


def test_without_entail_the_value_is_lost():
    """The measured behaviour this exists for (audits/alias_probe.py), so the test above means something."""
    from transformers import AutoConfig

    if not _have("Qwen3-4B"):
        print("skip (no local model)")
        return
    core.set_mode("off")
    rope_alias.install()
    try:
        assert AutoConfig.from_pretrained(MODELS["Qwen3-4B"], rope_theta=12345.0).rope_parameters["rope_theta"] \
            == 1000000
        cfg = AutoConfig.from_pretrained(MODELS["Qwen3-4B"])
        cfg.rope_scaling = dict(YARN)
        assert "rope_theta" not in cfg.rope_parameters
    finally:
        rope_alias.uninstall()


def test_the_same_value_stated_again_is_quiet():
    """vLLM's patch_rope_parameters writes config.rope_theta back with the value it just read."""
    from transformers import AutoConfig

    if not _have("Qwen3-4B"):
        print("skip (no local model)")
        return
    with _On() as on:
        cfg = AutoConfig.from_pretrained(MODELS["Qwen3-4B"])
        cfg.rope_theta = 1000000  # what the checkpoint already says
        assert not on.new()
        cfg.rope_theta = 10000.0
        cfg.rope_theta = 10000.0
        assert len(on.new()) == 1 and cfg.rope_parameters["rope_theta"] == 10000.0
        cfg.rope_parameters["rope_theta"] = 500000.0  # the new spelling, in place
        cfg.rope_theta = 10000.0  # a stale restatement must not undo it
        assert cfg.rope_parameters["rope_theta"] == 500000.0


def test_refuse_raises_only_when_something_would_be_lost():
    from transformers import AutoConfig

    if not _have("Qwen3-4B"):
        print("skip (no local model)")
        return
    with _On("refuse"):
        cfg = AutoConfig.from_pretrained(MODELS["Qwen3-4B"])
        cfg.rope_theta = 1000000  # nothing lost
        try:
            cfg.rope_scaling = dict(YARN)
        except core.RoleError as e:
            assert "rope_parameters" in str(e) and "policy refuses mismatches" in str(e), e
        else:
            raise AssertionError("refuse let a lossy rope_scaling write through")


def test_classes_that_read_the_old_name_are_left_alone():
    from transformers import EsmConfig

    with _On() as on:
        c = EsmConfig(vocab_size=33, position_embedding_type="rotary")
        c.rope_theta = 500.0
        assert c.rope_theta == 500.0 and not on.new()


def test_install_is_reversible():
    from transformers.configuration_utils import PreTrainedConfig

    s, f = PreTrainedConfig.__setattr__, PreTrainedConfig.__dict__["from_dict"].__func__
    assert rope_alias.install() == 1 and rope_alias.install() == 0
    assert rope_alias.uninstall() == 1 and rope_alias.uninstall() == 0
    assert PreTrainedConfig.__setattr__ is s and PreTrainedConfig.__dict__["from_dict"].__func__ is f


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            out = fn()
            print("ok", name, "" if out is None else f"({out} checked)")
