"""Tests for the autoinstall shim's target table. Run: python tests/test_autoinstall.py

The shim is loaded under another module name with ENTAIL off, so it only builds its table and installs nothing.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(os.path.dirname(HERE), "entail", "adapters", "autoinstall", "sitecustomize.py")


def _load(**env):
    keys = ("ENTAIL", "ENTAIL_ONLY", "ENTAIL_SKIP", "ENTAIL_SEED", "ENTAIL_LEDGER", "ENTAIL_SOURCE", "ENTAIL_PROBE")
    old = {k: os.environ.pop(k, None) for k in keys}
    os.environ.update({"ENTAIL": "off", **env})
    try:
        spec = importlib.util.spec_from_file_location("sitecustomize_under_test", SHIM)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if old[k] is not None:
                os.environ[k] = old[k]


def test_rope_alias_goes_in_before_any_model_config_exists():
    """@strict copies __setattr__ into each config class when it is created, so the hook has to be in place when
    configuration_utils finishes, before the first model config module is imported."""
    t = _load().TARGETS
    assert t["transformers.configuration_utils"] == ["entail.adapters.rope_alias"], t


def test_only_keeps_the_named_adapters():
    t = _load(ENTAIL_ONLY="rope_alias").TARGETS
    assert t == {"transformers.configuration_utils": ["entail.adapters.rope_alias"]}, t
    t = _load(ENTAIL_ONLY="rope_alias, sglang_adapter").TARGETS
    assert set(t) == {"transformers.configuration_utils", "sglang.srt.model_executor.model_runner"}, t


def test_only_matches_the_module_not_the_function():
    """Entries such as 'entail.adapters.vllm_seed:install_loader' are kept by the module name."""
    t = _load(ENTAIL_SEED="1", ENTAIL_ONLY="vllm_seed").TARGETS
    flat = [a for adapters in t.values() for a in adapters]
    assert "entail.adapters.vllm_seed:install_loader" in flat and all("vllm_seed" in a for a in flat), flat


def test_skip_leaves_out_one_entry_or_a_whole_module():
    t = _load(ENTAIL_SKIP="comfyui:install_buffer_guard").TARGETS
    assert "comfy.model_patcher" not in t and t["comfy.model_base"] == ["entail.adapters.comfyui:install_schedule_check"]
    assert t["comfy.sd"] == ["entail.adapters.comfyui"], t  # a bare entry is the module's install()
    t = _load(ENTAIL_SKIP="comfyui:install").TARGETS
    assert "comfy.sd" not in t and "comfy.sample" in t, t
    t = _load(ENTAIL_SKIP="comfyui").TARGETS
    assert not any("comfyui" in a for adapters in t.values() for a in adapters), t


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
