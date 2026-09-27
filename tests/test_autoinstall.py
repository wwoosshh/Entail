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
    assert t["transformers.configuration_utils"] == ["entail.adapters.rope_alias",
                                                     "entail.adapters.transformers_config"], t
    assert t["vllm.v1.attention.selector"] == ["entail.adapters.vllm_attention"], t


def test_only_keeps_the_named_adapters():
    t = _load(ENTAIL_ONLY="rope_alias").TARGETS
    assert t == {"transformers.configuration_utils": ["entail.adapters.rope_alias"]}, t
    t = _load(ENTAIL_ONLY="rope_alias, sglang_adapter").TARGETS
    assert set(t) == {"transformers.configuration_utils", "sglang.srt.model_executor.model_runner"}, t


def test_only_matches_the_module_not_the_function():
    """Entries such as 'entail.adapters.vllm_serve:install_render' are kept by the module name."""
    t = _load(ENTAIL_ONLY="vllm_serve").TARGETS
    flat = [a for adapters in t.values() for a in adapters]
    assert "entail.adapters.vllm_serve:install_render" in flat and all("vllm_serve" in a for a in flat), flat


def test_the_research_switches_add_nothing():
    """Fault injection, the ledger and the probe are research tools outside the package (M9.3): their switches
    leave the library's table as it is."""
    plain = _load().TARGETS
    assert _load(ENTAIL_SEED="corrupt_at_load", ENTAIL_LEDGER="/tmp/l", ENTAIL_PROBE="sglang_cache").TARGETS == plain
    flat = [a for adapters in plain.values() for a in adapters]
    assert not any(t in a for a in flat for t in ("_seed", "_ledger", "_probe")), flat


def test_skip_leaves_out_one_entry_or_a_whole_module():
    t = _load(ENTAIL_SKIP="comfyui:install_nodes").TARGETS
    assert "nodes" not in t
    assert t["comfy.sample"] == ["entail.adapters.comfyui:install_sampling"], t
    assert t["comfy.sd"] == ["entail.adapters.comfyui"], t  # a bare entry is the module's install()
    t = _load(ENTAIL_SKIP="comfyui:install").TARGETS
    assert "comfy.sd" not in t and "comfy.sample" in t, t
    t = _load(ENTAIL_SKIP="comfyui").TARGETS
    left = [a for adapters in t.values() for a in adapters]
    assert not any(a.partition(":")[0].endswith(".comfyui") for a in left), t
    assert "entail.adapters.vllm_serve:install_parsers" in left, "another module is not left out with it"


def test_no_engine_specific_repair_is_in_the_core_table():
    """The ComfyUI repair is an official DLC since P4 (dlc/comfyui): its entries come from the DLC, not the core."""
    flat = [a for adapters in _load().TARGETS.values() for a in adapters]
    assert not any("comfyui_repair" in a for a in flat), flat


def test_the_hook_watches_imports_once_when_its_file_is_imported_twice():
    """The .pth imports the hook as entail.adapters.autoinstall.sitecustomize; a folder on PYTHONPATH can bring the
    same file again as sitecustomize (P4, ComfyUI with entail 0.3.0's .pth and a development checkout): one finder."""
    import sys

    first, second = _load(), _load()
    keep_meta, keep_run = list(sys.meta_path), os.environ.get("ENTAIL_RUN_ID")
    try:
        first.activate()
        second.activate()
        second.activate()
        finders = [f for f in sys.meta_path if type(f).__name__ == "_PatchAfterImport"]
        assert len(finders) == 1 and isinstance(finders[0], first._PatchAfterImport), finders
    finally:
        sys.meta_path[:] = keep_meta
        if keep_run is None:
            os.environ.pop("ENTAIL_RUN_ID", None)
        else:
            os.environ["ENTAIL_RUN_ID"] = keep_run


def test_put_on_pythonpath_it_runs_the_sitecustomize_it_hides():
    """As `sitecustomize` it hides the one Python would have run (Ubuntu's apport hook); it runs that one too."""
    import sys
    import tempfile

    other = tempfile.mkdtemp()
    with open(os.path.join(other, "sitecustomize.py"), "w", encoding="utf-8") as f:
        f.write("import os\nos.environ['ENTAIL_TEST_HIDDEN_RAN'] = '1'\n")
    keep, mode = list(sys.path), os.environ.get("ENTAIL")
    sys.path[:] = [os.path.dirname(SHIM), other] + keep
    os.environ.pop("ENTAIL_TEST_HIDDEN_RAN", None)
    os.environ["ENTAIL"] = "off"                            # build the table only; nothing is activated
    try:
        spec = importlib.util.spec_from_file_location("sitecustomize", SHIM)   # as Python imports it from PYTHONPATH
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        assert os.environ.get("ENTAIL_TEST_HIDDEN_RAN") == "1"
        os.environ.pop("ENTAIL_TEST_HIDDEN_RAN", None)
        _load()                                             # brought in under another name (the .pth): nothing to run
        assert "ENTAIL_TEST_HIDDEN_RAN" not in os.environ
    finally:
        sys.path[:] = keep
        os.environ.pop("ENTAIL_TEST_HIDDEN_RAN", None)
        if mode is None:
            os.environ.pop("ENTAIL", None)
        else:
            os.environ["ENTAIL"] = mode


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
