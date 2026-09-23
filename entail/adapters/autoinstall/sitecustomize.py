"""Install the adapters in every Python process, including the ones an engine spawns for itself.

Two ways in, same module:
  - pip-installed: `entail-autoinstall.pth` in site-packages imports this module at start-up when ENTAIL is set
    (see entail/hook.py). Nothing to configure beyond `ENTAIL=load`.
  - from a source checkout: CPython imports `sitecustomize` at start-up if it is on the path, so
        PYTHONPATH=<checkout>:<checkout>/entail/adapters/autoinstall ENTAIL=load python serve.py
    does the same without installing anything.
Either way this reaches the parent process and every child it spawns, with no change to engine code and no
change to the user's script. Nothing happens unless ENTAIL is `load` or `debug`.

An engine is usually imported long after start-up, and patching a half-imported package is a good way to break
it. So this waits for the exact module that defines the class to finish executing, and patches then.
"""
import importlib.util
import os
import sys

# module that must finish importing -> ["adapter module[:function]", ...] to run once it has
TARGETS = {
    # Before any model config class exists: each one copies the __setattr__ it inherits when it is created.
    "transformers.configuration_utils": ["entail.adapters.rope_alias"],
    "transformers.modeling_utils": ["entail.adapters.transformers_adapter"],
    "sglang.srt.model_executor.model_runner": ["entail.adapters.sglang_adapter"],
    "sglang.srt.managers.schedule_batch": ["entail.adapters.sglang_cache_contract"],
    "vllm.model_executor.model_loader.utils": ["entail.adapters.vllm_layout"],
    "vllm.v1.core.kv_cache_manager": ["entail.adapters.vllm_cache_contract"],
    # ComfyUI: the LoRA check sits where LoRAs are applied; the node hook only adds the file name to the message.
    "comfy.sd": ["entail.adapters.comfyui"],
    "comfy.sample": ["entail.adapters.comfyui:install_sampling"],
    "nodes": ["entail.adapters.comfyui:install_nodes"],
    # A sampling schedule stays with its own object: record it where it is set, guard the loader that moved it,
    # and check it at the first model call.
    "comfy.model_sampling": ["entail.adapters.comfyui:install_schedule_record"],
    "comfy.model_patcher": ["entail.adapters.comfyui:install_buffer_guard"],
    "comfy.model_base": ["entail.adapters.comfyui:install_schedule_check"],
}
# A one-shot probe of SGLang's request bookkeeping, used while writing the cache contract.
if os.environ.get("ENTAIL_PROBE") == "sglang_cache":
    TARGETS["sglang.srt.managers.schedule_batch"].append("entail.adapters.sglang_cache_probe")
# Comparing against the checkpoint file costs a little I/O, so it is opt-in for now.
if os.environ.get("ENTAIL_SOURCE"):
    TARGETS["vllm.model_executor.model_loader.utils"].append("entail.adapters.vllm_source")
# Scaffolding for testing the layout check against a planted defect; installed first so the check sees it.
if os.environ.get("ENTAIL_SEED"):
    TARGETS["vllm.model_executor.model_loader.utils"].insert(0, "entail.adapters.vllm_seed")
    TARGETS["vllm.model_executor.model_loader.weight_utils"] = ["entail.adapters.vllm_seed:install_loader"]
    TARGETS["vllm.v1.core.kv_cache_manager"].insert(0, "entail.adapters.vllm_seed:install_blocks")
# The D-arm ledger is a measurement, not a check, so it is only installed when asked for.
if os.environ.get("ENTAIL_LEDGER"):
    TARGETS["vllm.model_executor.model_loader.utils"].insert(0, "entail.adapters.vllm_ledger:install_loader")
    TARGETS["vllm.model_executor.layers.quantization.utils.layer_utils"] = \
        ["entail.adapters.vllm_ledger:install_replace"]
    # vllm/model_executor/utils.py holds a second function with the same name, and the online quantisation
    # methods use that one. Both have to be wrapped or the ledger records nothing for them.
    TARGETS["vllm.model_executor.utils"] = ["entail.adapters.vllm_ledger:install_replace_core"]
# ENTAIL_ONLY=rope_alias,sglang_adapter installs just those adapters (to measure one of them on its own).
if os.environ.get("ENTAIL_ONLY"):
    _only = {s.strip() for s in os.environ["ENTAIL_ONLY"].split(",") if s.strip()}
    TARGETS = {mod: [a for a in adapters if a.partition(":")[0].rsplit(".", 1)[-1] in _only]
               for mod, adapters in TARGETS.items()}
    TARGETS = {mod: adapters for mod, adapters in TARGETS.items() if adapters}
# ENTAIL_SKIP=comfyui:install_buffer_guard leaves out single entries (to measure what the others do without them);
# a bare module name leaves out all of its entries.
if os.environ.get("ENTAIL_SKIP"):
    _skip = {s.strip() for s in os.environ["ENTAIL_SKIP"].split(",") if s.strip()}

    def _skipped(adapter):
        name, _, func = adapter.partition(":")
        base = name.rsplit(".", 1)[-1]
        return base in _skip or f"{base}:{func or 'install'}" in _skip

    TARGETS = {mod: [a for a in adapters if not _skipped(a)] for mod, adapters in TARGETS.items()}
    TARGETS = {mod: adapters for mod, adapters in TARGETS.items() if adapters}
_done = set()


def _install(adapter):
    try:
        name, _, func = adapter.partition(":")
        mod = __import__(name, fromlist=["install"])
        getattr(mod, func or "install")()
        if os.environ.get("ENTAIL_VERBOSE"):
            print(f"[entail] installed {adapter} in pid {os.getpid()}", flush=True)
    except Exception as e:  # never break the host program because a check could not be installed
        print(f"[entail] could not install {adapter}: {type(e).__name__}: {e}", flush=True)


class _PatchAfterImport:
    """A finder that claims nothing; it only wraps the real loader so we learn when a module has finished."""

    def find_spec(self, name, path=None, target=None):
        if name not in TARGETS or name in _done:
            return None
        _done.add(name)  # keeps the lookup below from coming back here
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module
        adapters = TARGETS[name]

        def exec_module(module):
            orig_exec(module)
            for adapter in adapters:
                _install(adapter)

        try:
            spec.loader.exec_module = exec_module
        except AttributeError:  # a loader that does not allow this: leave the import alone
            return None
        return spec


def install_now():
    """For a process that has already imported its engine."""
    for name, adapters in TARGETS.items():
        if name not in sys.modules:
            continue
        for adapter in adapters:
            if adapter not in _done:
                _done.add(adapter)
                _install(adapter)


def activate():
    """Watch for the target modules and patch the ones already imported. Safe to call more than once."""
    if not any(isinstance(f, _PatchAfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchAfterImport())
    install_now()


if (os.environ.get("ENTAIL", "off") in ("load", "debug") or os.environ.get("ENTAIL_LEDGER")
        or os.environ.get("ENTAIL_SEED")):
    activate()
