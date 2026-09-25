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
    # rope_alias first: both wrap from_dict, and the key coverage check reads the config rope_alias has settled.
    "transformers.configuration_utils": ["entail.adapters.rope_alias", "entail.adapters.transformers_config"],
    "transformers.modeling_utils": ["entail.adapters.transformers_adapter"],
    "sglang.srt.model_executor.model_runner": ["entail.adapters.sglang_adapter"],
    "sglang.srt.managers.schedule_batch": ["entail.adapters.sglang_cache_contract"],
    "vllm.model_executor.model_loader.utils": ["entail.adapters.vllm_layout", "entail.adapters.vllm_loader"],
    # Patched as soon as the selector has run, so attention.py imports the wrapped name.
    "vllm.v1.attention.selector": ["entail.adapters.vllm_attention"],
    "vllm.v1.core.kv_cache_manager": ["entail.adapters.vllm_cache_contract"],
    "vllm.v1.core.sched.scheduler": ["entail.adapters.vllm_identity"],
    "vllm.entrypoints.pooling.scoring.io_processor": ["entail.adapters.vllm_scoring"],
    "sglang.kernels.ops.quantization.fp8_kernel": ["entail.adapters.sglang_fp8_tile"],
    # the fused-MoE config lookup: wrapped right after its module runs, so the kernel module binds the wrapped name
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config":
        ["entail.adapters.sglang_fp8_tile:install_moe"],
    # The request boundary: the parser manager, the renderer and the chat server, each as soon as it has run.
    "vllm.parser.parser_manager": ["entail.adapters.vllm_serve:install_parsers"],
    "vllm.renderers.hf": ["entail.adapters.vllm_serve:install_render"],
    "vllm.entrypoints.openai.chat_completion.serving": ["entail.adapters.vllm_serve:install_serving"],
    # The chat template where transformers' tokenizers apply it (a script's, SGLang's server; M9.3), and SGLang's
    # server when it renders with a conversation template of its own instead.
    # ... and the tokenizer itself against the model's vocabulary (M15.3): one key, both adapters (a dict literal
    # keeps only the last value of a repeated key, which silently dropped the second adapter once).
    "transformers.tokenization_utils_base": ["entail.adapters.transformers_template",
                                             "entail.adapters.transformers_tokenizer"],
    "sglang.srt.entrypoints.openai.serving_chat": ["entail.adapters.sglang_serve"],
    # ComfyUI (M6.2): the loaders keep what a checkpoint declares with its model, the LoRA contract sits where LoRAs
    # are applied, the prediction and latent scale are decided at sampling; the node hook only names the LoRA file.
    "comfy.sd": ["entail.adapters.comfyui"],
    "comfy.sample": ["entail.adapters.comfyui:install_sampling"],
    "nodes": ["entail.adapters.comfyui:install_nodes"],
    # ENGINE-SPECIFIC repair of ComfyUI's own defect (Comfy-Org/ComfyUI#16490): a sampling schedule stays with its own
    # object - recorded where it is set, the loader that moved it guarded, checked at the first model call.
    "comfy.model_sampling": ["entail.adapters.comfyui_repair:install_schedule_record"],
    "comfy.model_patcher": ["entail.adapters.comfyui_repair:install_buffer_guard"],
    "comfy.model_base": ["entail.adapters.comfyui_repair:install_schedule_check"],
    # diffusers (M6.2): single files, local folders and a VAE put in later, and LoRAs.
    "diffusers.loaders.single_file": ["entail.adapters.diffusers_adapter"],
    "diffusers.pipelines.pipeline_utils": ["entail.adapters.diffusers_adapter:install_pipeline"],
    "diffusers.loaders.lora_pipeline": ["entail.adapters.diffusers_adapter:install_lora"],
}
# Comparing against the checkpoint file costs a little I/O, so it is opt-in for now.
if os.environ.get("ENTAIL_SOURCE"):
    TARGETS["vllm.model_executor.model_loader.utils"].append("entail.adapters.vllm_source")
# Fault injection, the layout ledger and the bookkeeping probe used to measure entail are research tools, not part
# of the library: they live in the development workspace, with a hook of their own that adds them to this table.
# ENTAIL_ONLY=rope_alias,sglang_adapter installs just those adapters (to measure one of them on its own).
if os.environ.get("ENTAIL_ONLY"):
    _only = {s.strip() for s in os.environ["ENTAIL_ONLY"].split(",") if s.strip()}
    TARGETS = {mod: [a for a in adapters if a.partition(":")[0].rsplit(".", 1)[-1] in _only]
               for mod, adapters in TARGETS.items()}
    TARGETS = {mod: adapters for mod, adapters in TARGETS.items() if adapters}
# ENTAIL_SKIP=comfyui_repair:install_buffer_guard leaves out single entries (to measure what the others do without them);
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
            _say(f"[entail] installed {adapter} in pid {os.getpid()}")
    except Exception as e:  # never break the host program because a check could not be installed
        _say(f"[entail] could not install {adapter}: {type(e).__name__}: {e}")


def _say(text):
    """Printed, and kept in the project's entail_logs folder (M6.4)."""
    try:
        from entail import record
        record.say(text)
    except Exception:  # noqa: BLE001 - the message itself must get out
        print(text, flush=True)


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
    """Watch for the target modules and patch the ones already imported. Safe to call more than once. The log folder
    is fixed here, where the program starts, so the processes an engine spawns write to the same one (M6.4)."""
    try:
        from entail import record
        record.log_dir()
    except Exception:  # noqa: BLE001 - a log folder that cannot be named does not stop the checks
        pass
    if not any(isinstance(f, _PatchAfterImport) for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchAfterImport())
    install_now()


if os.environ.get("ENTAIL", "off") in ("load", "debug"):
    activate()
