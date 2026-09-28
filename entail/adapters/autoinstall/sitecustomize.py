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
import time

# module that must finish importing -> ["adapter module[:function]", ...] to run once it has
TARGETS = {
    # Before any model config class exists: each one copies the __setattr__ it inherits when it is created.
    # rope_alias first: both wrap from_dict, and the key coverage check reads the config rope_alias has settled.
    "transformers.configuration_utils": ["entail.adapters.rope_alias", "entail.adapters.transformers_config"],
    "transformers.modeling_utils": ["entail.adapters.transformers_adapter", "entail.adapters.transformers_stops"],
    "sglang.srt.model_executor.model_runner": ["entail.adapters.sglang_adapter"],
    "sglang.srt.managers.schedule_batch": ["entail.adapters.sglang_cache_contract"],
    "vllm.model_executor.model_loader.utils": ["entail.adapters.vllm_layout", "entail.adapters.vllm_loader",
                                               "entail.adapters.vllm_pairing",
                                               "entail.adapters.vllm_kernel_reference"],
    # the engine's dummy runs (profile, capture warm-ups) are marked so the kernel comparison lands on real input;
    # 0.30 has two GPU model runners and one for encoder-only models, and the worker picks one
    "vllm.v1.worker.gpu_model_runner": ["entail.adapters.vllm_kernel_reference:install_dummy_run"],
    "vllm.v1.worker.gpu.model_runner": ["entail.adapters.vllm_kernel_reference:install_dummy_run"],
    "vllm.v1.worker.mm_encoder_model_runner": ["entail.adapters.vllm_kernel_reference:install_dummy_run"],
    # Patched as soon as the selector has run, so attention.py imports the wrapped name.
    "vllm.v1.attention.selector": ["entail.adapters.vllm_attention"],
    # engine functions with no definition of their own, held against entail's (M19 L3; entail/definitions.py):
    # wrapped as soon as the defining module has loaded, so every later import gets the wrapped name
    "vllm.model_executor.layers.fused_moe.fused_moe": ["entail.adapters.function_reference:install"],
    "vllm.model_executor.layers.quantization.utils.fp8_utils": ["entail.adapters.function_reference:install"],
    "sglang.kernels.ops.attention.fla.fused_gdn_gating": ["entail.adapters.function_reference:install"],
    # the model runner's bookkeeping every step goes through (M19 L3.3d): positions and slot mappings
    "vllm.v1.worker.gpu.input_batch": ["entail.adapters.function_reference:install"],
    "vllm.v1.worker.gpu.block_table": ["entail.adapters.function_reference:install"],
    # the engine's own paths against each other on probe requests, once the offline engine is up (M19 L3.3c)
    "vllm.entrypoints.llm": ["entail.adapters.vllm_paths"],
    # the two safety modes turn an optimization off in the engine's arguments, before they are built (product P3)
    "vllm.engine.arg_utils": ["entail.adapters.vllm_safe"],
    "sglang.srt.server_args": ["entail.adapters.sglang_safe"],
    "sglang.srt.entrypoints.engine": ["entail.adapters.sglang_paths"],
    "vllm.v1.core.kv_cache_manager": ["entail.adapters.vllm_cache_contract"],
    "vllm.v1.core.sched.scheduler": ["entail.adapters.vllm_identity"],
    "vllm.entrypoints.pooling.scoring.io_processor": ["entail.adapters.vllm_scoring"],
    # the stop set (M15.8): vLLM keeps it on the input processor, SGLang on its model config
    "vllm.v1.engine.input_processor": ["entail.adapters.vllm_stops"],
    "sglang.srt.configs.model_config": ["entail.adapters.sglang_stops"],
    # a LoRA adapter's config against what the engine reads of it (M17.1): vLLM reads it on the worker,
    # SGLang when the adapter object is built
    "vllm.lora.peft_helper": ["entail.adapters.vllm_lora"],
    "sglang.srt.lora.lora": ["entail.adapters.sglang_lora"],
    # a store's key against the fields that shaped the item (M17.2): vLLM's request block hashes, transformers'
    # beam-search cache reorder
    "vllm.v1.request": ["entail.adapters.vllm_cache_key"],
    "transformers.generation.utils": ["entail.adapters.transformers_beam"],
    # every Triton kernel launch in the process, engine-independent (M17.3): what the kernel is told about its
    # tensors' strides
    "triton.runtime.jit": ["entail.adapters.triton_launch"],
    "sglang.kernels.ops.quantization.fp8_kernel": ["entail.adapters.sglang_fp8_tile"],
    # the fused-MoE config lookup: wrapped right after its module runs, so the kernel module binds the wrapped name
    "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config":
        ["entail.adapters.sglang_fp8_tile:install_moe"],
    # The request boundary: the parser manager, the renderer and the chat server, each as soon as it has run.
    "vllm.parser.parser_manager": ["entail.adapters.vllm_serve:install_parsers", "entail.adapters.vllm_parse"],
    # where a multimodal item's placeholder is bound, against the model's declared markup (M18.4)
    "vllm.multimodal.processing.processor": ["entail.adapters.vllm_multimodal"],
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
    # (The repair of ComfyUI's own defect, Comfy-Org/ComfyUI#16490, left the core in product track P4: it is the
    # official DLC entail-dlc-comfyui, whose entries come in below with the other DLCs'.)
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
# Official DLCs (product track P4; LIBRARY_DESIGN.md 13.7): packages outside the core add entries to this table through
# the entry point group entail.dlc (entail/dlc.py). They are looked up only when entail is on, ENTAIL_DLC=off leaves
# them out, and ENTAIL_ONLY and ENTAIL_SKIP below apply to their entries too (by module: repair:install_buffer_guard).
DLC_OF = {}   # entry -> the DLC it belongs to: the core installs those itself (dlc.install)


def add_dlcs():
    try:
        from entail import dlc
        for module, entries in dlc.targets().items():
            for entry, name in entries:
                if entry not in TARGETS.setdefault(module, []):
                    TARGETS[module].append(entry)
                DLC_OF[entry] = name
    except Exception as e:  # noqa: BLE001 - a DLC that cannot be read never stops the host program
        print(f"[entail] could not look up DLCs: {type(e).__name__}: {e}", flush=True)


if os.environ.get("ENTAIL", "off") in ("load", "debug"):
    add_dlcs()
# ENTAIL_ONLY=rope_alias,sglang_adapter installs just those adapters (to measure one of them on its own).
if os.environ.get("ENTAIL_ONLY"):
    _only = {s.strip() for s in os.environ["ENTAIL_ONLY"].split(",") if s.strip()}
    TARGETS = {mod: [a for a in adapters if a.partition(":")[0].rsplit(".", 1)[-1] in _only]
               for mod, adapters in TARGETS.items()}
    TARGETS = {mod: adapters for mod, adapters in TARGETS.items() if adapters}
# ENTAIL_SKIP=comfyui:install_nodes leaves out single entries (to measure what the others do without them);
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
    if adapter in DLC_OF:          # a DLC's entry: the core installs it, and records a failure (entail/dlc.py)
        try:
            from entail import dlc
            dlc.install(DLC_OF[adapter], adapter)
            if os.environ.get("ENTAIL_VERBOSE"):
                _say(f"[entail] installed {adapter} (DLC {DLC_OF[adapter]}) in pid {os.getpid()}")
        except Exception as e:  # noqa: BLE001 - debug mode raises in dlc.install; the host program still goes on here
            _say(f"[entail] could not install {adapter}: {type(e).__name__}: {e}")
        return
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
        if os.environ.get("ENTAIL_VERBOSE"):
            _say(f"[entail] claimed {name} in pid {os.getpid()} (loaded already: {name in sys.modules}; "
                 f"spec: {spec is not None and spec.loader is not None})")
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
        folder = record.log_dir()
        # the launch id: the first process that turned entail on names it, the engine's child processes inherit it,
        # and a line one of them said is not said again by another (record.said_in_this_launch; M15.6 review)
        first = "ENTAIL_RUN_ID" not in os.environ
        os.environ.setdefault("ENTAIL_RUN_ID", f"{os.getpid()}-{int(time.time())}")
        if first and "start" not in os.environ.get("ENTAIL_QUIET", "").replace(" ", "").split(","):
            # one line, once per launch, that it took effect: nothing else is printed until the first model loads
            # (field test, entail#7). On stderr, so a program whose output is read stays as it was
            print(f"[entail] on ({os.environ.get('ENTAIL')}): what it finds goes to "
                  f"{folder or 'no folder (ENTAIL_LOG_DIR=off)'}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a log folder that cannot be named does not stop the checks
        pass
    mine = any(isinstance(f, _PatchAfterImport) for f in sys.meta_path)
    other = any(type(f).__name__ == "_PatchAfterImport" and not isinstance(f, _PatchAfterImport) for f in sys.meta_path)
    if other and not mine:
        # another copy of this file is active here: the .pth imports it as entail.adapters.autoinstall.sitecustomize,
        # and a folder on PYTHONPATH can bring it again as sitecustomize; one of them watches the imports
        return
    if not mine:
        sys.meta_path.insert(0, _PatchAfterImport())
    install_now()


if os.environ.get("ENTAIL", "off") in ("load", "debug"):
    activate()


def _run_the_hidden_sitecustomize():
    """Put on PYTHONPATH, this file is imported as `sitecustomize` and hides the one Python would have run instead
    (Ubuntu's /usr/lib/python3.X/sitecustomize.py installs apport's exception hook; environments and users have their
    own). Run that one too, as it would have run. Nothing to do when the .pth brought this file in (it is not
    `sitecustomize` then), or when there is no other."""
    here = os.path.dirname(os.path.realpath(__file__))
    for entry in sys.path:
        folder = os.path.realpath(entry or os.getcwd())
        if folder == here:
            continue
        path = os.path.join(folder, "sitecustomize.py")
        if os.path.isfile(path) and os.path.realpath(path) != os.path.realpath(__file__):
            try:
                spec = importlib.util.spec_from_file_location("_hidden_sitecustomize", path)
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
            except Exception as e:  # noqa: BLE001 - its failure would not have stopped Python either
                print(f"[entail] the sitecustomize this hook hides ({path}) failed: {type(e).__name__}: {e}",
                      flush=True)
            return path
    return None


if __name__ == "sitecustomize":
    _run_the_hidden_sitecustomize()
