"""Adapter v2 for vLLM's custom ops against their own native definitions (LIBRARY_DESIGN.md 11 M18; ROADMAP M18.2;
kernel_reference_contract.py; vllm#42016).

  hook         vllm.model_executor.model_loader.utils.process_weights_after_loading: the model is built. Every
               CustomOp module whose dispatched forward is a kernel path (custom_op.py keeps it as _forward_method;
               it is forward_native when custom ops are off, as under torch.compile by default) gets that method
               wrapped. The wrapper compares once per (op class and module, configuration, input pattern) per
               process, on the first real call, and then puts the original back, so the steady state costs nothing.
               vllm.v1.worker.gpu_model_runner.GPUModelRunner._dummy_run (install_dummy_run; and the second GPU
               runner's capture_model and profile_cudagraph_memory, which capture outside _dummy_run): the engine's
               own profile run and capture warm-ups feed zeros to every layer; calls made inside them decide
               nothing and are not counted, so the comparison lands on the first real input.
  read_choice  the kernel's and the definition's outputs on a slice of the real input (ROWS rows of the token
               dimension, cloned before the kernel touched its arguments), the definition run in the input dtype
               and in float32, the op's tensors put back afterwards (a definition may convert its own cache to the
               query's dtype: rotary's _match_cos_sin_cache_dtype); the rule is in the core.
  handles      the op's dispatched method (M19 L3): a kernel that differs from its definition is resolved by sending
               the op to its definition, the meaning-keeping consumer (principle 7) - from the very call that was
               compared (the slice is decided before the real input is computed) and for every later call of that
               module in the process. Offered only when the engine runs without CUDA graphs: captured graphs replay
               the kernel whatever the module's method says, so there it stays broken, with that reason. Under
               ENTAIL_POLICY=refuse (or KernelReference=refuse) nothing is switched and the mismatch is broken.
Not compared, each said once as unknown: ops that override forward() (the mamba mixers, static sink attention:
stateful, not row-wise); ops that hold the engine's state (a KV cache, an index buffer, a forward that reads the
forward context: DeepSeek's sparse attention indexer - re-running it on a slice would write the engine's buffers
from mismatched inputs); every op when the process is one rank of several (a definition that all-reduces would run
collectives out of step); every op enabled under torch.compile (the wrapper cannot run inside the compiled graph,
and steps aside if traced); ops whose arguments share no token dimension to cut, or cannot be cut and are too
large to clone whole; ops enabled in vLLM's registry but not reached from the model's modules (held by helpers
that are not modules, as the linear kernels' QuantFP8); ops whose definition raises on the input. Under a stop
policy the decision raises at the op's first real call, once. A slice of 64 rows can take another kernel
configuration than the engine's full launch; a defect that shows only at large row counts is not seen here. vLLM
caches rotary modules process-wide, so one instance serves every layer; each instance is wrapped once.
"""
import os

import torch

from .. import core, kernel_reference_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "kernel:vllm.custom_op"
CONSUMER = "vllm.custom_op"
_ORIG = {}             # "loader" -> process_weights_after_loading, <runner module> -> {method in DUMMY: original}
RUNNERS = ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner",
           "vllm.v1.worker.mm_encoder_model_runner")     # the model runner classes 0.30 picks one of
DUMMY = ("_dummy_run", "capture_model", "profile_cudagraph_memory")   # the runner methods that feed the model
#                                                                        the engine's own dummy input
_DECIDED = set()       # keys (module, class, configuration, input pattern) decided in this process
_TRIED = {}            # key -> real calls that decided nothing (dummy or identity inputs)
_WRAPPED = {}          # id(module) -> (module, original _forward_method)
_STATS = {}            # counts by reason: instrumented, native, overrides_forward, stateful, unreached, decided
_STATE = {"dummy": 0}  # depth of vLLM's dummy runs (profile run, capture warm-ups) in this process
_REPAIRED = {}         # id(module) -> True: the op was sent to its definition (a resolved mismatch)
TRIES = 64             # real calls a key may decide nothing on before it is given up: the dummy runs are not counted,
#                        and a real slice decides unless every row is one token; a single-token request through a
#                        rotary instance shared by 32 layers gives 32 such calls, so 64 covers two of them


def hooks():
    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "kernel"),
            Hook("vllm.v1.worker.gpu_model_runner.GPUModelRunner._dummy_run", "kernel"),
            Hook("vllm.v1.worker.gpu.model_runner.GPUModelRunner._dummy_run", "kernel"),
            Hook("vllm.v1.worker.gpu.model_runner.GPUModelRunner.capture_model", "kernel"),
            Hook("vllm.v1.worker.mm_encoder_model_runner.MMEncoderModelRunner._dummy_run", "kernel")]


def handles():
    return {}


def _count(k: str, by: int = 1) -> None:
    _STATS[k] = _STATS.get(k, 0) + by


def _overrides_forward(module) -> bool:
    """Whether a class between the module's own and CustomOp defines forward (the dispatch is bypassed there)."""
    for cls in type(module).__mro__:
        if cls.__name__ == "CustomOp":
            return False
        if "forward" in cls.__dict__:
            return True
    return False


def _is_native(fwd) -> bool:
    name = getattr(fwd, "__name__", "")
    inner = getattr(fwd, "__wrapped__", None)
    return name == "forward_native" or getattr(inner, "__name__", "") == "forward_native"


def _is_tensor(x) -> bool:
    return hasattr(x, "shape") and hasattr(x, "dtype") and hasattr(x, "dim")


def stateful(module):
    """Why the op holds the engine's state and must not be re-run on a slice: a KV cache, an index buffer the
    engine writes, or a dispatched forward that reads the forward context. None when it is a plain function of
    its arguments and its parameters."""
    for k, v in vars(module).items():
        try:
            if not _is_tensor(v) and not callable(v) and hasattr(v, "kv_cache"):
                return f"holds {k}, a KV cache of the engine"
        except Exception:  # noqa: BLE001 - an attribute whose access fails is not engine state
            continue
        if _is_tensor(v) and not v.is_floating_point() and v.dim() >= 1:
            return f"holds {k}, an index buffer of the engine"
    try:
        import inspect

        src = inspect.getsource(module._forward_method)
        if "forward_context" in src or "kv_cache" in src:
            return "its dispatched forward reads the forward context or a KV cache"
    except Exception:  # noqa: BLE001 - no source (a compiled or built-in path): nothing to say
        pass
    return None


def is_candidate(module) -> bool:
    """A CustomOp instance dispatching to a kernel path: it has the dispatched method and a native definition,
    does not bypass the dispatch and holds none of the engine's state."""
    fwd = getattr(module, "_forward_method", None)
    if fwd is None or not callable(getattr(module, "forward_native", None)):
        return False
    if not any(cls.__name__ == "CustomOp" for cls in type(module).__mro__):
        return False
    if _overrides_forward(module):
        _count("overrides_forward")
        return False
    if _is_native(fwd):
        _count("native")
        return False
    if stateful(module) is not None:
        _count("stateful")
        return False
    return True


def scalars(module) -> tuple:
    """The instance's configuration: its public scalar attributes (eps, head_size, rotary_dim, is_neox_style ...)
    and short tuples of scalars (an MRoPE section), sorted by name."""
    out = []
    for k, v in vars(module).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (bool, int, float, str)):
            out.append((k, v))
        elif isinstance(v, (tuple, list)) and len(v) <= 8 and all(isinstance(x, (bool, int, float, str)) for x in v):
            out.append((k, tuple(v)))
    return tuple(sorted(out))


def pattern(args, kwargs) -> tuple:
    def one(x):
        if hasattr(x, "dtype") and hasattr(x, "dim"):
            return ("t", str(x.dtype), int(x.dim()))
        if isinstance(x, (tuple, list)):
            return (type(x).__name__,) + tuple(one(y) for y in x)
        return (type(x).__name__,)

    return tuple(one(a) for a in args) + tuple((k, one(v)) for k, v in sorted(kwargs.items()))


def key_of(module, name, args, kwargs) -> tuple:
    return (type(module).__module__, name, scalars(module), pattern(args, kwargs))


def capturing() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def token_hint():
    """The engine's own token count for this forward (vLLM's batch descriptor), or None."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        if not is_forward_context_available():
            return None
        desc = getattr(get_forward_context(), "batch_descriptor", None)
        n = getattr(desc, "num_tokens", None)
        return int(n) if isinstance(n, int) and n > 0 else None
    except Exception:  # noqa: BLE001
        return None


def batch_invariant() -> bool:
    try:
        import vllm.envs as envs

        return bool(envs.VLLM_BATCH_INVARIANT)
    except Exception:  # noqa: BLE001
        return False


def world_size() -> int:
    try:
        import torch.distributed as dist

        return int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1
    except Exception:  # noqa: BLE001
        return 1


def compile_mode():
    """(vLLM's compilation mode: 0 is eager, and the registry's count of enabled custom ops by name)."""
    try:
        from vllm.config.vllm import get_cached_compilation_config

        cfg = get_cached_compilation_config()
        return int(getattr(cfg, "mode", 0) or 0), dict(getattr(cfg, "enabled_custom_ops", None) or {})
    except Exception:  # noqa: BLE001
        return 0, {}


def graph_mode():
    """vLLM's cudagraph mode by name (NONE under enforce_eager), or None when it cannot be read. It is read when the
    model is built (instrument): the current config is set then, and vLLM 0.22 raises when it is asked during a
    forward."""
    try:
        from vllm.config import get_current_vllm_config

        mode = getattr(get_current_vllm_config().compilation_config, "cudagraph_mode", None)
        return None if mode is None else str(getattr(mode, "name", mode)).upper()
    except Exception:  # noqa: BLE001
        return None


def graphs_off() -> bool:
    """Whether the engine runs without CUDA graphs (enforce_eager, or a cudagraph mode of none). Only then does a
    switch of an op's dispatched method reach every later call; captured graphs replay the kernel regardless. Not
    known to be off: no repair is offered."""
    return (_STATE.get("cudagraph_mode") or graph_mode()) == "NONE"


def prepare(args, kwargs):
    """The slices, cut and cloned before the kernel runs on the real input (a kernel may work in place):
    (args, kwargs, n, rows, bytes), or None when the arguments share no token dimension."""
    krc = kernel_reference_contract
    n = krc.rows_of(args, kwargs, token_hint())
    if n is None:
        return None
    rows = min(krc.ROWS, n)
    a, k = krc.sliced(args, n, rows), krc.sliced(kwargs, n, rows)
    return a, k, n, rows, krc.nbytes(a) + krc.nbytes(k)


def snapshot(module) -> list:
    """The op's tensors (its parameters, buffers and tensor attributes, and its submodules'), to be put back
    after the definition has run."""
    stores = []
    for _, m in module.named_modules():
        for store in (getattr(m, "_parameters", None), getattr(m, "_buffers", None), vars(m)):
            if isinstance(store, dict):
                stores.append((store, dict(store)))
    return stores


def restore(stores) -> int:
    """Put back every tensor the runs replaced, and drop every tensor they added. Returns how many."""
    changed = 0
    for store, kept in stores:
        for k, v in kept.items():
            cur = store.get(k, kept)
            if cur is not v and (_is_tensor(v) or _is_tensor(cur)):
                store[k] = v
                changed += 1
        for k in [k for k in store if k not in kept and _is_tensor(store[k])]:
            del store[k]
            changed += 1
    return changed


def read_choice(module, orig, a, k):
    """(kernel output, definition output in the input dtype, definition output in float32 or None, tensors of the
    op's state put back, whether the kernel path called the definition on this input) on the prepared slices;
    every run gets its own clone. The last: vLLM's unquantized fused-MoE forward_cuda is `return
    self.forward_native(...)`, and RMSNorm falls back to it for some sizes - then the comparison holds the
    definition against itself, and says so."""
    krc = kernel_reference_contract
    stores = snapshot(module)
    called = {"n": 0}
    native = module.forward_native
    own = "forward_native" in vars(module)                   # an instance attribute of its own (a test's stand-in)

    def spy(*sa, **sk):
        called["n"] += 1
        return native(*sa, **sk)

    with torch.no_grad():
        try:
            module.forward_native = spy                      # an instance attribute, shadowing the class's
            try:
                k_out = orig(*krc.cast(a), **krc.cast(k))
            finally:
                if own:
                    module.forward_native = native
                elif vars(module).get("forward_native") is spy:
                    del module.forward_native
            n_out = module.forward_native(*krc.cast(a), **krc.cast(k))
            try:
                r_out = module.forward_native(*krc.cast(a, torch.float32), **krc.cast(k, torch.float32))
            except Exception:  # noqa: BLE001 - a definition that refuses float32 (a kernel-backed native path)
                r_out = None
        finally:
            changed = restore(stores)
    return k_out, n_out, r_out, changed, called["n"] > 0


def _decide(module, name, orig, cut, call: int) -> bool:
    """Compare on this call. Returns whether something was decided (a decision or an unknown was recorded); False
    when the input decides nothing (zeros, one repeated row, an identity) and the next call should try again."""
    from .. import load

    krc = kernel_reference_contract
    consumer = f"vllm.{name}"

    def unknown(why):
        load.enforce([load.cannot_check(BOUNDARY, consumer, "KernelReference", f"{name}: {why}")])
        return True

    def again(why):
        if call < TRIES:
            return False
        return unknown(f"{why}, on each of its first {call} real calls; not compared")

    if cut is None:
        return unknown("its arguments share no token dimension to cut (no tensor, or metadata such as cu_seqlens), "
                       "so its kernel is not compared")
    if cut == "failed":
        return True
    a, k, n, rows, size = cut
    if rows == n and size > krc.BUDGET:
        return unknown(f"its arguments cannot be cut (their first dimension is not the token count) and hold "
                       f"{size >> 20} MB: too large to compare whole")
    inputs = krc.tensors_of(list(a)) + krc.tensors_of(k)
    if krc.uniform_rows(inputs):
        return again("every row of the input is the same token (a dummy input)")
    k_out, n_out, r_out, changed, via_definition = read_choice(module, orig, a, k)
    if not krc.tensors_of(k_out):
        return unknown("the kernel returns no floating tensor to compare")
    reference, native = (r_out, n_out) if r_out is not None else (n_out, None)
    cmp = krc.compare(k_out, reference, native, inputs)
    why = krc.vacuous(cmp)
    if why is not None:
        return again(why)
    where = (f"{name}'s dispatched {getattr(orig, '__name__', 'kernel')} ({type(module).__module__}) on {rows} rows "
             f"of its call {call}" + (" (batch-invariant mode: the definition's own ops are vLLM's kernels)"
                                      if batch_invariant() else ""))
    extra = f"; the runs replaced {changed} tensors of the op's state, put back" if changed else ""
    if via_definition:
        extra += "; the dispatched method calls the definition itself on this input (not an independent kernel)"
    repair = None
    if graphs_off() and not via_definition:
        repair = (f"{name} sent to its definition (forward_native) from this call on, for every later call of this "
                  f"module in the process")
    elif cmp.violations or cmp.nonfinite:
        extra += "; not repaired: CUDA graphs are captured with the kernel and replay it whatever the op dispatches"
    if krc.resolved(krc.check(BOUNDARY, consumer, name, cmp, where, extra=extra, repair=repair)):
        _REPAIRED[id(module)] = True
    return True


def wrap(module) -> bool:
    """Wrap the module's dispatched forward so its first real call (per class, configuration and input pattern)
    is compared with the definition; afterwards the original is put back on this module. Returns whether it was
    wrapped."""
    if id(module) in _WRAPPED or not is_candidate(module):
        return False
    orig = module._forward_method
    name = type(module).__name__

    def run(*args, **kwargs):
        if torch.compiler.is_compiling():        # traced: the graph holds the kernel alone
            return orig(*args, **kwargs)
        if _STATE["dummy"] or capturing():
            return orig(*args, **kwargs)
        try:
            key = key_of(module, name, args, kwargs)
        except Exception:  # noqa: BLE001 - never the engine's problem (principle 12)
            return orig(*args, **kwargs)
        if key in _DECIDED:
            module._forward_method = orig
            return orig(*args, **kwargs)
        from .. import load

        # the slice is cut and decided before the real input is computed, so a resolved mismatch is kept from this
        # very call's output (the kernel on the real input would already have gone to the engine otherwise)
        cut = load.safely(BOUNDARY, f"vllm.{name}", "KernelReference", lambda: prepare(args, kwargs), default="failed")
        call = _TRIED.get(key, 0) + 1
        _TRIED[key] = call
        try:
            decided = load.safely(BOUNDARY, f"vllm.{name}", "KernelReference",
                                  lambda: _decide(module, name, orig, cut, call), default=True)
        except core.RoleError:                   # the policy stops here: once, and the wrapper steps aside
            _DECIDED.add(key)
            module._forward_method = orig
            raise
        except Exception:  # noqa: BLE001 - said once by safely; not again on every call
            decided = True
        if decided:
            _DECIDED.add(key)
            _count("decided")
            if _REPAIRED.get(id(module)):
                module._forward_method = module.forward_native
                _count("sent_to_definition")
                return module.forward_native(*args, **kwargs)
            module._forward_method = orig
        return orig(*args, **kwargs)

    module._forward_method = run
    _WRAPPED[id(module)] = (module, orig)
    _count("instrumented")
    return True


def _registry_name(module):
    return getattr(type(module), "name", None) or type(module).__name__


def instrument(model) -> int:
    """Wrap every candidate custom op of the model. Returns how many were wrapped; the counts (wrapped, dispatching
    to the native definition, overriding forward, stateful, unreached) go to the record as one line, so a run
    where nothing was compared says why."""
    from .. import load

    def unknown(why):
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "KernelReference", why)])

    n, seen = 0, {}
    ws = world_size()
    mode, enabled = compile_mode()
    _STATE["cudagraph_mode"] = graph_mode()
    if ws > 1:
        unknown(f"custom ops are not compared on one rank of {ws}: a definition that all-reduces would run its "
                f"collectives out of step with the other ranks")
    elif mode != 0 and enabled:
        unknown(f"custom ops enabled under torch.compile (mode {mode}: "
                f"{', '.join(f'{k} x{v}' for k, v in sorted(enabled.items()))}) are not compared: the wrapper "
                f"cannot run inside the compiled graph")
    elif mode == 0:
        for _, m in model.named_modules():
            if any(cls.__name__ == "CustomOp" for cls in type(m).__mro__):
                seen[_registry_name(m)] = seen.get(_registry_name(m), 0) + 1
            if wrap(m):
                n += 1
        unreached = {k: v - seen.get(k, 0) for k, v in enabled.items() if v > seen.get(k, 0)}
        if unreached:
            _count("unreached", sum(unreached.values()))
            unknown(f"{sum(unreached.values())} custom op instances ({', '.join(f'{k} x{v}' for k, v in sorted(unreached.items()))}) "
                    f"dispatch to a kernel but are not reached from the model's modules (held by helpers that are "
                    f"not modules): not compared")
    try:
        load._write({"pid": os.getpid(), "boundary": BOUNDARY, "kernel_reference": {
            "instrumented": n, "native": _STATS.get("native", 0), "overrides_forward": _STATS.get("overrides_forward", 0),
            "stateful": _STATS.get("stateful", 0), "unreached": _STATS.get("unreached", 0), "model": type(model).__name__,
            "tries": TRIES, "world_size": ws, "compile_mode": mode, "batch_invariant": batch_invariant(),
            "dummy_hook": [m for m in RUNNERS if m in _ORIG]}})
    except Exception:  # noqa: BLE001 - a record that cannot be written is only a missing line
        pass
    return n


def install():
    try:
        from vllm.model_executor.model_loader import utils as loader_utils
    except ImportError:
        return 0
    if "loader" in _ORIG:
        return 0
    _ORIG["loader"] = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        out = _ORIG["loader"](model, model_config, target_device, *a, **kw)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "KernelReference", lambda: instrument(model))
        return out

    loader_utils.process_weights_after_loading = wrapped
    return 1


def _marked(orig):
    def run(self, *a, **kw):
        _STATE["dummy"] += 1
        try:
            return orig(self, *a, **kw)
        finally:
            _STATE["dummy"] -= 1
    return run


def install_dummy_run():
    """Mark vLLM's dummy runs (profile run, CUDA-graph capture and warm-ups), whose calls decide nothing: each
    method in DUMMY that the model runner class defines, for every runner class already imported (0.30 has two
    runners for GPUs and one for encoder-only models; the worker picks one). The second GPU runner captures and
    profiles its graphs outside `_dummy_run` (capture_model, profile_cudagraph_memory), so those are marked too.
    Returns how many classes were hooked by this call."""
    import sys

    n = 0
    for modname in RUNNERS:
        mod = sys.modules.get(modname)
        if mod is None or modname in _ORIG:
            continue
        for cls in list(vars(mod).values()):
            if not (isinstance(cls, type) and cls.__module__ == modname and "_dummy_run" in vars(cls)):
                continue
            _ORIG[modname] = {m: vars(cls)[m] for m in DUMMY if m in vars(cls)}
            for m, orig in _ORIG[modname].items():
                setattr(cls, m, _marked(orig))
            _ORIG[modname + ":class"] = cls
            n += 1
            break
    return n


def uninstall():
    for module, orig in list(_WRAPPED.values()):
        module._forward_method = orig
    _WRAPPED.clear()
    n = 0
    if "loader" in _ORIG:
        from vllm.model_executor.model_loader import utils as loader_utils

        loader_utils.process_weights_after_loading = _ORIG.pop("loader")
        n += 1
    for modname in RUNNERS:
        if modname in _ORIG:
            cls = _ORIG.pop(modname + ":class")
            for m, orig in _ORIG.pop(modname).items():
                setattr(cls, m, orig)
            n += 1
    return n


def stats():
    out = {"instrumented": 0, "native": 0, "overrides_forward": 0}
    out.update(_STATS)
    out["decided"] = len(_DECIDED)
    out.update(kernel_reference_contract.stats(BOUNDARY))
    return out


def reset():
    for module, orig in list(_WRAPPED.values()):
        module._forward_method = orig
    _WRAPPED.clear()
    _DECIDED.clear()
    _TRIED.clear()
    _STATS.clear()
    _REPAIRED.clear()
    _STATE["dummy"] = 0
    kernel_reference_contract.reset(BOUNDARY)
