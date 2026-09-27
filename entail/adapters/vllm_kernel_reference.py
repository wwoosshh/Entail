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
               compared (the slice is decided before the real input is computed) and for every later call of every
               module of that op class and configuration in the process (one layer's decision stands for the layers
               that share its configuration, so the repair reaches them all). Offered when the engine runs without
               CUDA graphs, or when every graph captured so far holds the kernel only at size classes where it was
               held to the definition and matched (M19 L3.3a); otherwise captured graphs replay the kernel whatever
               the module's method says, so it stays broken, with that reason. Under ENTAIL_POLICY=refuse (or
               KernelReference=refuse) nothing is switched and the mismatch is broken.
  warm-ups     (M19 L3.3a) a call inside vLLM's dummy runs - the profile run, the compile warm-ups, the warm-up runs
               before each capture - is decided on a probe made from it (kernel_reference_contract.probed: the
               call's shapes, dtypes and strides, made-up token values; rotary positions drawn from the op's own
               cos/sin cache range), once per size class, on the engine's own row count up to WARM_ROWS, before the
               graph of that size is captured. A capture-time call is noted (its size class), not compared.
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
_REPAIRED = set()      # op configurations (module, class, configuration) sent to their definition
_WARM = set()          # (key, size class) decided on a warm-up probe (M19 L3.3a)
_CAPTURED = {}         # op configuration -> size classes captured into a CUDA graph with the kernel
_PASSED = {}           # op configuration -> size classes where the kernel was held to the definition and matched
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


def config_of(module, name) -> tuple:
    """The op's class and configuration: modules that share it are decided, and repaired, together."""
    return (type(module).__module__, name, scalars(module))


def key_of(module, name, args, kwargs) -> tuple:
    return config_of(module, name) + (pattern(args, kwargs),)


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
    """Whether the engine runs without CUDA graphs (enforce_eager, or a cudagraph mode of none). Then a switch of an
    op's dispatched method reaches every later call; captured graphs replay the kernel regardless."""
    return (_STATE.get("cudagraph_mode") or graph_mode()) == "NONE"


def repairable(cfg) -> bool:
    """Whether sending the op to its definition now keeps every later call right: the engine runs without CUDA
    graphs, or every graph captured so far in this process holds the kernel only at size classes where it was held
    to its definition and matched (those graphs keep a kernel that is right at their size; every later call and
    capture gets the definition)."""
    if graphs_off():
        return True
    return _CAPTURED.get(cfg, set()) <= _PASSED.get(cfg, set())


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


def _device_of(obj):
    ts = kernel_reference_contract.all_tensors(obj)
    return ts[0].device if ts else "cpu"


def positions(module, orig):
    """The probe's fill for the arguments only the op knows the range of: a rotary op's positions index its cos/sin
    cache, one row per position, and a warm-up's positions are zeros (rotary at position 0 is the identity and
    decides nothing), so they are drawn from the cache's rows. Every other integer argument keeps the engine's
    values."""
    krc = kernel_reference_contract
    try:
        names = list(__import__("inspect").signature(orig).parameters)
    except (TypeError, ValueError):
        names = []
    cache = getattr(module, "cos_sin_cache", None)
    limit = int(cache.shape[0]) if _is_tensor(cache) and cache.dim() >= 1 else 0

    def fill(key, t):
        name = names[key] if isinstance(key, int) and key < len(names) else key
        if limit and isinstance(name, str) and "position" in name and not t.is_floating_point():
            return torch.randint(0, limit, tuple(t.shape), generator=krc.generator(t.device), device=t.device,
                                 dtype=t.dtype)
        return None

    return fill


def prepare_warm(module, orig, args, kwargs):
    """A probe made from a warm-up call (kernel_reference_contract.probed): (args, kwargs, n, rows, bytes) on the
    engine's own row count up to WARM_ROWS, or None when the arguments share no token dimension or hold none."""
    krc = kernel_reference_contract
    n = krc.rows_of(args, kwargs, token_hint())
    if not n:
        return None
    rows = min(krc.WARM_ROWS, n)
    gen = krc.generator(_device_of((args, kwargs)))
    fill = positions(module, orig)
    a, k = krc.probed(args, n, rows, gen, fill), krc.probed(kwargs, n, rows, gen, fill)
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


def _decide(module, name, orig, cut, call: int, warm: bool = False) -> bool:
    """Compare on this call. Returns whether something was decided (a decision or an unknown was recorded); False
    when the input decides nothing (zeros, one repeated row, an identity) and the next call should try again.
    `warm`: the cut is a probe made from a warm-up call (prepare_warm), whose values are made up."""
    from .. import load

    krc = kernel_reference_contract
    consumer = f"vllm.{name}"
    cfg = config_of(module, name)

    def unknown(why):
        load.enforce([load.cannot_check(BOUNDARY, consumer, "KernelReference", f"{name}: {why}")])
        return True

    def again(why):
        if warm or call < TRIES:
            return False
        return unknown(f"{why}, on each of its first {call} real calls; not compared")

    if cut is None:
        if warm:
            return False
        return unknown("its arguments share no token dimension to cut (no tensor, or metadata such as cu_seqlens), "
                       "so its kernel is not compared")
    if cut == "failed":
        return True
    a, k, n, rows, size = cut
    if rows == n and size > krc.BUDGET:
        return unknown(f"its arguments cannot be cut (their first dimension is not the token count) and hold "
                       f"{size >> 20} MB: too large to compare whole")
    inputs = krc.tensors_of(list(a)) + krc.tensors_of(k)
    if not warm and krc.uniform_rows(inputs):
        return again("every row of the input is the same token (a dummy input)")
    k_out, n_out, r_out, changed, via_definition = read_choice(module, orig, a, k)
    if not krc.tensors_of(k_out):
        return unknown("the kernel returns no floating tensor to compare")
    reference, native = (r_out, n_out) if r_out is not None else (n_out, None)
    cmp = krc.compare(k_out, reference, native, inputs)
    why = krc.vacuous(cmp)
    if why is not None:
        return again(why)
    if warm:
        where = (f"{name}'s dispatched {getattr(orig, '__name__', 'kernel')} ({type(module).__module__}) on a probe "
                 f"of {rows} rows made from vLLM's warm-up call of {n} rows (the engine's shapes, dtypes and "
                 f"strides; the values made up, seed {krc.SEED}), before any graph of that size is captured")
    else:
        where = (f"{name}'s dispatched {getattr(orig, '__name__', 'kernel')} ({type(module).__module__}) on {rows} "
                 f"rows of its call {call}")
    if batch_invariant():
        where += " (batch-invariant mode: the definition's own ops are vLLM's kernels)"
    extra = f"; the runs replaced {changed} tensors of the op's state, put back" if changed else ""
    if via_definition:
        extra += "; the dispatched method calls the definition itself on this input (not an independent kernel)"
    repair = None
    if not via_definition and repairable(cfg):
        repair = (f"{name} sent to its definition (forward_native) from this call on, for every later call of every "
                  f"module of this configuration in the process")
    elif cmp.violations or cmp.nonfinite:
        extra += ("; not repaired: CUDA graphs were captured with the kernel at sizes where it was not held to its "
                  "definition, and replay it whatever the op dispatches")
    decisions = krc.check(BOUNDARY, consumer, name, cmp, where, extra=extra, repair=repair)
    if krc.resolved(decisions):
        _send_all(cfg)
    elif decisions and decisions[0].verdict.name == "PASS":
        _PASSED.setdefault(cfg, set()).add(krc.bucket(n))
    return True


def _send_all(cfg) -> int:
    """The repair: every wrapped module of this op configuration dispatches to its definition from now on (a module
    whose wrapper already stepped aside is switched here too). Returns how many modules were switched."""
    _REPAIRED.add(cfg)
    n = 0
    for module, _orig in list(_WRAPPED.values()):
        try:
            if config_of(module, type(module).__name__) == cfg:
                module._forward_method = module.forward_native
                n += 1
        except Exception:  # noqa: BLE001 - a module that cannot be read keeps its dispatch
            continue
    return n


def _warm_up(module, name, orig, args, kwargs) -> None:
    """Decide on a probe made from this warm-up call, once per (key, size class) and at most WARM_CLASSES classes
    per op configuration (M19 L3.3a)."""
    from .. import load

    krc = kernel_reference_contract
    try:
        n = krc.rows_of(args, kwargs, token_hint())
        if not n:
            return
        key = (key_of(module, name, args, kwargs), krc.bucket(n))
        cfg = config_of(module, name)
    except Exception:  # noqa: BLE001 - never the engine's problem (principle 12)
        return
    if key in _WARM or sum(1 for w in _WARM if w[0][:3] == cfg) >= krc.WARM_CLASSES:
        return
    _WARM.add(key)
    cut = load.safely(BOUNDARY, f"vllm.{name}", "KernelReference", lambda: prepare_warm(module, orig, args, kwargs),
                      default="failed")
    load.safely(BOUNDARY, f"vllm.{name}", "KernelReference",
                lambda: _decide(module, name, orig, cut, 0, warm=True), default=True)
    _count("warm_decided")


def _note_capture(module, name, args, kwargs) -> None:
    """A call inside a CUDA graph capture: the graph of that size class holds whatever the op dispatches now."""
    try:
        n = kernel_reference_contract.rows_of(args, kwargs, token_hint())
    except Exception:  # noqa: BLE001
        n = None
    _CAPTURED.setdefault(config_of(module, name), set()).add(kernel_reference_contract.bucket(n or 1))


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
        try:
            cfg = config_of(module, name)
        except Exception:  # noqa: BLE001 - never the engine's problem (principle 12)
            return orig(*args, **kwargs)
        if cfg in _REPAIRED:                     # another module of this configuration was decided and repaired
            module._forward_method = module.forward_native
            _count("sent_to_definition")
            return module.forward_native(*args, **kwargs)
        if capturing():
            _note_capture(module, name, args, kwargs)
            return orig(*args, **kwargs)
        if _STATE["dummy"]:                      # the engine's own warm-up: decided on a probe made from it
            _warm_up(module, name, orig, args, kwargs)
            if cfg in _REPAIRED:
                module._forward_method = module.forward_native
                _count("sent_to_definition")
                return module.forward_native(*args, **kwargs)
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
            if cfg in _REPAIRED:
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
    _WARM.clear()
    _CAPTURED.clear()
    _PASSED.clear()
    _STATE["dummy"] = 0
    _STATE.pop("cudagraph_mode", None)
    kernel_reference_contract.reset(BOUNDARY)
