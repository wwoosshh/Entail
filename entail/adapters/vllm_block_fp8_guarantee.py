"""Adapter v2 for the guarantee profile's one boundary in vLLM 0.30: the dense block FP8 matmul on Triton
(entail/guarantee.py holds every rule; ROADMAP M19 L5.4a; docs/semantic-guarantee-design.ko.md).

  hooks        the two producers and the consumer, where they are defined, plus CUDA graph capture and replay:
                 fp8_utils.per_token_group_quant_fp8      the activation quantizer QuantFP8.forward_cuda calls
                 Fp8BlockScaledMMLinearKernel.process_weights_after_loading
                                                          the weight processing every block FP8 linear kernel
                                                          inherits (the Triton one among them)
                 fp8_utils.w8a8_triton_block_scaled_mm    the consumer (the custom op w8a8_triton_block_scaled_mm_func
                                                          imports it by name at each call, so it gets the wrapper)
                 torch.cuda.CUDAGraph.capture_begin / capture_end / replay
  read_choice  what a producer made, from its return value or the layer it filled: the tensors, the quantizer's
               group and scale layout, the kernel's weight block and the block the checkpoint's quantization config
               declares (vLLM's current config, when there is one), the layer's name.
  handles      the consumer's call goes through guarantee.gate, which may hand the kernel the scale the producer
               paired with its value or the declared block (before dispatch), hand on the reference's output, or
               refuse. Nothing here compares or decides.
Installed only by ENTAIL=guarantee, ENTAIL=structure or its integrity experiments structure_writes and
structure_inkernel (adapters/autoinstall/sitecustomize.py; the Triton launch hook only for the structural ones), or by
a harness calling install().
A producer called while torch.compile traces issues nothing: compiled code does not run these hooks at run time,
and the gate refuses a value without an issue.
"""
import functools
import inspect
import sys

from .. import guarantee
from .base import Hook

engine = "vllm"
versions = "vLLM 0.30.0"
FP8_UTILS = "vllm.model_executor.layers.quantization.utils.fp8_utils"
BLOCK_KERNEL = "vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel"
_WRAPPED = {}          # (holder, name) -> original
_STATS = {}


def hooks():
    return [Hook(guarantee.ACTIVATION_PRODUCER.replace(":", "."), "kernel"),
            Hook(guarantee.WEIGHT_PRODUCER.replace(":", "."), "load"),
            Hook(guarantee.CONSUMER.replace(":", "."), "kernel")]


def handles():
    return {"gate": "the consumer's call goes through guarantee.gate (repairs before dispatch, the reference's "
                    "output, or a refusal)"}


def _count(k):
    _STATS[k] = _STATS.get(k, 0) + 1


def _compiling():
    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def read_choice(kind, *args):
    """What a producer made, as (tensors, group or block, layout, source)."""
    if kind == "activation":
        bound, out = args
        x_q, x_s = out
        layout = ("tma_aligned" if bound.get("tma_aligned_scales") else
                  "column_major" if bound.get("column_major_scales") else "row_major")
        source = {"use_ue8m0": bound.get("use_ue8m0"), "eps": bound.get("eps"),
                  "dtype": str(x_q.dtype).replace("torch.", "")}
        return (x_q, x_s), int(bound["group_size"]), layout, source
    kernel, layer = args
    params = kernel._get_layer_params(layer)
    gs = kernel.weight_group_shape
    block = (int(gs.row), int(gs.col)) if hasattr(gs, "row") else (int(gs[0]), int(gs[1]))
    source = {"layer": getattr(layer, "prefix", None) or type(layer).__name__, "scale_attr": params.block_scale_attr,
              "kernel": type(kernel).__name__}
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        qc = getattr(cfg, "quant_config", None)
        wbs = getattr(qc, "weight_block_size", None)
        if wbs is not None:
            source["declared_block"] = [int(x) for x in wbs]
            source["declared_by"] = f"{type(qc).__name__}.weight_block_size (the checkpoint's quantization_config)"
        mc = getattr(cfg, "model_config", None)
        if mc is not None:
            source["model"] = getattr(mc, "model", None)
            source["revision"] = getattr(mc, "revision", None)
    except Exception:  # noqa: BLE001 - outside an engine (a harness) there is no current config
        pass
    return (params.weight, params.block_scale), block, "blocked", source


def _wrap_quantizer(mod, name):
    orig = getattr(mod, name)
    if getattr(orig, "__entail_guarantee__", False):
        return False
    sig = inspect.signature(orig)
    names = list(sig.parameters)
    defaults = {k: p.default for k, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}

    @functools.wraps(orig)
    def run(*args, **kwargs):
        out = orig(*args, **kwargs)
        if _compiling():
            return out
        try:
            bound = dict(defaults)
            bound.update(zip(names, args))
            bound.update(kwargs)
            (x_q, x_s), group, layout, source = read_choice("activation", bound, out)
            guarantee.issue_activation(x_q, x_s, group, layout=layout, source=source)
            _count("activation_issued")
        except Exception:  # noqa: BLE001 - principle 12: never the engine's problem; the gate then refuses
            _count("activation_issue_failed")
        return out

    run.__entail_guarantee__ = True
    setattr(mod, name, run)
    _WRAPPED[(mod, name)] = orig
    return True


def _wrap_consumer(mod, name):
    orig = getattr(mod, name)
    if getattr(orig, "__entail_guarantee__", False):
        return False
    sig = inspect.signature(orig)
    names = list(sig.parameters)
    defaults = {k: p.default for k, p in sig.parameters.items() if p.default is not inspect.Parameter.empty}

    @functools.wraps(orig)
    def run(*args, **kwargs):
        a = dict(defaults)
        a.update(zip(names, args))
        a.update(kwargs)
        _count("consumer_calls")
        return guarantee.gate(orig, a["A"], a["B"], a["As"], a["Bs"], a["block_size"], a["output_dtype"])

    run.__entail_guarantee__ = True
    setattr(mod, name, run)
    _WRAPPED[(mod, name)] = orig
    return True


def install_fp8_utils():
    mod = sys.modules.get(FP8_UTILS)
    if mod is None:
        return 0
    n = int(_wrap_quantizer(mod, "per_token_group_quant_fp8")) + int(_wrap_consumer(mod, "w8a8_triton_block_scaled_mm"))
    guarantee.installed(guarantee.ACTIVATION_PRODUCER)
    guarantee.installed(guarantee.CONSUMER)
    return n


def install_weights():
    mod = sys.modules.get(BLOCK_KERNEL)
    cls = getattr(mod, "Fp8BlockScaledMMLinearKernel", None) if mod is not None else None
    if cls is None:
        return 0
    orig = cls.__dict__["process_weights_after_loading"]
    if getattr(orig, "__entail_guarantee__", False):
        return 0

    @functools.wraps(orig)
    def run(self, layer):
        out = orig(self, layer)
        try:
            (w, w_s), block, _layout, source = read_choice("weight", self, layer)
            guarantee.issue_weight(w, w_s, block, source=source)
            _count("weight_issued")
        except Exception:  # noqa: BLE001 - principle 12; the gate refuses a weight without an issue
            _count("weight_issue_failed")
        return out

    run.__entail_guarantee__ = True
    cls.process_weights_after_loading = run
    _WRAPPED[(cls, "process_weights_after_loading")] = orig
    guarantee.installed(guarantee.WEIGHT_PRODUCER)
    return 1


def install_graphs():
    mod = sys.modules.get("torch.cuda.graphs")     # installed while torch.cuda is still importing: read the class
    G = getattr(mod, "CUDAGraph", None) if mod is not None else None   # from its own module, not torch.cuda
    if G is None:
        return 0
    if getattr(G.replay, "__entail_guarantee__", False):
        return 0
    begin, end, replay = G.capture_begin, G.capture_end, G.replay

    @functools.wraps(begin)
    def capture_begin(self, *a, **k):
        guarantee.capture_begin(self)
        return begin(self, *a, **k)

    @functools.wraps(end)
    def capture_end(self, *a, **k):
        try:
            return end(self, *a, **k)
        finally:
            guarantee.capture_end(self)

    @functools.wraps(replay)
    def run_replay(self, *a, **k):
        guarantee.before_replay(self)
        out = replay(self, *a, **k)
        guarantee.after_replay(self)
        return out

    for name, fn in (("capture_begin", capture_begin), ("capture_end", capture_end), ("replay", run_replay)):
        fn.__entail_guarantee__ = True
        _WRAPPED[(G, name)] = getattr(G, name)
        setattr(G, name, fn)
    guarantee.installed(guarantee.GRAPH_HOOK)
    return 1


_KERNEL_KEYS = {}      # id(JITFunction) -> (qualified name, source cache key)
_PARAMS = {}           # id(JITFunction) -> (it, parameter names, constexpr names)


def _kernel_key(fn, consts):
    k = _KERNEL_KEYS.get(id(fn))
    if k is None:
        f = getattr(fn, "fn", None)
        name = f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', '?')}"
        k = _KERNEL_KEYS[id(fn)] = (name, str(getattr(fn, "cache_key", "")))
    return k + (consts,)


def read_launch(fn, args, kwargs, grid):
    """One Triton launch as the core reads it: its arguments bound to the kernel's parameter names (constexpr ones
    apart, they are folded into the IR), the grid as a tuple."""
    sig = _PARAMS.get(id(fn))
    if sig is None or sig[0] is not fn:
        sig = _PARAMS[id(fn)] = (fn, [p.name for p in fn.params],
                                 {p.name for p in fn.params if getattr(p, "is_constexpr", False)})
    _fn, names, constexpr = sig
    bound = dict(zip(names, args))
    bound.update({k: v for k, v in kwargs.items() if k in names})
    values = {k: v for k, v in bound.items() if k not in constexpr}
    consts = tuple(sorted((k, repr(v)) for k, v in bound.items() if k in constexpr)) + \
        tuple(sorted((k, repr(v)) for k, v in kwargs.items() if k not in names))
    g = grid(dict(bound, **kwargs)) if callable(grid) else grid
    g = tuple(int(x) for x in (g if isinstance(g, (tuple, list)) else (g,)))
    return values, consts, g


_WRITTEN = {}          # kernel key (with its constexprs) -> the pointer arguments it writes, or None (any of them)


def _written(fn, args, kwargs, grid, consts):
    """The names of the arguments a Triton kernel writes through, read once per kernel and configuration from its
    TTIR (kernel_ir.written_args); None when that cannot be told (then every tensor argument counts)."""
    key = _kernel_key(fn, consts)
    if key not in _WRITTEN:
        try:
            from .. import kernel_ir

            _WRITTEN[key] = kernel_ir.written_args(fn.warmup(*args, grid=grid, **kwargs).asm["ttir"])
        except Exception:  # noqa: BLE001 - not read: every tensor argument is taken as written
            _WRITTEN[key] = None
            _count("written_args_unknown")
    return _WRITTEN[key]


def install_triton():
    """The launch hook check "static" (ENTAIL=structure and its integrity experiments) needs: every eager Triton
    launch while a gate is open is offered to the core, which decides whether it goes ahead (guarantee.consumer_launch;
    with integrity "inkernel", guarantee.consumer_launch_inkernel, which may hand back the rewritten kernel's launch).
    With integrity "writes" every launch, inside a gate or not, is also offered to writeguard.check_launch, which
    refuses one that writes - by its own TTIR (kernel_ir.written_args) - through a tensor overlapping bytes a producer
    issued (other than the gate's own operands); reading them passes. Compile-only warm-ups, and the kernels entail
    rewrote itself, pass through."""
    mod = sys.modules.get("triton.runtime.jit")
    J = getattr(mod, "JITFunction", None) if mod is not None else None
    if J is None or getattr(J.run, "__entail_guarantee__", False):
        return 0
    orig = J.run

    @functools.wraps(orig)
    def run(self, *args, grid, warmup, **kwargs):
        if warmup or getattr(self, "entail_source_sha256", None) is not None:
            return orig(self, *args, grid=grid, warmup=warmup, **kwargs)
        ctx = guarantee._ctx()
        if guarantee.plan().integrity == "writes":
            from .. import writeguard

            if writeguard.live():
                allowed = (ctx["A"], ctx["B"], ctx["As"], ctx["Bs"]) if ctx is not None else ()
                values, consts, _g = read_launch(self, args, kwargs, grid)
                names = _written(self, args, kwargs, grid, consts)
                targets = list(values.values()) if names is None else [values.get(n) for n in names]
                writeguard.check_launch(getattr(getattr(self, "fn", None), "__name__", "?"), targets,
                                        allowed)   # raises Refused
        if ctx is None:
            return orig(self, *args, grid=grid, warmup=warmup, **kwargs)
        go = True
        try:
            values, consts, g = read_launch(self, args, kwargs, grid)
            if ctx.get("inkernel"):
                go = guarantee.consumer_launch_inkernel(self, _kernel_key, values, consts, g, args, kwargs, grid)
            else:
                go = guarantee.consumer_launch(_kernel_key(self, consts), values, g,
                                               lambda: self.warmup(*args, grid=grid, **kwargs).asm["ttir"])
        except Exception:  # noqa: BLE001 - principle 12: never the engine's problem; the gate sees no verdict and
            _count("launch_read_failed")   # hands on the reference's output
            go = False if ctx.get("inkernel") else go
        if callable(go):
            _count("launch_rewritten")
            return go()
        if not go:
            _count("launch_withheld")
            return None
        return orig(self, *args, grid=grid, warmup=warmup, **kwargs)

    run.__entail_guarantee__ = True
    J.run = run
    _WRAPPED[(J, "run")] = orig
    guarantee.installed(guarantee.TRITON_HOOK)
    return 1


def install():
    """Everything whose module has loaded (a harness, or install_now after the imports)."""
    return install_fp8_utils() + install_weights() + install_graphs() + install_triton()


def uninstall():
    for (holder, name), orig in list(_WRAPPED.items()):
        setattr(holder, name, orig)
    _WRAPPED.clear()
    for h in guarantee.REQUIRED_HOOKS:
        guarantee.uninstalled(h)


def stats():
    out = dict(_STATS)
    out.update(guarantee.stats())
    return out
