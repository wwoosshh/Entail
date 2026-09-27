"""Adapter v2 for Triton kernel launches, engine-independent (LIBRARY_DESIGN.md 4.8; ROADMAP M17.3;
kernel_launch_contract.py).

  hook         triton.runtime.jit.JITFunction.run: every `kernel[grid](*args, **kwargs)` of a @triton.jit kernel
               launched eagerly, by any engine in the process (3.7 and 3.8), goes through it. Kernels Inductor
               generates for a compiled forward and AOT-compiled kernels do not. The parameters' names come from the
               kernel's own signature (JITFunction.params), so the launch's arguments are bound to names without
               knowing the kernel.
  read_choice  the bound arguments of one launch: parameter name -> argument; and (M19 L3.3b), for a tensor the
               launch cannot say the kernel knows the layout of, the launch run on copies twice over - as given, and
               with those tensors laid out again (kernel_launch_contract.relaid: the innermost dimension contiguous,
               every other stride kept, so the stride arguments still hold; or wholly contiguous for a kernel that
               takes no stride at all) - and a second time in that layout for the kernel's own noise.
  handles      the launch's tensors (M19 L3.3b): a kernel that writes other values as given than on the relaid
               copies reads the tensor as if it were contiguous; resolved by launching it, for every later launch of
               that layout pattern, on relaid copies of those tensors and copying what it wrote back into them.
Each (kernel, stride pattern of its tensor arguments) is decided once per process: the pattern is per tensor its
rank and innermost stride (1, 0 or the strided value), not its shape, so the decode shapes of a server share one
pattern and a strided tensor at a new shape is still seen. A kernel is looked at for its first LIMIT strided
patterns only. Compile-only warm-ups (JITFunction.warmup, `warmup=True`) launch nothing and are skipped; the
autotuner's benchmark launches are ordinary launches, memoised after the first. Cost: a pattern key per launch (a
few microseconds; within the noise of the S4 CUDA-graph measurement, testbed/results/m17/m55_v2) - the 4.8-6% that
the first M17.3 measurement showed at batch 32 was the record file being opened per line, not this hook. The
three extra launches run once per pattern, only for a tensor the launch does not tell the kernel the layout of.
Not run twice (the launch-only rule decides, as before): a launch inside a CUDA graph capture (principle 6: the
graph replays carry no Python), tensors that share storage (copies would part them), a launch whose tensors hold
more than kernel_reference_contract.BUDGET, a layout that keeping the outer strides cannot give (a transposed
tensor), and a comparison where every value is zero (it decides nothing).
"""
from .. import core, kernel_launch_contract, kernel_reference_contract
from .base import Hook

engine = "triton"
versions = "3.7.1, 3.8.0"
_ORIG = None
_SEEN = set()
_COUNT = {}        # id(kernel) -> strided patterns decided; past LIMIT the kernel's strided launches are not looked at
LIMIT = 8
_REPAIR = {}       # layout key -> ({location: whole}, written locations): what a resolved launch pattern relays
_STATS = {}


def hooks():
    return [Hook("triton.runtime.jit.JITFunction.run", "kernel")]


def _names(fn):
    try:
        return [p.name for p in fn.params]
    except AttributeError:
        return []


def value_params(fn):
    """The kernel's non-constexpr parameters: the ones whose integers can be strides."""
    try:
        return [p.name for p in fn.params if not getattr(p, "is_constexpr", False)]
    except AttributeError:
        return []


def told_params(fn):
    """The parameters whose integers count as strides told: the value parameters, and the constexpr parameters
    with a stride-like name (kernels often take strides as constexprs for specialisation: vLLM's and SGLang's
    causal_conv1d declare every stride `tl.constexpr`; the first review fix dropped all constexprs and said unknown
    on those launches). Other constexprs (BLOCK_*) and launch options (num_warps, never among the parameters) do
    not count."""
    try:
        return [p.name for p in fn.params
                if not getattr(p, "is_constexpr", False) or kernel_launch_contract.stride_like(p.name)]
    except AttributeError:
        return []


def read_choice(fn, args, kwargs):
    """parameter name -> argument for one launch (positional by the kernel's signature, then keywords)."""
    bound = dict(zip(_names(fn), args))
    bound.update(kwargs)
    return bound


def handles(fn):
    return {"relay": "the launch runs on relaid copies of the tensors a resolved decision names, written back"}


def _layout_key(fn, args, kwargs):
    """(kernel, per tensor argument: position, rank, innermost stride) - shape-free."""
    parts = []
    for i, v in enumerate(list(args) + list(kwargs.values())):
        if hasattr(v, "stride") and hasattr(v, "shape"):
            inner = kernel_launch_contract.innermost_stride(v)
            try:
                rank = len(v.shape)
            except TypeError:
                rank = None
            parts.append((i, rank, inner[2] if inner else None))
    return (id(fn), tuple(parts))


def _strided(key):
    return any(s not in (None, 0, 1) for _, _, s in key[1])


def should_look(fn, args, kwargs):
    """The pattern key when this launch is the first of its pattern and within the kernel's cap, else None."""
    return _look(fn, _layout_key(fn, args, kwargs))


def _look(fn, key):
    if key in _SEEN:
        return None
    strided = _strided(key)
    if strided and _COUNT.get(id(fn), 0) >= LIMIT:
        return None
    _SEEN.add(key)
    if strided:
        _COUNT[id(fn)] = _COUNT.get(id(fn), 0) + 1
    return key


def _count(k, by=1):
    _STATS[k] = _STATS.get(k, 0) + by


def _is_tensor(v):
    return hasattr(v, "stride") and hasattr(v, "shape") and callable(getattr(v, "stride", None))


def _capturing():
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _tensor_locations(args, kwargs):
    """Where each tensor argument sits: ("a", position) or ("k", name)."""
    return ([("a", i) for i, v in enumerate(args) if _is_tensor(v)]
            + [("k", k) for k, v in kwargs.items() if _is_tensor(v)])


def _get(args, kwargs, loc):
    return args[loc[1]] if loc[0] == "a" else kwargs[loc[1]]


def _with(args, kwargs, repl):
    """The launch's arguments with the tensors at the given locations replaced."""
    a = [repl.get(("a", i), v) for i, v in enumerate(args)]
    k = {n: repl.get(("k", n), v) for n, v in kwargs.items()}
    return a, k


def _locations(fn, args, kwargs, names):
    """The locations of the named parameters in this launch, or None when one is not found."""
    params = _names(fn)
    out = []
    for n in names:
        if n in kwargs:
            out.append(("k", n))
        elif n in params and params.index(n) < len(args):
            out.append(("a", params.index(n)))
        else:
            return None
    return out


def _span(t):
    """The byte range a tensor's elements occupy: (start, end), or None when it cannot be read."""
    try:
        if t.numel() == 0:
            return None
        extent = 1 + sum((int(n) - 1) * abs(int(s)) for n, s in zip(t.shape, t.stride()))
        start = int(t.data_ptr())
        return start, start + extent * int(t.element_size())
    except Exception:  # noqa: BLE001
        return None


def _margin(t) -> int:
    try:
        return int(max((int(s) for s in t.stride()), default=0))
    except Exception:  # noqa: BLE001
        return 0


def differential(launch, args, kwargs, relay):
    """Run the launch three times on copies of every tensor argument: as given, then twice with the tensors at the
    `relay` locations ({location: whole}) laid out again (kernel_launch_contract.relaid). Returns (compare of the
    as-given launch's floating tensors against the first relaid launch's, the second relaid launch as the noise
    floor; the locations the kernel wrote), or None when the comparison cannot stand for the real launch or decides
    nothing: a layout cannot be made, a tensor the kernel writes shares memory with another argument (copies would
    part what the kernel reads through the other), or every value is zero. Tensors that only share storage and are
    only read (two halves of one projection) are copied apart safely."""
    krc, klc = kernel_reference_contract, kernel_launch_contract
    locs = _tensor_locations(args, kwargs)
    runs, written = [], []
    for relaid in (False, True, True):
        repl = {}
        for loc in locs:
            t = _get(args, kwargs, loc)
            if relaid and loc in relay:
                c = klc.relaid(t, _margin(t), whole=relay[loc])
                if c is None:
                    return None
            else:
                c = krc.kept(t, _margin(t))
            repl[loc] = c
        a, k = _with(args, kwargs, repl)
        launch(a, k)
        if not relaid:
            written = [loc for loc in locs if not bool((repl[loc] == _get(args, kwargs, loc)).all())]
        runs.append([repl[loc] for loc in locs if repl[loc].is_floating_point()])
    spans = {loc: _span(_get(args, kwargs, loc)) for loc in locs}
    for w in written:
        for loc in locs:
            if loc != w and spans[w] and spans[loc] and spans[w][0] < spans[loc][1] and spans[loc][0] < spans[w][1]:
                return None
    if not runs[0]:
        return None
    cmp = krc.compare(runs[0], runs[1], runs[2])
    if cmp.scale == 0.0 and cmp.diff == 0.0 and not cmp.nonfinite:
        return None
    return cmp, written


def launch_relaid(launch, args, kwargs, relay, written=()):
    """The repair: launch on relaid copies of the tensors at the `relay` locations and copy what the kernel wrote
    (the `written` locations) back into the caller's tensors."""
    klc = kernel_launch_contract
    repl = {}
    for loc, whole in relay.items():
        t = _get(args, kwargs, loc)
        c = klc.relaid(t, _margin(t), whole=whole)
        if c is None:
            return launch(args, kwargs)
        repl[loc] = c
    a, k = _with(args, kwargs, repl)
    out = launch(a, k)
    for loc, c in repl.items():
        if loc in written:
            _get(args, kwargs, loc).copy_(c)
    return out


def _where(fn):
    f = getattr(fn, "fn", fn)
    module = getattr(f, "__module__", "") or ""
    eng = module.split(".", 1)[0] or "kernel"
    return eng, getattr(f, "__name__", "kernel"), module


def _decide(fn, args, kwargs, key, launch=None):
    """Decide one launch pattern: by running it twice (M19 L3.3b) when a tensor's layout cannot be told from the
    launch and the launch can be run on copies, else from the launch alone (M17.3). `launch(args, kwargs)` runs the
    kernel (the original JITFunction.run)."""
    eng, name, module = _where(fn)
    bound = read_choice(fn, args, kwargs)
    told = set(told_params(fn))
    boundary, consumer, where = f"kernel:{eng}.{name}", f"{eng}.{name}", f"{module}.{name} launch"
    found = kernel_launch_contract.classify(bound, told)
    cands = [f for f in found if f["status"] != "told"]
    if cands and launch is not None and not _capturing():
        locs = _locations(fn, args, kwargs, [f["name"] for f in cands])
        tensors = [_get(args, kwargs, loc) for loc in _tensor_locations(args, kwargs)]
        if locs is not None and kernel_reference_contract.nbytes(tensors) <= kernel_reference_contract.BUDGET:
            relay = {loc: f["status"] == "assumed" for loc, f in zip(locs, cands)}
            got = differential(launch, args, kwargs, relay)
            if got is not None:
                cmp, written = got
                names = [f["name"] for f in cands]
                repair = (f"{', '.join(names)} laid out contiguously in the innermost dimension (every other stride "
                          f"kept{'; wholly contiguous for a kernel told no stride' if any(relay.values()) else ''}) "
                          f"for every launch of {name} with this layout pattern, and what it wrote copied back")
                ds = kernel_launch_contract.check_relaid(boundary, consumer, name, names, cmp, where, owner=key,
                                                          repair=repair)
                _count("run_twice")
                if ds and ds[0].verdict.name == "RESOLVED":
                    _REPAIR[key] = (relay, [loc for loc in written if loc in relay])
                return
    kernel_launch_contract.check(boundary, consumer, name, bound, where, owner=key, ints_from=told)


def install():
    global _ORIG
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = JITFunction.run
    orig = _ORIG

    def run(self, *args, **kwargs):
        if not kwargs.get("warmup") and core.mode() in ("load", "debug"):
            try:
                key = _layout_key(self, args, kwargs)
            except Exception:  # noqa: BLE001 - never the engine's problem (principle 12)
                return orig(self, *args, **kwargs)
            launch = lambda a, k: orig(self, *a, **k)  # noqa: E731
            if key not in _REPAIR and _look(self, key) is not None:
                from .. import load

                load.safely("kernel:triton", "triton.launch", "Layout",
                            lambda: _decide(self, args, kwargs, key, launch))
            fix = _REPAIR.get(key)
            if fix is not None:
                _count("relaid_launches")
                return launch_relaid(launch, args, kwargs, fix[0], fix[1])
        return orig(self, *args, **kwargs)

    JITFunction.run = run
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from triton.runtime.jit import JITFunction

    JITFunction.run = _ORIG
    _ORIG = None
    _SEEN.clear()
    _COUNT.clear()
    _REPAIR.clear()
    return 1


def stats():
    out = {"kernels_seen": len(_COUNT), "patterns_decided": len(_SEEN), "patterns_relaid": len(_REPAIR)}
    out.update(_STATS)
    return out


def reset():
    _SEEN.clear()
    _COUNT.clear()
    _REPAIR.clear()
    _STATS.clear()
