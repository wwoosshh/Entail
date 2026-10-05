"""kernel_check: the one rule (kernel_types) at every Triton launch the engine makes (ROADMAP M19 L6; ENTAIL=types).

Meanings are attached where values are made and travel with them by their memory:
  the activation quantizer     x_q [token, hidden] (a value), x_s [token, hidden / group] (its scale)
  the weight loader            w [feature, hidden] (a value), w_s [feature / block, hidden / block] (its scale)
  every proven launch          the output's axes, as the rule inferred them from what the kernel stored
  a layer, while it runs       what it declares about the values it is handed (declarations.py: a rotary layer's
                               query and key - tokens, heads, and in each head its rotation pairs), for the launches
                               inside it, by shape and dtype
At every Triton launch (JITFunction.run) the tensors handed to the kernel are looked up, the kernel's IR is read once
per launch configuration (kernel, constexprs, shapes, strides, integer arguments, grid, and which arguments carry which
meanings), and the verdict is recorded. The output of a proven launch gets the inferred meaning, so the next kernel
that reads it sees named axes. Nothing is refused and no result is changed: this is the checker as a reporter
(ROADMAP M19 L6 step 2); what to do with a verdict is the policy's business.
"""
import bisect
import datetime
import functools
import inspect
import json
import os
import sys
import threading
import time
import weakref

from . import graph_types, kernel_ir, kernel_types
from .kernel_types import Axis, Meaning

_FACTS = {}            # (device, storage start) -> list of [data_ptr, shape, stride, dtype, fact dict, weakref]
_WCOUNT = {}           # (device, storage start) -> how many writes the launches seen made into that storage
_ALIEN = {}            # (device, storage start) -> writes the launches seen made into its memory through another storage
_UNWRITTEN = [0]       # parameters found holding elements nothing wrote (lifetime.loaded), in this process
_ALIASED = set()       # storages whose bytes overlap another live storage holding a meaning (found when registered)
_EXTENT = {}           # (device, storage start) -> storage end (bytes), for the storages that hold a meaning
_STARTS = {}           # device -> sorted storage starts in _EXTENT (to find two live values in one memory)
_VERDICTS = {}         # launch configuration key -> (Verdict, output argument name, inferred meaning)
_STATS = {}
_WRAPPED = {}
_SERIAL = [0]
_SCOPES = []           # the layers running now that declare what the values they are handed mean (scope_push)
_LOCK = threading.RLock()
_PATH = None


def _count(k, by=1):
    _STATS[k] = _STATS.get(k, 0) + by


_IS_COMPILING = []


def _compiling():
    try:
        if not _IS_COMPILING:
            import torch

            _IS_COMPILING.append(torch.compiler.is_compiling)
        return bool(_IS_COMPILING[0]())
    except Exception:  # noqa: BLE001
        return False


def _storage_key(t):
    off = t.storage_offset()
    return (t.get_device(), t.data_ptr() - off * t.element_size())     # the device's index (-1 on the CPU)


# --- the facts: a meaning on a tensor's memory, gone when the tensor is -----------------------------------------------

def attach(t, names, kind="value", serial=0, pair=0, groups=None, sums=None, pending=(), basis=None, label=None,
           pointers=None, packed=None, life="value", register=True):
    """Give the tensor `t` a meaning: a name per axis (None for an axis that means nothing the rule can pair); for an
    integer tensor the basis of its numbers; a label a loaded stride can refer to; for a table of pointers the
    tensors it points to (each with its own meaning attached); for a tensor packed for one kernel the packed form
    and the sizes it was packed from (its axes then mean nothing elementwise). `life` says how long the content
    stays what the meaning says (M19 L7): "const" (nothing writes it after this - a loaded weight), "state" (its
    owner updates it in place - a cache, a table), "value" (made once and read). `register` keeps the storage's
    bytes for the one-memory-one-value check (left out for the outputs a launch's meaning is inferred for: torch's
    allocator never gives out the same bytes twice while both live)."""
    return _attach_at(t, _storage_key(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, names, kind,
                      serial, pair, groups, sums, pending, basis, label, pointers, packed, life, register)


def _attach_at(t, key, ptr, shape, stride, dtype, names, kind="value", serial=0, pair=0, groups=None, sums=None,
               pending=(), basis=None, label=None, pointers=None, packed=None, life="value", register=True):
    """attach, for a tensor whose storage key, address, shape, strides and dtype the caller has read already."""
    fact = {"names": list(names), "kind": kind, "serial": serial, "pair": pair,
            "groups": list(groups or [1] * len(names)), "sums": dict(sums or {}), "pending": list(pending),
            "basis": basis, "label": label, "packed": packed, "life": life, "born": _version_of(t, key), "key": key,
            "pointers": None if pointers is None else [_snapshot(x) for x in pointers]}
    if register:
        _register_extent(t, key)
    sig = _sig_of(fact)
    entry = [ptr, shape, stride, dtype, fact, None, sig]
    with _LOCK:
        lst = _FACTS.setdefault(key, [])
        for e in lst:
            if e[0] == ptr and e[1] == shape and e[2] == stride and e[3] == dtype:
                ref = e[5]
                if ref is not None and ref() is t:
                    # the same tensor given a meaning again (a buffer an engine reuses every step): its entry and
                    # the reference that drops it when the tensor goes stay; the meaning is the new one
                    e[4], e[6] = fact, sig
                    _count("attached")
                    return fact
        lst[:] = [e for e in lst if not (e[0] == ptr and e[1] == shape and e[2] == stride and e[3] == dtype)]
        lst.append(entry)

    def gone(key=key, entry=entry):
        with _LOCK:
            lst = _FACTS.get(key)
            if lst is not None:
                lst[:] = [e for e in lst if e is not entry]
                if not lst:
                    _FACTS.pop(key, None)
                    _drop_extent(key)
    try:
        entry[5] = weakref.ref(t, lambda _ref, gone=gone: gone())
    except TypeError:
        pass
    _count("attached")
    return fact


def _sig_of(fact):
    """What of a meaning a launch's verdict depends on (the part of the launch key a meaning gives)."""
    sig = (tuple(fact["names"]), fact["kind"], tuple(fact["groups"]), fact["pair"] != 0)
    view = fact.get("view")
    return sig if view is None else sig + ((tuple(view["shape"]), tuple(view["stride"])),)


# --- a layer, while it runs: what it declares about the values it is handed (M22.4) ---------------------------------

def scope_push(fact_for):
    """A layer that declares what the values it is handed mean starts running. Until scope_pop, a tensor handed to a
    kernel that has no meaning of its own gets fact_for(shape, stride, dtype) (None: the layer says nothing of it).
    Eager execution shows a kernel's arguments, not the copies a layer makes of its values on the way (a contiguous
    copy of a query), so a value is known by its shape and dtype; the meaning serves the launches inside the layer
    and is not attached to the tensor."""
    _SCOPES.append(fact_for)


def scope_pop():
    if _SCOPES:
        _SCOPES.pop()


def _scope_fact(shape, stride, dtype):
    for fact_for in reversed(_SCOPES):
        if fact_for is None:
            continue
        try:
            f = fact_for(shape, stride, dtype)
        except Exception:  # noqa: BLE001 - never the engine's problem
            f = None
        if f is not None:
            return f
    return None


def scoped_fact(names, view_shape, view_stride):
    """The fact a layer gives a value while it runs: axis names over a finer view of the value's memory (the same
    elements, an axis split into named parts - a head's features into its rotation pairs)."""
    n = len(names)
    return {"names": list(names), "kind": "value", "serial": 0, "pair": 0, "groups": [1] * n, "sums": {},
            "pending": [], "basis": None, "label": None, "packed": None, "life": "value", "born": None, "key": None,
            "pointers": None, "view": {"shape": [int(x) for x in view_shape], "stride": [int(x) for x in view_stride]},
            "scope": True}


# --- lifetime (M19 L7): one memory holds one live value; a constant is not written after it is made ---------------

def _version_of(t, key=None):
    """The content version of a tensor's memory: the writes the launches seen made into its storage, torch's own
    counter of in-place writes (shared by every view of the storage), and the writes the launches seen made into its
    memory through another storage (another value written over it)."""
    try:
        # an inference tensor (made under torch.inference_mode, as an engine's activations are) has no version
        # counter, and asking for it raises: only the launches' counts follow its writes
        v = 0 if t.is_inference() else int(t._version)
    except Exception:  # noqa: BLE001
        v = 0
    k = key if key is not None else _storage_key(t)
    return (_WCOUNT.get(k, 0), v, _ALIEN.get(k, 0))


def _register_extent(t, key):
    """A storage that holds a meaning, by its bytes; another live storage over the same bytes is two values in one
    memory (the allocator never gives that out while both live: an alias)."""
    if key in _EXTENT:
        return
    try:
        nbytes = int(t.untyped_storage().nbytes())
    except Exception:  # noqa: BLE001
        return
    if nbytes <= 0:
        return
    dev, start = key
    end = start + nbytes
    starts = _STARTS.setdefault(dev, [])
    i = bisect.bisect_left(starts, start)
    other = None
    if i > 0 and _EXTENT.get((dev, starts[i - 1]), 0) > start:
        other = (dev, starts[i - 1])
    elif i < len(starts) and starts[i] < end:
        other = (dev, starts[i])
    _EXTENT[key] = end
    starts.insert(i, start)
    if other is not None:
        _ALIASED.update((key, other))
        _count("life_alias")
        _write({"kind": "types_life", "verdict": "violation",
                "why": f"two live values share memory: a storage of {nbytes} bytes at {start:#x} overlaps another "
                       f"at {other[1]:#x} ({_EXTENT[other] - other[1]} bytes)"})
        _broken("attach", kernel_types.Verdict(
            "violation", f"two live values share memory: a new value's storage ({nbytes} bytes at {start:#x}) "
                         f"overlaps a live one at {other[1]:#x}"), stop=False)


def _drop_extent(key):
    _ALIASED.discard(key)
    if _EXTENT.pop(key, None) is None:
        return
    dev, start = key
    starts = _STARTS.get(dev, [])
    i = bisect.bisect_left(starts, start)
    if i < len(starts) and starts[i] == start:
        starts.pop(i)


def set_life(t, life):
    """Say how long a tensor's content stays what its meaning says (the fact it already has, or an unnamed one)."""
    f = fact_of(t)
    if f is None:
        attach(t, [None] * t.dim(), life=life)
        return
    with _LOCK:
        for e in _FACTS.get(_storage_key(t), []):
            if e[0] == t.data_ptr() and e[1] == tuple(t.shape) and e[2] == tuple(t.stride()) and e[3] == t.dtype:
                e[4]["life"] = life
                e[4]["born"] = _version_of(t)
                return
    attach(t, [None] * t.dim(), life=life)


def set_unwritten(t, n):
    """Say that `n` elements of a tensor were never written while it was made (lifetime.loaded): whoever reads it
    reads no value there."""
    if fact_of(t) is None:
        attach(t, [None] * t.dim())
    _UNWRITTEN[0] += 1
    with _LOCK:
        for e in _FACTS.get(_storage_key(t), []):
            if e[0] == t.data_ptr() and e[1] == tuple(t.shape) and e[2] == tuple(t.stride()) and e[3] == t.dtype:
                e[4]["unwritten"] = int(n)
                return


def _life_check(kernel, tensors, facts, written):
    """Before a launch runs: a tensor it reads that holds elements nothing wrote, a constant it writes, or a constant
    it reads that something changed since it was made, is a violation (the value the reader gets is not the one its
    meaning names). Returns the reasons."""
    bad = []
    for k, t in tensors.items():
        f = facts.get(k)
        if f is None:
            continue
        if f.get("unwritten") and (written is None or k not in written):
            bad.append(f"the kernel reads {k}, but {f['unwritten']} of its elements were never written while the "
                       f"model was loaded")
        born = f.get("born")
        if f.get("life") != "const":
            # any value: a write through another storage into its memory replaced it with another value (no such
            # write has happened in this process while _ALIEN is empty: nothing to compare)
            if _ALIEN and born is not None and len(born) > 2 and (written is None or k not in written):
                now = _version_of(t)
                if now[2] != born[2]:
                    bad.append(f"the kernel reads {k}, but since it was made a kernel wrote another value into its "
                               f"memory (through another tensor's storage over the same bytes)")
            continue
        if written is not None and k in written:
            bad.append(f"the kernel writes {k}, a constant (a weight) since it was loaded")
            continue
        now = _version_of(t, f.get("key"))
        if born is not None and tuple(born) != tuple(now[:len(born)]):
            how = ("a kernel this process saw" if now[0] != born[0] else
                   "a kernel writing another value over its memory" if len(born) > 2 and now[2] != born[2] else
                   "an in-place torch operation or a copy")
            bad.append(f"{k} is a constant (a weight), but it was written after it was loaded (by {how})")
    return bad


def _note_writes(tensors, written, keys=None):
    """For a launch about to run: each tensor it writes moves its storage's version, and the version of every other
    storage holding a meaning whose bytes the write's tensor spans (another value written over it). Returns the
    reasons: a write that reaches another live value's memory. `keys`: the tensors' storage keys, read already."""
    bad = []
    for k in (written or ()):
        t = tensors.get(k)
        if t is not None:
            key = keys[k] if keys is not None and k in keys else _storage_key(t)
            _WCOUNT[key] = _WCOUNT.get(key, 0) + 1
            if key in _ALIASED:
                others = _overlapping(t, key)
                for other in others:
                    _ALIEN[other] = _ALIEN.get(other, 0) + 1
                if others:
                    bad.append(f"the kernel writes {k}, but its bytes reach the memory of {len(others)} other live "
                               f"value(s) (another storage over the same bytes)")
    return bad


def _overlapping(t, own):
    """The other storages holding a meaning that the bytes of `t` reach."""
    try:
        if t.numel() == 0:
            return []
        lo = t.data_ptr()
        span = sum((int(n) - 1) * int(s) for n, s in zip(t.shape, t.stride()) if int(n) > 0) + 1
        hi = lo + span * t.element_size()
    except Exception:  # noqa: BLE001
        return []
    dev = own[0]
    starts = _STARTS.get(dev) or []
    out = []
    j = bisect.bisect_left(starts, hi) - 1
    while j >= 0:
        k = (dev, starts[j])
        end = _EXTENT.get(k, 0)
        if end <= lo:
            break                  # storages are disjoint unless aliased: the ones before end earlier still
        if k != own:
            out.append(k)
        j -= 1
    return out


def _snapshot(t):
    """A pointer table's target: its layout and its fact, as they are now."""
    f = fact_of(t)
    return {"shape": tuple(int(x) for x in t.shape), "stride": tuple(int(x) for x in t.stride()),
            "dtype": str(t.dtype), "fact": f}


def _meaning_from(shape, stride, f, kind=None):
    view = f.get("view") if f is not None else None
    if view is not None:               # the same memory as a finer view, whose axes the fact names
        shape, stride = view["shape"], view["stride"]
    shape, stride = tuple(shape), tuple(stride)
    if f is None or len(f["names"]) != len(shape):
        return Meaning(tuple(Axis(None, n) for n in shape), shape, stride, kind or "value")
    origins = f.get("origins") or [0] * len(shape)
    axes = tuple(Axis(nm, n, g, o) for nm, n, g, o in zip(f["names"], shape, f["groups"], origins))
    return Meaning(axes, shape, stride, kind or f["kind"], f["serial"], f["pair"], basis=f.get("basis"),
                   label=f.get("label"))


def fact_of(t):
    return _fact_at(t, _storage_key(t), t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)


def _fact_at(t, key, ptr, shape, stride, dtype):
    """fact_of, for a tensor whose storage key, address, shape, strides and dtype the caller has read already."""
    lst = _FACTS.get(key)
    if not lst:
        return None
    for e in lst:
        if e[0] == ptr and e[1] == shape and e[2] == stride and e[3] == dtype:
            return e[4]
    return _view_of(t, lst, ptr, shape, stride, dtype)


def _fact_sig_at(t, key, ptr, shape, stride, dtype):
    """(_fact_at, its part of a launch key)."""
    lst = _FACTS.get(key)
    if not lst:
        return None, None
    for e in lst:
        if e[0] == ptr and e[1] == shape and e[2] == stride and e[3] == dtype:
            return e[4], e[6]
    f = _view_of(t, lst, ptr, shape, stride, dtype)
    return f, (None if f is None else _sig_of(f))


def _view_of(t, lst, ptr, shape, stride, dtype):
    # a view of a tensor with a meaning (a slice, a selected row or column, a transpose, a merge or a split of its
    # axes): each axis of the view that walks one axis of the tensor means what that axis means, from the
    # coordinate the view starts at; an axis that walks two merged axes means what the vocabulary says the merge
    # means; anything else means nothing the rule can pair
    for e in reversed(lst):
        if e[3] != dtype or e[4].get("pointers") is not None:
            continue
        f = _view_fact(ptr, tuple(shape), tuple(stride), t.element_size(), e)
        if f is not None:
            return f
    return None


def _view_fact(ptr, shape, stride, esize, e):
    base, eshape, estride, _dtype, ef = e[0], e[1], e[2], e[3], e[4]
    delta = ptr - base
    if delta < 0 or delta % esize:
        return None
    off = delta // esize
    names, groups = ef["names"], ef.get("groups") or [1] * len(eshape)
    order = sorted((i for i in range(len(eshape)) if eshape[i] > 1 and estride[i] > 0), key=lambda i: -estride[i])
    start = [0] * len(eshape)
    for i in order:
        start[i], off = divmod(off, estride[i])
        if start[i] >= eshape[i]:
            return None
    if off:
        return None
    vnames, vgroups, vorigins, used = [], [], [], set()
    for n, s in zip(shape, stride):
        if n <= 1:
            vnames.append(None)
            vgroups.append(1)
            vorigins.append(0)
            continue
        hit = next((i for i in order if estride[i] == s and i not in used), None)
        if hit is not None and start[hit] + n <= eshape[hit]:
            used.add(hit)
            vnames.append(names[hit] if hit < len(names) else None)
            vgroups.append(groups[hit] if hit < len(groups) else 1)
            vorigins.append(start[hit])
            continue
        # two adjacent axes walked as one (a merge): the vocabulary names it
        merged = None
        for a, b in zip(order, order[1:]):
            if estride[b] == s and estride[a] == estride[b] * eshape[b] and a not in used and b not in used and \
                    start[b] == 0 and n <= (eshape[a] - start[a]) * eshape[b]:
                merged = (a, b)
                break
        if merged is not None:
            a, b = merged
            used.update(merged)
            vnames.append(kernel_types.MERGES.get((names[a], names[b])))
            vgroups.append(1)
            vorigins.append(start[a] * eshape[b])
            continue
        vnames.append(None)              # a split of an axis, or a stride the tensor does not have
        vgroups.append(1)
        vorigins.append(0)
    if not any(vnames):
        return None
    f = dict(ef)
    f["names"], f["groups"], f["origins"] = vnames, vgroups, vorigins
    f["view_of"] = (base, tuple(eshape))
    return f


def meaning_of(t):
    """The Meaning of a tensor handed to a kernel: its layout from the tensor, its axes from its fact (unnamed
    without one)."""
    f = fact_of(t)
    return _meaning_from(t.shape, t.stride(), f), f


def _next_serial():
    _SERIAL[0] += 1
    return _SERIAL[0]


# --- the producers --------------------------------------------------------------------------------------------------

def issue_pair(value, scale, value_names, scale_names, scale_groups):
    """A quantized value and its scale, as one producer made them."""
    s = _next_serial()
    p = _next_serial()
    attach(value, value_names, "value", s, p)
    attach(scale, scale_names, "scale", p, s, scale_groups)
    return s, p


def _wrap(holder, name, make):
    orig = getattr(holder, name) if not isinstance(holder, type) else holder.__dict__[name]
    if getattr(orig, "__entail_types__", False):
        return 0
    run = make(orig)
    run.__entail_types__ = True
    setattr(holder, name, run)
    _WRAPPED[(holder, name)] = orig
    return 1


def install_fp8_utils():
    mod = sys.modules.get("vllm.model_executor.layers.quantization.utils.fp8_utils")
    if mod is None:
        return 0

    def make(orig):
        sig = inspect.signature(orig)
        names = list(sig.parameters)

        @functools.wraps(orig)
        def run(*args, **kwargs):
            out = orig(*args, **kwargs)
            if _compiling():
                return out
            try:
                bound = dict(zip(names, args))
                bound.update(kwargs)
                x_q, x_s = out
                g = int(bound["group_size"])
                issue_pair(x_q, x_s, ["token", "hidden"], ["token", "hidden"], [1, g])
                _count("activation_issued")
            except Exception:  # noqa: BLE001 - never the engine's problem
                _count("activation_issue_failed")
            return out
        return run
    return _wrap(mod, "per_token_group_quant_fp8", make)


def install_weights():
    mod = sys.modules.get("vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel")
    cls = getattr(mod, "Fp8BlockScaledMMLinearKernel", None) if mod is not None else None
    if cls is None:
        return 0

    def make(orig):
        @functools.wraps(orig)
        def run(self, layer):
            out = orig(self, layer)
            try:
                params = self._get_layer_params(layer)
                gs = self.weight_group_shape
                block = (int(gs.row), int(gs.col)) if hasattr(gs, "row") else (int(gs[0]), int(gs[1]))
                issue_pair(params.weight, params.block_scale, ["feature", "hidden"], ["feature", "hidden"],
                           [block[0], block[1]])
                _count("weight_issued")
            except Exception:  # noqa: BLE001
                _count("weight_issue_failed")
            return out
        return run
    return _wrap(cls, "process_weights_after_loading", make)


def install_marlin_weights():
    """FP8 weights repacked for Marlin (vLLM's default on a GPU without FP8 units): the packed weight and its
    permuted scales are one issue, with the sizes they were packed from."""
    mod = sys.modules.get("vllm.model_executor.kernels.linear.scaled_mm.marlin")
    cls = getattr(mod, "MarlinFP8ScaledMMLinearKernel", None) if mod is not None else None
    if cls is None:
        return 0

    def make(orig):
        @functools.wraps(orig)
        def run(self, layer):
            out = orig(self, layer)
            try:
                w = layer.weight
                sname = "weight_scale_inv" if getattr(layer, "weight_scale_inv", None) is not None else "weight_scale"
                sc = getattr(layer, sname)
                size_n, size_k = int(layer.output_size_per_partition), int(layer.input_size_per_partition)
                bs = getattr(layer, "weight_block_size", None)
                group = int(bs[1]) if bs is not None else size_k
                serial, pair = _next_serial(), _next_serial()
                packed = {"form": "marlin_fp8", "size_k": size_k, "size_n": size_n, "group": group}
                attach(w, [None] * w.dim(), "value", serial, pair, packed=packed)
                attach(sc, ["hidden", "feature"] if sc.dim() == 2 else [None] * sc.dim(), "scale", pair, serial,
                       [group, 1] if sc.dim() == 2 else None, packed=dict(packed, form="marlin_permuted"))
                _count("weight_issued")
            except Exception:  # noqa: BLE001
                _count("weight_issue_failed")
            return out
        return run
    return _wrap(cls, "process_weights_after_loading", make)


# --- the compiled graph ---------------------------------------------------------------------------------------------

_GRAPHS = []


def _graph_key(config, prefix=""):
    """One key per engine configuration and compiled part: the configuration's own hash (what vLLM keys its compile
    cache by), which both the compile backend and the loader of a cached compiled model know."""
    try:
        h = config.compute_hash()
    except Exception:  # noqa: BLE001
        h = "unknown"
    return f"graph:{_checker_version()}:{h}:{prefix}"


def _graph(graph, example_inputs):
    v = graph_types.check_graph(graph, example_inputs, fact_of)
    _GRAPHS.append(v)
    _count(f"graph_{v['verdict']}")
    _write(dict(v, kind="types_graph", cached=False))
    _tallied(BOUNDARY_TYPES, v["verdict"])
    for x in v.get("violations", []):
        _broken(f"graph.{x.get('op')}", kernel_types.Verdict("violation", x.get("why", "")))
    return v


def install_compile():
    """vLLM's compile backend receives the model's graph with its real inputs: the rule over it, once, before it is
    compiled. The verdict is kept with vLLM's compile cache so a later process that loads the compiled model from
    disk still reports it."""
    mod = sys.modules.get("vllm.compilation.backends")
    cls = getattr(mod, "VllmBackend", None) if mod is not None else None
    if cls is None:
        return 0

    def make(orig):
        @functools.wraps(orig)
        def call(self, graph, example_inputs):
            v = None
            try:
                v = _graph(graph, example_inputs)
            except Broken:
                raise
            except Exception as e:  # noqa: BLE001 - never the engine's problem
                _count("graph_hook_failed")
                _write({"kind": "types_graph", "verdict": "unproven", "why": f"the check raised {type(e).__name__}: {e}"})
            out = orig(self, graph, example_inputs)
            try:
                if v is not None:
                    _cache_put(_graph_key(self.vllm_config, getattr(self, "prefix", "")), {"graph": v})
            except Exception:  # noqa: BLE001
                _count("graph_cache_write_failed")
            return out
        return call
    return _wrap(cls, "__call__", make)


def install_compile_cache():
    """A compiled model loaded from vLLM's cache skips the backend, so the graph is not seen: when its verdict is in
    the cache it is reported again; when it is not (the cache predates the rule), the load is declined once so the
    model compiles and the rule sees it."""
    mod = sys.modules.get("vllm.compilation.decorators")
    if mod is None or not hasattr(mod, "_try_load_aot_compiled_fn"):
        return 0

    def make(orig):
        @functools.wraps(orig)
        def load(self, path, *args, **kwargs):
            if _UNWRITTEN[0]:
                # parameters hold elements nothing wrote: a cached graph would run without any reader seeing them
                # (no module runs, no graph is checked), so the model is compiled again and its graph checked
                _count("graph_cache_declined")
                _write({"kind": "types_graph", "verdict": "deferred", "path": str(path),
                        "why": "parameters hold elements nothing wrote; the model is compiled again so the graph "
                               "check sees which of them it reads"})
                return None
            try:
                head = _graph_key(self.vllm_config)
                olds = [rec for key, rec in _cache().items() if key.startswith(head)]
                if not olds:
                    _count("graph_cache_declined")
                    _write({"kind": "types_graph", "verdict": "deferred", "why": "the compiled model in vLLM's cache "
                            "has no verdict from this checker; it is compiled once more so the rule sees its graph",
                            "path": str(path)})
                    return None
                for old in olds:
                    v = old.get("graph")
                    if v:
                        _GRAPHS.append(v)
                        _count(f"graph_{v.get('verdict', 'unproven')}")
                        _count("graph_from_cache")
                        _write(dict(v, kind="types_graph", cached=True))
                        _tallied(BOUNDARY_TYPES, v.get("verdict", "unproven"))
            except Exception:  # noqa: BLE001
                _count("graph_cache_failed")
            return orig(self, path, *args, **kwargs)
        return load
    return _wrap(mod, "_try_load_aot_compiled_fn", make)


# --- the launches ---------------------------------------------------------------------------------------------------

def _record_path():
    global _PATH
    if _PATH is None:
        named = os.environ.get("ENTAIL_TYPES_RECORD")
        if named:
            _PATH = named
        else:
            folder = os.environ.get("ENTAIL_LOG_DIR", "entail_logs")
            _PATH = os.path.join(folder, f"types-{datetime.date.today().isoformat()}.jsonl")
        try:
            os.makedirs(os.path.dirname(_PATH) or ".", exist_ok=True)
        except OSError:
            pass
    return _PATH


def _write(line):
    try:
        with open(_record_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    except OSError:
        _count("record_failed")


_CACHE = None          # verdicts of earlier processes: key -> record (entail_logs/types-cache.jsonl)


def _cache_path():
    named = os.environ.get("ENTAIL_TYPES_CACHE")
    if named:
        return named
    return os.path.join(os.environ.get("ENTAIL_LOG_DIR", "entail_logs"), "types-cache.jsonl")


_VERSION = None


def _checker_version():
    """The checker's own code, hashed: a verdict of an older checker is not reused."""
    global _VERSION
    if _VERSION is None:
        import hashlib

        h = hashlib.sha256()
        here = os.path.dirname(os.path.abspath(__file__))
        # the rule, and what decides the meanings it is handed (the producers' code and their data files): a
        # verdict cached under other meanings is not one for these
        names = ["kernel_ir.py", "kernel_types.py", "kernel_check.py", "graph_types.py",
                 "declarations.py", os.path.join("adapters", "vllm_declarations.py"),
                 os.path.join("adapters", "sglang_declarations.py"), os.path.join("adapters", "vllm_index_meanings.py")]
        try:
            names += sorted(os.path.join("data", f) for f in os.listdir(os.path.join(here, "data"))
                            if f.endswith(".json"))
        except OSError:
            pass
        for name in names:
            try:
                with open(os.path.join(here, name), "rb") as f:
                    h.update(f.read())
            except OSError:
                h.update(name.encode())
        _VERSION = h.hexdigest()[:16]
    return _VERSION


def _cache():
    """The verdicts earlier processes decided, read once: a launch configuration is decided once per machine, not
    once per process (the checker's code, the kernel's IR text, the meanings, the integer arguments and the grid
    make the key)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        try:
            with open(_cache_path(), encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        _CACHE[rec["key"]] = rec
                    except (ValueError, KeyError):
                        continue
        except OSError:
            pass
    return _CACHE


def _cache_put(key, rec):
    _cache()[key] = rec
    try:
        os.makedirs(os.path.dirname(_cache_path()) or ".", exist_ok=True)
        with open(_cache_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(rec, key=key), ensure_ascii=False, default=str) + "\n")
    except OSError:
        _count("cache_write_failed")


_PROFILE = os.environ.get("ENTAIL_TYPES_PROFILE") == "1"


def _tick(name, t0):
    if _PROFILE:
        import time

        t1 = time.perf_counter_ns()
        _STATS[f"ns_{name}"] = _STATS.get(f"ns_{name}", 0) + (t1 - t0)
        return t1
    return t0


_FN = {}               # id(fn) -> (fn, parameter names, their set, constexpr names, their set, constexpr part by values,
#                        which parameters are constexpr)
_ESIZE = {}            # dtype -> bytes per element
_MISSING = object()
_HOT = {}              # torch, and the adapter's kernel key, imported once for the launch path


def _read(fn, args, kwargs, grid, Tensor):
    """read_launch's reading of one launch (vllm_block_fp8_guarantee.read_launch: the arguments by name, the
    constexpr part of the kernel's key, the grid) in one pass, as (tensors, scalars, constexpr part, grid); the
    constexpr part's text is made once per set of values."""
    e = _FN.get(id(fn))
    if e is None or e[0] is not fn:
        names = [p.name for p in fn.params]
        flags = [bool(getattr(p, "is_constexpr", False)) for p in fn.params]
        cnames = tuple(n for n, c in zip(names, flags) if c)
        e = _FN[id(fn)] = (fn, names, set(names), cnames, set(cnames), {}, flags)
    _fn, names, nset, cnames, cset, by_values, flags = e
    tensors, scalars, cvals = {}, {}, {}
    for name, c, v in zip(names, flags, args):
        if c:
            cvals[name] = v
        elif isinstance(v, Tensor):
            tensors[name] = v
        elif isinstance(v, (int, float)):
            scalars[name] = float(v) if isinstance(v, float) else int(v)
    extra = ()
    if kwargs:
        ex = []
        for k, v in kwargs.items():
            if k not in nset:
                ex.append((k, v))
            elif k in cset:
                cvals[k] = v
            elif isinstance(v, Tensor):
                tensors[k] = v
            elif isinstance(v, (int, float)):
                scalars[k] = float(v) if isinstance(v, float) else int(v)
        extra = tuple(ex)
    try:
        ck = (tuple([(type(v), v) for v in [cvals.get(n, _MISSING) for n in cnames]]),
              tuple([(k, type(v), v) for k, v in extra]))
        consts = by_values.get(ck)
        if consts is None:
            consts = by_values[ck] = tuple(sorted((k, repr(v)) for k, v in cvals.items())) + \
                tuple(sorted((k, repr(v)) for k, v in extra))
    except TypeError:                    # a constexpr value that cannot be a key: its text, as read_launch makes it
        consts = tuple(sorted((k, repr(v)) for k, v in cvals.items())) + tuple(sorted((k, repr(v)) for k, v in extra))
    if callable(grid):
        meta = dict(zip(names, args))
        meta.update(kwargs)
        grid = grid(meta)
    g = tuple(int(x) for x in (grid if isinstance(grid, (tuple, list)) else (grid,)))
    return tensors, scalars, consts, g


def _launch(fn, args, kwargs, grid):
    if not _HOT:
        import torch

        from .adapters.vllm_block_fp8_guarantee import _kernel_key
        _HOT.update(torch=torch, Tensor=torch.Tensor, kernel_key=_kernel_key)
    torch, Tensor, _kernel_key = _HOT["torch"], _HOT["Tensor"], _HOT["kernel_key"]

    t0 = time.perf_counter_ns() if _PROFILE else 0
    tensors, scalars, consts, g = _read(fn, args, kwargs, grid, Tensor)
    if not tensors:
        return
    t0 = _tick("read_launch", t0)
    # each tensor read once: its storage key, address, shape, strides and dtype serve the meaning, the key and the
    # lifetime checks
    facts, keys, rows = {}, {}, []
    for k, t in tensors.items():
        ptr, shape, stride, dtype = t.data_ptr(), tuple(t.shape), t.stride(), t.dtype
        es = _ESIZE.get(dtype)
        if es is None:
            es = _ESIZE[dtype] = t.element_size()
        skey = (t.get_device(), ptr - t.storage_offset() * es)
        keys[k] = skey
        f, fs = _fact_sig_at(t, skey, ptr, shape, stride, dtype)
        if f is None and _SCOPES:
            f = _scope_fact(shape, stride, dtype)
            if f is not None:
                fs = _sig_of(f)
        facts[k] = f
        rows.append((k, dtype, shape, stride, fs))
    t0 = _tick("facts", t0)
    sig = tuple(rows)
    kkey = _kernel_key(fn, consts)
    key = (kkey, sig, tuple(scalars.items()), g)
    hit = _VERDICTS.get(key)
    t0 = _tick("key", t0)
    if hit is None:
        import hashlib

        name = str(kkey[0])
        out_name, inferred = None, None
        cached = False
        short = None
        try:
            ttir = fn.warmup(*args, grid=grid, **kwargs).asm["ttir"]
            short = hashlib.sha256(ttir.encode()).hexdigest()[:8]
            dump = os.environ.get("ENTAIL_TYPES_DUMP_TTIR")
            if dump:                     # a research aid: the IR the verdict was decided on, to replay it offline
                try:
                    os.makedirs(dump, exist_ok=True)
                    with open(os.path.join(dump, f"{name.rsplit('.', 1)[-1]}-{short}.ttir"), "w",
                              encoding="utf-8") as f:
                        f.write(ttir)
                except OSError:
                    pass
            ckey = hashlib.sha256(json.dumps([_checker_version(), ttir, [list(x) for x in sig],
                                              sorted(scalars.items()), list(g)], default=str).encode()).hexdigest()
            old = _cache().get(ckey)
            if old is not None:
                v = kernel_types.Verdict(**{k: old["verdict"].get(k) for k in ("verdict", "why", "checks", "programs",
                                                                               "seconds", "example", "inferred")})
                out_name, inferred, cached = old.get("output"), old.get("inferred"), True
            else:
                written = kernel_ir.written_args(ttir)
                meanings = {}
                pointers = {}
                # the serials in a cached configuration are those of the first launch; the rule only compares them
                for k, t in tensors.items():
                    f = facts[k]
                    m = _meaning_from(t.shape, t.stride(), f)
                    if written is not None and k in written:
                        m.kind = "output"
                    meanings[k] = m
                    if f is not None and f.get("pointers") is not None:
                        m.kind = "pointers"
                        names = []
                        for i, tgt in enumerate(f["pointers"]):
                            nm = f"@{k}[{i}]"
                            tm = _meaning_from(tgt["shape"], tgt["stride"], tgt["fact"])
                            if written is not None and written == {k}:
                                pass
                            meanings[nm] = tm
                            names.append(nm)
                        pointers[k] = names
                if written is None:
                    # a store through a pointer the kernel loaded from a table: the table's targets are its outputs
                    if pointers:
                        for k in pointers:
                            for nm in pointers[k]:
                                meanings[nm].kind = "output"
                        written = set()
                        for k in pointers:
                            written |= set(pointers[k])
                    else:
                        v = kernel_types.Verdict("unproven", "the IR does not say which arguments the kernel writes")
                if written is not None and not written:
                    v = kernel_types.Verdict("unproven", "the kernel writes no argument")
                elif written is not None:
                    out_name = sorted(written)
                    v = kernel_types.check_launch(ttir, meanings, scalars, g, pointers=pointers or None)
                    inferred = v.inferred
                _cache_put(ckey, {"kernel": name, "output": out_name, "inferred": inferred, "verdict": v.to_json()})
        except Exception as e:  # noqa: BLE001 - the checker failed: nothing is claimed
            v = kernel_types.Verdict("unproven", f"the check raised {type(e).__name__}: {e}")
        hit = _VERDICTS[key] = (v, out_name, inferred)
        _count(f"verdict_{v.verdict}")
        _tallied(BOUNDARY_TYPES, v.verdict)
        _count("verdict_from_cache" if cached else "verdict_decided")
        _write({"kind": "types_launch", "kernel": name, "grid": list(g), "scalars": scalars,
                "tensors": {k: {"dtype": str(t.dtype), "shape": list(t.shape), "stride": list(t.stride()),
                                "fact": facts[k]} for k, t in tensors.items()},
                "output": out_name, "verdict": v.to_json(), "cached": cached, "ttir": short,
                "captured": bool(torch.cuda.is_current_stream_capturing())})
    v, out_name, inferred = hit
    _count(f"launch_{v.verdict}")
    t0 = _tick("decide", t0)
    written = set(out_name) if out_name else None
    life = _life_check(str(kkey[0]), tensors, facts, written)
    life += _note_writes(tensors, written, keys)
    for why in life:
        _count("life_violation")
        _write({"kind": "types_life", "kernel": str(kkey[0]), "verdict": "violation", "why": why})
        _broken(str(kkey[0]), kernel_types.Verdict("violation", why), boundary=BOUNDARY_LIFE)
    if v.verdict == "violation":
        _broken(str(kkey[0]), v)
    t0 = _tick("life", t0)
    if v.verdict == "proven" and inferred:
        for name, inf in inferred.items():
            t = tensors.get(name)
            if t is None or not isinstance(inf, dict) or "coverage" in inf:
                continue                     # a partly covered output keeps no meaning from this launch
            if (facts.get(name) or {}).get("view") is not None:
                continue                     # a layer said what it means (its axes are a view's, not the tensor's)
            names = [inf.get(f"axis_{i}") for i in range(t.dim())]
            names = [n if isinstance(n, str) else None for n in names]
            if not any(names):
                continue
            pending = inf.get("pending_scales") or []
            row = next((r for r in rows if r[0] == name), None)
            if row is None:
                attach(t, names, "value", _next_serial() if pending else 0, pending[0] if len(pending) == 1 else 0,
                       sums=inf.get("sums") or {}, pending=pending, register=False)
            else:
                _attach_at(t, keys[name], t.data_ptr(), row[2], row[3], row[1], names, "value",
                           _next_serial() if pending else 0, pending[0] if len(pending) == 1 else 0,
                           sums=inf.get("sums") or {}, pending=pending, register=False)
            _count("inferred_attached")
    _tick("attach_inferred", t0)


class Broken(RuntimeError):
    """A kernel launch whose values do not pair on their meanings, stopped before it runs (ENTAIL_ON_BROKEN=stop)."""


_REPORTED = set()
# the boundaries the platform shows these under (data/nodes.json: "^kernel:" is the Kernels node)
BOUNDARY_TYPES = "kernel:types"      # the rule at a kernel launch or a compiled graph
BOUNDARY_LIFE = "kernel:life"        # memory nothing wrote, read (M19 L7)


def _broken(kernel, v, stop=True, where=None, boundary=None):
    """A violation is broken: reported once per launch configuration, the run goes on - unless the policy stops
    at what is broken (ENTAIL_ON_BROKEN=stop), when the launch is refused before the kernel runs. `where` names a
    reader that is not a kernel (a module), as it is. The first report is also a decision in the record, at
    `boundary` (default: kernel:life for a module reader, kernel:types for a kernel)."""
    key = (kernel, v.why)
    at = where or kernel.rsplit('.', 1)[-1]
    stops = stop and os.environ.get("ENTAIL_ON_BROKEN", "report") == "stop"
    if key not in _REPORTED:
        _REPORTED.add(key)
        _count("broken_reported")
        if os.environ.get("ENTAIL_QUIET") not in ("all",):
            sys.stderr.write(f"entail: broken at {at}: {v.why}\n")
        _decision(boundary or (BOUNDARY_LIFE if where is not None else BOUNDARY_TYPES), at, v.why, stops)
    if stops:
        _count("broken_stopped")
        raise Broken(f"entail stopped {'a kernel launch' if where is None else 'a read'}: {at}: {v.why}")


def _decision(boundary, consumer, why, refused=False):
    """A broken read or launch as a decision in the record (record.write_json), counted for its boundary (tally):
    what the platform's Kernels node shows. Never the engine's problem."""
    try:
        from . import record, tally

        record.write_json({"pid": os.getpid(), "boundary": boundary, "consumer": consumer,
                           "name": "Lifetime" if boundary == BOUNDARY_LIFE else "KernelMeaning",
                           "verdict": "refused" if refused else "broken", "blocking": bool(refused),
                           "rule": "lifetime" if boundary == BOUNDARY_LIFE else "kernel_types", "resolution": None,
                           "handle": None, "target": None, "note": why, "lost_by": None, "declared": None,
                           "chosen": None, "observed": None, "conflict": []})
        if refused:
            tally.refused(boundary)
        else:
            tally.broken(boundary)
    except Exception:  # noqa: BLE001
        _count("decision_record_failed")


def _tallied(boundary, verdict):
    """One decided launch configuration or compiled graph, counted for the platform: proven/checked as passed,
    unproven/possible as not decided (skipped); a violation is counted by _decision."""
    try:
        from . import tally

        c = tally.counts(boundary)
        c["checks"] += 1
        if verdict in ("proven", "checked"):
            tally.passed(boundary, ["kernel_types"])
        elif verdict != "violation":
            c["skipped"] += 1
        tally.tick(boundary)
    except Exception:  # noqa: BLE001
        _count("tally_failed")
    _ended()


_ENDED = [False]


def _ended():
    """Once per process: at exit, one line on what was checked and what was not read (the record, and stderr)."""
    if _ENDED[0]:
        return
    _ENDED[0] = True
    import atexit
    atexit.register(_end_line)


def end_text():
    s = _STATS
    n = sum(s.get(f"verdict_{k}", 0) for k in ("proven", "violation", "unproven", "possible"))
    g = sum(s.get(f"graph_{k}", 0) for k in ("checked", "violation", "unproven"))
    return (f"checked {n} kernel launch configurations ({s.get('verdict_proven', 0)} proven, "
            f"{s.get('verdict_violation', 0)} broken, "
            f"{s.get('verdict_unproven', 0) + s.get('verdict_possible', 0)} not decided) and {g} compiled graphs "
            f"({s.get('graph_violation', 0)} broken); not read: the inside of C++ kernels (only what their "
            f"arguments mean) and the engine's own Python computation")


def _end_line():
    try:
        from . import record

        text = end_text()
        record.write_json({"pid": os.getpid(), "said": BOUNDARY_TYPES, "text": text})
        if os.environ.get("ENTAIL_QUIET") not in ("all",) and \
                "types" not in os.environ.get("ENTAIL_QUIET", "").replace(" ", "").split(","):
            sys.stderr.write(f"entail: types: {text}\n")
    except Exception:  # noqa: BLE001 - exiting: nothing to report to
        pass


def install_triton():
    mod = sys.modules.get("triton.runtime.jit")
    J = getattr(mod, "JITFunction", None) if mod is not None else None
    if J is None or getattr(J.run, "__entail_types__", False):
        return 0
    orig = J.run

    @functools.wraps(orig)
    def run(self, *args, grid, warmup, **kwargs):
        if not warmup and not _compiling():
            t0 = time.perf_counter_ns() if _PROFILE else 0
            try:
                _launch(self, args, kwargs, grid)
            except Broken:
                raise                                  # the policy chose to stop: the kernel does not run
            except Exception:  # noqa: BLE001 - never the engine's problem
                _count("launch_hook_failed")
            _tick("hook_total", t0)
            _count("hook_calls")
        return orig(self, *args, grid=grid, warmup=warmup, **kwargs)

    run.__entail_types__ = True
    J.run = run
    _WRAPPED[(J, "run")] = orig
    return 1


def install():
    return install_fp8_utils() + install_weights() + install_marlin_weights() + install_triton() + \
        install_compile() + install_compile_cache()


def uninstall():
    for (holder, name), orig in list(_WRAPPED.items()):
        setattr(holder, name, orig)
    _WRAPPED.clear()


def stats():
    return dict(_STATS)


def summary():
    """Every launch configuration decided, with its verdict (plain data for a report)."""
    out = []
    for (kkey, sig, scalars, g), (v, out_name, inferred) in _VERDICTS.items():
        out.append({"kernel": str(kkey[0]).rsplit(".", 1)[-1], "grid": list(g), "output": out_name,
                    "verdict": v.verdict, "why": v.why[:300], "checks": v.checks, "seconds": round(v.seconds, 3),
                    "named_inputs": [k for k, _d, _s, _st, f in sig if f is not None],
                    "tensors": len(sig), "inferred": inferred})
    return {"stats": dict(_STATS), "launches": out, "graphs": list(_GRAPHS)}
