"""kernel_check: the one rule (kernel_types) at every Triton launch the engine makes (ROADMAP M19 L6; ENTAIL=types).

Meanings are attached where values are made and travel with them by their memory:
  the activation quantizer     x_q [token, hidden] (a value), x_s [token, hidden / group] (its scale)
  the weight loader            w [feature, hidden] (a value), w_s [feature / block, hidden / block] (its scale)
  every proven launch          the output's axes, as the rule inferred them from what the kernel stored
At every Triton launch (JITFunction.run) the tensors handed to the kernel are looked up, the kernel's IR is read once
per launch configuration (kernel, constexprs, shapes, strides, integer arguments, grid, and which arguments carry which
meanings), and the verdict is recorded. The output of a proven launch gets the inferred meaning, so the next kernel
that reads it sees named axes. Nothing is refused and no result is changed: this is the checker as a reporter
(ROADMAP M19 L6 step 2); what to do with a verdict is the policy's business.
"""
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
_VERDICTS = {}         # launch configuration key -> (Verdict, output argument name, inferred meaning)
_STATS = {}
_WRAPPED = {}
_SERIAL = [0]
_LOCK = threading.RLock()
_PATH = None


def _count(k, by=1):
    _STATS[k] = _STATS.get(k, 0) + by


def _compiling():
    try:
        import torch

        return bool(torch.compiler.is_compiling())
    except Exception:  # noqa: BLE001
        return False


def _storage_key(t):
    off = t.storage_offset()
    return (str(t.device), t.data_ptr() - off * t.element_size())


# --- the facts: a meaning on a tensor's memory, gone when the tensor is -----------------------------------------------

def attach(t, names, kind="value", serial=0, pair=0, groups=None, sums=None, pending=(), basis=None, label=None,
           pointers=None, packed=None):
    """Give the tensor `t` a meaning: a name per axis (None for an axis that means nothing the rule can pair); for an
    integer tensor the basis of its numbers; a label a loaded stride can refer to; for a table of pointers the
    tensors it points to (each with its own meaning attached); for a tensor packed for one kernel the packed form
    and the sizes it was packed from (its axes then mean nothing elementwise)."""
    key = _storage_key(t)
    fact = {"names": list(names), "kind": kind, "serial": serial, "pair": pair,
            "groups": list(groups or [1] * len(names)), "sums": dict(sums or {}), "pending": list(pending),
            "basis": basis, "label": label, "packed": packed,
            "pointers": None if pointers is None else [_snapshot(x) for x in pointers]}
    entry = [t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, fact, None]
    with _LOCK:
        lst = _FACTS.setdefault(key, [])
        lst[:] = [e for e in lst if not (e[0] == entry[0] and e[1] == entry[1] and e[2] == entry[2]
                                        and e[3] == entry[3])]
        lst.append(entry)

    def gone(key=key, entry=entry):
        with _LOCK:
            lst = _FACTS.get(key)
            if lst is not None:
                lst[:] = [e for e in lst if e is not entry]
                if not lst:
                    _FACTS.pop(key, None)
    try:
        entry[5] = weakref.finalize(t, gone)
    except TypeError:
        pass
    _count("attached")
    return fact


def _snapshot(t):
    """A pointer table's target: its layout and its fact, as they are now."""
    f = fact_of(t)
    return {"shape": tuple(int(x) for x in t.shape), "stride": tuple(int(x) for x in t.stride()),
            "dtype": str(t.dtype), "fact": f}


def _meaning_from(shape, stride, f, kind=None):
    shape, stride = tuple(shape), tuple(stride)
    if f is None or len(f["names"]) != len(shape):
        return Meaning(tuple(Axis(None, n) for n in shape), shape, stride, kind or "value")
    origins = f.get("origins") or [0] * len(shape)
    axes = tuple(Axis(nm, n, g, o) for nm, n, g, o in zip(f["names"], shape, f["groups"], origins))
    return Meaning(axes, shape, stride, kind or f["kind"], f["serial"], f["pair"], basis=f.get("basis"),
                   label=f.get("label"))


def fact_of(t):
    lst = _FACTS.get(_storage_key(t))
    if not lst:
        return None
    ptr, shape, stride, dtype = t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype
    for e in lst:
        if e[0] == ptr and e[1] == shape and e[2] == stride and e[3] == dtype:
            return e[4]
    # a view of a tensor with a meaning (a slice, a selected row or column, a transpose, a merge or a split of its
    # axes): each axis of the view that walks one axis of the tensor means what that axis means, from the
    # coordinate the view starts at; an axis that walks two merged axes means what the vocabulary says the merge
    # means; anything else means nothing the rule can pair
    for e in reversed(lst):
        if e[3] != dtype or e[4].get("pointers") is not None:
            continue
        f = _view_fact(ptr, shape, stride, t.element_size(), e)
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
                 os.path.join("adapters", "vllm_declarations.py"), os.path.join("adapters", "vllm_index_meanings.py")]
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


def _launch(fn, args, kwargs, grid):
    import hashlib

    import torch

    from .adapters.vllm_block_fp8_guarantee import _kernel_key, read_launch

    t0 = time.perf_counter_ns() if _PROFILE else 0
    values, consts, g = read_launch(fn, args, kwargs, grid)
    tensors = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        return
    scalars = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in values.items()
               if isinstance(v, (int, float)) and not isinstance(v, torch.Tensor)}
    t0 = _tick("read_launch", t0)
    facts = {k: fact_of(t) for k, t in tensors.items()}
    t0 = _tick("facts", t0)
    sig = tuple((k, str(t.dtype), tuple(t.shape), tuple(t.stride()),
                 None if facts[k] is None else (tuple(facts[k]["names"]), facts[k]["kind"], tuple(facts[k]["groups"]),
                                                facts[k]["pair"] != 0))
                for k, t in tensors.items())
    kkey = _kernel_key(fn, consts)
    key = (kkey, sig, tuple(sorted(scalars.items())), g)
    hit = _VERDICTS.get(key)
    t0 = _tick("key", t0)
    if hit is None:
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
                    m, f = meaning_of(t)
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
        _count("verdict_from_cache" if cached else "verdict_decided")
        _write({"kind": "types_launch", "kernel": name, "grid": list(g), "scalars": scalars,
                "tensors": {k: {"dtype": str(t.dtype), "shape": list(t.shape), "stride": list(t.stride()),
                                "fact": facts[k]} for k, t in tensors.items()},
                "output": out_name, "verdict": v.to_json(), "cached": cached, "ttir": short,
                "captured": bool(torch.cuda.is_current_stream_capturing())})
    v, out_name, inferred = hit
    _count(f"launch_{v.verdict}")
    if v.verdict == "violation":
        _broken(str(kkey[0]), v)
    t0 = _tick("decide", t0)
    if v.verdict == "proven" and inferred:
        for name, inf in inferred.items():
            t = tensors.get(name)
            if t is None or not isinstance(inf, dict) or "coverage" in inf:
                continue                     # a partly covered output keeps no meaning from this launch
            names = [inf.get(f"axis_{i}") for i in range(t.dim())]
            names = [n if isinstance(n, str) else None for n in names]
            if not any(names):
                continue
            pending = inf.get("pending_scales") or []
            attach(t, names, "value", _next_serial() if pending else 0, pending[0] if len(pending) == 1 else 0,
                   sums=inf.get("sums") or {}, pending=pending)
            _count("inferred_attached")
    _tick("attach_inferred", t0)


class Broken(RuntimeError):
    """A kernel launch whose values do not pair on their meanings, stopped before it runs (ENTAIL_ON_BROKEN=stop)."""


_REPORTED = set()


def _broken(kernel, v):
    """A violation is broken: reported once per launch configuration, the run goes on - unless the policy stops
    at what is broken (ENTAIL_ON_BROKEN=stop), when the launch is refused before the kernel runs."""
    key = (kernel, v.why)
    if key not in _REPORTED:
        _REPORTED.add(key)
        _count("broken_reported")
        if os.environ.get("ENTAIL_QUIET") not in ("all",):
            sys.stderr.write(f"entail: broken at {kernel.rsplit('.', 1)[-1]}: {v.why}\n")
    if os.environ.get("ENTAIL_ON_BROKEN", "report") == "stop":
        _count("broken_stopped")
        raise Broken(f"entail stopped a kernel launch: {kernel.rsplit('.', 1)[-1]}: {v.why}")


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
