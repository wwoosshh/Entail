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
import weakref

from . import kernel_ir, kernel_types
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

def attach(t, names, kind="value", serial=0, pair=0, groups=None, sums=None, pending=()):
    """Give the tensor `t` a meaning: a name per axis (None for an axis that means nothing the rule can pair)."""
    key = _storage_key(t)
    fact = {"names": list(names), "kind": kind, "serial": serial, "pair": pair,
            "groups": list(groups or [1] * len(names)), "sums": dict(sums or {}), "pending": list(pending)}
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


def fact_of(t):
    lst = _FACTS.get(_storage_key(t))
    if not lst:
        return None
    ptr, shape, stride, dtype = t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype
    for e in lst:
        if e[0] == ptr and e[1] == shape and e[2] == stride and e[3] == dtype:
            return e[4]
    return None


def meaning_of(t):
    """The Meaning of a tensor handed to a kernel: its layout from the tensor, its axes from its fact (unnamed
    without one)."""
    f = fact_of(t)
    shape, stride = tuple(int(x) for x in t.shape), tuple(int(x) for x in t.stride())
    if f is None or len(f["names"]) != len(shape):
        return Meaning(tuple(Axis(None, n) for n in shape), shape, stride, "value"), None
    axes = tuple(Axis(nm, n, g) for nm, n, g in zip(f["names"], shape, f["groups"]))
    return Meaning(axes, shape, stride, f["kind"], f["serial"], f["pair"]), f


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


def _cache():
    """The verdicts earlier processes decided, read once: a launch configuration is decided once per machine, not
    once per process (the kernel's IR text, the meanings, the integer arguments and the grid make the key)."""
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


def _launch(fn, args, kwargs, grid):
    import hashlib

    import torch

    from .adapters.vllm_block_fp8_guarantee import _kernel_key, read_launch

    values, consts, g = read_launch(fn, args, kwargs, grid)
    tensors = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
    if not tensors:
        return
    scalars = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in values.items()
               if isinstance(v, (int, float)) and not isinstance(v, torch.Tensor)}
    facts = {k: fact_of(t) for k, t in tensors.items()}
    sig = tuple((k, str(t.dtype), tuple(t.shape), tuple(t.stride()),
                 None if facts[k] is None else (tuple(facts[k]["names"]), facts[k]["kind"], tuple(facts[k]["groups"]),
                                                facts[k]["pair"] != 0))
                for k, t in tensors.items())
    kkey = _kernel_key(fn, consts)
    key = (kkey, sig, tuple(sorted(scalars.items())), g)
    hit = _VERDICTS.get(key)
    if hit is None:
        name = str(kkey[0])
        out_name, inferred = None, None
        cached = False
        try:
            ttir = fn.warmup(*args, grid=grid, **kwargs).asm["ttir"]
            ckey = hashlib.sha256(json.dumps([ttir, [list(x) for x in sig], sorted(scalars.items()), list(g)],
                                             default=str).encode()).hexdigest()
            old = _cache().get(ckey)
            if old is not None:
                v = kernel_types.Verdict(**{k: old["verdict"].get(k) for k in ("verdict", "why", "checks", "programs",
                                                                               "seconds", "example", "inferred")})
                out_name, inferred, cached = old.get("output"), old.get("inferred"), True
            else:
                written = kernel_ir.written_args(ttir)
                meanings = {}
                # the serials in a cached configuration are those of the first launch; the rule only compares them
                for k, t in tensors.items():
                    m, _f = meaning_of(t)
                    if written is not None and k in written:
                        m.kind = "output"
                    meanings[k] = m
                if written is None:
                    v = kernel_types.Verdict("unproven", "the IR does not say which arguments the kernel writes")
                elif not written:
                    v = kernel_types.Verdict("unproven", "the kernel writes no argument")
                else:
                    out_name = sorted(written)
                    v = kernel_types.check_launch(ttir, meanings, scalars, g)
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
                "output": out_name, "verdict": v.to_json(), "cached": cached,
                "captured": bool(torch.cuda.is_current_stream_capturing())})
    v, out_name, inferred = hit
    _count(f"launch_{v.verdict}")
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


def install_triton():
    mod = sys.modules.get("triton.runtime.jit")
    J = getattr(mod, "JITFunction", None) if mod is not None else None
    if J is None or getattr(J.run, "__entail_types__", False):
        return 0
    orig = J.run

    @functools.wraps(orig)
    def run(self, *args, grid, warmup, **kwargs):
        if not warmup and not _compiling():
            try:
                _launch(self, args, kwargs, grid)
            except Exception:  # noqa: BLE001 - never the engine's problem
                _count("launch_hook_failed")
        return orig(self, *args, grid=grid, warmup=warmup, **kwargs)

    run.__entail_types__ = True
    J.run = run
    _WRAPPED[(J, "run")] = orig
    return 1


def install():
    return install_fp8_utils() + install_weights() + install_triton()


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
    return {"stats": dict(_STATS), "launches": out}
