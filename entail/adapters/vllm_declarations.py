"""Adapter v2: what vLLM 0.30 already declares about the values it makes, read where vLLM makes them and turned into
meanings (ROADMAP M19 L6 step 1, ENTAIL=types; entail/kernel_check.py holds the facts, entail/kernel_types.py the
rule). Nothing here is written for one model, one quantization method or one bug: vLLM declares, this reads.

  hooks        model_loader.utils.process_weights_after_loading: before it runs, every parameter's declaration (its
                   vLLM parameter class and output_dim / input_dim / packed_dim / packed_factor; a MoE parameter's
                   quant_method and is_transposed; the layer's kind); after it runs, the tensors the layers now
                   hold get their meanings - matched by name, and by shape to what was declared (the same shape,
                   or a transpose; a repacked tensor keeps only its pairing with its scale)
               the functions of the data file (the MoE entry, the router, the block alignment, the activation
                   quantizer): their tensor arguments and what they return
  read_choice  the declarations, from the data file data/vllm_declarations.json (the layer kinds' axis names, which
               scale belongs to which value, the functions' arguments and returns, relations, merges)
  handles      none: meanings are attached, nothing is changed or decided here
A tensor that already has a meaning (attached by a more specific producer) keeps it.
"""
import functools
import sys

from .. import kernel_check, kernel_types

engine = "vllm"
versions = "vLLM 0.30.0"
_WRAPPED = {}
_STATS = {}
_TABLE = []


def _table():
    if not _TABLE:
        import json
        import os

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                            "vllm_declarations.json")
        with open(path, encoding="utf-8") as f:
            _TABLE.append(json.load(f))
    return _TABLE[0]


def hooks():
    from .base import Hook

    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "load")] + \
        [Hook(k.replace(":", "."), "request") for k in _table()["functions"]]


def read_choice(kind, obj=None):
    """The declarations of the data file: kind is one of layers, scales, functions, relations, merges."""
    return _table().get(kind)


def handles():
    return {}


def _count(k):
    _STATS[k] = _STATS.get(k, 0) + 1


def stats():
    return dict(_STATS)


def _wrap(holder, name, make):
    orig = getattr(holder, name, None)
    if orig is None or getattr(orig, "__entail_types__", False):
        return 0
    run = make(orig)
    run.__entail_types__ = True
    setattr(holder, name, run)
    _WRAPPED[(holder, name)] = orig
    return 1


def declare():
    for a, op, b, result in read_choice("relations") or []:
        kernel_types.relate(a, op, b, result)
    for outer, inner, result in read_choice("merges") or []:
        kernel_types.merge(outer, inner, result)


# --- a value and its scale ------------------------------------------------------------------------------------------

def _has(t):
    try:
        return kernel_check.fact_of(t) is not None
    except Exception:  # noqa: BLE001
        return True


def _attach(t, names, basis=None, **kw):
    try:
        if t is not None and hasattr(t, "data_ptr") and t.is_cuda and t.dim() == len(names) and not _has(t):
            kernel_check.attach(t, names, kind="index" if basis else kw.pop("kind", "value"), basis=basis, **kw)
            _count("attached")
            return True
    except Exception:  # noqa: BLE001 - never the engine's problem
        _count("attach_failed")
    return False


def _scale_axes(value_shape, value_names, scale_shape):
    """A scale's axes by broadcast against its value's, aligned from the first axis: the same size is the same axis,
    a size that divides is that axis in groups, size 1 is no coordinate. None when they do not align."""
    if len(scale_shape) > len(value_shape):
        if all(int(n) == 1 for n in scale_shape[len(value_shape):]):
            scale_shape = scale_shape[:len(value_shape)]
        else:
            return None
    names, groups = [], []
    for i, n in enumerate(scale_shape):
        n, v = int(n), int(value_shape[i])
        if n == v:
            names.append(value_names[i])
            groups.append(1)
        elif n == 1:
            names.append(None)
            groups.append(1)
        elif n > 0 and v % n == 0:
            names.append(value_names[i])
            groups.append(v // n)
        elif n > 0 and (n - 1) * -(-v // n) < v:
            names.append(value_names[i])          # groups of ceil(v / n), the last one short
            groups.append(-(-v // n))
        else:
            return None
    return names + [None] * (len(scale_shape) - len(names)), groups + [1] * (len(scale_shape) - len(groups))


def _issue(value, value_names, scale):
    """A value and its scale, as one producer made them: their meanings and their pairing."""
    if value is None or scale is None or not hasattr(scale, "shape"):
        return False
    if not (hasattr(value, "data_ptr") and value.is_cuda and scale.is_cuda):
        return False
    if _has(value) or _has(scale):
        return False
    axes = _scale_axes(tuple(value.shape), list(value_names), tuple(scale.shape))
    if axes is None or value.dim() != len(value_names):
        _count("scale_not_aligned")
        return False
    names, groups = axes
    try:
        kernel_check.issue_pair(value, scale, list(value_names), names, groups)
        _count("issued")
        return True
    except Exception:  # noqa: BLE001
        _count("attach_failed")
        return False


# --- the weights, after loading -------------------------------------------------------------------------------------

def _kind(module):
    layers = read_choice("layers") or {}
    for cls in type(module).__mro__:
        if cls.__name__ in layers:
            return cls.__name__, layers[cls.__name__]
    return None, None


def _declaration(name, p):
    d = {"shape": tuple(int(n) for n in p.shape), "dtype": str(p.dtype), "cls": type(p).__name__}
    for attr in ("output_dim", "input_dim", "packed_dim", "packed_factor", "pack_factor", "quant_method",
                 "is_transposed"):
        try:
            v = getattr(p, attr, None)
        except Exception:  # noqa: BLE001 - a property that raises
            v = None
        if v is not None:
            d[attr] = v.value if hasattr(v, "value") else v
    return d


def _names(kind_row, d, shape):
    """The axis names a declaration gives a tensor of `shape` (None: it does not say)."""
    if kind_row is None:
        return None
    if "moe" in kind_row:
        layout = kind_row["moe_transposed"] if d.get("is_transposed") else kind_row["moe"]
        return list(layout) if len(shape) == 3 else None
    out, inn = d.get("output_dim"), d.get("input_dim")
    if out is None and inn is None:
        return None
    names = [None] * len(shape)
    if out is not None and 0 <= int(out) < len(shape):
        names[int(out)] = kind_row["out"]
    if inn is not None and 0 <= int(inn) < len(shape):
        names[int(inn)] = kind_row["in"]
    return names


def _matched(before, after_shape):
    """The names of a processed tensor from its declaration: the same shape keeps them, a transpose swaps them."""
    names, shape = before
    if tuple(after_shape) == tuple(shape):
        return list(names)
    if len(shape) == 2 and tuple(after_shape) == (shape[1], shape[0]) and shape[0] != shape[1]:
        return [names[1], names[0]]
    if len(shape) == 3 and tuple(after_shape) == (shape[0], shape[2], shape[1]) and shape[1] != shape[2]:
        return [names[0], names[2], names[1]]
    return None


def _snapshot(model):
    """Every parameter's declaration, by module and name, before the weights are processed."""
    snap = {}
    for mname, module in model.named_modules():
        kname, row = _kind(module)
        params = dict(module.named_parameters(recurse=False))
        if not params:
            continue
        decl = {}
        for pname, p in params.items():
            d = _declaration(pname, p)
            d["names"] = _names(row, d, d["shape"])
            decl[pname] = d
        snap[mname] = (kname, row, decl)
    return snap


def _apply(model, snap):
    scales_of = read_choice("scales") or {}
    modules = dict(model.named_modules())
    for mname, (kname, row, decl) in snap.items():
        module = modules.get(mname)
        if module is None:
            continue
        now = {}
        for pname in decl:
            t = getattr(module, pname, None)
            if t is not None and hasattr(t, "shape"):
                now[pname] = t
        done = set()
        for vname, snames in scales_of.items():
            if vname not in now or vname not in decl or decl[vname]["names"] is None:
                continue
            names = _matched((decl[vname]["names"], decl[vname]["shape"]), tuple(now[vname].shape))
            for sname in snames:
                if sname in now and names is not None:
                    if _issue(now[vname], names, now[sname]):
                        done.update((vname, sname))
                    break
        for pname, t in now.items():
            if pname in done or decl[pname]["names"] is None:
                continue
            names = _matched((decl[pname]["names"], decl[pname]["shape"]), tuple(t.shape))
            if names is not None and _attach(t, names):
                _count("weight_named")


def install_loader():
    declare()
    mod = sys.modules.get("vllm.model_executor.model_loader.utils")
    if mod is None or not hasattr(mod, "process_weights_after_loading"):
        return 0

    def make(orig):
        @functools.wraps(orig)
        def run(model, *a, **k):
            try:
                snap = _snapshot(model)
            except Exception:  # noqa: BLE001
                snap = None
                _count("snapshot_failed")
            out = orig(model, *a, **k)
            if snap is not None:
                try:
                    _apply(model, snap)
                except Exception:  # noqa: BLE001
                    _count("apply_failed")
            return out
        return run
    return _wrap(mod, "process_weights_after_loading", make)


# --- the functions: their arguments and what they return -----------------------------------------------------------

def _wrap_function(holder, name, spec):
    import inspect

    orig = getattr(holder, name, None)
    if orig is None or getattr(orig, "__entail_types__", False):
        return 0
    try:
        params = list(inspect.signature(orig).parameters)
    except (TypeError, ValueError):
        return 0
    moe = (read_choice("layers") or {}).get("FusedMoE", {}).get("moe")

    @functools.wraps(orig)
    def run(*a, **k):
        bound = {}
        try:
            bound = dict(zip(params, a))
            bound.update(k)
            for left, right in spec.get("pairs", []):
                value, scale = bound.get(left), bound.get(right)
                if value is not None and moe and len(getattr(value, "shape", ())) == 3:
                    _issue(value, moe, scale)
            for arg, meaning in spec.get("args", {}).items():
                t = bound.get(arg)
                if t is None or not hasattr(t, "dim"):
                    continue
                if meaning == "moe_weight":
                    if moe and t.dim() == 3:
                        _attach(t, list(moe))
                    continue
                names, basis = meaning
                _attach(t, list(names), basis)
        except Exception:  # noqa: BLE001
            _count("attach_failed")
        out = orig(*a, **k)
        returns = spec.get("returns")
        if returns:
            try:
                outs = out if isinstance(out, tuple) else (out,)
                for i, meaning in enumerate(returns):
                    if i >= len(outs) or outs[i] is None or not hasattr(outs[i], "dim"):
                        continue
                    t = outs[i]
                    if isinstance(meaning, dict) and "like" in meaning:
                        src = bound.get(meaning["like"])
                        f = kernel_check.fact_of(src) if src is not None and hasattr(src, "dim") else None
                        scale_i = next((j for j, mm in enumerate(returns) if isinstance(mm, dict) and
                                        mm.get("scale_of") == i), None)
                        scale = outs[scale_i] if scale_i is not None and scale_i < len(outs) else None
                        if f is not None and t is not src and t.dim() == len(f["names"]):
                            if scale is not None and hasattr(scale, "dim"):
                                _issue(t, f["names"], scale)
                            else:
                                _attach(t, list(f["names"]))
                    elif isinstance(meaning, list):
                        names, basis = meaning
                        _attach(t, list(names), basis)
            except Exception:  # noqa: BLE001
                _count("attach_failed")
        return out

    run.__entail_types__ = True
    setattr(holder, name, run)
    _WRAPPED[(holder, name)] = orig
    return 1


def install_functions():
    declare()
    n = 0
    for key, spec in (read_choice("functions") or {}).items():
        module, _, name = key.rpartition(":")
        holder = sys.modules.get(module)
        if holder is not None and "." in name:
            cname, _, name = name.partition(".")
            holder = getattr(holder, cname, None)
        if holder is not None:
            n += _wrap_function(holder, name, spec)
    return n


def install():
    declare()
    return install_loader() + install_functions()


def uninstall():
    for (holder, name), orig in list(_WRAPPED.items()):
        setattr(holder, name, orig)
    _WRAPPED.clear()
