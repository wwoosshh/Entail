"""Adapter v2 for engine functions that carry no definition of their own (M19 L3; entail/definitions.py; the rule is
kernel_reference_contract's, the same one the custom ops are held to).

  hooks        the module that defines each function in definitions.DEFINITIONS; the autoinstall shim runs install()
               as soon as that module has finished loading, so every later `from module import name` gets the wrapped
               function. A caller that bound the name before that is not reached.
  read_choice  the function's and its definition's outputs on a slice of the first real call's input: the arguments
               the definition names as holding the token dimension are cut to ROWS rows and copied with their strides
               kept (a kernel that misreads a layout misreads the slice as well); every other argument is passed as
               it is (weights, scales, options: read, not written, by the registered functions). The definition runs
               in its noise dtype (the inputs' own, or the one it names) and in float32.
  handles      the function's name in its module: a mismatch the core resolves sends the function to its definition
               (computed in float32, returned in the function's own output dtype), from the very call that was
               compared and for every later call in the process. Not offered when the function was called inside a
               CUDA graph capture in this process (the graph replays the kernel whatever the name says); it then stays
               broken with that reason. A repaired function called inside a later capture gets its definition
               captured when the definition allows it (Definition.capturable), else the kernel, said once as broken.
               ENTAIL_POLICY=refuse repairs nothing.
Decided once per function and process, on its first real call: vLLM's dummy runs (marked by the kernel reference
adapter), captures, torch.compile tracing and calls whose rows are all the same decide nothing, and a later call
with another input pattern or launch configuration is not compared. A call the definition does not cover
(NotImplementedError) is unknown, said once. In an engine that replays captured graphs for every real call the
function may never be called on real input: nothing is decided then, and the record shows it instrumented only.
"""
import functools
import inspect
import sys

import torch

from .. import core, definitions, kernel_reference_contract
from .base import Hook

engine = "vllm, sglang"
versions = "vLLM 0.30.0, SGLang 0.5.20"
BOUNDARY = "kernel:definition"
TRIES = 64             # calls on which the definition's output decides nothing (all zeros, or the input itself)
#                        before the function is given up; calls whose rows are all one row (an engine's dummy batch:
#                        SGLang marks none of its capture warm-ups) are not counted
_WRAPPED = {}          # target -> (module, name, original, state)
_STATS = {}


def hooks():
    return [Hook(d.target.replace(":", "."), "kernel") for d in definitions.DEFINITIONS]


def handles():
    return {"send_to_definition": "the wrapper calls the definition in place of the function"}


def _count(k, by=1):
    _STATS[k] = _STATS.get(k, 0) + by


def _capturing():
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


def _dummy():
    """Inside vLLM's own profile run or capture warm-up (the kernel reference adapter marks them)."""
    vk = sys.modules.get("entail.adapters.vllm_kernel_reference")
    return bool(vk is not None and getattr(vk, "_STATE", {}).get("dummy"))


def _prepare(sig, d, args, kwargs):
    """(the bound arguments with the token-dimension ones cut to ROWS rows, n, rows); None when those arguments are
    missing or disagree on the token count; "empty" for a call with no token."""
    ba = sig.bind(*args, **kwargs)
    ba.apply_defaults()
    held = [k for k in d.rows if isinstance(ba.arguments.get(k), torch.Tensor) and ba.arguments[k].dim() >= 1]
    if not held:
        return None
    n = int(ba.arguments[held[0]].shape[0])
    if any(int(ba.arguments[k].shape[0]) != n for k in held):
        return None
    if n == 0:
        return "empty"
    rows = min(kernel_reference_contract.ROWS, n)
    cut = {k: (kernel_reference_contract.kept(v[:rows]) if k in held else v) for k, v in ba.arguments.items()}
    return cut, n, rows


def _fresh(bound, d):
    """Each run gets its own copies of the cut arguments (a kernel may work in place)."""
    return {k: (kernel_reference_contract.kept(v) if k in d.rows and isinstance(v, torch.Tensor) else v)
            for k, v in bound.items()}


def read_choice(d, orig, bound):
    """(the function's output, the definition's in its noise dtype, the definition's in float32) on the slice."""
    noise = getattr(torch, d.noise) if d.noise else None
    with torch.no_grad():
        k_out = orig(**_fresh(bound, d))
        n_out = d.fn(**_fresh(bound, d), _dtype=noise)
        r_out = d.fn(**_fresh(bound, d), _dtype=torch.float32)
    return k_out, n_out, r_out


def _decide(d, name, orig, cut, st):
    """True once the function is decided (compared, or given up); False to try again on the next real call."""
    from .. import load

    krc = kernel_reference_contract
    consumer = f"{d.engine}.{name}"

    def unknown(why):
        load.enforce([load.cannot_check(BOUNDARY, consumer, "KernelReference", f"{name}: {why}")])
        return True

    if cut is None:
        return unknown("the arguments that hold the token dimension are missing or disagree on it; not compared")
    if cut == "failed":
        return True
    bound, n, rows = cut
    inputs = [bound[k] for k in d.rows if isinstance(bound.get(k), torch.Tensor) and bound[k].is_floating_point()]
    if krc.uniform_rows(inputs):
        _count("uniform_calls")
        return False      # an engine's dummy batch (one row over and over): decides nothing and is not counted
    st["tries"] += 1
    try:
        k_out, n_out, r_out = read_choice(d, orig, bound)
    except NotImplementedError as e:
        return unknown(f"its definition does not cover this call ({e}); not compared")
    cmp = krc.compare(k_out, r_out, n_out, inputs)
    why = krc.vacuous(cmp)
    if why is not None:
        return unknown(f"{why}, on each of its first {st['tries']} real calls; not compared") \
            if st["tries"] >= TRIES else False
    where = f"{name} ({d.target}) on {rows} of {n} rows of its first real call, against entail's definition ({d.source})"
    repair, extra = None, ""
    if not st["captured"]:
        repair = f"{name} sent to entail's definition from this call on, for every later call in the process"
    elif cmp.violations or cmp.nonfinite:
        extra = "; not repaired: the function was called inside a CUDA graph capture in this process, and the graph " \
                "replays the kernel whatever the name says"
    if krc.resolved(krc.check(BOUNDARY, consumer, name, cmp, where, extra=extra, repair=repair)):
        st["repaired"] = True
        st["cmp"], st["where"] = cmp, where
        _count("sent_to_definition")
    return True


def _captured_after_repair(d, name, st):
    """A repaired function called inside a CUDA graph capture whose definition cannot be captured: the graph gets the
    kernel, and the repair no longer holds for what the graph replays; said once, as the mismatch it is."""
    from .. import load

    if st.get("said_capture"):
        return
    st["said_capture"] = True
    extra = ("; not repaired inside a CUDA graph capture: the definition synchronises with the host and cannot be "
             "captured, so the graph holds the kernel")
    load.safely(BOUNDARY, f"{d.engine}.{name}", "KernelReference",
                lambda: kernel_reference_contract.check(BOUNDARY, f"{d.engine}.{name}", name, st["cmp"], st["where"],
                                                        extra=extra))


def wrap(module, name, d):
    """Put the checking wrapper in place of module.name. False when it is there already."""
    orig = getattr(module, name)
    if getattr(orig, "__entail_definition__", None) is not None:
        return False
    sig = inspect.signature(orig)
    st = {"done": False, "repaired": False, "captured": False, "tries": 0}
    consumer = f"{d.engine}.{name}"

    def repaired(args, kwargs):
        _count("definition_calls")
        return d.fn(*args, **kwargs, _dtype=torch.float32)

    @functools.wraps(orig)
    def run(*args, **kwargs):
        if st["repaired"]:
            if d.capturable or not _capturing():
                return repaired(args, kwargs)
            _captured_after_repair(d, name, st)
            return orig(*args, **kwargs)
        if st["done"] or torch.compiler.is_compiling():
            return orig(*args, **kwargs)
        if _capturing():
            st["captured"] = True
            return orig(*args, **kwargs)
        if _dummy():
            return orig(*args, **kwargs)
        from .. import load

        cut = load.safely(BOUNDARY, consumer, "KernelReference", lambda: _prepare(sig, d, args, kwargs),
                          default="failed")
        if cut == "empty":
            return orig(*args, **kwargs)
        try:
            decided = load.safely(BOUNDARY, consumer, "KernelReference", lambda: _decide(d, name, orig, cut, st),
                                  default=True)
        except core.RoleError:                   # the policy stops here: once, and the wrapper steps aside
            st["done"] = True
            raise
        if decided:
            st["done"] = True
            _count("decided")
        if st["repaired"]:
            return repaired(args, kwargs)
        return orig(*args, **kwargs)

    run.__entail_definition__ = d
    setattr(module, name, run)
    _WRAPPED[d.target] = (module, name, orig, st)
    _count("instrumented")
    return True


def install():
    """Wrap every registered function whose module has loaded. Returns how many were wrapped now."""
    n = 0
    for d in definitions.DEFINITIONS:
        modname, fname = d.target.split(":")
        mod = sys.modules.get(modname)
        if mod is not None and d.target not in _WRAPPED and hasattr(mod, fname) and wrap(mod, fname, d):
            n += 1
    return n


def uninstall():
    for module, name, orig, _st in list(_WRAPPED.values()):
        setattr(module, name, orig)
    _WRAPPED.clear()


def stats():
    out = dict(_STATS)
    out.update(kernel_reference_contract.stats(BOUNDARY))
    return out


def reset():
    uninstall()
    _STATS.clear()
    kernel_reference_contract.reset(BOUNDARY)
