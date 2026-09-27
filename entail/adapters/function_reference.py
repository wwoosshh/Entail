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
               compared and for every later call in the process. Not offered when a CUDA graph captured in this
               process holds the kernel at a size class where it was not held to the definition and matched (the
               graph replays the kernel whatever the name says); it then stays broken with that reason. A repaired
               function called inside a later capture gets its definition captured when the definition allows it
               (Definition.capturable), else the kernel, said once as broken. ENTAIL_POLICY=refuse repairs nothing.
  warm-ups     (M19 L3.3a) a call inside the engine's warm-ups - vLLM's dummy runs (marked by the kernel reference
               adapter), and, before the function's first real decision, any call whose rows are all one row (SGLang
               marks none of its capture warm-ups) - is decided on a probe made from it: the call's shapes, dtypes
               and strides, the token arguments' values made up (seeded normal values, or Definition.probe for the
               arguments whose range only the function knows: expert ids, positive scales), once per size class, on
               the engine's own row count up to WARM_ROWS. So the function is decided before any graph captures it,
               and at each size the engine warms up (a launch configuration that changes with the batch size is seen).
Decided once more per function and process on its first real call (ROWS rows of it). Captures and torch.compile
tracing decide nothing. A call the definition does not cover (NotImplementedError) is unknown, said once. Once the
real call is decided, nothing on the hot path looks at the rows again (a warm-up is then only a marked dummy run).
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


def _held(sig, d, args, kwargs):
    """(the bound arguments, the token-dimension ones present, their row count); the count is None when they are
    missing or disagree."""
    ba = sig.bind(*args, **kwargs)
    ba.apply_defaults()
    held = [k for k in d.rows if isinstance(ba.arguments.get(k), torch.Tensor) and ba.arguments[k].dim() >= 1]
    if not held:
        return ba.arguments, held, None
    n = int(ba.arguments[held[0]].shape[0])
    if any(int(ba.arguments[k].shape[0]) != n for k in held):
        return ba.arguments, held, None
    return ba.arguments, held, n


def _whole(bound):
    """Copies of every tensor argument of the call (strides kept), for a definition compared on the whole call; None
    when they hold more than BUDGET."""
    krc = kernel_reference_contract
    if krc.nbytes([v for v in bound.values() if isinstance(v, torch.Tensor)]) > krc.BUDGET:
        return None
    return {k: (krc.kept(v) if isinstance(v, torch.Tensor) else v) for k, v in bound.items()}


def _prepare(sig, d, args, kwargs):
    """(the bound arguments with the token-dimension ones cut to ROWS rows, n, rows); None when those arguments are
    missing or disagree on the token count; "empty" for a call with no token. A definition compared on the whole call
    (Definition.whole) gets copies of every tensor, uncut."""
    bound, held, n = _held(sig, d, args, kwargs)
    if n is None:
        return None
    if n == 0:
        return "empty"
    if d.whole:
        cut = _whole(bound)
        return (cut, n, n) if cut is not None else None
    rows = min(kernel_reference_contract.ROWS, n)
    cut = {k: (kernel_reference_contract.kept(v[:rows]) if k in held else v) for k, v in bound.items()}
    return cut, n, rows


def _prepare_warm(sig, d, args, kwargs):
    """A probe made from a warm-up call: the token-dimension arguments cut to the engine's own row count up to
    WARM_ROWS, their values made up (Definition.probe first, then seeded normal values for the floating ones);
    (bound, n, rows), or None/"empty" as _prepare. A definition compared on the whole call keeps the warm-up's own
    values (its index structures are valid as the engine made them)."""
    krc = kernel_reference_contract
    bound, held, n = _held(sig, d, args, kwargs)
    if n is None:
        return None
    if n == 0:
        return "empty"
    if d.whole:
        cut = _whole(bound)
        return (cut, n, n) if cut is not None else None
    rows = min(krc.WARM_ROWS, n)
    gen = krc.generator(bound[held[0]].device)
    cut = {k: (krc.kept(v[:rows]) if k in held else v) for k, v in bound.items()}
    given = d.probe(cut, gen) if d.probe is not None else {}
    for k in held:
        if k in given:
            cut[k] = given[k]
        elif cut[k].is_floating_point():
            cut[k] = krc.made_up(cut[k], gen)
    return cut, n, rows


def _uniform_call(sig, d, args, kwargs) -> bool:
    """Whether the call's token arguments hold one row over and over: an engine's dummy batch."""
    bound, held, n = _held(sig, d, args, kwargs)
    if not n or n < 2:
        return False
    return kernel_reference_contract.uniform_rows([bound[k] for k in held if bound[k].is_floating_point()])


def _warm_call(sig, d, args, kwargs) -> bool:
    """A call to decide on a probe made from it: one row over and over (an engine's warm-up that is not marked).
    Never raises (principle 12)."""
    try:
        return _uniform_call(sig, d, args, kwargs)
    except Exception:  # noqa: BLE001
        return False


def _fresh(bound, d):
    """Each run gets its own copies of the cut arguments (a kernel may work in place) - of every tensor for a
    definition compared on the whole call - and its own output buffers where the function would write the engine's
    (Definition.fresh)."""
    copy = (lambda k, v: isinstance(v, torch.Tensor)) if d.whole else \
        (lambda k, v: k in d.rows and isinstance(v, torch.Tensor))
    b = {k: (kernel_reference_contract.kept(v) if copy(k, v) else v) for k, v in bound.items()}
    if d.fresh is not None:
        b.update(d.fresh(b))
    return b


def _outputs(out, b, d):
    """What a run produced: its return value and the arguments the definition declares it writes."""
    got = [] if out is None else (list(out) if isinstance(out, (tuple, list)) else [out])
    return got + [b[w] for w in d.writes]


def read_choice(d, orig, bound):
    """(the function's outputs, the definition's in its noise dtype, the definition's in float32) on the slice."""
    noise = getattr(torch, d.noise) if d.noise else None

    def run(fn, **extra):
        b = _fresh(bound, d)
        return _outputs(fn(**b, **extra), b, d)

    with torch.no_grad():
        k_out = run(orig)
        n_out = run(d.fn, _dtype=noise)
        r_out = run(d.fn, _dtype=torch.float32)
    return k_out, n_out, r_out


def _decide(d, name, orig, cut, st, warm: bool = False):
    """True once the function is decided (compared, or given up); False to try again on the next real call.
    `warm`: the cut is a probe made from a warm-up call (_prepare_warm), whose values are made up."""
    from .. import load

    krc = kernel_reference_contract
    consumer = f"{d.engine}.{name}"

    def unknown(why):
        if warm and st.get("said_warm_unknown"):
            return True
        st["said_warm_unknown"] = warm or st.get("said_warm_unknown", False)
        load.enforce([load.cannot_check(BOUNDARY, consumer, "KernelReference", f"{name}: {why}")])
        return True

    if cut is None:
        return True if warm else unknown("the arguments that hold the token dimension are missing or disagree on "
                                         "it; not compared")
    if cut == "failed":
        return True
    bound, n, rows = cut
    inputs = [bound[k] for k in d.rows if isinstance(bound.get(k), torch.Tensor) and bound[k].is_floating_point()]
    if not warm and krc.uniform_rows(inputs):
        _count("uniform_calls")
        return False      # an engine's dummy batch (one row over and over): decides nothing and is not counted
    if not warm:
        st["tries"] += 1
    try:
        k_out, n_out, r_out = read_choice(d, orig, bound)
    except NotImplementedError as e:
        return unknown(f"its definition does not cover this call ({e}); not compared")
    cmp = krc.compare_exact(k_out, r_out) if d.exact else krc.compare(k_out, r_out, n_out, inputs)
    why = krc.vacuous(cmp)
    if why is not None:
        if warm:
            return True
        return unknown(f"{why}, on each of its first {st['tries']} real calls; not compared") \
            if st["tries"] >= TRIES else False
    if warm:
        where = (f"{name} ({d.target}) on a probe of {rows} rows made from the engine's warm-up call of {n} rows (its "
                 f"shapes, dtypes and strides; the token values made up, seed {krc.SEED}), before any graph of that "
                 f"size is captured, against entail's definition ({d.source})")
    else:
        where = (f"{name} ({d.target}) on {rows} of {n} rows of its first real call, against entail's definition "
                 f"({d.source})")
    repair, extra = None, ""
    if st["captured"] <= st["passed"]:
        repair = f"{name} sent to entail's definition from this call on, for every later call in the process"
        if st["captured"]:
            repair += " (graphs captured before hold the kernel at the sizes where it matched)"
    elif cmp.violations or cmp.nonfinite:
        extra = "; not repaired: a CUDA graph captured in this process holds the kernel at a size where it was not " \
                "held to its definition, and replays it whatever the name says"
    decisions = krc.check(BOUNDARY, consumer, name, cmp, where, extra=extra, repair=repair)
    if krc.resolved(decisions):
        st["repaired"] = True
        st["cmp"], st["where"] = cmp, where
        _count("sent_to_definition")
    elif decisions and decisions[0].verdict.name == "PASS":
        st["passed"].add(krc.bucket(n))
    return True


def _warm_up(d, name, orig, sig, args, kwargs, st) -> None:
    """Decide on a probe made from this warm-up call, once per size class and at most WARM_CLASSES classes."""
    from .. import load

    krc = kernel_reference_contract
    try:
        _b, _h, n = _held(sig, d, args, kwargs)
    except Exception:  # noqa: BLE001 - never the engine's problem (principle 12)
        return
    if not n or krc.bucket(n) in st["warm"] or len(st["warm"]) >= krc.WARM_CLASSES:
        return
    st["warm"].add(krc.bucket(n))
    consumer = f"{d.engine}.{name}"
    cut = load.safely(BOUNDARY, consumer, "KernelReference", lambda: _prepare_warm(sig, d, args, kwargs),
                      default="failed")
    if cut == "empty":
        return
    load.safely(BOUNDARY, consumer, "KernelReference", lambda: _decide(d, name, orig, cut, st, warm=True),
                default=True)
    _count("warm_decided")


def _note_capture(sig, d, args, kwargs, st) -> None:
    try:
        _b, _h, n = _held(sig, d, args, kwargs)
    except Exception:  # noqa: BLE001
        n = None
    st["captured"].add(kernel_reference_contract.bucket(n or 1))


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
    """Put the checking wrapper in place of module.name (a module's function, or a class's method: the wrapper then
    receives the instance as its first argument). False when it is there already."""
    orig = module.__dict__[name] if isinstance(module, type) else getattr(module, name)
    if getattr(orig, "__entail_definition__", None) is not None:
        return False
    sig = inspect.signature(orig)
    st = {"done": False, "repaired": False, "captured": set(), "passed": set(), "warm": set(), "tries": 0}
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
        if torch.compiler.is_compiling():
            return orig(*args, **kwargs)
        if _capturing():
            _note_capture(sig, d, args, kwargs, st)
            return orig(*args, **kwargs)
        if _dummy() or (not st["done"] and _warm_call(sig, d, args, kwargs)):
            _warm_up(d, name, orig, sig, args, kwargs, st)
            if st["repaired"]:
                return repaired(args, kwargs)
            return orig(*args, **kwargs)
        if st["done"]:
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


def _holder(d):
    """(the object that holds the target's name, the name): the module for "module:function", the class for
    "module:Class.method" (M19 L3.3d: a method is looked up on its class at every call, so wrapping it there reaches
    every instance); None when the module has not loaded or does not have it."""
    modname, attr = d.target.split(":")
    holder = sys.modules.get(modname)
    if holder is None:
        return None
    if "." in attr:
        cls, attr = attr.split(".", 1)
        holder = getattr(holder, cls, None)
        if holder is None:
            return None
    return (holder, attr) if hasattr(holder, attr) else None


def install():
    """Wrap every registered function whose module has loaded. Returns how many were wrapped now."""
    n = 0
    for d in definitions.DEFINITIONS:
        where = _holder(d)
        if where is not None and d.target not in _WRAPPED and wrap(where[0], where[1], d):
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
