"""Shared core of entail: modes, the side table of facts, boundary checks, load-time checks.

Modes (environment variable ENTAIL, or set_mode()):
  off    default. tag() stores nothing and boundaries only read one global per call.
  load   load-time checks (check_props, check_config_keys, check_tied) are active; boundaries attach what they
         declare to their results and written arguments, so the meaning travels, but check nothing.
  debug  everything is checked, including every @boundary call: the arguments' meaning and what was written.
Errors are always RoleError and name the boundary, the argument, the fact kind, what was expected and what came.

Policy (environment variable ENTAIL_POLICY, or set_policy()): what to do when a declaration and the real
thing disagree.
  resolve  default. Keep the meaning intact by fixing the situation - send the value to a consumer that honours
           the declaration, convert it to the form the consumer expects, or recompute it - and say so in one
           line. Stop only when no such fix exists. (RESEARCH_PLAN.md 0 and 9, decided 2026-09-23.)
  refuse   stop at the first disagreement, as before.
"""
import functools
import inspect
import os
import weakref

from .facts import Invalidated, KernelCaps, ModelProps

_MODE = os.environ.get("ENTAIL", "off")
_POLICY = os.environ.get("ENTAIL_POLICY", "resolve")
_FACTS = {}  # id(tensor) -> {fact class name: fact}; entries are removed when the tensor is collected


class RoleError(RuntimeError):
    pass


def set_mode(mode):
    global _MODE
    if mode not in ("off", "load", "debug"):
        raise ValueError(mode)
    _MODE = mode


def mode():
    return _MODE


def set_policy(policy):
    global _POLICY
    if policy not in ("resolve", "refuse"):
        raise ValueError(policy)
    _POLICY = policy


def policy():
    return _POLICY


def tag(t, *facts):
    """Attach facts to a value. Returns the value so it can be used inline."""
    if _MODE == "off":
        return t
    key = id(t)
    if key not in _FACTS:
        _FACTS[key] = {}
        weakref.finalize(t, _FACTS.pop, key, None)
    for f in facts:
        _FACTS[key][type(f).__name__] = f
    return t


def facts_of(t):
    return dict(_FACTS.get(id(t), {}))


def _is_fact(x):
    return hasattr(x, "__dataclass_fields__")


def _kind(want):
    """Name of the fact class a declaration refers to; None for a predicate (any fact is passed to it)."""
    if isinstance(want, tuple):
        return type(want[0]).__name__
    if _is_fact(want):
        return type(want).__name__
    return None


def _accepts(want, got):
    if isinstance(want, tuple):
        return got in want
    if _is_fact(want):
        return got == want
    return bool(want(got))


class carry:
    """returns=carry("x"): the result means what argument x meant (a copy, a view, a reshape of it)."""

    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f"carry({self.arg!r})"


def _facts(declared, out=None, bound=None):
    """A declaration as a tuple of facts: None -> (), one fact -> (fact,), a tuple stays. A function stands for a
    meaning that depends on the values, e.g. the scale of a quantized result or the valid length of a cache after
    a write: it is called with the result and the arguments by name."""
    if declared is None:
        return ()
    if callable(declared) and not _is_fact(declared) and not isinstance(declared, carry):
        return _facts(declared(out, bound or {}))
    return tuple(declared) if isinstance(declared, tuple) else (declared,)


def _written_by_schema(fn):
    """The arguments a torch.library custom op declares it writes (mutates_args, `Tensor(a!)` in its schema), or
    None when fn is not such an op."""
    schema = getattr(getattr(fn, "_opoverload", None), "_schema", None)
    if schema is None:
        return None
    return {a.name for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write}


def _replace(value, facts, why):
    """The value now holds something else: its old facts go and `facts` take their place. With nothing declared in
    their place, the old facts are kept as an Invalidated marker that names who made them untrue."""
    have = _FACTS.get(id(value))
    old = sorted(k for k in (have or {}) if k != "Invalidated")
    new = {type(f).__name__: f for f in facts} if facts else (
        {"Invalidated": Invalidated(kind=",".join(old), why=why)} if old else {})
    if have is not None:  # rewrite the entry in place: the value's finalizer is already registered
        have.clear()
        have.update(new)
    elif new:
        tag(value, *new.values())


def _attach(out, returns, bound, where):
    if isinstance(returns, list):  # one declaration per element of a tuple result
        if not isinstance(out, (tuple, list)) or len(out) != len(returns):
            raise RoleError(f"{where}: declares {len(returns)} results, but returned {type(out).__name__}")
        for o, r in zip(out, returns):
            _attach(o, r, bound, where)
        return
    if isinstance(returns, carry):
        facts = tuple(facts_of(bound.get(returns.arg)).values())
    else:
        facts = _facts(returns, out, bound)
    if not facts:
        return
    try:
        tag(out, *facts)
    except TypeError:
        raise RoleError(f"{where}: declares what its result means, but returned {type(out).__name__}") from None


def _check_arg(where, arg, want, kwargs):
    if arg not in kwargs:
        raise RoleError(f"{where}: argument {arg!r} must be passed by keyword at this boundary")
    kind = _kind(want)
    have = facts_of(kwargs[arg])
    got = have.get(kind) if kind else next(iter(have.values()), None)
    if got is None:
        dead = have.get("Invalidated")
        if dead is not None and (kind is None or kind in dead.kind.split(",")):
            raise RoleError(f"{where}: argument {arg!r} had a {kind or dead.kind} declaration, but {dead.why} "
                            f"made it untrue. Declare the transform.")
        raise RoleError(f"{where}: argument {arg!r} carries no {kind or 'fact'} declaration")
    if not _accepts(want, got):
        raise RoleError(f"{where}: argument {arg!r} {kind}: expected {want}, got {got}")


def boundary(name=None, *, returns=None, writes=None, **expected):
    """Declare a boundary between components: what each argument must mean (keyword=fact), what the result means
    (returns=), and which arguments it writes in place and what they mean afterwards (writes={arg: fact or None}).

      @boundary(w=(Layout("q8_0", packing="interleaved"),))                     what it takes
      @boundary(buf=INTERLEAVED, writes={"buf": SPLIT})                         rewrites buf in place
      @boundary(x=Quantized("float8_e4m3fn", 0.5), returns=Quantized("float32"))  produces a new value
      @boundary(returns=carry("x"))                                             the result means what x meant

    - Declared and written arguments are passed by keyword: the keyword is the role marker (src=, dst=). Passing
      one by position is an error, in debug mode, as a missing declaration is.
    - In load and debug mode the result and the written arguments carry what is declared, so the meaning reaches
      the next boundary without anyone tagging it again. A written argument declared with None keeps its old facts
      as an Invalidated marker naming this boundary: nothing is silently dropped.
    - In debug mode the arguments are checked, and so is the writing: an argument declared as written must have been
      written, and no other argument may have been (tensor version counters). On a torch.library custom op, its own
      declaration (mutates_args) must agree with `writes`; torch runs an op whose mutates_args is wrong without a
      word, so this is where it shows.
    """
    writes = dict(writes or {})

    def deco(fn):
        where = name or getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)
        by_schema = _written_by_schema(fn)
        if by_schema is not None:
            if writes and set(writes) != by_schema:
                raise RoleError(f"{where}: the op declares it writes {sorted(by_schema)} (mutates_args), the "
                                f"boundary says {sorted(writes)}")
            for arg in by_schema:
                writes.setdefault(arg, None)
        try:
            sig = inspect.signature(getattr(fn, "_init_fn", fn))
        except (TypeError, ValueError):
            sig = None
        def dependent(r):
            return isinstance(r, carry) or (callable(r) and not _is_fact(r))

        needs_binding = bool(writes) or dependent(returns) or (
            isinstance(returns, list) and any(dependent(r) for r in returns))
        # What the meaning attached after the call needs by name. When all of it came by keyword (as roles do),
        # binding the whole signature is skipped: it is most of the cost in load mode.
        by_name = set(writes) | {r.arg for r in (returns if isinstance(returns, list) else [returns])
                                 if isinstance(r, carry)}
        any_function = any(callable(m) and not _is_fact(m) and not isinstance(m, carry)
                           for m in [*writes.values(), *(returns if isinstance(returns, list) else [returns])])

        def bind(args, kwargs, debug):
            if not debug and not any_function and by_name <= kwargs.keys():
                return kwargs
            if sig is None:
                return dict(kwargs)
            try:
                return dict(sig.bind(*args, **kwargs).arguments)
            except TypeError:
                return dict(kwargs)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _MODE == "off":
                return fn(*args, **kwargs)
            debug = _MODE == "debug"
            if debug:
                for arg, want in expected.items():
                    _check_arg(where, arg, want, kwargs)
                for arg in writes:
                    if arg not in kwargs:
                        raise RoleError(f"{where}: argument {arg!r} is written here, so it must be passed by keyword "
                                        f"(the keyword is its role)")
            bound = bind(args, kwargs, debug) if (needs_binding or debug) else {}
            before = {n: v._version for n, v in bound.items() if hasattr(v, "_version")} if debug else {}
            out = fn(*args, **kwargs)
            for n, v0 in before.items():
                changed = bound[n]._version != v0
                if n in writes and not changed:
                    raise RoleError(f"{where}: declares it writes {n!r}, but {n!r} was not written")
                if n not in writes and changed:
                    raise RoleError(f"{where}: wrote into {n!r}, which it does not declare "
                                    f"(it declares writes: {sorted(writes) or 'none'})")
            for arg, meaning in writes.items():
                if arg in bound:
                    _replace(bound[arg], _facts(meaning, out, bound), f"{where} wrote into it")
            if returns is not None:
                _attach(out, returns, bound, where)
            return out
        return wrapper
    return deco


def _load_active():
    return _MODE in ("load", "debug")


def require(cond, where, message, at="debug"):
    """A contract between facts that one declaration cannot express (e.g. decode position == valid KV length).
    Checked in debug mode, or also in load mode when at="load"."""
    active = _MODE == "debug" or (at == "load" and _MODE == "load")
    if active and not cond:
        raise RoleError(f"{where}: {message}")


def check_props(props: ModelProps, caps: KernelCaps, where: str):
    """Every declared model property must be honoured by the chosen kernel (fail loud, at load time)."""
    if not _load_active():
        return
    missing = []
    if props.softcap is not None and not caps.softcap:
        missing.append(f"softcap={props.softcap}")
    if props.sliding_window is not None and not caps.sliding_window:
        missing.append(f"sliding_window={props.sliding_window}")
    if missing:
        raise RoleError(f"{where}: kernel does not honour declared model properties: {', '.join(missing)}")


def check_config_keys(config: dict, known: set, where: str):
    """Keys the consumer does not recognise are an error, not silently kept."""
    if not _load_active():
        return
    from .coverage import count  # the same count as a LoRA's modules: given, and not taken by the consumer

    left = list(count(config, known).left)
    if left:
        raise RoleError(f"{where}: unrecognised config keys {left}")


def check_tied(declared_tie: bool, embed, head, where: str):
    """A declared tie must match the tensors: if both are present and differ, the declaration is wrong."""
    if not _load_active():
        return
    if declared_tie and head is not None and not (embed.shape == head.shape and bool((embed == head).all())):
        raise RoleError(f"{where}: config declares tied embeddings but the checkpoint holds a different head")
