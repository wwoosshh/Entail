"""Shared core of entail: modes, the side table of facts, boundary checks, load-time checks.

Modes (environment variable ENTAIL, or set_mode()):
  off    default. tag() stores nothing and boundaries only read one global per call.
  load   load-time checks (check_props, check_config_keys, check_tied) are active; boundaries are not.
  debug  everything is checked, including every @boundary call.
Errors are always RoleError and name the boundary, the argument, the fact kind, what was expected and what came.

Policy (environment variable ENTAIL_POLICY, or set_policy()): what to do when a declaration and the real
thing disagree.
  resolve  default. Keep the meaning intact by fixing the situation - send the value to a consumer that honours
           the declaration, convert it to the form the consumer expects, or recompute it - and say so in one
           line. Stop only when no such fix exists. (RESEARCH_PLAN.md 0 and 9, decided 2026-09-23.)
  refuse   stop at the first disagreement, as before.
"""
import functools
import os
import weakref

from .facts import KernelCaps, ModelProps

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


def boundary(name=None, **expected):
    """Declare what each named argument must carry. Checked only in debug mode.

    Example: @boundary(q=Positions("absolute"), w=(Layout("q8_0", packing="interleaved"),))
    A missing declaration is an error (declarations are mandatory at boundaries).
    """
    def deco(fn):
        where = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _MODE == "debug":
                for arg, want in expected.items():
                    if arg not in kwargs:
                        raise RoleError(f"{where}: argument {arg!r} must be passed by keyword at this boundary")
                    kind = _kind(want)
                    have = facts_of(kwargs[arg])
                    got = have.get(kind) if kind else next(iter(have.values()), None)
                    if got is None:
                        dead = have.get("Invalidated")
                        if dead is not None and (kind is None or dead.kind == kind):
                            raise RoleError(f"{where}: argument {arg!r} had a {dead.kind} declaration, but "
                                            f"{dead.why} made it untrue. Declare the transform.")
                        raise RoleError(f"{where}: argument {arg!r} carries no {kind or 'fact'} declaration")
                    if not _accepts(want, got):
                        raise RoleError(f"{where}: argument {arg!r} {kind}: expected {want}, got {got}")
            return fn(*args, **kwargs)
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
    unknown = sorted(set(config) - set(known))
    if unknown:
        raise RoleError(f"{where}: unrecognised config keys {unknown}")


def check_tied(declared_tie: bool, embed, head, where: str):
    """A declared tie must match the tensors: if both are present and differ, the declaration is wrong."""
    if not _load_active():
        return
    if declared_tie and head is not None and not (embed.shape == head.shape and bool((embed == head).all())):
        raise RoleError(f"{where}: config declares tied embeddings but the checkpoint holds a different head")
