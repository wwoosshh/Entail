"""Shared core of entail: modes, the side table of facts, the 0.3.0 load-time checks.

Modes (environment variable ENTAIL, or set_mode()):
  off    default. tag() stores nothing and boundaries only read one global per call.
  load   load-time contracts run (load.py); boundaries attach what they declare to their results and written
         arguments, so the meaning travels, but check nothing.
  debug  everything is checked, including every @boundary call: the arguments' meaning and what was written.

Policy (environment variable ENTAIL_POLICY, or set_policy()): whether entail may fix it when a declaration and
the real thing disagree.
  resolve  default. Keep the meaning intact by fixing the situation - send the value to a consumer that honours
           the declaration, convert it to the form the consumer expects, or recompute it - and say so in one
           line. (RESEARCH_PLAN.md 0 and 9, decided 2026-09-23.)
  refuse   fix nothing.
What is not fixed is reported as broken and the run goes on; it stops only with ENTAIL_ON_BROKEN=stop or in debug
mode (policies.py; the researcher's decision of 2026-09-24, ROADMAP M5.4). The 0.3.0 checks below (check_props,
check_config_keys, check_tied, require) are assertions a caller makes explicitly, and still raise.

The side table maps a value to the facts it carries. A fact of the vocabulary is kept as a Fact envelope, so it
says where it came from (a boundary's declaration, a tag); facts_of() gives the plain values, envelopes_of() the
envelopes. @boundary and carry live in boundaries.py (M4.1) and are re-exported here.
"""
import os
import weakref

from .facts import VOCABULARY, Certainty, Fact, Invalidated, KernelCaps, ModelProps, Source

_MODE = os.environ.get("ENTAIL", "off")
_POLICY = os.environ.get("ENTAIL_POLICY", "resolve")
_FACTS = {}  # id(value) -> {fact name: Fact envelope, or the raw marker}; removed when the value is collected


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
    """Attach facts to a value: fact values (wrapped as declared by a tag) or Fact envelopes. Returns the value so
    it can be used inline."""
    if _MODE == "off":
        return t
    key = id(t)
    if key not in _FACTS:
        _FACTS[key] = {}
        weakref.finalize(t, _FACTS.pop, key, None)
    for f in facts:
        if isinstance(f, Fact):
            _FACTS[key][f.name] = f
        elif type(f).__name__ in VOCABULARY:
            _FACTS[key][type(f).__name__] = Fact(type(f).__name__, f, Source("boundary", "tag"), Certainty.DECLARED)
        else:   # a marker outside the vocabulary (Invalidated, Base)
            _FACTS[key][type(f).__name__] = f
    return t


def facts_of(t):
    """{fact name: value} for what a value carries."""
    return {k: (v.value if isinstance(v, Fact) else v) for k, v in _FACTS.get(id(t), {}).items()}


def envelopes_of(t):
    """{fact name: Fact} for the vocabulary facts a value carries (markers such as Invalidated are left out)."""
    return {k: v for k, v in _FACTS.get(id(t), {}).items() if isinstance(v, Fact)}


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


__all__ = ["RoleError", "set_mode", "mode", "set_policy", "policy", "tag", "facts_of", "envelopes_of", "require",
           "check_props", "check_config_keys", "check_tied", "Invalidated", "boundary", "carry"]

from .boundaries import boundary, carry  # noqa: E402  (boundaries imports this module; keep this last)
