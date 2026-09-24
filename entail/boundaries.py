"""boundaries: signatures on code boundaries (LIBRARY_DESIGN.md 4.6; ROADMAP M4.1).

A boundary states what it takes, what its result means and what it writes in place, with keyword-only role
markers, so the meaning a producer declares reaches the next consumer without anyone tagging it again:

    @boundary(takes={"buf": INTERLEAVED}, writes={"buf": SPLIT})     # or buf=INTERLEAVED, the 2.2 spelling
    def reorder(*, buf): ...

    @boundary(returns=Positions("absolute"))                          # what the result means
    @boundary(returns=carry("x"))                                     # the result means what x meant
    @boundary(writes={"counter": advance()})                          # the write moves the buffer to its next Epoch
    @boundary(agree={"Epoch": ("mask", "offset")})                    # these arguments must carry the same fact

A declaration of what an argument must be is a fact (must agree), a tuple of facts (one of them), or a predicate.

By mode (core.mode()):
  off    nothing but one global read per call
  load   the meaning travels: results and written arguments carry what is declared, as Fact envelopes whose Source
         names the boundary ("boundary: reorder.writes.buf") - the chain the diagnosis follows (M7)
  debug  every check is also a Decision (contracts.decide), with the policy in force:
         - each declared argument: the fact it carries, which its producer declared, against what this boundary
           takes; a converter below repairs a mismatch it knows (resolved: the converted value is what the function
           receives), otherwise refused; a value that carries nothing is unknown, and stops in debug mode
         - agree: the named arguments carry the same fact of that kind
         - writes: tensor version counters against the declaration (a missing or an undeclared write is refused)
A declared argument passed by position has no role marker and is refused. A torch custom op's own schema
(mutates_args) must agree with `writes`; that is checked once, when the function is decorated. Debug mode stops at
what it cannot repair, so these are refused (contracts.unrepaired; outside debug mode the same outcome would be
broken and reported, M5.4). A declaration that does not fit the call's shape - a tuple of results declared and
something else returned, a result that cannot carry facts - is misuse of the API and raises as before.

Passes are counted per boundary (PASSES), not recorded one by one: a boundary can run millions of times. Anything
else goes to the ledger through load.enforce, once per distinct outcome (REPEATS counts the rest), and a blocking
decision raises RoleError before the function runs.
"""
import functools
import inspect
from dataclasses import replace
from typing import Callable, Dict, Optional, Tuple

from . import core
from .contracts import RULES, Contract, Decision, Resolution, Verdict, decide, unrepaired
from .core import RoleError, facts_of, tag
from .facts import VOCABULARY, Certainty, Epoch, Fact, Invalidated, Source
from .kv_contract import KvExtent, check_extent

PASSES: Dict[Tuple[str, str], int] = {}     # (boundary, fact name) -> checks that passed
REPEATS: Dict[tuple, int] = {}              # an outcome already recorded once -> how many times it came again


class carry:
    """returns=carry("x"): the result means what argument x meant (a copy, a view, a reshape of it)."""

    def __init__(self, arg):
        self.arg = arg

    def __repr__(self):
        return f"carry({self.arg!r})"


class advance:
    """writes={"buf": advance()}: the write moves the buffer to its next Epoch (the owner is kept)."""

    def __repr__(self):
        return "advance()"


# --- converters: the repairs a code boundary can make on the value itself ----------------------------------------

def _to_dense(value, have, want):
    return value.contiguous()


def _dequantize(value, have, want):
    import torch

    out = value.to(torch.float32) * have.scale
    return out if want.dtype == "float32" else out.to(getattr(torch, want.dtype))


def _to_absolute(value, have, want):
    return value + have.offset


def _only_differ_in(have, want, name):
    return all(getattr(have, f) == getattr(want, f) for f in have.__dataclass_fields__ if f != name
               and getattr(want, f) is not None)


# vocabulary name -> [(Resolution, function(value, have, want))]; offered to decide() per call, never registered
# globally, so load-time contracts (M3) do not pick up repairs that only make sense on a value in hand.
CONVERTERS = {
    "Layout": [(Resolution("make the tensor contiguous", "layout.contiguous",
                           when=lambda d, c: d.value.kind == "strided" and c.value.kind == "dense"
                           and _only_differ_in(d.value, c.value, "kind")), _to_dense)],
    "Quantized": [(Resolution("dequantize with its scale", "quantized.dequantize",
                              when=lambda d, c: d.value.scale is not None and c.value.scale is None
                              and c.value.dtype in ("float32", "bfloat16", "float16")), _dequantize)],
    "Positions": [(Resolution("add the chunk's offset", "positions.to_absolute",
                              when=lambda d, c: d.value.frame == "chunk_relative" and d.value.offset is not None
                              and c.value.frame == "absolute"), _to_absolute)],
}


# --- what a value carries --------------------------------------------------------------------------------------

def _envelope(value, where):
    name = type(value).__name__
    if name in VOCABULARY:
        return Fact(name, value, Source("boundary", where), Certainty.DECLARED)
    return value


def _is_fact(x):
    return hasattr(x, "__dataclass_fields__")


def _kind(want):
    if isinstance(want, tuple):
        return type(want[0]).__name__
    if _is_fact(want):
        return type(want).__name__
    return None


def _declared(declared, out=None, bound=None):
    """A declaration as a tuple of fact values: None -> (), one fact -> (fact,), a tuple stays. A function stands for
    a meaning that depends on the values (the scale of a quantized result, the length of a cache after a write)."""
    if declared is None:
        return ()
    if callable(declared) and not _is_fact(declared) and not isinstance(declared, (carry, advance)):
        return _declared(declared(out, bound or {}))
    return tuple(declared) if isinstance(declared, tuple) else (declared,)


def _written_by_schema(fn):
    """The arguments a torch.library custom op declares it writes (mutates_args), or None when fn is not one."""
    schema = getattr(getattr(fn, "_opoverload", None), "_schema", None)
    if schema is None:
        return None
    return {a.name for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write}


def _replace(value, facts, why):
    """The value now holds something else: its old facts go and `facts` take their place. With nothing declared in
    their place, the old facts are kept as an Invalidated marker that names who made them untrue."""
    have = core._FACTS.get(id(value))
    old = sorted(k for k in (have or {}) if k != "Invalidated")
    new = {(f.name if isinstance(f, Fact) else type(f).__name__): f for f in facts} if facts else (
        {"Invalidated": Invalidated(kind=",".join(old), why=why)} if old else {})
    if have is not None:   # rewrite the entry in place: the value's finalizer is already registered
        have.clear()
        have.update(new)
    elif new:
        tag(value, *new.values())


def _attach(out, returns, bound, where):
    if isinstance(returns, list):   # one declaration per element of a tuple result
        if not isinstance(out, (tuple, list)) or len(out) != len(returns):
            raise RoleError(f"{where}: declares {len(returns)} results, but returned {type(out).__name__}")
        for o, r in zip(out, returns):
            _attach(o, r, bound, where)
        return
    if isinstance(returns, carry):
        facts = tuple(core.envelopes_of(bound.get(returns.arg)).values())
    else:
        facts = tuple(_envelope(f, f"{where}.returns") for f in _declared(returns, out, bound))
    if not facts:
        return
    try:
        tag(out, *facts)
    except TypeError:
        raise RoleError(f"{where}: declares what its result means, but returned {type(out).__name__}") from None


def _advanced(value, where):
    now = facts_of(value).get("Epoch")
    nxt = Epoch(0 if now is None else now.version + 1, None if now is None else now.owner)
    return (_envelope(nxt, where),)


# --- deciding --------------------------------------------------------------------------------------------------

def _record(decisions, where):
    """Passes are counted; anything else goes to the ledger once per distinct outcome; a blocking one stops."""
    from . import load

    first = []
    for d in decisions:
        if d.verdict is Verdict.PASS:
            PASSES[(where, d.name)] = PASSES.get((where, d.name), 0) + 1
            continue
        key = (where, d.name, d.verdict, d.rule, str(d.target), d.note, d.blocking)
        if key in REPEATS and not d.blocking:
            REPEATS[key] += 1
            continue
        REPEATS.setdefault(key, 0)
        first.append(d)
    if first:
        load.enforce(first)


def _fact(name, value, where):
    return Fact(name, value, Source("boundary", where), Certainty.DECLARED)


def _check_arg(where, arg, want, value, policy):
    """(decision, value to pass on) for one declared argument."""
    kind = _kind(want)
    have = core.envelopes_of(value)
    raw = facts_of(value)
    boundary = f"boundary:{where}"
    if kind is None:   # a predicate over the first fact the value carries (named by its `kind`, if it has one)
        got = next((v for k, v in raw.items() if k in VOCABULARY), None)
        name = type(got).__name__ if got is not None else getattr(want, "kind", "Layout")
        contract = Contract(boundary, where, (name,), (name,))
        if got is None:
            return Decision(contract, name, Verdict.UNKNOWN, RULES["undeclared"],
                            blocking=policy.stops_unknown(name, True), note=f"argument {arg} carries no fact"), value
        ok = bool(want(got))
        verdict, blocking = (Verdict.PASS, False) if ok else unrepaired(policy, name)
        d = Decision(contract, name, verdict, RULES["match"] if ok else RULES["predicate"], declared=have.get(name),
                     blocking=blocking, note=f"argument {arg}")
        return d, value
    contract = Contract(boundary, where, (kind,), (kind,))
    carried = have.get(kind)
    if carried is None and "Invalidated" in raw and kind in raw["Invalidated"].kind.split(","):
        dead = raw["Invalidated"]
        return Decision(contract, kind, Verdict.UNKNOWN, RULES["invalidated"],
                        blocking=policy.stops_unknown(kind, True) or policy.mode == "debug",
                        note=f"argument {arg}: made untrue by {dead.why}; declare what it holds now",
                        lost_by=dead.why), value
    options = want if isinstance(want, tuple) else (want,)
    chosen = next((w for w in options if carried is not None and carried.value == w), None)
    if chosen is None and carried is not None:   # a form this boundary takes that a converter can reach
        for w in options:
            c = _fact(kind, w, f"{where}.takes.{arg}")
            if any(r.applies(carried, c) for r, _ in CONVERTERS.get(kind, [])):
                chosen = w
                break
    chosen = options[0] if chosen is None else chosen
    c = _fact(kind, chosen, f"{where}.takes.{arg}")
    offered = [r for r, _ in CONVERTERS.get(kind, [])]
    (d,) = decide(contract, {kind: carried} if carried is not None else {}, {kind: c}, policy,
                  resolutions={kind: offered})
    if len(options) > 1 and d.verdict in (Verdict.BROKEN, Verdict.REFUSED):
        d = replace(d, note=f"argument {arg}; it takes one of {', '.join(str(o) for o in options)}")
    else:
        d = replace(d, note=f"argument {arg}")
    if d.verdict is Verdict.RESOLVED:
        r, fn = next((r, f) for r, f in CONVERTERS[kind] if r.handle == d.handle)
        # a converter changes the value, so it goes from what the value carried to what this boundary takes
        # (Resolution.describe words the load-time kind, which changes the consumer's choice towards the declaration)
        d = replace(d, resolution=f"{r.name} ({carried.value} -> {chosen})")
        value = fn(value, carried.value, chosen)
        tag(value, _fact(kind, chosen, f"{where}.takes.{arg} ({d.resolution})"))
    return d, value


def _check_agree(where, kind, args, kwargs, policy):
    """The named arguments carry the same fact of `kind`; the first one's is what the others must match."""
    contract = Contract(f"boundary:{where}", where, (kind,), (kind,))
    first = core.envelopes_of(kwargs.get(args[0])).get(kind)
    out = []
    for other in args[1:]:
        f = core.envelopes_of(kwargs.get(other)).get(kind)
        if first is None or f is None:
            out.append(Decision(contract, kind, Verdict.UNKNOWN, RULES["undeclared"], declared=first, chosen=f,
                                blocking=policy.stops_unknown(kind, True),
                                note=f"{args[0]} and {other} must carry the same {kind}"))
            continue
        same = first.value == f.value
        verdict, blocking = (Verdict.PASS, False) if same else unrepaired(policy, kind)
        out.append(Decision(contract, kind, verdict, RULES["match"] if same else RULES["disagree"], declared=first,
                            chosen=f, blocking=blocking, note=f"{args[0]} and {other} must carry the same {kind}"))
    return out


def _write_decision(where, arg, rule, policy):
    contract = Contract(f"boundary:{where}", where, ("Epoch",), ("Epoch",))
    verdict, blocking = unrepaired(policy, "Epoch")
    return Decision(contract, "Epoch", verdict, RULES[rule], blocking=blocking, note=f"argument {arg}")


# --- the decorator ---------------------------------------------------------------------------------------------

def boundary(name=None, *, takes=None, returns=None, writes=None, agree=None, **expected):
    """Declare a boundary between components (see the module docstring). `takes` and keyword arguments both say
    what an argument must mean; `returns` what the result means; `writes` which arguments it writes in place and
    what they hold afterwards (a fact, a function of the result and the arguments, advance(), or None when the
    old meaning simply stops being true); `agree` which arguments must carry the same fact of a kind."""
    expected = {**(takes or {}), **expected}
    writes = dict(writes or {})
    agree = dict(agree or {})

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
        # what the meaning attached after the call needs by name; when all of it came by keyword (as roles do),
        # binding the whole signature is skipped: it is most of the cost in load mode
        by_name = set(writes) | {r.arg for r in (returns if isinstance(returns, list) else [returns])
                                 if isinstance(r, carry)}
        any_function = any(callable(m) and not _is_fact(m) and not isinstance(m, (carry, advance))
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
            mode = core.mode()
            if mode == "off":
                return fn(*args, **kwargs)
            debug = mode == "debug"
            if debug:
                from . import policies
                policy = policies.current()
                decisions = []
                for arg, want in expected.items():
                    if arg not in kwargs:
                        kind = _kind(want) or "Layout"
                        verdict, blocking = unrepaired(policy, kind)
                        decisions.append(Decision(Contract(f"boundary:{where}", where, (kind,)), kind,
                                                  verdict, RULES["positional"], blocking=blocking,
                                                  note=f"argument {arg}"))
                        continue
                    d, kwargs[arg] = _check_arg(where, arg, want, kwargs[arg], policy)
                    decisions.append(d)
                for kind, names in agree.items():
                    decisions += _check_agree(where, kind, names, kwargs, policy)
                for arg in writes:
                    if arg not in kwargs:
                        verdict, blocking = unrepaired(policy, "Epoch")
                        decisions.append(Decision(Contract(f"boundary:{where}", where, ("Epoch",)), "Epoch",
                                                  verdict, RULES["positional"], blocking=blocking,
                                                  note=f"argument {arg} is written here"))
                _record(decisions, where)
            bound = bind(args, kwargs, debug) if (needs_binding or debug) else {}
            before = {n: v._version for n, v in bound.items() if hasattr(v, "_version")} if debug else {}
            out = fn(*args, **kwargs)
            if before:
                bad = []
                for n, v0 in before.items():
                    changed = bound[n]._version != v0
                    if n in writes and not changed:
                        bad.append(_write_decision(where, n, "write_missing", policy))
                    elif n not in writes and changed:
                        bad.append(_write_decision(where, n, "write_undeclared", policy))
                _record(bad, where)
            for arg, meaning in writes.items():
                if arg in bound:
                    if isinstance(meaning, advance):
                        _replace(bound[arg], _advanced(bound[arg], f"{where}.writes.{arg}"), "")
                    else:
                        _replace(bound[arg], tuple(_envelope(f, f"{where}.writes.{arg}")
                                                   for f in _declared(meaning, out, bound)),
                                 f"{where}, which wrote into it")
            if returns is not None:
                _attach(out, returns, bound, where)
            return out
        return wrapper
    return deco


__all__ = ["RoleError", "boundary", "carry", "advance", "facts_of", "tag", "KvExtent", "check_extent", "CONVERTERS",
           "PASSES", "REPEATS"]
