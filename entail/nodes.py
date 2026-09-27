"""nodes: custom nodes - high-level checks a developer puts on points of their own project (the workshop; ROADMAP
product track P5; LIBRARY_DESIGN.md 13.7).

A custom node is a named point of the developer's program ("rag.answer") and the validators that check a value there.
A validator is a plain function of the value and keyword parameters that says what it found:

    import json
    from entail import nodes

    @nodes.validator("json_object")
    def json_object(value, keys=()):
        try:
            obj = json.loads(value)
        except ValueError as e:
            return nodes.broken(f"not JSON: {e}")
        missing = [k for k in keys if k not in obj]
        return nodes.broken(f"missing keys {missing}") if missing else nodes.ok()

    answer = nodes.check("rag.answer", answer, json_object, keys=("title", "body"))   # the value comes back as is

    @nodes.watch("rag.answer", json_object, keys=("title", "body"))    # checks what the function returns
    def ask(question): ...

What a validator found is a decision of the core (fact Check, vocabulary v12) at boundary node:<node>/<validator>, and
is recorded and reported like the core's own: a check that holds is counted, not written; one that does not is broken
(the program goes on; ENTAIL_ON_BROKEN=stop or ENTAIL_POLICY=Check=stop stops it, as for the core's checks), once per
distinct finding; one the validator could not decide is unknown. The platform (entail serve) shows each custom node in
the user-code flow, and can turn one off.

The core runs every validator itself, and a validator never breaks the program (principle 12):
  - an exception is recorded as unknown ("the validator failed", with the exception); after two failures the validator
    is left out for the rest of the process; debug mode raises
  - a call that takes longer than the validator's budget (BUDGET_MS unless it says otherwise) is said once and the
    validator is left out for the process. The budget is measured when the call returns, so a validator that may
    never return (a network call, a lock) should say hard=True: it then runs on a daemon thread and the program
    stops waiting at the budget - the validator is recorded as not returning, left out, and abandoned (Python
    cannot stop a thread; it may go on in the background). A hard validator costs a thread start per call
  - nothing runs when entail is off (ENTAIL unset or off) or ENTAIL_NODES=off, and a node turned off in the log
    folder's nodes.json ({"off": [...]}: what the platform writes; read again when it changes) is skipped
The trust boundary is the DLCs' (entail/dlc.py): a validator is Python code in the program's process with the
program's rights; entail does not sandbox it. Packages of custom nodes attach through the entry point group
entail.nodes - an object with name, version, requires and validators ({name: function}) - only when installed, and
ENTAIL_NODES=name,name lists which may; the developer's own nodes are their own code, imported by their program.
"""
import functools
import json
import os
import re
import time
from typing import Callable, Dict, List, NamedTuple, Optional

GROUP = "entail.nodes"
BUDGET_MS = 50.0          # a validator's time per call (measured in P5: the examples' validators take microseconds)
_NODE = re.compile(r"^[a-z0-9][a-z0-9_-]*(\.[a-z0-9][a-z0-9_-]*)*$")
_CHECK = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VALIDATORS: Dict[str, Callable] = {}
_FAILED: Dict[str, int] = {}
_LEFT_OUT: Dict[str, str] = {}    # validator -> why it no longer runs in this process
_OFF = {"at": 0.0, "stamp": None, "nodes": frozenset()}
_PACKAGES: Optional[List[dict]] = None


class Result(NamedTuple):
    holds: Optional[bool]
    detail: Optional[str] = None


def ok(detail: Optional[str] = None) -> Result:
    """The value has the property (detail: what was checked, optional)."""
    return Result(True, detail)


def broken(why: str) -> Result:
    """The value does not have the property: `why` says what was expected and what was seen."""
    return Result(False, str(why))


def unknown(why: str) -> Result:
    """The validator could not tell (the value it needed was missing, say)."""
    return Result(None, str(why))


def valid_node(name) -> bool:
    """Whether `name` can name a custom node: dotted lower-case letters, digits, "_" and "-" ("rag.answer")."""
    return isinstance(name, str) and bool(_NODE.match(name))


def validator(name: str, budget_ms: Optional[float] = None, hard: bool = False):
    """Register a validator under `name` (lower-case letters, digits, "_" and "-"); `budget_ms`: its time per call;
    `hard`: run it on a thread and stop waiting at the budget (for one that may never return)."""
    if not isinstance(name, str) or not _CHECK.match(name):
        raise ValueError(f"nodes.validator: {name!r} is not lower-case letters, digits, '_' and '-'")

    def deco(fn):
        fn._entail_check = name
        fn._entail_budget_ms = BUDGET_MS if budget_ms is None else float(budget_ms)
        fn._entail_hard = bool(hard)
        _VALIDATORS.setdefault(name, fn)
        return fn

    return deco


def _call_within(fn, value, params, budget_ms: float):
    """Run fn on a daemon thread; (finished, result or exception)."""
    import threading

    box = {}

    def work():
        try:
            box["out"] = fn(value, **params)
        except BaseException as e:  # noqa: BLE001 - handed back to the caller, which decides
            box["err"] = e

    t = threading.Thread(target=work, name="entail-validator", daemon=True)
    t.start()
    t.join(budget_ms / 1000.0)
    if t.is_alive():
        return False, None
    if "err" in box:
        raise box["err"]
    return True, box.get("out")


def _active() -> bool:
    from . import core

    return core.mode() in ("load", "debug") and (os.environ.get("ENTAIL_NODES") or "").strip().lower() != "off"


def off_nodes() -> frozenset:
    """The nodes turned off in the log folder's nodes.json, read again when the file changes (looked at once a
    second at most)."""
    now = time.monotonic()
    if now - _OFF["at"] < 1.0:
        return _OFF["nodes"]
    _OFF["at"] = now
    from . import record

    folder = record.log_dir()
    path = os.path.join(folder, "nodes.json") if folder else None
    try:
        st = os.stat(path) if path else None
    except OSError:
        st = None
    stamp = (st.st_mtime_ns, st.st_size) if st else None
    if stamp != _OFF["stamp"]:
        _OFF["stamp"] = stamp
        names = frozenset()
        if stamp:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                names = frozenset(n for n in data.get("off", []) if isinstance(n, str))
            except (OSError, ValueError, AttributeError):
                names = frozenset()
        _OFF["nodes"] = names
    return _OFF["nodes"]


def _resolve(v) -> Optional[Callable]:
    if callable(v):
        return v
    if isinstance(v, str):
        packages()
        return _VALIDATORS.get(v)
    return None


def _name_of(fn) -> str:
    name = getattr(fn, "_entail_check", None) or getattr(fn, "__name__", "validator")
    name = re.sub(r"[^a-z0-9_-]", "_", str(name).lower()).strip("_") or "validator"
    return name


def check(node: str, value, *validators, **params):
    """Run the validators on `value` at the custom node `node` (dotted lower-case names) and return `value` as it
    was. A validator is a function (registered with @validator or not) or a registered name."""
    if not _active() or node in off_nodes():
        return value
    if not valid_node(node):
        _say(f"node:{node}", f"not a node name (dotted lower-case letters, digits, '_' and '-'): nothing checked")
        return value
    for v in validators:
        fn = _resolve(v)
        if fn is None:
            _say(f"node:{node}", f"no validator {v!r}: nothing checked by it")
            continue
        _run(node, fn, value, params)
    return value


def watch(node: str, *validators, **params):
    """Decorate a function: its result is checked at `node` by the validators, and returned as it was."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            out = fn(*a, **kw)
            check(node, out, *validators, **params)
            return out

        return wrapper

    return deco


def _run(node: str, fn, value, params) -> None:
    from . import core, load, tally
    from .contracts import RULES, Contract, Decision, unrepaired
    from .facts import Certainty, Check, Fact, Source
    from . import policies

    name = _name_of(fn)
    key = f"{node}/{name}"
    if key in _LEFT_OUT:
        return
    boundary = f"node:{key}"
    budget = getattr(fn, "_entail_budget_ms", BUDGET_MS)
    t0 = time.perf_counter()
    try:
        if getattr(fn, "_entail_hard", False):
            finished, got = _call_within(fn, value, params, budget)
            if not finished:
                why = (f"the validator {name} did not return within its budget of {budget:g} ms: the program stopped "
                       f"waiting, and it is left out for the rest of this process (its thread may still run)")
                _LEFT_OUT[key] = why
                tally.counts(boundary)["checks"] += 1
                load.enforce([load.cannot_check(boundary, node, "Check", why)])
                return
        else:
            got = fn(value, **params)
    except Exception as e:  # noqa: BLE001 - principle 12: a validator never breaks the program
        if core.mode() == "debug":
            raise
        _FAILED[key] = _FAILED.get(key, 0) + 1
        why = (f"the validator {name} failed ({type(e).__name__}: {str(e)[:200]})"
               + ("" if _FAILED[key] < 2 else "; it is left out for the rest of this process"))
        if _FAILED[key] >= 2:
            _LEFT_OUT[key] = why
        tally.counts(boundary)["checks"] += 1
        if tally.first(boundary, "cannot_check", key=type(e).__name__):
            load.enforce([load.cannot_check(boundary, node, "Check", why)])
        return
    ms = (time.perf_counter() - t0) * 1000.0
    if ms > budget:
        _LEFT_OUT[key] = f"took {ms:.1f} ms, over its budget of {budget:g} ms"
        _say(boundary, f"the validator {name} took {ms:.1f} ms, over its budget of {budget:g} ms: it is left out for "
                       f"the rest of this process (@validator(budget_ms=...) gives it more)")
    if isinstance(got, bool):
        got = Result(got, None if got else "the check does not hold")
    elif not isinstance(got, Result):
        got = Result(None, f"the validator returned {type(got).__name__}, not ok(), broken() or unknown()")
    counts = tally.counts(boundary)
    counts["checks"] += 1
    if got.holds is True:
        tally.passed(boundary, ["node_check"])
        tally.tick(boundary)
        return
    if got.holds is None:
        if tally.first(boundary, "cannot_check", key=got.detail):
            load.enforce([load.cannot_check(boundary, node, "Check", f"{name}: {got.detail}")])
        tally.tick(boundary)
        return
    policy = policies.current()
    verdict, blocking = unrepaired(policy, "Check")
    contract = Contract(boundary, node, ("Check",), ())
    declared = Fact("Check", Check(name, True), Source("user", f"{node}: {name}"), Certainty.DECLARED)
    seen = Fact("Check", Check(name, False, got.detail), Source("probe", f"{name} on the value at {node}"),
                Certainty.VERIFIED)
    d = Decision(contract, "Check", verdict, RULES["node_check"], declared=declared, chosen=seen, blocking=blocking,
                 note=f"{node}: {name}: {got.detail}")
    if blocking:
        tally.refused(boundary)
        load.enforce([d])
    elif tally.first(boundary, "node_check", key=got.detail):
        tally.broken(boundary)
        load.enforce([d])
    else:
        tally.broken(boundary)
    tally.tick(boundary)


def packages(refresh: bool = False) -> List[dict]:
    """The workshop packages installed here (entry point group entail.nodes), attached the way DLCs are (their core
    range, ENTAIL_NODES=name,name): each gives validators ({name: function}), registered under their names."""
    global _PACKAGES
    if _PACKAGES is not None and not refresh:
        return _PACKAGES
    from . import __version__, dlc

    out = []
    setting = (os.environ.get("ENTAIL_NODES") or "").strip().lower()
    listed = None if setting in ("", "on", "all") else {n.strip() for n in setting.split(",") if n.strip()}
    if setting != "off":
        from importlib import metadata

        try:
            eps = list(metadata.entry_points(group=GROUP))
        except TypeError:
            eps = list(metadata.entry_points().get(GROUP, []))
        for ep in eps:
            info = {"entry_point": ep.name, "dist": getattr(getattr(ep, "dist", None), "name", None),
                    "attached": False}
            if listed is not None and ep.name not in listed:
                info["why"] = "not listed in ENTAIL_NODES (not imported)"
                out.append(info)
                continue
            try:
                obj = ep.load()
                name = getattr(obj, "name", None)
                info.update(name=name, version=str(getattr(obj, "version", "?")),
                            requires=str(getattr(obj, "requires", "")))
                found = dict(getattr(obj, "validators", {}) or {})
                if name != ep.name:
                    info["why"] = f"its entry point is named {ep.name!r}, not {name!r}"
                elif not dlc.version_ok(info["requires"], __version__):
                    info["why"] = f"it needs entail {info['requires']}, this is {__version__}"
                elif not all(isinstance(k, str) and _CHECK.match(k) and callable(v) for k, v in found.items()):
                    info["why"] = "its validators are not {name: function} with lower-case names"
                else:
                    taken = [k for k in found if k in _VALIDATORS and _VALIDATORS[k] is not found[k]]
                    for k, fn in found.items():
                        if k not in _VALIDATORS:
                            try:
                                fn._entail_check = k
                                fn._entail_budget_ms = getattr(fn, "_entail_budget_ms", BUDGET_MS)
                            except AttributeError:
                                pass
                            _VALIDATORS[k] = fn
                    info.update(attached=True, why="", validators=sorted(found), taken=taken)
            except Exception as e:  # noqa: BLE001 - a package that cannot be read never reaches the program
                info["why"] = f"could not be loaded ({type(e).__name__}: {e})"
            out.append(info)
    _PACKAGES = out
    for info in out:
        if not info["attached"] and "not listed" not in info.get("why", ""):
            _say(f"nodes:{info.get('name') or info['entry_point']}", f"not attached: {info['why']}")
        elif info.get("taken"):
            _say(f"nodes:{info['name']}", f"validators {info['taken']} were already registered by another package or "
                                          f"the program: the first one stays")
    return out


def _say(where: str, text: str) -> None:
    try:
        from . import load

        load.say(where, text)
    except Exception:  # noqa: BLE001
        print(f"[entail] {where}: {text}", flush=True)


def reset() -> None:
    """Forget what was registered, failed and read (tests)."""
    global _PACKAGES
    _VALIDATORS.clear()
    _FAILED.clear()
    _LEFT_OUT.clear()
    _OFF.update(at=0.0, stamp=None, nodes=frozenset())
    _PACKAGES = None
