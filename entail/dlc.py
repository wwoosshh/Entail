"""dlc: official DLCs - packages outside the core that attach through the entry point group `entail.dlc`
(LIBRARY_DESIGN.md 4.8, 13.7; ROADMAP product track P4).

A DLC's entry point names an object (a module or a class) that says:
  name      the DLC's name: lower-case letters, digits, "_" and "-"
  version   its version
  requires  the core versions it works with: comparisons joined by commas (">=1.3,<3")
  engines   {engine: versions} it was made for (shown, not enforced)
  targets   {module: ["package.module:function", ...]}: what to install once that module is imported, as the core's
            start-up hook does for its own adapters (adapters/autoinstall/sitecustomize.py)
  nodes     (optional) the platform's nodes for its boundaries, in data/nodes.json's form; a DLC's boundaries start
            with "dlc:<name>." and its node ids may not be the core's
  facts     (optional) the vocabulary names its checks decide (entail/facts.py); a DLC that only repairs has none
The core looks the DLCs up once per process, and only when entail is on (ENTAIL=load or debug) and ENTAIL_DLC is not
"off" (the platform, which runs apart from the engines, looks them up for their nodes). A DLC whose range does not hold
the core's version is not attached, and that is said once. The core installs every entry itself (install): an entry
that raises is recorded at dlc:<name>.install - as unknown for the DLC's first fact, or said when it decides none -
with the exception, and the program goes on; an entry that failed twice in a process is left out for the rest of it;
debug mode raises, as it does for the core's own adapters (principle 12). A DLC reports through the core (load.say, load.enforce, decisions) and does not
change the core's rules.
"""
import importlib
import os
import re
from typing import Dict, List, Optional, Tuple

GROUP = "entail.dlc"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_FOUND: Optional[List[dict]] = None
_FAILED: Dict[str, int] = {}     # entry -> failures in this process


def _numbers(text: str) -> Tuple[int, ...]:
    """1.3.0 -> (1, 3, 0); what follows the numbers (rc1, .dev0, +local) is not compared."""
    out = []
    for part in str(text).strip().split("."):
        m = re.match(r"\d+", part)
        if not m:
            break
        out.append(int(m.group()))
    return tuple(out)


def version_ok(requires: str, version: str) -> bool:
    """Whether `version` meets `requires`: comparisons (>=, <=, >, <, ==, !=) joined by commas; empty holds all."""
    have = _numbers(version)
    for clause in filter(None, (c.strip() for c in str(requires or "").split(","))):
        m = re.match(r"^(>=|<=|==|!=|>|<)\s*([0-9][0-9A-Za-z.+-]*)$", clause)
        if not m:
            return False
        op, want = m.group(1), _numbers(m.group(2))
        width = max(len(have), len(want))
        a, b = have + (0,) * (width - len(have)), want + (0,) * (width - len(want))
        if not {">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b, "==": a == b, "!=": a != b}[op]:
            return False
    return True


def _entry_points():
    from importlib import metadata

    try:
        return list(metadata.entry_points(group=GROUP))
    except TypeError:          # Python 3.10's selectable form is there; older dict form, kept for safety
        return list(metadata.entry_points().get(GROUP, []))


def _read(ep, core_version: str) -> dict:
    info = {"entry_point": ep.name, "dist": getattr(getattr(ep, "dist", None), "name", None), "attached": False}
    try:
        obj = ep.load()
    except Exception as e:  # noqa: BLE001 - a DLC that cannot be imported is reported, never raised into the program
        info["why"] = f"could not be loaded ({type(e).__name__}: {e})"
        return info
    name = getattr(obj, "name", None)
    info.update(name=name, version=str(getattr(obj, "version", "?")), requires=str(getattr(obj, "requires", "")),
                engines=dict(getattr(obj, "engines", {}) or {}), facts=list(getattr(obj, "facts", ()) or ()))
    targets = getattr(obj, "targets", {}) or {}
    nodes = list(getattr(obj, "nodes", []) or [])
    if not isinstance(name, str) or not _NAME.match(name):
        info["why"] = f"its name {name!r} is not lower-case letters, digits, '_' and '-'"
    elif not isinstance(targets, dict) or not all(isinstance(v, (list, tuple)) and all(isinstance(e, str) for e in v)
                                                  for v in targets.values()):
        info["why"] = "its targets are not {module: [\"package.module:function\", ...]}"
    elif not all(isinstance(f, str) and f in _vocabulary() for f in info["facts"]):
        info["why"] = f"its facts {info['facts']} are not all names of entail's vocabulary"
    elif not version_ok(info["requires"], core_version):
        info["why"] = f"it needs entail {info['requires']}, this is {core_version}"
    else:
        info.update(attached=True, why="", targets={m: list(v) for m, v in targets.items()}, nodes=nodes)
    return info


def _vocabulary():
    from .facts import VOCABULARY

    return VOCABULARY


def found(refresh: bool = False) -> List[dict]:
    """The DLCs installed in this environment, each with whether it is attached and, if not, why. Each DLC that is
    not attached is said once, when this first runs in a process."""
    global _FOUND
    if _FOUND is not None and not refresh:
        return _FOUND
    from . import __version__

    out, seen = [], set()
    if (os.environ.get("ENTAIL_DLC") or "").strip().lower() != "off":
        for ep in _entry_points():
            info = _read(ep, __version__)
            if info.get("attached") and info["name"] in seen:
                info.update(attached=False, why=f"another DLC is already attached as {info['name']!r}")
            if info.get("attached"):
                seen.add(info["name"])
            out.append(info)
    _FOUND = out
    for info in out:
        if not info["attached"]:
            _say(f"dlc:{info.get('name') or info['entry_point']}", f"not attached: {info['why']}")
    return out


def targets() -> Dict[str, List[Tuple[str, str]]]:
    """{module: [(entry, DLC name), ...]} of the attached DLCs, for the start-up hook's table."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    for info in found():
        if info["attached"]:
            for module, entries in info["targets"].items():
                out.setdefault(module, []).extend((e, info["name"]) for e in entries)
    return out


def nodes(core_ids=()) -> List[dict]:
    """The attached DLCs' nodes (data/nodes.json form). A node without an id or patterns, or whose id another node
    already has, is left out and said once."""
    out, ids = [], set(core_ids)
    for info in found():
        if not info["attached"]:
            continue
        for n in info["nodes"]:
            nid = n.get("id") if isinstance(n, dict) else None
            if not nid or not isinstance(n.get("patterns"), list) or nid in ids:
                _say(f"dlc:{info['name']}", f"a node was left out ({nid!r}: no id or patterns, or the id is taken)")
                continue
            ids.add(nid)
            out.append({**n, "dlc": info["name"]})
    return out


def install(name: str, entry: str) -> int:
    """Install one entry of DLC `name` ("package.module:function"; the function defaults to install), the core's way:
    an exception is recorded once as unknown at dlc:<name>.install and the program goes on; after two failures the
    entry is left out for this process; debug mode raises."""
    if _FAILED.get(entry, 0) >= 2:
        return 0
    try:
        module, _, func = entry.partition(":")
        return getattr(importlib.import_module(module), func or "install")() or 0
    except Exception as e:  # noqa: BLE001 - principle 12: a DLC that cannot install never breaks the program
        from . import core, load

        if core.mode() == "debug":
            raise
        _FAILED[entry] = _FAILED.get(entry, 0) + 1
        why = (f"the DLC {name}'s entry {entry} failed ({type(e).__name__}: {e}); the program goes on without it"
               + ("" if _FAILED[entry] < 2 else ", and it is left out for the rest of this process"))
        facts = next((i.get("facts") for i in (_FOUND or []) if i.get("name") == name), None) or []
        try:
            if facts:
                load.enforce([load.cannot_check(f"dlc:{name}.install", entry, facts[0], why)])
            else:
                _say(f"dlc:{name}.install", why)
        except Exception:  # noqa: BLE001 - the report itself must not break the program either
            print(f"[entail] dlc:{name}.install: {why}", flush=True)
        return 0


def _say(where: str, text: str) -> None:
    try:
        from . import load

        load.say(where, text)
    except Exception:  # noqa: BLE001
        print(f"[entail] {where}: {text}", flush=True)


def reset() -> None:
    """Forget what was found and failed (tests)."""
    global _FOUND
    _FOUND = None
    _FAILED.clear()
