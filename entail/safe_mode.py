"""safe_mode: the two safety modes (LIBRARY_DESIGN.md 13.6; ROADMAP product track P3; THEORY.md 2.1 item 3; the
researcher's words, 2026-09-27: "일종의 윈도우의 안전모드부팅과 같은 상태", "1번은 나눠서 진행해").

  explicit safe mode   ENTAIL_SAFE=all: every optimization the engine declares does not change results
                       (data/safe_mode.json: CUDA graphs, the prefix cache, speculative decoding, custom kernels) is
                       turned off, so a fault that stays is outside them and one that goes is inside them
  selective safe path  ENTAIL_SAFE=auto, the default: when the engine's own paths disagree (path_contract), the
                       optimizations that disagreement points at become this configuration's candidates; each next
                       start turns the next one off, until the paths agree (that one is the cause, and stays off) or
                       none is left (the cause is outside them: they go back on, and that is reported once). Nothing
                       stops: the run that found the disagreement goes on as it was; the next start changes.
  off                  ENTAIL_SAFE=off: neither.
The mode comes from ENTAIL_SAFE, else from safe_mode.json in the log folder ({"mode": ...}; the platform writes it),
else "auto". What a mode turned off is a decision like any other (resolved, fact SafeMode, at start:<engine>.safe_mode)
and what it found is said there too, so the platform shows both on the engine's self-check node. The selective safe
path keeps its state per configuration in safe_paths.json in the log folder. Adapters read which of the options their
engine was given and turn them with their handle; the rules are here.
"""
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from . import record

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "safe_mode.json")
MODES = ("off", "auto", "all")
STORE = "safe_paths.json"
_TABLE = None
LAST: Dict[str, dict] = {}   # engine -> what this process's start asked for and turned off (the path check reads it)


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        with open(DATA, encoding="utf-8") as f:
            _TABLE = json.load(f)
    return _TABLE


def features(engine: str) -> Dict[str, Tuple[str, object]]:
    """feature -> (the engine option that turns it off, the value that does), for one engine."""
    return {f: (spec["option"], spec["safe"]) for f, spec in table()["engines"].get(engine, {}).items()}


def mode() -> str:
    """off, auto or all: ENTAIL_SAFE, else the log folder's safe_mode.json, else auto."""
    v = (os.environ.get("ENTAIL_SAFE") or "").strip().lower()
    if v in MODES:
        return v
    folder = record.log_dir()
    if folder:
        try:
            with open(os.path.join(folder, "safe_mode.json"), encoding="utf-8") as f:
                m = json.load(f).get("mode")
            if m in MODES:
                return m
        except (OSError, ValueError, AttributeError):
            pass
    return "auto"


def config_key(engine: str, parts: dict) -> str:
    """The key of one engine configuration (engine, version, model, dtype, ...): what the selective safe path keeps
    its state under. The optimizations themselves are not part of it - they are what it turns."""
    text = json.dumps({"engine": engine, **parts}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _store_path() -> Optional[str]:
    folder = record.log_dir()
    return os.path.join(folder, STORE) if folder else None


def load_store() -> dict:
    path = _store_path()
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_store(store: dict) -> None:
    path = _store_path()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        pass   # a store that cannot be written only means the next start does not change (principle 12)


def plan(engine: str, key: str, enabled: Dict[str, bool]) -> List[Tuple[str, str]]:
    """What to turn off at this start: [(feature, why)], why "all" (explicit) or "path" (selective)."""
    m = mode()
    if m == "off":
        return []
    names = features(engine)
    if m == "all":
        return [(f, "all") for f in names if enabled.get(f)]
    entry = load_store().get(key)
    if not entry:
        return []
    if entry.get("status") == "found":
        return [(f, "path") for f in entry.get("off", []) if enabled.get(f)]
    if entry.get("status") == "searching":
        left = [f for f in entry.get("candidates", []) if f not in entry.get("tried", [])]
        return [(left[0], "path")] if left and enabled.get(left[0]) else []
    return []


def decisions(engine: str, boundary: str, consumer: str, items: Sequence[Tuple[str, str]], where: str) -> list:
    """One resolved decision per optimization turned off, for the adapter's handle `safe_mode` (target: the option
    and the value that turns it off)."""
    from .contracts import RULES, Contract, Decision, Verdict
    from .facts import Certainty, Fact, SafeMode, Source

    names = features(engine)
    contract = Contract(boundary, consumer, ("SafeMode",), ("SafeMode",))
    out = []
    for feature, why in items:
        option, safe = names[feature]
        source = Source("user", "ENTAIL_SAFE=all") if why == "all" else \
            Source("probe", f"{STORE}: the engine's paths disagreed with {feature} on")
        declared = Fact("SafeMode", SafeMode(feature, False, why), source, Certainty.DECLARED)
        chosen = Fact("SafeMode", SafeMode(feature, True), Source("engine", where), Certainty.VERIFIED)
        note = ("the explicit safe mode: a fault that stays with every such optimization off is outside them"
                if why == "all" else "the selective safe path: this configuration's paths disagreed with it on")
        out.append(Decision(contract, "SafeMode", Verdict.RESOLVED, RULES["safe_mode" if why == "all" else "safe_path"],
                            declared=declared, chosen=chosen, resolution=f"{option} = {safe!r}", handle="safe_mode",
                            target=(option, safe), note=note))
    return out


def started(engine: str, key: str, enabled: Dict[str, bool], off: Sequence[str], model: str = "") -> None:
    """What this process's start asked for and what it turned off (the engine's path check reads it)."""
    LAST[engine] = {"key": key, "enabled": dict(enabled), "off": list(off), "model": model, "mode": mode()}


def _candidates(pairs: Sequence[str], on: Dict[str, bool]) -> List[str]:
    out = []
    for p in pairs:
        for f in table()["candidates"].get(p, []):
            if on.get(f) and f not in out:
                out.append(f)
    return out


def after_self_check(engine: str, disagreeing: Sequence[str], boundary: str, consumer: str, where: str) -> list:
    """The engine's paths came to `disagreeing` (the pairs that disagreed; empty: all agreed). Moves the selective
    safe path of this start's configuration on and says what it found; returns the decisions to record (a broken
    one when every candidate was tried in vain)."""
    from . import load

    ctx = LAST.get(engine)
    if not ctx:
        return []
    if ctx["mode"] == "all":
        if disagreeing:
            load.say(boundary, f"every optimization {engine} declares does not change results is off, and its paths "
                               f"still disagree ({', '.join(disagreeing)}): the cause is outside them")
        return []
    if ctx["mode"] != "auto":
        return []
    store = load_store()
    key, entry, out = ctx["key"], store.get(ctx["key"]), []
    if not disagreeing:
        if entry and entry.get("status") == "searching" and ctx["off"]:
            entry.update(status="found", off=list(ctx["off"]), updated=time.time())
            save_store(store)
            load.say(boundary, f"the paths agree with {', '.join(ctx['off'])} off: that is the cause in this "
                               f"configuration, which starts without it from now on")
        return out
    on_at_start = {f: bool(ctx["enabled"].get(f)) for f in features(engine)}
    if entry is None or entry.get("status") == "outside":
        cands = _candidates(disagreeing, on_at_start)
        if entry is not None and entry.get("status") == "outside":
            return out          # already reported: the cause is outside the candidates
        if not cands:
            store[key] = {"engine": engine, "model": ctx["model"], "pairs": list(disagreeing), "candidates": [],
                          "tried": [], "off": [], "status": "outside", "updated": time.time()}
            save_store(store)
            load.say(boundary, f"the paths disagree ({', '.join(disagreeing)}) and no optimization they point at is "
                               f"on: nothing to turn off")
            return out
        store[key] = {"engine": engine, "model": ctx["model"], "pairs": list(disagreeing), "candidates": cands,
                      "tried": [], "off": [], "status": "searching", "updated": time.time()}
        save_store(store)
        load.say(boundary, f"the paths disagree ({', '.join(disagreeing)}): the next start turns {cands[0]} off to "
                           f"find the cause (1 of {len(cands)} candidates)")
        return out
    if entry.get("status") == "found":        # the cause was off and the paths disagree again: search the others
        entry.update(status="searching", tried=list(entry.get("off", [])), off=[], updated=time.time())
    tried = list(entry.get("tried", []))
    for f in ctx["off"]:
        if f not in tried:
            tried.append(f)
    entry["tried"] = tried
    left = [f for f in entry.get("candidates", []) if f not in tried]
    if left:
        entry["updated"] = time.time()
        save_store(store)
        load.say(boundary, f"the paths still disagree with {', '.join(ctx['off']) or 'nothing'} off: the next start "
                           f"turns {left[0]} off ({len(tried) + 1} of {len(entry['candidates'])} candidates)")
        return out
    entry.update(status="outside", off=[], updated=time.time())
    save_store(store)
    from .contracts import RULES, Contract, Decision, Verdict
    from .facts import Certainty, Fact, SafeMode, Source

    first = entry["candidates"][0]
    out.append(Decision(Contract(boundary, consumer, ("SafeMode",), ("SafeMode",)), "SafeMode", Verdict.BROKEN,
                        RULES["safe_path_outside"],
                        declared=Fact("SafeMode", SafeMode(first, False, "path"), Source("probe", STORE),
                                      Certainty.DECLARED),
                        chosen=Fact("SafeMode", SafeMode(first, True), Source("engine", where), Certainty.VERIFIED),
                        note=f"tried {', '.join(tried)} off, one per start; the paths still disagree "
                             f"({', '.join(disagreeing)}), so the cause is outside them and they are on again"))
    return out
