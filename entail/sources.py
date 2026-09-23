"""sources: read what artifacts already declare, and turn it into facts (LIBRARY_DESIGN.md 4.2).

The re-investigation found that the meaning of a model is usually written down somewhere - config.json,
scheduler_config.json, safetensors metadata (ModelSpec), GGUF keys, quantization configs - but consumers often do not
read it and guess or default instead (THEORY.md 2.1). This module is the reading half of the fix.

Must:
  - give every fact its Source (which file, which key) and its Certainty
  - keep name aliases as data (ALIASES): `modelspec.prediction_type`, a `v_pred` key and `scheduler_config`
    `prediction_type` are one fact
  - never choose silently between sources that disagree: `pick` and `merge` return the conflicts, and the ledger
    records them
Must not:
  - look at what an engine chose (adapters do that) or decide anything (contracts do that)

`pick` and `merge` were built in M1 (the verdict table needs them); `read_all` and the readers in M2.1
(readers.py, with key names and value spellings in data/aliases.json).
"""
from dataclasses import dataclass, fields
from typing import Dict, List, Protocol, Sequence, Tuple

from . import readers as _readers
from .facts import Certainty, Fact, Source

# Which source wins when two disagree (LIBRARY_DESIGN.md 11). A conflict is always recorded. "boundary" (a code
# signature) ranks with the explicit declarations; kinds not listed ("engine", "data") rank last.
DEFAULT_PRECEDENCE = ("user", "manifest", "boundary", "file", "config", "probe", "default")

ALIASES = _readers.ALIASES
ReadResult = _readers.ReadResult


class Reader(Protocol):
    """One kind of artifact: hf_config, hf_template, diffusers, safetensors, gguf (readers.py)."""
    name: str

    def applies_to(self, path: str) -> bool:
        ...

    def read(self, path: str) -> "ReadResult":
        ...


READERS: List[Reader] = list(_readers.READERS)


@dataclass(frozen=True)
class Conflict:
    """Two or more sources said different things about one fact; `chosen` is what the precedence picked."""
    name: str
    facts: Tuple[Fact, ...]
    chosen: Fact


def compatible(a, b) -> bool:
    """Two values of one fact do not contradict: same class, and every field both of them fill is equal."""
    if type(a) is not type(b):
        return False
    return all(getattr(a, f.name) == getattr(b, f.name) for f in fields(a)
               if getattr(a, f.name) is not None and getattr(b, f.name) is not None)


def _rank(fact, precedence):
    kind = fact.source.kind
    return precedence.index(kind) if kind in precedence else len(precedence)


_STRENGTH = (Certainty.UNKNOWN, Certainty.DEFAULTED, Certainty.INFERRED, Certainty.DECLARED, Certainty.VERIFIED)


def _combine(ordered):
    """Compatible candidates, best first, combined field by field: a field comes from the best candidate that fills
    it, so nothing any source said is dropped. Every contributing source is named, and the certainty is the
    weakest of theirs (one inferred field makes the whole fact inferred)."""
    best = ordered[0]
    names = [f.name for f in fields(best.value)]
    filled, contributors = {}, []
    for fact in ordered:
        added = False
        for n in names:
            v = getattr(fact.value, n)
            if v is not None and filled.get(n) is None:
                filled[n], added = v, True
        if added:
            contributors.append(fact)
    if len(contributors) == 1:
        return contributors[0]
    value = type(best.value)(**{n: filled.get(n) for n in names})
    certainty = min((f.certainty for f in contributors), key=_STRENGTH.index)
    where = "; ".join(f"{f.source.kind}: {f.source.where}" for f in contributors)
    return Fact(best.name, value, Source(best.source.kind, where), certainty)


def pick(candidates, precedence=DEFAULT_PRECEDENCE):
    """What the sources say about one name, and the candidates involved in a disagreement.

    Compatible candidates are combined (see _combine), so a source that fills a field nobody else fills is never
    dropped. When candidates contradict each other, the precedence picks one and all of them are returned as the
    conflict. Returns (fact or None, conflict tuple)."""
    candidates = tuple(candidates)
    known = [f for f in candidates if f.certainty is not Certainty.UNKNOWN]
    if not known:
        return (candidates[0] if candidates else None), ()
    ordered = sorted(known, key=lambda f: _rank(f, precedence))   # stable: equal ranks keep their order
    if all(compatible(a.value, b.value) for a in ordered for b in ordered):
        return _combine(ordered), ()
    return ordered[0], tuple(known)


def read_all(path: str, manifest_dirs: Sequence[str] = ()) -> "ReadResult":
    """Every fact any reader finds in the artifact at `path` (a file or a model folder), plus the facts of a manifest
    for it when one is found in `manifest_dirs`. A reader that fails becomes a problem: reading never breaks the
    caller (principle 12)."""
    result = ReadResult()
    for reader in READERS:
        try:
            if not reader.applies_to(path):
                continue
            result.extend(reader.read(path))
        except Exception as e:  # noqa: BLE001 - reported, never raised
            result.problems.append(f"{reader.name}: {path}: {type(e).__name__}: {e}")
    if manifest_dirs:
        from . import manifest as _manifest
        try:
            m = _manifest.find(path, manifest_dirs)
        except Exception as e:  # noqa: BLE001
            result.problems.append(f"manifest: {path}: {type(e).__name__}: {e}")
            m = None
        if m is not None:
            result.facts.extend(m.facts)
    return result


def merge(facts: List[Fact], precedence=DEFAULT_PRECEDENCE):
    """One fact per vocabulary name, chosen by `precedence`, and the Conflicts. Returns (dict, list)."""
    by_name: Dict[str, List[Fact]] = {}
    for fact in facts:
        by_name.setdefault(fact.name, []).append(fact)
    chosen, conflicts = {}, []
    for name, group in by_name.items():
        best, conflict = pick(group, precedence)
        chosen[name] = best
        if conflict:
            conflicts.append(Conflict(name, conflict, best))
    return chosen, conflicts
