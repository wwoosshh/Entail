"""signatures: what the boundaries entail wraps in engine code declare (LIBRARY_DESIGN.md 4.6; ROADMAP M4.2).

entail cannot put @boundary on an engine's source, but it owns the wrappers it installs there, so it can sign them.
Such a signature is data (data/signatures.json), one row per producer - e.g. a vLLM quantisation method, whose
process_weights_after_loading leaves each weight in the layout its apply reads:

  {"producer": "vllm.quant_method.Fp8PerTensorOnlineLinearMethod",
   "writes": "process_weights_after_loading", "reads": "apply", "value": "weight",
   "facts": {"Layout": {"kind": "strided", "dtype": "float8_e4m3fn", "orientation": "in_out",
                        "scale_granularity": "per_tensor"}},
   "moves": "identity", "evidence": "measured", "version": "0.30.0", "ref": "..."}

  producer   engine.role.name of the component that writes the value
  writes     its step that leaves the value so; `reads` the step that takes it so (often the same component's kernel)
  facts      vocabulary name -> fields, checked by the vocabulary class itself (closed sets)
  moves      where the step puts the values it was given (MOVES); a sample of values is checked against it
  evidence, version, ref   as in caps.json: measured (healthy runs and planted defects) or code (read, not run)

`not_values` names producers whose layers hold no such value (an embedding table, KV cache scales), each with the
reason, so that leaving them out is a statement rather than a silence. Declared facts come out as envelopes whose
source names the step, "boundary: <producer>.<writes>.writes.<value>" (the M4.1 spelling), so the facts a wrapper
attaches to a value say where they came from. The rules that compare a signature with the data are in load.py; this
module only reads the table. Pure Python; no engine is imported here.
"""
import json
import os
from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

from .caps import EVIDENCE
from .facts import ADDED_IN, READABLE_VERSIONS, VOCAB_VERSION, Certainty, Fact, Source, vocabulary_class

SCHEMA = 1
MOVES = frozenset({"identity"})   # a new kind of move needs the code that maps its indices (observe.moved)
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "signatures.json")
_FIELDS = ("producer", "writes", "reads", "value", "facts", "moves", "evidence", "version", "ref")


@dataclass(frozen=True)
class Signature:
    producer: str
    writes: str
    reads: str
    value: str
    facts: Tuple[Tuple[str, object], ...]   # (vocabulary name, fact value), sorted by name
    moves: Optional[str]
    evidence: str
    version: str
    ref: str

    def declared(self, name: str) -> Optional[Fact]:
        """What the producer's step declares it leaves in the value, as an envelope; None when it says nothing."""
        return self._fact(name, f"{self.producer}.{self.writes}.writes.{self.value}")

    def taken(self, name: str) -> Optional[Fact]:
        """What the reading step takes (the same declaration, from the consumer's side)."""
        return self._fact(name, f"{self.producer}.{self.reads}.takes.{self.value}")

    def _fact(self, name, where):
        value = dict(self.facts).get(name)
        return None if value is None else Fact(name, value, Source("boundary", where), Certainty.DECLARED)


@dataclass(frozen=True)
class Table:
    rows: Tuple[Signature, ...]
    not_values: Tuple[Tuple[str, str], ...] = ()   # (producer, why its layers hold no such value)
    path: str = ""

    def lookup(self, producer: str) -> Optional[Signature]:
        return next((r for r in self.rows if r.producer == producer), None)

    def why_not(self, producer: str) -> Optional[str]:
        return dict(self.not_values).get(producer)


def _value(name, data, vocab_version, where):
    cls = vocabulary_class(name)   # raises for a name outside the vocabulary
    known = {f.name for f in fields(cls)}
    if not isinstance(data, dict) or set(data) - known:
        raise ValueError(f"signatures: {where}: {name} takes the fields {sorted(known)}, got {data!r}")
    for fld in data:
        added = ADDED_IN.get((name, fld))
        if added is not None and vocab_version < added:
            raise ValueError(f"signatures: {where}: {name}.{fld} is not in vocabulary v{vocab_version} "
                             f"(added in v{added})")
    try:
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})
    except (TypeError, ValueError) as e:
        raise ValueError(f"signatures: {where}: {e}") from None


def from_rows(rows, not_values=None, vocab_version: int = VOCAB_VERSION, path: str = "") -> Table:
    """A table from row dicts, checked: known fields, a producer named engine.role.name once, values in the
    vocabulary, a known move and evidence kind, and a ref for every row."""
    built = []
    for r in rows:
        extra, missing = set(r) - set(_FIELDS), {"producer", "writes", "value", "facts", "evidence", "ref"} - set(r)
        if extra or missing:
            raise ValueError(f"signatures: row {r.get('producer')!r}: unknown fields {sorted(extra)}, missing "
                             f"{sorted(missing)}")
        producer = r["producer"]
        if not isinstance(producer, str) or producer.count(".") < 2 or not all(producer.split(".")):
            raise ValueError(f"signatures: producer must be named engine.role.name, got {producer!r}")
        if any(b.producer == producer for b in built):
            raise ValueError(f"signatures: {producer} appears twice")
        if not isinstance(r["facts"], dict) or not r["facts"]:
            raise ValueError(f"signatures: {producer}: facts must name at least one vocabulary fact")
        facts = tuple(sorted((n, _value(n, v, vocab_version, producer)) for n, v in r["facts"].items()))
        moves = r.get("moves")
        if moves is not None and moves not in MOVES:
            raise ValueError(f"signatures: {producer}: moves {moves!r} is not one of {sorted(MOVES)}")
        if r["evidence"] not in EVIDENCE:
            raise ValueError(f"signatures: {producer}: evidence {r['evidence']!r} is not one of {list(EVIDENCE)}")
        if not isinstance(r["ref"], str) or not r["ref"].strip():
            raise ValueError(f"signatures: {producer}: every row needs a ref (where the evidence is)")
        built.append(Signature(producer, r["writes"], r.get("reads") or r["writes"], r["value"], facts, moves,
                               r["evidence"], r.get("version", ""), r["ref"]))
    nv = tuple(sorted((not_values or {}).items()))
    for producer, why in nv:
        if any(b.producer == producer for b in built):
            raise ValueError(f"signatures: {producer} is both signed and listed as holding no value")
        if not isinstance(why, str) or not why.strip():
            raise ValueError(f"signatures: {producer}: say why its layers hold no value")
    return Table(tuple(built), nv, path)


def load_table(path: Optional[str] = None) -> Table:
    """The signatures from a JSON file (default: the packaged data/signatures.json)."""
    path = path or DEFAULT_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"signatures: {path}: schema {data.get('schema')!r}, this library reads {SCHEMA}")
    if data.get("vocab_version") not in READABLE_VERSIONS:
        raise ValueError(f"signatures: {path}: written for vocabulary v{data.get('vocab_version')}, this library "
                         f"reads v{', v'.join(str(v) for v in sorted(READABLE_VERSIONS))}")
    return from_rows(data.get("rows", []), data.get("not_values", {}), data["vocab_version"], path)


_TABLE = None


def default_table() -> Table:
    global _TABLE
    if _TABLE is None:
        _TABLE = load_table()
    return _TABLE
