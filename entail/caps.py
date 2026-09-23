"""caps: what each consumer honours, kept as data with evidence (LIBRARY_DESIGN.md 4.4; ROADMAP M3.1).

A consumer is anything that acts on a fact: an attention backend, a kernel, a loader, a sampler. It is named
engine.role.name, e.g. "sglang.attention.flashinfer"; engine.role ("sglang.attention") is its group. The table lives
in data/caps.json, one row per consumer and fact field:

  {"consumer": "sglang.attention.flashinfer", "fact": "ModelProps.softcap", "honours": false,
   "evidence": "measured", "version": "sglang 0.5.20", "ref": "sweep/results/...json"}

What the table is for, and what it is not:
  - `uses` says what a consumer will actually use of a declared value: the fields it honours keep the declared
    value, the fields it drops become None. That is the `chosen` side of a contract (contracts.decide compares).
  - A consumer or a field missing from the table is unknown, never honouring (LIBRARY_DESIGN.md 7).
  - `route` names a consumer of the same group that honours every declared field, with measured evidence only:
    a resolution sends the value to a consumer that has been seen to honour it, never to one we only believe does
    (the table has been wrong from code reading before: vLLM FLEX_ATTENTION, sweep/RESULTS.md).
  - The rows are checked with data by `entail probe` (probes.py): run with the fact bound and removed; identical
    output means the consumer ignores it, and two control runs must agree first.

Pure Python; no engine is imported here.
"""
import json
import os
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

from .facts import VOCAB_VERSION, Certainty, Fact, Source, vocabulary_class

EVIDENCE = ("measured", "code", "documented")
# How sure a consumer's use is, by the weakest evidence behind it (Certainty, LIBRARY_DESIGN.md principle 3).
CERTAINTY = {"measured": Certainty.VERIFIED, "documented": Certainty.DECLARED, "code": Certainty.INFERRED}
_ORDER = ("code", "documented", "measured")   # weakest first
SCHEMA = 1
DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "caps.json")


def group_of(consumer: str) -> str:
    return consumer.rsplit(".", 1)[0]


def _fact_field(text):
    """("ModelProps", "softcap") from "ModelProps.softcap", checked against the vocabulary."""
    if not isinstance(text, str) or text.count(".") != 1:
        raise ValueError(f"caps: fact must be written Name.field, got {text!r}")
    name, fld = text.split(".")
    cls = vocabulary_class(name)   # raises for a name outside the vocabulary
    if fld not in {f.name for f in fields(cls)}:
        raise ValueError(f"caps: {name} has no field {fld!r}; its fields are {[f.name for f in fields(cls)]}")
    return name, fld


# A value that satisfies a class's required fields, so that one field can be checked on its own.
_PLACEHOLDER = {"Layout": {"kind": "dense"}, "Quantized": {"dtype": "float32"}, "Positions": {"frame": "absolute"},
                "Rotary": {"rope_type": "default"},
                "Prediction": {"kind": "eps"}, "LatentScale": {"scale": 1.0}, "Reduction": {"state": "R"},
                "Epoch": {"version": 0}, "Assumed": {"conditions": ()}, "Origin": {"setting": "s", "came_from": "user"}}


def _check_value(name, fld, value):
    """A value for one field, checked by the vocabulary class itself (closed sets, types)."""
    kw = dict(_PLACEHOLDER.get(name, {}))
    kw[fld] = tuple(value) if isinstance(value, list) else value
    try:
        vocabulary_class(name)(**kw)
    except (ValueError, TypeError) as e:
        raise ValueError(f"caps: {name}.{fld} = {value!r} is not a value of the vocabulary: {e}") from None


@dataclass(frozen=True)
class Capability:
    consumer: str   # e.g. "sglang.attention.flashinfer"
    fact: str       # vocabulary name and field, e.g. "ModelProps.softcap"
    honours: bool
    evidence: str   # one of EVIDENCE
    ref: str        # the result file, commit or document that shows it
    version: str = ""   # the engine version the evidence was taken on
    note: str = ""
    reads: object = None   # with honours false: the value the consumer uses instead, whatever is declared
    #                        (e.g. a kernel that reads every scale as "ue8m0"); None: it runs without the field

    def __post_init__(self):
        if not isinstance(self.consumer, str) or self.consumer.count(".") < 2 or not all(self.consumer.split(".")):
            raise ValueError(f"caps: consumer must be named engine.role.name, got {self.consumer!r}")
        name, fld = _fact_field(self.fact)
        if not isinstance(self.honours, bool):
            raise ValueError(f"caps: {self.consumer} {self.fact}: honours must be true or false, got {self.honours!r}")
        if self.reads is not None:
            if self.honours:
                raise ValueError(f"caps: {self.consumer} {self.fact}: 'reads' is for a consumer that does not honour "
                                 f"the declaration")
            _check_value(name, fld, self.reads)
        if self.evidence not in EVIDENCE:
            raise ValueError(f"caps: {self.consumer} {self.fact}: evidence {self.evidence!r} is not one of "
                             f"{list(EVIDENCE)}")
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise ValueError(f"caps: {self.consumer} {self.fact}: every row needs a ref (where the evidence is)")

    @property
    def name(self):
        return self.fact.split(".")[0]

    @property
    def field(self):
        return self.fact.split(".")[1]


@dataclass(frozen=True)
class Table:
    rows: Tuple[Capability, ...]
    prefer: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()   # (group, consumer names in order)
    path: str = ""

    def preferred(self, group: str) -> Tuple[str, ...]:
        return next((names for g, names in self.prefer if g == group), ())


def from_rows(rows, prefer=None, path="") -> Table:
    """A table from row dicts (or Capability objects), checked: known fields only, no duplicate cell, and every
    preferred consumer present in the table."""
    built, seen = [], set()
    allowed = {f.name for f in fields(Capability)}
    for r in rows:
        if not isinstance(r, Capability):
            extra = set(r) - allowed
            if extra:
                raise ValueError(f"caps: unknown row fields {sorted(extra)} in {r!r}")
            r = Capability(**r)
        if (r.consumer, r.fact) in seen:
            raise ValueError(f"caps: {r.consumer} {r.fact} appears twice")
        seen.add((r.consumer, r.fact))
        built.append(r)
    consumers = {r.consumer for r in built}
    pref = []
    for group, names in (prefer or {}).items():
        for n in names:
            if f"{group}.{n}" not in consumers:
                raise ValueError(f"caps: prefer lists {group}.{n}, which has no row")
        pref.append((group, tuple(names)))
    return Table(tuple(built), tuple(pref), path)


def load_table(path: Optional[str] = None) -> Table:
    """The capability table from a JSON file (default: the packaged data/caps.json)."""
    path = path or DEFAULT_PATH
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema") != SCHEMA:
        raise ValueError(f"caps: {path}: schema {data.get('schema')!r}, this library reads {SCHEMA}")
    if data.get("vocab_version") != VOCAB_VERSION:
        raise ValueError(f"caps: {path}: written for vocabulary v{data.get('vocab_version')}, this library reads "
                         f"v{VOCAB_VERSION}")
    return from_rows(data.get("rows", []), data.get("prefer", {}), path)


_TABLE = None


def default_table() -> Table:
    global _TABLE
    if _TABLE is None:
        _TABLE = load_table()
    return _TABLE


def lookup(table: Table, consumer: str, fact: str) -> Optional[Capability]:
    """The entry, or None - which the contracts read as an unknown consumer, not as a pass."""
    return next((r for r in table.rows if r.consumer == consumer and r.fact == fact), None)


def consumers(table: Table, group: str) -> List[str]:
    out = []
    for r in table.rows:
        if group_of(r.consumer) == group and r.consumer not in out:
            out.append(r.consumer)
    return out


def consumed(table: Table, group: str) -> Tuple[str, ...]:
    """The fact fields ("ModelProps.softcap" ...) the consumers of a group act on: the ones its rows mention."""
    out = []
    for r in table.rows:
        if group_of(r.consumer) == group and r.fact not in out:
            out.append(r.fact)
    return tuple(out)


def project(value, wanted: Tuple[str, ...]):
    """The part of a fact value a group acts on: the listed fields kept, the rest None. None when nothing is left."""
    if value is None:
        return None
    name = type(value).__name__
    keep = {f.split(".")[1] for f in wanted if f.split(".")[0] == name}
    required = set(_PLACEHOLDER.get(name, {}))   # e.g. Layout.kind: what the value is, kept whatever is wanted
    kept = {f.name: getattr(value, f.name) for f in fields(value) if f.name in keep | required}
    if all(v is None for k, v in kept.items() if k not in required):
        return None
    return type(value)(**{f.name: kept.get(f.name) for f in fields(value)})


@dataclass(frozen=True)
class Use:
    """What a consumer will use of a declared value (see `uses`)."""
    value: Optional[object]            # None when a field is unknown and none is known to be dropped
    dropped: Tuple[str, ...]           # declared fields the consumer runs without ("ModelProps.softcap")
    unknown: Tuple[str, ...]           # declared fields the table has no row for
    rows: Tuple[Capability, ...]       # the rows used

    @property
    def evidence(self):
        return min((r.evidence for r in self.rows), key=_ORDER.index) if self.rows else None


def uses(table: Table, consumer: str, declared) -> Use:
    """What `consumer` will actually use of `declared` (a fact value, e.g. ModelProps(softcap=50.0)).

    Fields the declaration leaves open are not looked at. A field the consumer honours keeps the declared value; a
    field it drops becomes None, or the value it reads instead (`reads`). A field with no row is unknown: if nothing
    is known to differ the whole use is unknown (value None); if something is, the use is known to differ and the
    unknown fields count as not honoured (a resolution then needs a consumer measured to honour all of them).
    Required fields of the class (Layout.kind ...) that have no row keep the declared value: they name what the
    value is, not something a consumer can drop."""
    name = type(declared).__name__
    rows, dropped, unknown, kept = [], [], [], {}
    for f in fields(declared):
        v = getattr(declared, f.name)
        if v is None:
            continue
        cell = lookup(table, consumer, f"{name}.{f.name}")
        if cell is None:
            if f.name in _PLACEHOLDER.get(name, {}):
                kept[f.name] = v
                continue
            unknown.append(f"{name}.{f.name}")
            continue
        rows.append(cell)
        if cell.honours:
            kept[f.name] = v
        else:
            if cell.reads is not None:
                kept[f.name] = tuple(cell.reads) if isinstance(cell.reads, list) else cell.reads
            if kept.get(f.name) != v:
                dropped.append(f"{name}.{f.name}")
    if unknown and not dropped:
        return Use(None, (), tuple(unknown), tuple(rows))
    value = type(declared)(**{f.name: kept.get(f.name) for f in fields(declared)})
    return Use(value, tuple(dropped), tuple(unknown), tuple(rows))


def chosen_fact(table: Table, consumer: str, declared_fact: Fact) -> Fact:
    """`uses` as a Fact for contracts.decide: source "engine", the consumer and the evidence in its address, and the
    certainty of the weakest evidence (unknown when the table cannot say)."""
    use = uses(table, consumer, declared_fact.value)
    if not use.rows and not use.unknown:
        raise ValueError(f"caps: the declared {declared_fact.name} fills no field; project it first")
    parts = [f"{r.field}: {'honours' if r.honours else ('reads ' + str(r.reads)) if r.reads is not None else 'drops'}"
             f" ({r.evidence})" for r in use.rows]
    parts += [f"{u.split('.')[1]}: not in the capability table" for u in use.unknown]
    where = f"{consumer} [{'; '.join(parts)}]"
    if use.value is None:
        return Fact(declared_fact.name, None, Source("engine", where), Certainty.UNKNOWN)
    certainty = CERTAINTY[use.evidence]
    if use.unknown and _STRENGTH.index(certainty) > _STRENGTH.index(Certainty.INFERRED):
        certainty = Certainty.INFERRED
    return Fact(declared_fact.name, use.value, Source("engine", where), certainty)


_STRENGTH = (Certainty.UNKNOWN, Certainty.DEFAULTED, Certainty.INFERRED, Certainty.DECLARED, Certainty.VERIFIED)


def route(table: Table, group: str, declared, exclude: Tuple[str, ...] = (), measured_only: bool = True
          ) -> Optional[str]:
    """The first consumer of `group` (in the table's preferred order) that honours every field `declared` fills,
    with measured evidence for each. Returns its short name ("triton"), or None."""
    name = type(declared).__name__
    wanted = [f"{name}.{f.name}" for f in fields(declared) if getattr(declared, f.name) is not None]
    for short in table.preferred(group):
        consumer = f"{group}.{short}"
        if short in exclude or consumer in exclude:
            continue
        cells = [lookup(table, consumer, w) for w in wanted]
        if all(c is not None and c.honours and (c.evidence == "measured" or not measured_only) for c in cells):
            return short
    return None


def as_dict(table: Table) -> Dict[str, Dict[str, Capability]]:
    """consumer -> {fact: Capability}: a view for printing."""
    out: Dict[str, Dict[str, Capability]] = {}
    for r in table.rows:
        out.setdefault(r.consumer, {})[r.fact] = r
    return out


def probe(engine: str, consumer: str, fact: str, model: str, **kw):
    """Check one entry against the data (`entail probe`); see probes.py."""
    from . import probes
    return probes.probe(engine, consumer, fact, model, **kw)
