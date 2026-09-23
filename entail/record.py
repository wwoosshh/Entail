"""record: the transfer ledger, and locating where meaning broke (LIBRARY_DESIGN.md 4.9, 12). Ledger in M1.2,
locating in M7.1.

Every decision is kept with its source chain, and printed as one line that names the fact, the declared value and
where it came from, the consumer and what it uses, the rule, the verdict, and what was changed. From the ledger the
library can later say where a problem lies (THEORY.md 2.1; the researcher's words: if meaning is carried exactly,
the place where it broke is the problem area):
  - meaning broke at a boundary                          -> that boundary
  - every checked boundary kept its meaning, output wrong -> not the plumbing: inside a layer (the model itself,
                                                            a compiler, a kernel, the hardware)
  - some boundaries could not be checked                 -> they and the layers beside them stay suspect, so the
                                                            precision depends on how densely meaning is checked (S1)
Diagnosis is secondary: it is what preservation makes possible, not a separate bug hunt.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Localization:
    broken_at: Optional[str]        # the boundary where meaning broke, if any
    all_intact: bool                # every checked boundary passed or was resolved
    unchecked: Tuple[str, ...]      # boundaries reported as "could not check"
    suspects: Tuple[str, ...]       # the layers where the fault must lie


def _shown(fact):
    value = "unknown" if fact.value is None else str(fact.value)
    return f"{value} ({fact.source}, {fact.certainty.value})"


def line(decision) -> str:
    """One line for one Decision."""
    d, consumer = decision, decision.contract.consumer
    declared = f"declared {_shown(d.declared)}" if d.declared is not None else "nothing declared"
    used = f"{consumer} uses {_shown(d.chosen)}" if d.chosen is not None else f"what {consumer} uses is unknown"
    text = f"[entail] {d.verdict.value} at {d.contract.boundary}: {d.name} {declared}; {used}; rule: {d.rule}"
    if d.resolution:
        text += f"; changed: {d.resolution}"
    if d.observed is not None:
        text += f"; the data shows {_shown(d.observed)}"
    if d.conflict:
        text += "; sources disagreed: " + ", ".join(_shown(f) for f in d.conflict)
    if d.blocking:
        text += "; stops here"
    return text


def _fact_json(fact):
    if fact is None:
        return None
    return {"name": fact.name, "kind": fact.kind, "value": None if fact.value is None else str(fact.value),
            "source": {"kind": fact.source.kind, "where": fact.source.where}, "certainty": fact.certainty.value}


@dataclass
class Ledger:
    decisions: List[object] = field(default_factory=list)   # contracts.Decision

    def add(self, decision) -> None:
        self.decisions.append(decision)

    def extend(self, decisions) -> None:
        for d in decisions:
            self.add(d)

    def blocking(self) -> List[object]:
        """Decisions that must stop the run before any output."""
        return [d for d in self.decisions if d.blocking]

    def lines(self) -> List[str]:
        return [line(d) for d in self.decisions]

    def to_json(self) -> dict:
        return {"decisions": [{
            "boundary": d.contract.boundary, "consumer": d.contract.consumer, "name": d.name, "verdict": d.verdict.value,
            "blocking": d.blocking, "rule": d.rule, "resolution": d.resolution, "handle": d.handle,
            "declared": _fact_json(d.declared), "chosen": _fact_json(d.chosen), "observed": _fact_json(d.observed),
            "conflict": [_fact_json(f) for f in d.conflict]} for d in self.decisions]}

    def locate(self) -> Localization:
        raise NotImplementedError("M7.1: locating where meaning broke")
