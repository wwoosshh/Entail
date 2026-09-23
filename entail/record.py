"""record: the transfer ledger, and locating where meaning broke (LIBRARY_DESIGN.md 4.9, 12). Built in M1.2 and M7.1.

Every decision is kept with its source chain. From the ledger the library can say where a problem lies
(THEORY.md 2.1, the researcher's words: "if meaning is carried exactly, the place where it broke is the problem"):
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


@dataclass
class Ledger:
    decisions: List[object] = field(default_factory=list)   # contracts.Decision

    def add(self, decision) -> None:
        raise NotImplementedError("M1.2: the ledger")

    def lines(self) -> List[str]:
        """One line per decision, e.g.
        `entail: prediction_type=v (file modelspec) vs sampler eps -> resolved: sampler set to v`"""
        raise NotImplementedError("M1.2: one-line reports")

    def to_json(self) -> dict:
        raise NotImplementedError("M1.2: JSON reports")

    def locate(self) -> Localization:
        raise NotImplementedError("M7.1: locating where meaning broke")
