"""contracts: compare what was declared with what a consumer chose, and decide (LIBRARY_DESIGN.md 4.5, 7). Built in M1.2.

The verdicts (LIBRARY_DESIGN.md 7):
  declared, consumer agrees                         -> PASS
  declared, consumer differs, a resolution exists   -> RESOLVED (route to a consumer that honours it, convert, or
                                                       recompute; one line in the ledger)
  declared, consumer differs, no resolution         -> REFUSED, before any output is produced
  declared, consumer not in the capability table    -> UNKNOWN (consumer)
  not declared                                      -> UNKNOWN (declaration): `require` for meaning-changing facts,
                                                       `report` for the rest (policy)
The rules live here and only here. Adapters supply the consumer's choice and the handles that carry out a
resolution; they never decide (principle 8). The earlier one-fact `contract.py` (reconcile) is folded in here in M1.2.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

from .facts import Fact


class Verdict(str, Enum):
    PASS = "pass"
    RESOLVED = "resolved"
    REFUSED = "refused"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Contract:
    """What one boundary needs.

    boundary          where, e.g. "load:sglang.attention_backend" or "container:vllm.allocate_slots"
    needs             vocabulary names the consumer at this boundary must honour
    meaning_changing  the subset whose absence may not be filled by a default (policy `require`)
    """
    boundary: str
    needs: Tuple[str, ...]
    meaning_changing: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Decision:
    """One verdict, with everything the ledger needs to explain it: the blame requirement of LIBRARY_DESIGN.md 7."""
    contract: Contract
    name: str
    declared: Optional[Fact]
    chosen: Optional[Fact]
    verdict: Verdict
    rule: str
    resolution: Optional[str] = None   # what was changed, when RESOLVED


class Resolution(Protocol):
    """One way to repair a mismatch for one vocabulary name. The adapter's handle carries it out."""
    name: str
    handle: str   # the adapter handle that performs it

    def applies(self, declared: Fact, chosen: Fact) -> bool:
        ...

    def describe(self, declared: Fact, chosen: Fact) -> str:
        ...


RESOLUTIONS: Dict[str, List[Resolution]] = {}   # vocabulary name -> resolutions; filled in M1.2 and later milestones


def decide(contract: Contract, declared: Dict[str, Fact], chosen: Dict[str, Fact], caps, policy) -> List[Decision]:
    """Apply the verdict table to every name the contract needs."""
    raise NotImplementedError("M1.2: the verdict table, resolutions registry and policy")
