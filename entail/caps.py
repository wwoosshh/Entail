"""caps: what each consumer honours, kept as data with evidence (LIBRARY_DESIGN.md 4.4). Built in M3.1.

A consumer is anything that acts on a fact: an attention backend, a kernel, a loader, a sampler. Example entry:
"sglang flashinfer attention does not honour ModelProps.softcap". Every entry says how that is known.

Must:
  - mark every entry with its evidence: "measured" (a probe), "code" (read in the source) or "documented"
  - be checkable by `probe`: run with the fact bound and unbound; identical outputs mean the consumer ignores it,
    and two control runs must agree first (the method of entail/audits/cap_probe.py)
  - treat a consumer missing from the table as unknown, never as honouring (LIBRARY_DESIGN.md 7)
"""
from dataclasses import dataclass
from typing import List, Optional

EVIDENCE = ("measured", "code", "documented")


@dataclass(frozen=True)
class Capability:
    consumer: str   # e.g. "sglang.attention.flashinfer"
    fact: str       # vocabulary name and field, e.g. "ModelProps.softcap"
    honours: bool
    evidence: str   # one of EVIDENCE
    ref: str        # the result file, commit or document that shows it


def load_table(path: str) -> List[Capability]:
    raise NotImplementedError("M3.1: capability tables as data files")


def lookup(table: List[Capability], consumer: str, fact: str) -> Optional[Capability]:
    """The entry, or None - which the contracts read as an unknown consumer, not as a pass."""
    raise NotImplementedError("M3.1")


def probe(engine: str, consumer: str, fact: str) -> Capability:
    """Check one entry against the data. `entail probe`."""
    raise NotImplementedError("M3.1: port audits/cap_probe.py")
