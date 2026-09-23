"""sources: read what artifacts already declare, and turn it into facts (LIBRARY_DESIGN.md 4.2). Built in M2.

The re-investigation found that the meaning of a model is usually written down somewhere - config.json,
scheduler_config.json, safetensors metadata (ModelSpec), GGUF keys, quantization configs - but consumers often do not
read it and guess or default instead (THEORY.md 2.1). This module is the reading half of the fix.

Must:
  - give every fact its Source (which file, which key) and its Certainty
  - keep name aliases as data (ALIASES): `modelspec.prediction_type`, a `v_pred` key and `scheduler_config`
    `prediction_type` are one fact
  - never choose silently between sources that disagree: `merge` returns the conflicts, and the ledger records them
Must not:
  - look at what an engine chose (adapters do that) or decide anything (contracts do that)
"""
from dataclasses import dataclass
from typing import List, Protocol, Tuple

from .facts import Fact

# Which source wins when two disagree (LIBRARY_DESIGN.md 11, decided 2026-09-23). A conflict is always recorded.
DEFAULT_PRECEDENCE = ("user", "manifest", "file", "config", "probe", "default")

# Other names for the same fact, as data: {vocabulary name: [(source kind, key or pattern), ...]}. Filled in M2.1.
ALIASES = {}


class Reader(Protocol):
    """One kind of artifact. Planned in M2.1: hf_config, safetensors_metadata, gguf, diffusers_configs,
    quantization_config."""
    name: str

    def applies_to(self, path: str) -> bool:
        ...

    def read(self, path: str) -> List[Fact]:
        ...


READERS: List[Reader] = []


@dataclass(frozen=True)
class Conflict:
    """Two or more sources said different things about one fact; `chosen` is what the precedence picked."""
    name: str
    facts: Tuple[Fact, ...]
    chosen: Fact


def read_all(path: str) -> List[Fact]:
    """Every fact that any reader finds in the artifact at `path` (a file or a model folder)."""
    raise NotImplementedError("M2.1: readers for HF configs, safetensors metadata, GGUF, diffusers, quantization")


def merge(facts: List[Fact], precedence=DEFAULT_PRECEDENCE):
    """One fact per vocabulary name, chosen by `precedence`, plus the list of Conflicts. Returns (facts, conflicts)."""
    raise NotImplementedError("M2.2: conflict detection and precedence")
