"""manifest: declaration files for artifacts that declare nothing (LIBRARY_DESIGN.md 4.3). Built in M2.3.

A model file often carries no declaration of what it means: of the researcher's 44 model files, 24 declared their
prediction type (reinvestigation/feasibility.md 9.3). A manifest supplies the missing declarations from outside,
the way a `.d.ts` file types a JavaScript library that has no types of its own.

Format: JSON, keyed by the artifact's SHA-256. Every fact carries its value, certainty and evidence.
Life cycle: `infer` writes a draft whose facts are `inferred`; a person reviews it; `pin` makes them `declared`.
Must not: silently override a declaration the artifact itself makes. A disagreement between the two is a Conflict.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .facts import Fact

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Manifest:
    sha256: str
    facts: Tuple[Fact, ...]
    evidence: Dict[str, List[str]] = field(default_factory=dict)  # vocabulary name -> result files, docs, probes
    pinned: bool = False


def load(path: str) -> Manifest:
    raise NotImplementedError("M2.3: manifest JSON schema and loader")


def find(artifact_path: str, search_dirs: List[str]) -> Optional[Manifest]:
    """The manifest whose sha256 matches the artifact, if any."""
    raise NotImplementedError("M2.3: lookup by file hash")


def infer(artifact_path: str) -> Manifest:
    """A draft from the artifact's own declarations, its configs and probes. Probe results are `inferred`."""
    raise NotImplementedError("M2.3: `entail infer`")


def pin(manifest: Manifest) -> Manifest:
    """After review: the reviewed facts become `declared`."""
    raise NotImplementedError("M2.3: pinning a reviewed manifest")
