"""manifest: declaration files for artifacts that declare nothing (LIBRARY_DESIGN.md 4.3; ROADMAP M2.3).

A model file often carries no declaration of what it means: of the researcher's 44 model files, 24 declared their
prediction type (reinvestigation/feasibility.md 9.3). A manifest supplies the missing declarations from outside,
the way a `.d.ts` file types a JavaScript library that has no types of its own.

Format (JSON, schema 1):
  {"schema": 1, "sha256": "<hash of the artifact>", "file": "<name, for people>", "pinned": false,
   "facts": [{"name": "Prediction", "value": {"kind": "v", "zsnr": null}, "certainty": "inferred",
              "evidence": ["..."]}, ...]}
  The key is the SHA-256 of the artifact file; for a model folder, of its config.json (or model_index.json).
  A fact whose value is null is a slot nobody has filled yet (certainty "unknown").
  "fingerprint" (M6.3, optional): the SHA-256 of the key file's size and its first and last 4 MiB. It only says
  which files a manifest could be for: `find` hashes a file in full only when some manifest's fingerprint matches it
  (or some manifest has none). Hashing a 6.9 GB checkpoint on every load cost 18 s (testbed/results/m63); files of one
  architecture have the same size, so the size alone does not tell them apart.

Life cycle: `infer` writes a draft from what the artifact itself declares, plus empty slots for the facts that
matter for its kind; a person reviews it and fills the slots; `pin` marks it reviewed and turns every filled
`inferred` fact into a `declared` one. Only a pinned manifest's facts count as declarations.
Must not: silently override what the artifact itself declares. A disagreement between the two is a conflict in
`sources.merge`, and the precedence (manifest before file) picks, with a record.
"""
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .facts import VOCABULARY, Certainty, Fact, Source, vocabulary_class

SCHEMA_VERSION = 1
SIDECAR = ".entail.json"   # a manifest may also sit next to its artifact: <artifact>.entail.json


@dataclass(frozen=True)
class Manifest:
    sha256: str
    facts: Tuple[Fact, ...]
    evidence: Dict[str, List[str]] = field(default_factory=dict)   # vocabulary name -> result files, docs, probes
    pinned: bool = False
    file: Optional[str] = None
    problems: Tuple[str, ...] = ()   # what the readers could not represent when the draft was made
    fingerprint: Optional[str] = None   # quick_fingerprint of the artifact (M6.3): which files it could be for


# --- values <-> JSON ------------------------------------------------------------------------------------------

def value_to_json(value):
    if value is None:
        return None
    out = {}
    for f in dataclasses.fields(value):
        v = getattr(value, f.name)
        out[f.name] = [list(x) if isinstance(x, tuple) else x for x in v] if isinstance(v, tuple) else v
    return out


def value_from_json(name, data):
    if data is None:
        return None
    cls = vocabulary_class(name)
    known = {f.name for f in dataclasses.fields(cls)}
    extra = sorted(set(data) - known)
    if extra:
        raise ValueError(f"manifest: {name} has no fields {extra}")
    args = {k: tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
            for k, v in data.items()}
    return cls(**args)


# --- hashing --------------------------------------------------------------------------------------------------

_HASHES = {}   # (path, size, mtime_ns) -> sha256, so an unchanged file is hashed once per process


def key_file(path):
    """The file whose hash keys an artifact: the file itself, or a model folder's config.json / model_index.json."""
    if os.path.isdir(path):
        for name in ("config.json", "model_index.json"):
            p = os.path.join(path, name)
            if os.path.isfile(p):
                return p
        raise ValueError(f"manifest: {path} is a folder without config.json or model_index.json")
    return path


_EDGE = 4 << 20   # bytes read at each end for the quick fingerprint


def quick_fingerprint(path):
    """SHA-256 of the key file's size and its first and last 4 MiB: cheap, and different for different weights."""
    path = key_file(path)
    size = os.path.getsize(path)
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(_EDGE))
        if size > _EDGE:
            f.seek(max(_EDGE, size - _EDGE))
            h.update(f.read(_EDGE))
    return h.hexdigest()


def sha256_of(path):
    path = key_file(path)
    st = os.stat(path)
    cache = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if cache not in _HASHES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        _HASHES[cache] = h.hexdigest()
    return _HASHES[cache]


# --- load, save, find -----------------------------------------------------------------------------------------

def to_json(m: Manifest) -> dict:
    return {"schema": SCHEMA_VERSION, "sha256": m.sha256, "file": m.file, "pinned": m.pinned,
            "fingerprint": m.fingerprint, "problems": list(m.problems),
            "facts": [{"name": f.name, "value": value_to_json(f.value), "certainty": f.certainty.value,
                       "evidence": list(m.evidence.get(f.name, []))} for f in m.facts]}


def from_json(data: dict, where: str) -> Manifest:
    if data.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"manifest {where}: schema {data.get('schema')!r}, this library reads {SCHEMA_VERSION}")
    sha = data.get("sha256")
    if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
        raise ValueError(f"manifest {where}: sha256 must be 64 lowercase hex digits, got {sha!r}")
    pinned = data.get("pinned") is True
    facts, evidence = [], {}
    for entry in data.get("facts", []):
        name = entry.get("name")
        if name not in VOCABULARY:
            raise ValueError(f"manifest {where}: unknown fact name {name!r}")
        value = value_from_json(name, entry.get("value"))
        certainty = Certainty(entry.get("certainty", "unknown")) if value is not None else Certainty.UNKNOWN
        if value is not None and certainty is Certainty.UNKNOWN:
            certainty = Certainty.INFERRED   # a slot a person filled in; it counts once the manifest is pinned
        if not pinned and certainty is Certainty.DECLARED:
            certainty = Certainty.INFERRED   # an unreviewed draft declares nothing
        facts.append(Fact(name, value, Source("manifest", f"{where}#{name}"), certainty))
        if entry.get("evidence"):
            evidence[name] = list(entry["evidence"])
    fp = data.get("fingerprint")
    if fp is not None and not (isinstance(fp, str) and len(fp) == 64 and all(c in "0123456789abcdef" for c in fp)):
        raise ValueError(f"manifest {where}: fingerprint must be 64 lowercase hex digits, got {fp!r}")
    return Manifest(sha, tuple(facts), evidence, pinned, data.get("file"), tuple(data.get("problems", [])), fp)


def load(path: str) -> Manifest:
    with open(path, encoding="utf-8") as f:
        return from_json(json.load(f), path)


def save(m: Manifest, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json(m), f, ensure_ascii=False, indent=1)


_FINGERPRINTS = {}   # search dir -> (its mtime, the fingerprints its manifests record; None for one without)


def _fingerprints(d):
    try:
        mtime = os.stat(d).st_mtime_ns
    except OSError:
        return set()
    cached = _FINGERPRINTS.get(d)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    out = set()
    for name in os.listdir(d):
        if name.endswith(".json"):
            try:
                with open(os.path.join(d, name), encoding="utf-8") as f:
                    out.add(json.load(f).get("fingerprint"))
            except (OSError, ValueError, AttributeError):
                out.add(None)   # unreadable here: not ruled out, find() reports it when it loads it
    _FINGERPRINTS[d] = (mtime, out)
    return out


def find(artifact_path: str, search_dirs: Sequence[str]) -> Optional[Manifest]:
    """The manifest for this artifact: a sidecar `<artifact>.entail.json`, or `<sha256>.json` in a search dir.
    A manifest whose sha256 does not match the artifact is not used (it describes another file). The artifact is
    hashed in full only when a sidecar exists or a manifest in the search dirs could be for it: its fingerprint
    matches, or it records none (written before M6.3)."""
    if not os.path.isfile(key_file(artifact_path) + SIDECAR):
        prints = set().union(*(_fingerprints(d) for d in search_dirs)) if search_dirs else set()
        if None not in prints and quick_fingerprint(artifact_path) not in prints:
            return None
    sha = sha256_of(artifact_path)
    candidates = [key_file(artifact_path) + SIDECAR] + [os.path.join(d, f"{sha}.json") for d in search_dirs]
    for p in candidates:
        if os.path.isfile(p):
            m = load(p)
            if m.sha256 == sha:
                return m
    return None


# --- draft and pin --------------------------------------------------------------------------------------------

def relevant_names(path: str) -> Tuple[str, ...]:
    """The facts that matter for this kind of artifact, so a draft can show the empty slots."""
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "model_index.json")):
            return ("Prediction", "LatentScale")
        return ("ModelProps", "Rotary", "Template")
    if str(path).endswith(".gguf"):
        return ("ModelProps", "Rotary", "Template")
    if str(path).endswith(".safetensors"):
        from .readers import safetensors_header
        keys, _ = safetensors_header(path)
        if any(k.startswith(("model.diffusion_model.", "first_stage_model.")) for k in keys):
            return ("Prediction", "LatentScale")
    return ()


def infer(artifact_path: str) -> Manifest:
    """A draft: what the artifact itself declares (with where it says so), and an empty slot for every relevant
    fact it does not declare. Nothing is guessed here; a probe's result would enter as `inferred`."""
    from .sources import merge, read_all
    result = read_all(artifact_path)
    chosen, conflicts = merge(result.facts)
    facts, evidence = [], {}
    for name, fact in chosen.items():
        facts.append(Fact(name, fact.value, Source("manifest", f"draft#{name}"), fact.certainty))
        evidence[name] = [f"the artifact declares it: {fact.source.where}"]
    for c in conflicts:
        evidence.setdefault(c.name, []).append(
            "the artifact's own statements disagree: " + "; ".join(f"{f.source.where} says {f.value}" for f in c.facts))
    for name in relevant_names(artifact_path):
        if name not in chosen:
            facts.append(Fact(name, None, Source("manifest", f"draft#{name}"), Certainty.UNKNOWN))
            evidence[name] = ["not declared by the artifact: fill in after review"]
    return Manifest(sha256_of(artifact_path), tuple(facts), evidence, False, os.path.basename(artifact_path),
                    tuple(result.problems), quick_fingerprint(artifact_path))


def pin(m: Manifest) -> Manifest:
    """After review: the manifest is pinned, and every filled fact counts as declared."""
    facts = tuple(f if f.value is None else dataclasses.replace(f, certainty=Certainty.DECLARED) for f in m.facts)
    return dataclasses.replace(m, facts=facts, pinned=True)
