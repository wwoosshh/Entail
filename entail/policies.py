"""policies: how strict the library is (LIBRARY_DESIGN.md 4.10; ROADMAP M1.2).

(It is `policies`, not `policy`: the package already exports the function `entail.policy`.) The older code in
core.py still reads its mode and policy through `core.set_mode` / `core.set_policy`; the new contracts take a
Policy. Both read the same environment variables, ENTAIL and ENTAIL_POLICY.

Environment:
  ENTAIL                     off | load | debug                 mode
  ENTAIL_POLICY              resolve | refuse                   a mismatch with a resolution: repair it, or stop
  ENTAIL_UNKNOWN             report | require | stop            a meaning-changing fact nobody declares
  ENTAIL_UNKNOWN_OTHER       report | require | stop            any other fact nobody declares
  ENTAIL_FALSE_DECLARATION   refuse | use_data                  a declaration the data contradicts
  ENTAIL_SOURCE_CONFLICT     record | stop                      sources that disagree
  ENTAIL_FACT_POLICY         Name=setting,...                   per fact, e.g. "Prediction=refuse,Template=report"
"""
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from .facts import VOCABULARY

MODES = ("off", "load", "debug")
ON_MISMATCH = ("resolve", "refuse")
ON_UNKNOWN = ("report", "require", "stop")
ON_FALSE_DECLARATION = ("refuse", "use_data")
ON_SOURCE_CONFLICT = ("record", "stop")


def _one_of(label, value, allowed):
    if value not in allowed:
        raise ValueError(f"policy {label}: {value!r} is not one of {list(allowed)}")


@dataclass(frozen=True)
class Policy:
    mode: str = "off"
    on_mismatch: str = "resolve"                  # resolution first; refuse only when none exists (principle 7)
    on_unknown_meaning_changing: str = "require"  # no silent default for a fact that changes the meaning (principle 4)
    on_unknown_other: str = "report"
    on_false_declaration: str = "refuse"
    on_source_conflict: str = "record"
    overrides: Tuple[Tuple[str, str], ...] = ()   # (vocabulary name, a mismatch or unknown setting)

    def __post_init__(self):
        _one_of("mode", self.mode, MODES)
        _one_of("on_mismatch", self.on_mismatch, ON_MISMATCH)
        _one_of("on_unknown_meaning_changing", self.on_unknown_meaning_changing, ON_UNKNOWN)
        _one_of("on_unknown_other", self.on_unknown_other, ON_UNKNOWN)
        _one_of("on_false_declaration", self.on_false_declaration, ON_FALSE_DECLARATION)
        _one_of("on_source_conflict", self.on_source_conflict, ON_SOURCE_CONFLICT)
        for name, setting in self.overrides:
            if name not in VOCABULARY:
                raise ValueError(f"policy override: unknown fact name {name!r}; vocabulary has {sorted(VOCABULARY)}")
            _one_of(f"override for {name}", setting, ON_MISMATCH + ON_UNKNOWN)

    def _override(self, name, allowed) -> Optional[str]:
        return next((s for n, s in self.overrides if n == name and s in allowed), None)

    def mismatch_setting(self, name: str) -> str:
        return self._override(name, ON_MISMATCH) or self.on_mismatch

    def unknown_setting(self, name: str, meaning_changing: bool) -> str:
        default = self.on_unknown_meaning_changing if meaning_changing else self.on_unknown_other
        return self._override(name, ON_UNKNOWN) or default


def from_env(environ: Optional[Mapping[str, str]] = None) -> Policy:
    env = os.environ if environ is None else environ
    overrides = []
    for item in filter(None, (p.strip() for p in env.get("ENTAIL_FACT_POLICY", "").split(","))):
        name, sep, setting = item.partition("=")
        if not sep:
            raise ValueError(f"ENTAIL_FACT_POLICY: expected Name=setting, got {item!r}")
        overrides.append((name.strip(), setting.strip()))
    return Policy(mode=env.get("ENTAIL", "off"), on_mismatch=env.get("ENTAIL_POLICY", "resolve"),
                  on_unknown_meaning_changing=env.get("ENTAIL_UNKNOWN", "require"),
                  on_unknown_other=env.get("ENTAIL_UNKNOWN_OTHER", "report"),
                  on_false_declaration=env.get("ENTAIL_FALSE_DECLARATION", "refuse"),
                  on_source_conflict=env.get("ENTAIL_SOURCE_CONFLICT", "record"), overrides=tuple(overrides))
