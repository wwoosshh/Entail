"""policies: how strict the library is (LIBRARY_DESIGN.md 4.10; ROADMAP M1.2).

(It is `policies`, not `policy`: the package already exports the function `entail.policy`.) The older code in
core.py still reads its mode and policy through `core.set_mode` / `core.set_policy`; the new contracts take a
Policy. Both read the same environment variables, ENTAIL and ENTAIL_POLICY.

Environment:
  ENTAIL                     off | load | debug                 mode
  ENTAIL_POLICY              resolve | refuse                   a mismatch with a resolution: repair it, or not
  ENTAIL_ON_BROKEN           report | stop                      what is not repaired: report it and go on, or stop
  ENTAIL_UNKNOWN             report | require | stop            a meaning-changing fact nobody declares
  ENTAIL_UNKNOWN_OTHER       report | require | stop            any other fact nobody declares
  ENTAIL_FALSE_DECLARATION   refuse | use_data                  a declaration the data contradicts: not repaired, or
                                                                the data's value is used
  ENTAIL_SOURCE_CONFLICT     record | stop                      sources that disagree
  ENTAIL_FACT_POLICY         Name=setting,...                   per fact, e.g. "Prediction=refuse,Layout=stop"

The defaults do not stop (the researcher's decision of 2026-09-24, ROADMAP M5.4): what can be repaired is, and what
cannot is reported - as broken, or unknown - while the run goes on, so that entail adds no failure a user sees and
a false alarm of its own cannot stop an engine. Stopping is chosen: ENTAIL_ON_BROKEN=stop, a per-fact "Name=stop"
(stops when that fact is broken or unknown; "Name=report" reports both), require/stop for unknown facts, or debug
mode, which stops at what is broken and at a consumer nobody can read (as CI fails on a type error that a production
build only warns about). A fact nobody declares follows ENTAIL_UNKNOWN in every mode: `require` there is the
counterpart of TypeScript's noImplicitAny.
"""
import os
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Tuple

from .facts import VOCABULARY

MODES = ("off", "load", "debug")
ON_MISMATCH = ("resolve", "refuse")
ON_BROKEN = ("report", "stop")
ON_UNKNOWN = ("report", "require", "stop")
ON_FALSE_DECLARATION = ("refuse", "use_data")
ON_SOURCE_CONFLICT = ("record", "stop")


def _one_of(label, value, allowed):
    if value not in allowed:
        raise ValueError(f"policy {label}: {value!r} is not one of {list(allowed)}")


@dataclass(frozen=True)
class Policy:
    mode: str = "off"
    on_mismatch: str = "resolve"                  # resolution first (principle 7)
    on_broken: str = "report"                     # not repaired: reported, the run goes on (principle 7, M5.4)
    on_unknown_meaning_changing: str = "report"   # never a silent default: reported (principle 4; "require" until M5.4)
    on_unknown_other: str = "report"
    on_false_declaration: str = "refuse"
    on_source_conflict: str = "record"
    overrides: Tuple[Tuple[str, str], ...] = ()   # (vocabulary name, a mismatch or unknown setting)

    def __post_init__(self):
        _one_of("mode", self.mode, MODES)
        _one_of("on_mismatch", self.on_mismatch, ON_MISMATCH)
        _one_of("on_broken", self.on_broken, ON_BROKEN)
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

    def stops(self, name: str) -> bool:
        """Whether a mismatch of the fact `name` that is not repaired stops the run (refused) or is reported while
        the run goes on (broken). Debug mode always stops."""
        if self.mode == "debug":
            return True
        return (self._override(name, ON_BROKEN) or self.on_broken) == "stop"

    def stops_unknown(self, name: str, meaning_changing: bool) -> bool:
        """Whether a fact nobody declares stops the run: when its setting (require, stop) says so, in any mode."""
        return self.unknown_setting(name, meaning_changing) in ("require", "stop")


def current() -> Policy:
    """The policy in force in this process: the mode and the mismatch setting as core holds them (set_mode,
    set_policy and enable() change them at run time), everything else from the environment."""
    from . import core

    return replace(from_env(), mode=core.mode(), on_mismatch=core.policy())


def from_env(environ: Optional[Mapping[str, str]] = None) -> Policy:
    env = os.environ if environ is None else environ
    overrides = []
    for item in filter(None, (p.strip() for p in env.get("ENTAIL_FACT_POLICY", "").split(","))):
        name, sep, setting = item.partition("=")
        if not sep:
            raise ValueError(f"ENTAIL_FACT_POLICY: expected Name=setting, got {item!r}")
        overrides.append((name.strip(), setting.strip()))
    return Policy(mode=env.get("ENTAIL", "off"), on_mismatch=env.get("ENTAIL_POLICY", "resolve"),
                  on_broken=env.get("ENTAIL_ON_BROKEN", "report"),
                  on_unknown_meaning_changing=env.get("ENTAIL_UNKNOWN", "report"),
                  on_unknown_other=env.get("ENTAIL_UNKNOWN_OTHER", "report"),
                  on_false_declaration=env.get("ENTAIL_FALSE_DECLARATION", "refuse"),
                  on_source_conflict=env.get("ENTAIL_SOURCE_CONFLICT", "record"), overrides=tuple(overrides))
