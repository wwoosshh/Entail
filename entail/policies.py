"""policies: how strict the library is (LIBRARY_DESIGN.md 4.10). Built in M1.2.

Until M1.2 the running code still takes its mode and policy from `core.set_mode` / `core.set_policy`.
(It is `policies`, not `policy`: the package already exports the function `entail.policy`.)
"""
from dataclasses import dataclass
from typing import Tuple

MODES = ("off", "load", "debug")
ON_MISMATCH = ("resolve", "refuse")
ON_UNKNOWN = ("report", "require", "stop")


@dataclass(frozen=True)
class Policy:
    mode: str = "off"
    on_mismatch: str = "resolve"                  # resolution first; refuse only when none exists (principle 7)
    on_unknown_meaning_changing: str = "require"  # no silent default for a fact that changes the meaning (principle 4)
    on_unknown_other: str = "report"
    overrides: Tuple[Tuple[str, str], ...] = ()   # (vocabulary name, setting)


def from_env() -> Policy:
    """ENTAIL, ENTAIL_POLICY and the per-fact settings."""
    raise NotImplementedError("M1.2: policy from the environment")
