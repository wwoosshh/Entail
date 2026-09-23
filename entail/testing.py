"""testing: the test problems as fixtures, and a pytest plugin (LIBRARY_DESIGN.md 4.12, 9). Built in M7.2.

The problems are listed in the research folder, `testbed/PROBLEMS.md` (M0.4). They are used only to check that the
design stops the cause classes it claims to stop. They are not a hunt for new defects.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Problem:
    id: str                 # e.g. "rolebench-08", "field-rope-override", "market-L07"
    fact: str               # vocabulary name
    defect: str             # where the defective version lives
    fixed: Optional[str]    # where the fixed version lives, when there is one
    expected: str           # "resolved" or "refused" at a named site
    milestone: str          # the ROADMAP milestone that must make it pass


def problems() -> List[Problem]:
    raise NotImplementedError("M7.2: load testbed/PROBLEMS.md as fixtures")
