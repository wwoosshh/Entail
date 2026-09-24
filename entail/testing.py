"""testing: risk conditions drawn from declarations, the test problems as fixtures, and what the pytest plugin runs on
(LIBRARY_DESIGN.md 4.12, 9; ROADMAP M7.2).

A declared fact that changes what a component must do at a threshold is where a test belongs: a sliding window
starts masking after its last token, a scaled RoPE changes regime at the original context, an earlier answer's
reasoning matters only from the second turn. risk_conditions() turns what a model declares into such conditions -
the lengths just under, at and over each threshold, a second turn, a tool call - so a test runs where the meaning
is at stake instead of where the default path runs. Standard evaluations miss these cases because the trigger is a
length, a turn or a feature they do not reach (rolebench/README.md), not because nobody could compare.

problems() loads a list of test problems (JSON) - each a mechanism, where its defective and fixed versions live, and
the verdict the design requires - for a parametrized test to run them. This project's list is generated from
`testbed/PROBLEMS.md` in the research folder (testbed/m72_problems.py). They check that the design stops the cause
classes it claims to stop; they are not a hunt for new defects.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Problem:
    id: str                 # e.g. "rb-08", "fd-rope", "mk-L07"
    fact: str               # vocabulary name
    defect: str             # where the defective version lives
    fixed: Optional[str]    # where the fixed version lives, when there is one
    expected: str           # the verdict the design requires, e.g. "resolved", "broken (refused when strict)"
    milestone: str          # the ROADMAP milestone that must make it pass
    site: str = ""          # where it is checked: load, boundary, container, request, reuse
    result: str = ""        # what was measured, when it was


@dataclass(frozen=True)
class Condition:
    """One condition under which a declared fact decides what a component must do."""
    id: str                 # the test id, e.g. "length=4096"
    fact: str               # the declaration it comes from, e.g. "ModelProps.sliding_window"
    why: str                # e.g. "exactly the sliding window the model declares (4096)"
    params: Dict[str, object] = field(default_factory=dict)   # e.g. {"length": 4096}


def _lengths(n: int, fact: str, what: str) -> List[Condition]:
    return [Condition(f"length={k}", fact, f"{rel} {what} ({n})", {"length": k})
            for k, rel in ((n - 1, "one token under"), (n, "exactly"), (n + 1, "one token over"))]


def _one(facts, name):
    got = facts.get(name) if facts is not None else None
    return got[0].value if got else None


def risk_conditions(model_path: Optional[str] = None, facts=None, chunk_sizes: Sequence[int] = ()) -> List[Condition]:
    """The conditions under which what `model_path` declares (or `facts`, a load.Declared) is at stake:
      ModelProps.sliding_window        lengths one under, at and one over the window
      Rotary.original_max_position     the same around where a scaled RoPE (not "default") changes regime
      Valid.window                     the same around a declared context window (a manifest declares it)
      Template.reasoning_history       a second turn after an answer with reasoning
      Template.tool_call_format        a turn that calls a tool
      chunk_sizes (the engine's)       lengths around each chunk a prefill is cut into
    One id per condition; a length two facts share is kept once, under the first."""
    if facts is None:
        from . import load

        facts = load.declared(model_path) if model_path else None
    out: List[Condition] = []
    props, rotary = _one(facts, "ModelProps"), _one(facts, "Rotary")
    valid, template = _one(facts, "Valid"), _one(facts, "Template")
    if props is not None and props.sliding_window:
        out += _lengths(int(props.sliding_window), "ModelProps.sliding_window", "the sliding window the model declares")
    if rotary is not None and rotary.original_max_position and rotary.rope_type != "default":
        out += _lengths(int(rotary.original_max_position), "Rotary.original_max_position",
                        f"the context where {rotary.rope_type} RoPE scaling takes over")
    if valid is not None and valid.window:
        out += _lengths(int(valid.window), "Valid.window", "the context window declared")
    for n in chunk_sizes:
        out += _lengths(int(n), "Positions (chunks)", "one chunk of the prefill")
    if template is not None and template.reasoning_history:
        out.append(Condition("second_turn", "Template.reasoning_history",
                             f"a second turn after an answer with reasoning; the model declares "
                             f"'{template.reasoning_history}' for earlier reasoning", {"turns": 2, "reasoning": True}))
    if template is not None and template.tool_call_format:
        out.append(Condition("tool_call", "Template.tool_call_format",
                             f"a turn that calls a tool, in the {template.tool_call_format} format the model declares",
                             {"tool_call": template.tool_call_format}))
    seen, unique = set(), []
    for c in out:
        if c.id not in seen:
            seen.add(c.id)
            unique.append(c)
    return unique


def problems(path: Optional[str] = None) -> List[Problem]:
    """The test problems in a JSON file - a list of objects with Problem's fields - named here or by ENTAIL_PROBLEMS.
    None named: no problems (the collection lives with the project that measures it)."""
    path = path or os.environ.get("ENTAIL_PROBLEMS")
    if not path:
        return []
    with open(os.path.expanduser(path), encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("problems", data) if isinstance(data, dict) else data
    known = set(Problem.__dataclass_fields__)
    return [Problem(**{k: v for k, v in r.items() if k in known}) for r in rows]
