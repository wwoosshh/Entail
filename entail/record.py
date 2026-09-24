"""record: the transfer ledger, and locating where meaning broke (LIBRARY_DESIGN.md 4.9, 12). Ledger in M1.2,
locating in M7.1.

Every decision is kept with its source chain, and printed as one line that names the fact, the declared value and
where it came from, the consumer and what it uses, the rule, the verdict, and what was changed. From the ledger the
library can later say where a problem lies (THEORY.md 2.1; the researcher's words: if meaning is carried exactly,
the place where it broke is the problem area):
  - meaning broke at a boundary                          -> that boundary
  - every checked boundary kept its meaning, output wrong -> not the plumbing: inside a layer (the model itself,
                                                            a compiler, a kernel, the hardware)
  - some boundaries could not be checked                 -> they and the layers beside them stay suspect, so the
                                                            precision depends on how densely meaning is checked (S1)
Diagnosis is secondary: it is what preservation makes possible, not a separate bug hunt.

Where it goes (M6.4, the researcher's decision of 2026-09-24): every line entail says is printed, and - whenever
entail is on - also kept in the project, in `entail_logs/` in the folder the program was started from, where a
developer finds it without having asked for it beforehand:
  entail-<date>.log     the lines, each with its time and process
  record-<date>.jsonl   every decision as one JSON line (or the file ENTAIL_RECORD names, as before)
  .gitignore            so the folder stays out of the project's history
ENTAIL_LOG_DIR moves the folder, or turns the files off ("off"). A folder that cannot be written is said once, and the
run goes on (principle 12).
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

LOG_DIR_NAME = "entail_logs"
_READY = set()    # log folders made (with their .gitignore) in this process
_WARNED = set()   # files this process could not write, said once


def log_dir() -> Optional[str]:
    """The folder entail's log and record files go to, or None. ENTAIL_LOG_DIR names it ("off": none). Unset, it is
    entail_logs in the folder the program was started from, whenever entail is on (ENTAIL=load or debug, which
    entail.enable() sets too); code that only sets the mode (tests, harnesses) writes nothing. The first answer is put
    in the environment, so the processes an engine starts afterwards write to the same folder."""
    d = os.environ.get("ENTAIL_LOG_DIR")
    if d:
        return None if d.strip().lower() == "off" else d
    if os.environ.get("ENTAIL") not in ("load", "debug"):
        return None
    d = os.path.join(os.getcwd(), LOG_DIR_NAME)
    os.environ["ENTAIL_LOG_DIR"] = d
    return d


def _append(path, text) -> None:
    folder = os.path.dirname(path)
    try:
        if folder and folder not in _READY:
            os.makedirs(folder, exist_ok=True)
            ignore = os.path.join(folder, ".gitignore")
            if os.path.basename(folder) == LOG_DIR_NAME and not os.path.exists(ignore):
                with open(ignore, "w", encoding="utf-8") as f:
                    f.write("# written by entail: what it said about this project's runs, not part of the project\n*\n")
            _READY.add(folder)
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        if path not in _WARNED:
            _WARNED.add(path)
            print(f"[entail] could not write {path}: {e}; what entail says goes to the console only",
                  file=sys.stderr, flush=True)


def write_json(obj) -> None:
    """One JSON line: to the file ENTAIL_RECORD names, else to record-<date>.jsonl in the log folder, if any."""
    path = os.environ.get("ENTAIL_RECORD")
    if not path:
        folder = log_dir()
        if folder is None:
            return
        path = os.path.join(folder, f"record-{time.strftime('%Y-%m-%d')}.jsonl")
    _append(path, json.dumps(obj, ensure_ascii=False) + "\n")


def say(text: str) -> None:
    """A line entail says: printed, and kept in entail-<date>.log in the log folder with its time and process."""
    print(text, flush=True)
    folder = log_dir()
    if folder is not None:
        _append(os.path.join(folder, f"entail-{time.strftime('%Y-%m-%d')}.log"),
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} {text}\n")


@dataclass(frozen=True)
class Localization:
    broken_at: Optional[str]        # the boundary where meaning broke, if any
    all_intact: bool                # every checked boundary passed or was resolved
    unchecked: Tuple[str, ...]      # boundaries reported as "could not check"
    suspects: Tuple[str, ...]       # the layers where the fault must lie


def _shown(fact):
    value = "unknown" if fact.value is None else str(fact.value)
    return f"{value} ({fact.source}, {fact.certainty.value})"


def line(decision) -> str:
    """One line for one Decision."""
    d, consumer = decision, decision.contract.consumer
    declared = f"declared {_shown(d.declared)}" if d.declared is not None else "nothing declared"
    used = f"{consumer} uses {_shown(d.chosen)}" if d.chosen is not None else f"what {consumer} uses is unknown"
    text = f"[entail] {d.verdict.value} at {d.contract.boundary}: {d.name} {declared}; {used}; rule: {d.rule}"
    if d.resolution:
        text += f"; changed: {d.resolution}"
    if d.observed is not None:
        text += f"; the data shows {_shown(d.observed)}"
    if d.conflict:
        text += "; sources disagreed: " + ", ".join(_shown(f) for f in d.conflict)
    if getattr(d, "note", ""):
        text += f"; note: {d.note}"
    if d.blocking:
        text += "; stops here"
    elif d.verdict.value == "broken":
        text += "; reported, not stopped"
    return text


def _fact_json(fact):
    if fact is None:
        return None
    return {"name": fact.name, "kind": fact.kind, "value": None if fact.value is None else str(fact.value),
            "source": {"kind": fact.source.kind, "where": fact.source.where}, "certainty": fact.certainty.value}


def decision_json(d) -> dict:
    return {"boundary": d.contract.boundary, "consumer": d.contract.consumer, "name": d.name,
            "verdict": d.verdict.value, "blocking": d.blocking, "rule": d.rule, "resolution": d.resolution,
            "handle": d.handle, "target": None if getattr(d, "target", None) is None else str(d.target),
            "note": getattr(d, "note", ""), "declared": _fact_json(d.declared), "chosen": _fact_json(d.chosen),
            "observed": _fact_json(d.observed), "conflict": [_fact_json(f) for f in d.conflict]}


@dataclass
class Ledger:
    decisions: List[object] = field(default_factory=list)   # contracts.Decision

    def add(self, decision) -> None:
        self.decisions.append(decision)

    def extend(self, decisions) -> None:
        for d in decisions:
            self.add(d)

    def blocking(self) -> List[object]:
        """Decisions that must stop the run before any output."""
        return [d for d in self.decisions if d.blocking]

    def broken(self) -> List[object]:
        """Where meaning broke and nothing repaired it, whether the run stopped (refused) or went on (broken)."""
        return [d for d in self.decisions if d.verdict.value in ("broken", "refused")]

    def lines(self) -> List[str]:
        return [line(d) for d in self.decisions]

    def to_json(self) -> dict:
        return {"decisions": [decision_json(d) for d in self.decisions]}

    def locate(self) -> Localization:
        raise NotImplementedError("M7.1: locating where meaning broke")
