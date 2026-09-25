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
from typing import Dict, List, Optional, Sequence, Tuple

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


def say(text: str, console: bool = True) -> None:
    """A line entail says: printed (unless `console` is False: a non-blocking unknown under ENTAIL_QUIET=unknown),
    and kept in entail-<date>.log in the log folder with its time and process."""
    if console:
        print(text, flush=True)
    folder = log_dir()
    if folder is not None:
        _append(os.path.join(folder, f"entail-{time.strftime('%Y-%m-%d')}.log"),
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()} {text}\n")


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
            "note": getattr(d, "note", ""), "lost_by": getattr(d, "lost_by", None),
            "declared": _fact_json(d.declared), "chosen": _fact_json(d.chosen),
            "observed": _fact_json(d.observed), "conflict": [_fact_json(f) for f in d.conflict]}


@dataclass
class Ledger:
    decisions: List[object] = field(default_factory=list)   # contracts.Decision
    layers: List[dict] = field(default_factory=list)        # layers compared with a reference (diagnose.py, M7.1)

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

    def locate(self, output_wrong: Optional[bool] = None, passes: Optional[Dict[str, int]] = None,
               skipped: Sequence[str] = ()) -> "Localization":
        """Where the fault lies (locate() below), from these decisions and layer comparisons. `passes`: boundaries
        whose checks are counted rather than recorded, with how many held; `skipped`: boundaries whose every check
        was skipped (inside a captured graph)."""
        return locate([decision_json(d) for d in self.decisions], passes or {}, self.layers, output_wrong, skipped)


# --- locating where meaning broke (M7.1) ------------------------------------------------------------------------
#
# The rules (LIBRARY_DESIGN.md 12; the researcher's words: if meaning is carried exactly, the place where it broke
# is the problem area):
#   1. a boundary where meaning broke (broken, refused) is the problem area; the first one recorded comes first
#   2. a boundary entail could not check, or where a fact was unknown, vouches for nothing: it and the layers beside
#      it (who declared, who used) stay suspect
#   3. every checked boundary held and the output is wrong: the fault is not in the plumbing but inside a layer -
#      the model itself, a compiler, a kernel, the hardware. A layer compared with a reference on the same inputs
#      (diagnose.watch) narrows it: the one whose output its reference does not reproduce is where the fault lies;
#      one that agrees is cleared
# The precision is only as fine as the boundaries are dense (S1): the ledger can name only what was checked.

@dataclass(frozen=True)
class Localization:
    """Where the fault lies, as far as the ledger can say.
      broken_at   the first boundary where meaning broke (None: none broke)
      all_intact  at least one boundary was checked, and every checked one kept its meaning (passed, or repaired)
      unchecked   boundaries that could not be checked, or where a fact was unknown
      suspects    where the fault must lie, most specific first
    and what explains them: every broken boundary, the operations that made a fact untrue on the way, the
    boundaries that held, and the layers compared with a reference."""
    broken_at: Optional[str]
    all_intact: bool
    unchecked: Tuple[str, ...]
    suspects: Tuple[str, ...]
    broken: Tuple[str, ...] = ()
    lost_by: Tuple[str, ...] = ()
    intact: Tuple[str, ...] = ()
    layers: Tuple[str, ...] = ()
    why: Tuple[str, ...] = ()        # one sentence per suspect, in the same order

    def lines(self) -> List[str]:
        """What a person reads: the verdict first, then why."""
        out = []
        if self.broken_at is not None:
            out.append(f"[entail] where: meaning broke at {self.broken_at}" +
                       (f" (and at {', '.join(self.broken[1:])})" if len(self.broken) > 1 else ""))
        elif self.lost_by:
            out.append(f"[entail] where: meaning was lost on the way: {self.lost_by[0]}")
        elif self.all_intact:
            out.append(f"[entail] where: every checked boundary kept its meaning ({len(self.intact)} boundaries)")
        else:
            out.append("[entail] where: no boundary was checked, so the ledger cannot say where")
        out += [f"[entail]   suspect: {w}" for w in self.why]
        out += [f"[entail]   lost on the way: {op}" for op in self.lost_by]
        out += [f"[entail]   not checked: {u}" for u in self.unchecked]
        out += [f"[entail]   compared: {c}" for c in self.layers]
        return out

    def to_json(self) -> dict:
        return {"broken_at": self.broken_at, "all_intact": self.all_intact, "unchecked": list(self.unchecked),
                "suspects": list(self.suspects), "broken": list(self.broken), "lost_by": list(self.lost_by),
                "intact": list(self.intact), "layers": list(self.layers), "why": list(self.why)}


def _source(fact) -> Optional[str]:
    if not fact:
        return None
    src = fact.get("source") or {}
    return f"{src.get('kind')}: {src.get('where')}" if src.get("where") else src.get("kind")


def _worse(a: dict, b: dict) -> bool:
    """Comparison a says more against the layer than b: it differs where b agrees, or by more."""
    def key(c):
        rel = c.get("max_rel")
        return (not c.get("agrees"), float("inf") if rel is None else rel)
    return key(a) > key(b)


def _layer_text(c: dict) -> str:
    verdict = "agrees with" if c.get("agrees") else "differs from"
    where = f" (instance {c['instance']})" if c.get("instance") is not None and not c.get("agrees") else ""
    text = f"{c.get('layer')}{where} {verdict} {c.get('reference')} on the same inputs"
    if c.get("max_rel") is not None:
        text += f" (largest difference {c['max_rel']:.3g} of the reference's scale, tolerance {c.get('tol')}"
        text += f"; {c['calls']} calls compared)" if (c.get("calls") or 1) > 1 else ")"
    if c.get("note"):
        text += f"; {c['note']}"
    return text


def locate(rows: Sequence[dict], passes: Optional[Dict[str, int]] = None, layers: Sequence[dict] = (),
           output_wrong: Optional[bool] = None, skipped: Sequence[str] = ()) -> Localization:
    """Apply the rules above. `rows`: decisions as JSON (decision_json, or the lines of a record file); `passes`:
    boundary -> checks that held but were only counted; `layers`: comparisons with a reference; `output_wrong`: what
    the caller knows about the result (None: not known); `skipped`: boundaries none of whose checks ran."""
    order, status, facts, sides, lost, lost_ops = [], {}, {}, {}, [], {}
    rank = {"intact": 0, "unchecked": 1, "broken": 2}

    def see(boundary, state):
        if boundary not in status:
            order.append(boundary)
            status[boundary] = state
        elif rank[state] > rank[status[boundary]]:
            status[boundary] = state

    for r in rows:
        b, v = r.get("boundary"), r.get("verdict")
        if not b or not v:
            continue
        state = {"pass": "intact", "resolved": "intact", "broken": "broken", "refused": "broken"}.get(v, "unchecked")
        see(b, state)
        if state != "intact":
            facts.setdefault(b, []).append(f"{r.get('name')}: {r.get('rule')}")
        sides.setdefault(b, (_source(r.get("declared")), r.get("consumer")))
        if r.get("lost_by"):
            lost.append(f"{r.get('name')} at {b}, made untrue by {r['lost_by']}")
            lost_ops.setdefault(b, r["lost_by"])
    for b, n in (passes or {}).items():
        if n:
            see(b, "intact")
    for b in skipped:
        if b not in status:
            see(b, "unchecked")
            facts.setdefault(b, []).append("every check skipped (inside a captured graph)")

    broken = tuple(b for b in order if status[b] == "broken")
    unchecked = tuple(b for b in order if status[b] == "unchecked")
    intact = tuple(b for b in order if status[b] == "intact")
    checked = bool(broken or intact)
    suspects, why = [], []
    for b in broken:
        suspects.append(f"boundary {b}")
        why.append(f"meaning broke at {b}: {'; '.join(facts.get(b, []))}")
    for b in unchecked:
        if b in lost_ops:   # not a boundary that could not look: meaning was lost before it, by a named operation
            suspects.append(f"operation {lost_ops[b]} on the way to {b}")
            why.append(f"meaning was lost on the way to {b}: {lost_ops[b]} made what it needs untrue, and nothing "
                       f"said what the value holds after it ({'; '.join(facts.get(b, []))})")
            continue
        producer, consumer = sides.get(b, (None, None))
        named = ", ".join(x for x in (producer, consumer) if x)
        suspects.append(f"boundary {b} and beside it {named}" if named else f"boundary {b} and the layers beside it")
        why.append(f"{b} vouches for nothing ({'; '.join(facts.get(b, []))}), so it and "
                   f"{named or 'the layers beside it'} stay suspect")
    worst = {}   # layer -> [its worst comparison, how many calls were compared]: one differing call is enough
    for c in layers:
        if not c.get("layer"):
            continue
        seen = worst.setdefault(c["layer"], [c, 0])
        seen[1] += 1
        if _worse(c, seen[0]):
            seen[0] = c
    compared = [dict(c, calls=n) for c, n in worst.values()]
    differs = [c for c in compared if not c.get("agrees")]
    if output_wrong and not broken:
        if differs:
            for c in differs:
                suspects.append(f"inside {c['layer']}")
                why.append(f"inside {c['layer']}: {_layer_text(c)}")
        else:
            cleared = f"; cleared: {', '.join(c['layer'] for c in compared)}" if compared else ""
            suspects.append("inside a layer")
            why.append("the output is wrong but no checked boundary broke, so the fault is not in the plumbing: it "
                       "is inside a layer - the model itself, a compiler, a kernel, the hardware" + cleared)
    return Localization(broken_at=broken[0] if broken else None, all_intact=checked and not broken,
                        unchecked=tuple(f"{b} ({'; '.join(facts.get(b, []))})" for b in unchecked),
                        suspects=tuple(suspects), broken=broken, lost_by=tuple(lost), intact=intact,
                        layers=tuple(_layer_text(c) for c in compared), why=tuple(why))


def read_records(paths: Sequence[str], pid: Optional[int] = None):
    """(rows, passes, layers, skipped) from record files (record-<date>.jsonl, ENTAIL_RECORD): the decisions, the
    boundaries' counts, and the layer comparisons, of one process or of all of them (engines check in the processes
    they start). A line that is not JSON is left out."""
    rows, latest, layers = [], {}, []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for text in f:
                try:
                    obj = json.loads(text)
                except ValueError:
                    continue
                if not isinstance(obj, dict) or (pid is not None and obj.get("pid") != pid):
                    continue
                if "verdict" in obj:
                    rows.append(obj)
                elif "layer" in obj:
                    layers.append(obj)
                elif isinstance(obj.get("boundaries"), dict):
                    for b, counts in obj["boundaries"].items():   # counts so far: a process's latest line wins
                        if isinstance(counts, dict):
                            latest[(obj.get("pid"), b)] = counts
    passes: Dict[str, int] = {}
    skipped = set()
    for (_, b), counts in latest.items():
        passes[b] = passes.get(b, 0) + sum((counts.get("passed") or {}).values())
        if not counts.get("checks") and (counts.get("skipped") or counts.get("deferred")):
            skipped.add(b)
    return rows, passes, layers, sorted(skipped)
