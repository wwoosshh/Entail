"""graph: a launch's record lines as the nodes of the project's workflow (LIBRARY_DESIGN.md 13.3, 13.4; ROADMAP P1).

A node is one stage of the workflow (config, tokenizer, weights, KV cache, kernels, ...). Every boundary a decision
names belongs to one node, by the patterns in data/nodes.json (first match in the file's order). A node shows:
  state      the worst of its decisions and counts: refused > broken > unknown > unchecked > resolved > pass > none.
             "unchecked": its checks were all skipped (inside a captured graph); "none": nothing was decided there
  progress   decisions per verdict, checks counted by the boundaries' tally lines (passes are counted, not recorded),
             skipped checks, and the time spent checking (timing lines)
  decisions  what each decision said: the declared value and its source, what the consumer chose, the rule, the
             resolution, the note - where and why, as the ledger recorded it
The graph adds no rule: where meaning broke is record.locate's answer (LIBRARY_DESIGN.md 12), over the same lines.

Record lines of version 2 carry v, t and run (record.write_json); a version-1 line has no run, so each record file's
version-1 lines count as one launch of their own ("file:<name>").
"""
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from .. import record

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nodes.json")
STATES = ("none", "pass", "resolved", "unchecked", "unknown", "broken", "refused")   # better to worse
RANK = {s: i for i, s in enumerate(STATES)}
# A launch's own state: a node whose checks were all skipped (a request boundary in a run that sent no chat request)
# does not make a launch whose other nodes passed look unchecked; it counts only when nothing else was decided
RUN_RANK = {"none": 0, "unchecked": 1, "pass": 2, "resolved": 3, "unknown": 4, "broken": 5, "refused": 6}
_MODEL = None
_NODE_OF: Dict[str, str] = {}


def model() -> dict:
    """The node model from data/nodes.json, its patterns compiled: {"flows": {...}, "nodes": [...], "by_id": {...}}.
    The attached official DLCs' nodes (entail/dlc.py) go in before the catch-all, so their boundaries find them."""
    global _MODEL
    if _MODEL is None:
        with open(DATA, encoding="utf-8") as f:
            raw = json.load(f)
        core_nodes = [n for n in raw["nodes"] if n["patterns"] != [""]]
        catch_all = [n for n in raw["nodes"] if n["patterns"] == [""]]
        try:
            from .. import dlc

            extra = dlc.nodes(core_ids={n["id"] for n in raw["nodes"]})
        except Exception:  # noqa: BLE001 - a DLC that cannot be read never hides the core's nodes
            extra = []
        nodes = []
        for n in core_nodes + extra + catch_all:
            try:
                nodes.append(dict(n, compiled=[re.compile(p) for p in n["patterns"]]))
            except re.error:
                continue            # a DLC node whose pattern does not compile is left out
        _MODEL = {"flows": raw["flows"], "nodes": nodes, "by_id": {n["id"]: n for n in nodes}}
    return _MODEL


def reset_model() -> None:
    """Read the node model again on the next call (tests; a DLC installed while the platform runs)."""
    global _MODEL
    _MODEL = None
    _NODE_OF.clear()


def node_of(boundary: str) -> str:
    """The id of the node a boundary belongs to: a custom node's own ("node:rag.answer/json_object" ->
    "node:rag.answer", entail/nodes.py), else the first node whose pattern matches it ("other" at the end takes the
    rest)."""
    found = _NODE_OF.get(boundary)
    if found is None:
        if boundary.startswith("node:"):
            found = "node:" + boundary[5:].split("/", 1)[0]
        else:
            found = next(n["id"] for n in model()["nodes"] if any(p.match(boundary) for p in n["compiled"]))
        _NODE_OF[boundary] = found
    return found


def _custom_spec(nid: str, i: int) -> dict:
    """A custom node's place in the user-code flow: after the core's user-code node, in name order."""
    name = nid[5:]
    return {"id": nid, "flow": "user", "step": 1 + (i + 1) / 1000.0, "ko": name, "en": name, "custom": True,
            "patterns": [], "compiled": []}


def engine_of(boundary: str) -> Optional[str]:
    """The engine a boundary's name speaks of ("load:vllm.attention" -> "vllm"), or None ("kernel:triton"; a custom
    node's "node:app.answer/json_object", which names a point of the program, not an engine)."""
    if boundary.startswith("node:"):
        return None
    rest = boundary.split(":", 1)[1] if ":" in boundary else ""
    return rest.split(".", 1)[0] if "." in rest else None


def read_lines(paths: Iterable[str]) -> List[Tuple[str, dict]]:
    """(file, line) for every JSON object line of the record files; a line that is not JSON is left out."""
    out = []
    for path in paths:
        try:
            f = open(path, encoding="utf-8")
        except OSError:
            continue
        with f:
            for text in f:
                try:
                    obj = json.loads(text)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    out.append((path, obj))
    return out


def record_files(folder: str) -> List[str]:
    """The record files of a log folder (record-<date>.jsonl), oldest first."""
    try:
        names = sorted(n for n in os.listdir(folder) if n.startswith("record-") and n.endswith(".jsonl"))
    except OSError:
        return []
    return [os.path.join(folder, n) for n in names]


def launches(lines: Iterable[Tuple[str, dict]]) -> Dict[str, List[dict]]:
    """Record lines grouped by launch: the run id of version-2 lines, "file:<name>" for a file's version-1 lines."""
    out: Dict[str, List[dict]] = {}
    for path, obj in lines:
        key = obj.get("run") or f"file:{os.path.basename(path)}"
        out.setdefault(str(key), []).append(obj)
    return out


def _worse(a: str, b: str) -> str:
    return a if RANK[a] >= RANK[b] else b


def _verdict_state(v: str) -> str:
    return v if v in RANK else "unknown"


def graph(lines: List[dict]) -> dict:
    """The nodes of one launch (its lines, as launches() grouped them): per node its state, progress and boundaries,
    the flows in order with their edges, and where meaning broke (record.locate over the same lines)."""
    m = model()
    per_node: Dict[str, dict] = {}
    tally_latest: Dict[Tuple[object, str], dict] = {}
    rows, layers, times = [], [], []
    for obj in lines:
        if "verdict" in obj and obj.get("boundary"):
            rows.append(obj)
        elif isinstance(obj.get("boundaries"), dict):
            for b, counts in obj["boundaries"].items():
                if isinstance(counts, dict):
                    tally_latest[(obj.get("pid"), b)] = counts
        elif "layer" in obj:
            layers.append(obj)
        if obj.get("t") is not None:
            times.append(obj.get("t"))

    def node(b):
        nid = node_of(b)
        n = per_node.get(nid)
        if n is None:
            n = per_node[nid] = {"id": nid, "state": "none", "boundaries": set(), "verdicts": {}, "checks": 0,
                                 "passed": 0, "skipped": 0, "timed_calls": 0, "ms": 0.0, "said": 0}
        n["boundaries"].add(b)
        return n

    for r in rows:
        n = node(r["boundary"])
        s = _verdict_state(r.get("verdict"))
        n["verdicts"][s] = n["verdicts"].get(s, 0) + 1
        n["state"] = _worse(s, n["state"])
    passes: Dict[str, int] = {}
    skipped = set()
    for (_, b), counts in tally_latest.items():
        n = node(b)
        passed = sum((counts.get("passed") or {}).values())
        n["checks"] += int(counts.get("checks") or 0)
        n["passed"] += passed
        n["skipped"] += int(counts.get("skipped") or 0) + int(counts.get("deferred") or 0)
        passes[b] = passes.get(b, 0) + passed
        if passed:
            n["state"] = _worse("pass", n["state"])
        if not counts.get("checks") and (counts.get("skipped") or counts.get("deferred")):
            skipped.add(b)
            n["state"] = _worse("unchecked", n["state"])
    for obj in lines:
        if obj.get("timing"):
            n = node(obj["timing"])
            n["timed_calls"] += 1
            n["ms"] += float(obj.get("ms") or 0.0)
        elif obj.get("said"):
            node(str(obj["said"]))["said"] += 1
    located = record.locate(rows, passes, layers, None, sorted(skipped))
    # custom nodes (P5) come from the records: each gets a node in the user-code flow
    custom = [_custom_spec(nid, i) for i, nid in enumerate(sorted(n for n in per_node if n.startswith("node:")))]
    m = dict(m, nodes=list(m["nodes"]) + custom, by_id={**m["by_id"], **{c["id"]: c for c in custom}})
    # which flows a launch has: those of its nodes with data, not counting shared nodes (a Triton kernel names no
    # engine, so it must not bring the LLM flow into an image launch); a shared node joins the flows present, at the
    # end of those that are not its own, and brings its own flow only when nothing else is there
    present = [fid for fid in m["flows"]
               if any(n["flow"] == fid and not n.get("shared") and n["id"] in per_node for n in m["nodes"])]
    shared = [n for n in m["nodes"] if n.get("shared") and n["id"] in per_node]
    if not present and shared:
        present = [shared[0]["flow"]]
    flows = []
    for fid in present:
        names = m["flows"][fid]
        members = [n for n in m["nodes"] if n["flow"] == fid and not (n.get("shared") and n["id"] not in per_node
                                                                          and fid != n["flow"])]
        shown = [n for n in members if n["id"] in per_node or n["id"] not in ("other", "user")]
        shown.sort(key=lambda n: n["step"])
        shown += [n for n in shared if n["flow"] != fid]
        flows.append({"id": fid, "ko": names["ko"], "en": names["en"], "nodes": [n["id"] for n in shown],
                      "edges": [[a["id"], b["id"]] for a, b in zip(shown, shown[1:])]})
    nodes, placed = [], set()
    for fl in flows:
        for nid in fl["nodes"]:
            if nid in placed:        # a shared node in two flows is one node
                continue
            placed.add(nid)
            spec = m["by_id"][nid]
            n = per_node.get(nid) or {"id": nid, "state": "none", "boundaries": set(), "verdicts": {}, "checks": 0,
                                       "passed": 0, "skipped": 0, "timed_calls": 0, "ms": 0.0, "said": 0}
            nodes.append(dict(n, boundaries=sorted(n["boundaries"]), ms=round(n["ms"], 3), flow=fl["id"],
                              step=spec["step"], ko=spec["ko"], en=spec["en"], custom=bool(spec.get("custom"))))
    stamps = [t for t in times if isinstance(t, (int, float))]
    worst = "none"
    for n in nodes:
        if RUN_RANK[n["state"]] > RUN_RANK[worst]:
            worst = n["state"]
    # what the program itself said, in process, about where the fault lies (diagnose: it knew whether the output was
    # wrong, and which layers it compared), the latest such line
    said_located = [o["located"] for o in lines if isinstance(o.get("located"), dict)]
    return {"flows": flows, "nodes": nodes, "state": worst, "locate": located.to_json(),
            "located": said_located[-1] if said_located else None,
            "start": min(stamps) if stamps else None, "end": max(stamps) if stamps else None,
            "pids": sorted({o.get("pid") for o in lines if o.get("pid") is not None}),
            "engines": sorted({e for n in nodes for b in n["boundaries"] for e in [engine_of(b)] if e}),
            "decisions": len(rows)}


def node_detail(lines: List[dict], node_id: str) -> dict:
    """What one node of a launch holds: its decisions as recorded (declared value and source, the consumer's choice,
    rule, resolution, note), the boundaries' latest counts, the time spent checking and what entail said there."""
    decisions, counts, timing, said = [], {}, {}, []
    for obj in lines:
        b = obj.get("boundary") if "verdict" in obj else None
        if b and node_of(b) == node_id:
            decisions.append(obj)
        elif isinstance(obj.get("boundaries"), dict):
            for bb, c in obj["boundaries"].items():
                if node_of(bb) == node_id and isinstance(c, dict):
                    counts[(obj.get("pid"), bb)] = c
        elif obj.get("timing") and node_of(obj["timing"]) == node_id:
            t = timing.setdefault(obj["timing"], {"calls": 0, "ms": 0.0})
            t["calls"] += 1
            t["ms"] = round(t["ms"] + float(obj.get("ms") or 0.0), 3)
        elif obj.get("said") and node_of(str(obj["said"])) == node_id:
            said.append({"where": obj["said"], "text": obj.get("text"), "t": obj.get("t"), "pid": obj.get("pid")})
    per_boundary: Dict[str, dict] = {}
    for (_, b), c in counts.items():
        agg = per_boundary.setdefault(b, {"checks": 0, "passed": 0, "skipped": 0})
        agg["checks"] += int(c.get("checks") or 0)
        agg["passed"] += sum((c.get("passed") or {}).values())
        agg["skipped"] += int(c.get("skipped") or 0) + int(c.get("deferred") or 0)
    spec = model()["by_id"].get(node_id) or (_custom_spec(node_id, 0) if node_id.startswith("node:") else {})
    return {"id": node_id, "ko": spec.get("ko"), "en": spec.get("en"), "decisions": decisions,
            "counts": per_boundary, "timing": timing, "said": said, "custom": bool(spec.get("custom"))}


def summaries(groups: Dict[str, List[dict]]) -> List[dict]:
    """One line per launch, newest first: when it started and ended, its processes and engines, how many decisions
    and the worst state; for the run list also the node where meaning first broke ("where": its names), how many
    nodes are in each state (nodes with nothing decided are left out) and the flows drawn."""
    out = []
    for key, lines in groups.items():
        g = graph(lines)
        broken_at = g["locate"]["broken_at"]
        where = next(({"id": n["id"], "ko": n["ko"], "en": n["en"]} for n in g["nodes"]
                      if broken_at and broken_at in n["boundaries"]), None)
        states: Dict[str, int] = {}
        for n in g["nodes"]:
            if n["state"] != "none":
                states[n["state"]] = states.get(n["state"], 0) + 1
        out.append({"run": key, "start": g["start"], "end": g["end"], "pids": g["pids"], "engines": g["engines"],
                    "decisions": g["decisions"], "state": g["state"], "broken_at": broken_at, "where": where,
                    "states": states, "flows": [f["id"] for f in g["flows"]]})
    out.sort(key=lambda s: (s["start"] is None, -(s["start"] or 0)))
    return out
