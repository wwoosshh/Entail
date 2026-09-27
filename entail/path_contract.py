"""path_contract: the same request answered along two of the engine's own paths that mean the same thing (M19 L3.3c;
LIBRARY_DESIGN.md 11; the rule and its thresholds come from the L2 flow study, lowlevel/l2/compare.py).

An engine answers a request along paths that must not change what the answer means: the decode steps that read the
KV cache the engine wrote (and replay CUDA graphs) against a fresh prefill of the very same tokens; a request alone
against the same request in a batch; a prefix-cache hit against a cold run. Numerics may differ between paths (another
kernel, another reduction order) and can flip a near-tie or move a probability a little; a lost meaning - a cache
entry that stands for other tokens, a graph that replays stale buffers, a kernel that reads another layout at another
batch size - changes confident predictions and moves probabilities a lot. The rule is one, the same for every path:

  paths_disagree   along one of the path pairs, on some probe, a prediction the first path was confident of (top-1
                   ahead of top-2 by more than MARGIN in log-probability) changed, or the probability of a token both
                   paths kept moved by more than PDRIFT - compared only up to the first step where the two paths
                   chose different tokens (after it their contexts differ) - or a log-probability on either path is
                   not a finite number (NaN never equals anything, so two NaN paths do not agree; before this a
                   model whose every log-probability was NaN passed all three pairs, vllm#33560 on vLLM 0.16).

MARGIN and PDRIFT are the L2 study's (lowlevel/l2/RESULTS.md): set above the largest values healthy engines showed
(a confident-flip margin of 0.62 and a probability move of 0.176 over vLLM 0.30 with eight models and transformers
5.17 with three; a second pass over seven vLLM models and fifteen comparisons each gave no difference but a real one).
The records are the engines' greedy outputs with their top log-probabilities per step: {"tokens": [id, ...],
"logprobs": [{id: logprob}, ...], "prompt_tokens": [...], "prompt_logprobs": [{id: logprob} or None, ...]}; a forced
record is {"steps": [{id: logprob}, ...]}, the prompt log-probabilities of the generated positions. The contract
reads no engine: adapters bring the records. No repair is offered: a running engine cannot be given another path;
the disagreement is reported (broken), and a stop policy refuses to serve.
"""
import math
from typing import Dict, List, Optional

from . import tally as _tally

RULE_NAMES = ("paths_disagree",)
MARGIN = 1.0      # log-probability gap of a confident prediction (healthy maximum 0.62)
PDRIFT = 0.25     # probability move of a kept token (healthy maximum 0.176)
PAIRS = {"decode_prefill": "the decode steps against a fresh prefill of the same tokens",
         "alone_batched": "each request alone against the requests in one batch",
         "cold_cache": "a cold run against a run that reads the prefix cache"}
# The probes every engine is given (the same measurement for every adapter): fixed texts written for the L2 study
# (lowlevel/l2/probes.py) - two share a long prefix (the second can hit the prefix cache the first wrote), one is code.
PROBE_TOKENS = 12
TOP = 5
_STORY = (
    "The harbour town woke slowly on the first cold morning of the season. Fishing boats knocked against the pier while "
    "the tide pulled at their ropes, and the smell of salt and diesel drifted up the narrow streets. At the bakery on "
    "the corner, Mara lit the ovens before dawn, as her grandmother had done for forty years, and set out the long "
    "wooden trays that still carried the marks of a thousand loaves. The first customers were always the same: the "
    "harbour master, who wanted two rolls and the weather report, the teacher from the school on the hill, who bought "
    "a loaf for the staff room, and old Tomas, who never bought anything but stayed to talk about the ships he had "
    "sailed on as a young man. That morning Tomas arrived early and did not sit down. He had seen something in the "
    "water beyond the breakwater, he said, a shape that was not a boat and not a whale, moving against the current. ")
PROBES = [
    {"id": "story_a", "text": _STORY + "Mara wiped her hands on her apron and asked him what he thought it was. Tomas"},
    {"id": "story_b", "text": _STORY + "The harbour master laughed and said the old man had been dreaming again, but"},
    {"id": "code", "text": (
        "def merge_intervals(intervals):\n    \"\"\"Merge overlapping [start, end] intervals and return them sorted.\"\"\"\n"
        "    if not intervals:\n        return []\n    intervals = sorted(intervals, key=lambda iv: iv[0])\n"
        "    merged = [list(intervals[0])]\n    for start, end in intervals[1:]:\n        last = merged[-1]\n"
        "        if start <= last[1]:\n            last[1] = max(last[1], end)\n        else:\n")},
]
TARGET = "story_b"          # the probe read again from the prefix cache


def _top2(d: Dict[str, float]):
    vals = sorted(d.values(), reverse=True)
    top = max(d, key=d.get)
    return top, (vals[0] - vals[1]) if len(vals) > 1 else float("inf")


def _dp(la: float, lb: float) -> float:
    return abs(math.exp(la) - math.exp(lb))


def _nonfinite(steps) -> int:
    """How many log-probabilities in a list of {id: logprob} steps are NaN or infinite (empty steps are skipped)."""
    return sum(1 for d in steps or () if d for v in d.values() if not math.isfinite(v))


def generated(a: dict, b: dict) -> dict:
    """Two greedy generations of the same request: the first step where the chosen token differs, how confident the
    first was there, the largest move of a kept token's probability before it, and how many log-probabilities on
    each path are not finite."""
    ta, tb = a["tokens"], b["tokens"]
    n = min(len(ta), len(tb))
    first = next((i for i in range(n) if ta[i] != tb[i]), None)
    upto = first if first is not None else n
    drift = pdrift = 0.0
    for i in range(upto):
        tok = str(ta[i])
        if i >= len(a["logprobs"]) or i >= len(b["logprobs"]):
            break
        la, lb = a["logprobs"][i].get(tok), b["logprobs"][i].get(tok)
        if la is not None and lb is not None:
            drift = max(drift, abs(la - lb))
            pdrift = max(pdrift, _dp(la, lb))
    margin = None
    if first is not None and first < len(a["logprobs"]) and a["logprobs"][first]:
        _, margin = _top2(a["logprobs"][first])
    return {"first_diff": first, "base_margin_at_diff": margin, "drift": drift, "pdrift": pdrift, "steps": n,
            "nonfinite": [_nonfinite(a["logprobs"]), _nonfinite(b["logprobs"])]}


def prompt(a: dict, b: dict) -> Optional[dict]:
    """Teacher-forced prompt log-probabilities of the same tokens, position by position (one difference cannot
    cascade): where the top-1 changed and how confident the first was there, and the largest move of the probability
    of the token that follows."""
    pa, pb = a.get("prompt_logprobs") or [], b.get("prompt_logprobs") or []
    if not pa or not pb or a["prompt_tokens"] != b["prompt_tokens"]:
        return None
    toks = a["prompt_tokens"]
    flips: List[dict] = []
    drift = pdrift = 0.0
    for i in range(1, min(len(pa), len(pb))):
        da, db = pa[i], pb[i]
        if not da or not db:
            continue
        nxt = str(toks[i])
        if nxt in da and nxt in db:
            drift = max(drift, abs(da[nxt] - db[nxt]))
            pdrift = max(pdrift, _dp(da[nxt], db[nxt]))
        ta, ma = _top2(da)
        tb, _ = _top2(db)
        if ta != tb:
            flips.append({"pos": i, "base_margin": ma})
    return {"flips": flips, "max_flip_margin": max((f["base_margin"] for f in flips), default=0.0),
            "drift": drift, "pdrift": pdrift, "positions": min(len(pa), len(pb)) - 1,
            "nonfinite": [_nonfinite(pa[1:]), _nonfinite(pb[1:])]}


def decode_vs_prefill(run: Dict[str, dict], forced: Dict[str, dict]) -> Dict[str, dict]:
    """Per probe: the generated steps' distributions (the decode path) against the same tokens fed back in one fresh
    prefill (the forced steps), as a prompt comparison of the generated positions."""
    out = {}
    for pid, a in run.items():
        f = forced.get(pid)
        if f is None or not a.get("logprobs"):
            continue
        n = min(len(a["tokens"]), len(a["logprobs"]), len(f["steps"]))
        toks = [0] + list(a["tokens"][:n])
        pa = {"prompt_tokens": toks, "prompt_logprobs": [None] + list(a["logprobs"][:n])}
        pb = {"prompt_tokens": toks, "prompt_logprobs": [None] + list(f["steps"][:n])}
        out[pid] = {"generated": {"first_diff": None, "base_margin_at_diff": None, "drift": 0.0, "pdrift": 0.0,
                                  "steps": len(a["tokens"]), "nonfinite": [0, 0]},
                    "prompt": prompt(pb, pa)}
    return out


def pair(base: Dict[str, dict], other: Dict[str, dict]) -> Dict[str, dict]:
    """Per probe id: the generated and the prompt comparison of `other` against `base`."""
    return {pid: {"generated": generated(a, other[pid]), "prompt": prompt(a, other[pid])}
            for pid, a in base.items() if pid in other}


def verdict(cmp: Dict[str, dict], margin: float = None, pdrift: float = None) -> dict:
    """Whether some probe changed a confident prediction or moved a kept token's probability beyond the thresholds;
    the first such probe is named, with the largest values over all probes."""
    margin = MARGIN if margin is None else margin
    pdrift = PDRIFT if pdrift is None else pdrift
    worst_margin = worst_pdrift = 0.0
    found = None
    for pid, c in cmp.items():
        g, p = c["generated"], c["prompt"]
        reasons = []
        gm = (g["base_margin_at_diff"] or 0.0) if g["first_diff"] is not None else 0.0
        worst_margin = max(worst_margin, gm, (p or {}).get("max_flip_margin", 0.0))
        worst_pdrift = max(worst_pdrift, g["pdrift"], (p or {}).get("pdrift", 0.0))
        if g["first_diff"] is not None and gm > margin:
            reasons.append(f"generated token {g['first_diff']} changed where the first path was confident "
                           f"(margin {gm:.2f})")
        if g["pdrift"] > pdrift:
            reasons.append(f"a kept generated token's probability moved by {g['pdrift']:.3f}")
        if p is not None and p["max_flip_margin"] > margin:
            reasons.append(f"{len([f for f in p['flips'] if f['base_margin'] > margin])} positions changed a "
                           f"confident prediction (margin up to {p['max_flip_margin']:.2f})")
        if p is not None and p["pdrift"] > pdrift:
            reasons.append(f"a token's probability moved by {p['pdrift']:.3f}")
        bad = [x + y for x, y in zip(g.get("nonfinite") or (0, 0), (p or {}).get("nonfinite") or (0, 0))]
        if any(bad):
            reasons.append(f"log-probabilities are not finite numbers ({bad[0]} on the first path, {bad[1]} on the "
                           f"second)")
        if reasons and found is None:
            found = {"probe": pid, "why": reasons}
    return {"differs": found is not None, "probe": found and found["probe"], "why": found and found["why"],
            "margin": worst_margin, "pdrift": worst_pdrift, "probes": len(cmp)}


def check(boundary: str, consumer: str, paths: str, v: dict, where: str, owner=None, policy=None,
          record: bool = True) -> list:
    """Decide one path pair from verdict()'s result. Returns the decisions; with `record` they go through
    load.enforce (the run goes on, or stops where the policy says)."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty, Fact, PathAgreement, Source

    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("PathAgreement",), ("PathAgreement",))
    declared = Fact("PathAgreement", PathAgreement(paths=paths, probes=v["probes"], flip_margin=0.0, pdrift=0.0),
                    Source("engine", f"{PAIRS[paths]}: one request, one meaning"), Certainty.VERIFIED)
    held = Fact("PathAgreement", PathAgreement(paths=paths, probes=v["probes"], flip_margin=round(v["margin"], 6),
                                               pdrift=round(v["pdrift"], 6)), Source("engine", where),
                Certainty.VERIFIED)
    numbers = (f"{v['probes']} probes; largest confident-flip margin {v['margin']:.3g} (threshold {MARGIN:g}), largest "
               f"probability move {v['pdrift']:.3g} (threshold {PDRIFT:g})")
    if v["differs"]:
        verdict_, blocking = unrepaired(policy, "PathAgreement")
        d = Decision(contract, "PathAgreement", verdict_, RULES["paths_disagree"], declared=declared, chosen=held,
                     blocking=blocking, note=f"{where}: {PAIRS[paths]} disagree on probe {v['probe']!r}: "
                                            f"{'; '.join(v['why'])} ({numbers})")
    else:
        d = Decision(contract, "PathAgreement", Verdict.PASS, RULES["match"], declared=declared, chosen=held,
                     note=f"{where}: {PAIRS[paths]} agree ({numbers})")
    decisions = [d]
    if record:
        _tally.counts(boundary)["checks"] += 1
        if d.verdict is Verdict.PASS:
            _tally.passed(boundary, ["paths"])
        elif d.blocking:
            _tally.refused(boundary)
        else:
            _tally.broken(boundary)
        _tally.tick(boundary)
        load.enforce(decisions, once_for=owner)
    return decisions


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
