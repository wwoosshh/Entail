"""adapter_config_contract: what a LoRA adapter's config declares against what its consumer reads (ROADMAP M17.1).

A PEFT adapter's adapter_config.json is a declaration file like config.json: it says how the adapter's weights are to
be applied - the scale rule (use_rslora), per-module ranks and alphas, a bias on lora_B, transposed weights, fully
saved modules, activation tokens. Each engine reads a subset (data/adapter_config_keys.json names the code lines):
PEFT reads all of them, vLLM 0.30 the scale rule and a few it refuses loudly, SGLang 0.5.20 only the rank, alpha and
targets. A key declared with a value that changes how the weights apply, that the consumer never reads, is a meaning
lost at load: the adapter loads without a word and applies with the wrong scale (sglang#40835: an rsLoRA adapter
served 4-8 times too weak).

The rule is one, in the core, for every consumer and every key:

  adapter_key_dropped   the consumer does not read a declared key whose value is not its neutral value. Resolved
                        where the adapter can carry the value into the consumer (SGLang's `scaling` for use_rslora),
                        broken (reported, the run goes on; M5.4) or refused where the policy stops otherwise.
  adapter_key_unknown   a key entail does not know, declared with a value that is not empty: said once, unknown.

A key with its neutral value decides nothing (not reading a default changes nothing), nor does a key the weights carry
(which modules exist) or a training-time key. A key the consumer refuses loudly is not silent, so it passes with a
note. Nothing here reads a device; adapters run this under load.safely.
"""
import json
import os
from typing import Dict, List, Optional

from . import tally as _tally

RULE_NAMES = ("adapter_key_dropped", "adapter_key_unknown")
_TABLE = None
_EMPTY = (None, False, 0, "", "none", [], {})


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        with open(os.path.join(os.path.dirname(__file__), "data", "adapter_config_keys.json"), encoding="utf-8") as f:
            _TABLE = json.load(f)
    return _TABLE


def read(path_or_dict) -> Optional[dict]:
    """The declared keys: a dict as given, or the folder's adapter_config.json; None when there is none."""
    if isinstance(path_or_dict, dict):
        return dict(path_or_dict)
    p = os.path.join(os.path.expanduser(str(path_or_dict)), "adapter_config.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d if isinstance(d, dict) else None


def carried_value(consumer: str, key: str, declared: dict):
    """The value an adapter's handle sets into the consumer for a dropped key, computed HERE (principle 8: the rule
    and its arithmetic live in the core, the adapter only assigns): use_rslora -> the scaling lora_alpha / sqrt(r)
    that PEFT applies (tuners/lora/layer.py L278-279). None when the table says the consumer carries nothing."""
    import math

    if key not in table().get("carries", {}).get(consumer, {}):
        return None
    if key == "use_rslora":
        return (float(declared["lora_alpha"]) / math.sqrt(float(declared["r"]))) if declared.get(key) \
            else float(declared["lora_alpha"]) / float(declared["r"])
    return declared.get(key)


def _neutral(spec: dict, value) -> bool:
    """A value under which not reading the key changes nothing: the key's neutral value, and null (PEFT 0.21 writes
    `use_bdlora: null` and `arrow_config: null` into every file - a null declares nothing; the first real-engine
    run flagged it on every adapter, M17.1) or an empty container where the neutral value is one."""
    if value is None:
        return True
    n = spec.get("neutral")
    if value == n:
        return True
    return n in ({}, None, []) and value in ({}, [])


def classify(declared: dict, consumer: str, tbl: Optional[dict] = None) -> Dict[str, List[str]]:
    """Each declared key into taken / dropped / unknown for this consumer, by the table: taken = read by the
    consumer, refused loudly, neutral, carried by the weights or training-only; dropped = an inference key with a
    value that is not neutral and that the consumer does not read; unknown = not in the table (with a value that is
    not empty)."""
    tbl = tbl or table()
    keys, cons = tbl["keys"], tbl["consumers"].get(consumer)
    if cons is None:
        raise ValueError(f"adapter_config_keys.json names no consumer {consumer!r}")
    reads = cons.get("reads")
    reads_all = reads == "all"
    refuses = set(cons.get("refuses", ()))
    out: Dict[str, List[str]] = {"taken": [], "dropped": [], "unknown": [], "refused": []}
    for k, v in declared.items():
        spec = keys.get(k)
        if spec is None:
            if v not in _EMPTY:
                out["unknown"].append(k)
            continue
        kind = spec.get("kind")
        if kind in ("weights", "carried_by_weights", "train_only") or _neutral(spec, v) or reads_all \
                or k in (reads or ()):
            if k in refuses and not _neutral(spec, v):
                out["refused"].append(k)
            out["taken"].append(k)
        elif k in refuses:
            out["refused"].append(k)
            out["taken"].append(k)
        else:
            out["dropped"].append(k)
    return out


def check(boundary: str, consumer: str, engine: str, declared: Optional[dict], where: str,
          handles: Optional[dict] = None, owner=None, policy=None, record: bool = True) -> list:
    """Decide one consumer's reading of an adapter's declaration. `consumer`: a name in the table's consumers
    (peft, vllm, sglang). `declared`: the adapter_config.json dict (None or empty: nothing to decide). `handles`:
    the adapter's repairs, named apply_<key>; a dropped key with a handle is resolved (the handle gets the declared
    value), one without is broken. Returns the decisions; with `record` they go through load.enforce."""
    from dataclasses import replace

    from . import load, policies
    from .contracts import RULES, Contract, Decision, Resolution, Verdict, decide
    from .coverage import Coverage
    from .facts import Certainty, Fact, Source

    if not declared:
        return []
    policy = policy or policies.current()
    handles = handles or {}
    tbl = table()
    groups = classify(declared, consumer, tbl)
    contract = Contract(boundary, f"{engine}.{consumer}" if not consumer.startswith(engine) else consumer,
                        ("Coverage",), ("Coverage",))
    decisions: List[Decision] = []
    effective = [k for k in declared if k in tbl["keys"]]
    for k in groups["dropped"]:
        spec = tbl["keys"][k]
        v = declared[k]
        declared_fact = Fact("Coverage", Coverage(1, 1, ()), Source("file", f"{where}#{k}"), Certainty.DECLARED)
        chosen = Fact("Coverage", Coverage(1, 0, (k,)),
                      Source("engine", f"{consumer} reads {', '.join(tbl['consumers'][consumer].get('reads') or ())}"),
                      Certainty.VERIFIED)
        res = None
        if f"apply_{k}" in handles:
            target = carried_value(consumer, k, declared)
            target = v if target is None else target
            res = Resolution(f"carry the declared {k} into the consumer", f"apply_{k}",
                             target=lambda d, c, t=target: t)
        out = decide(contract, {"Coverage": declared_fact}, {"Coverage": chosen}, policy,
                     resolutions={"Coverage": [res]} if res else None)
        note = f"{k}={json.dumps(v)} is declared and {consumer} does not read it: {spec.get('effect', '')}".rstrip(": ")
        decisions += [replace(d, rule=RULES["adapter_key_dropped"], note=note + ("; " + d.note if d.note else ""))
                      for d in out]
    if groups["unknown"]:
        names = ", ".join(sorted(groups["unknown"]))
        decisions.append(Decision(contract, "Coverage", Verdict.UNKNOWN, RULES["adapter_key_unknown"],
                                  declared=Fact("Coverage", Coverage(len(groups["unknown"]), 0,
                                                                     tuple(sorted(groups["unknown"]))),
                                                Source("file", where), Certainty.DECLARED),
                                  note=f"keys entail does not know, with values: {names}"))
    if not groups["dropped"]:
        note = ""
        if groups["refused"]:
            note = f"{', '.join(sorted(groups['refused']))}: {consumer} refuses these loudly, not silently"
        taken = Fact("Coverage", Coverage(len(effective), len(effective), ()), Source("file", where),
                     Certainty.DECLARED)
        decisions.append(Decision(contract, "Coverage", Verdict.PASS, RULES["match"], declared=taken, chosen=taken,
                                  note=note))
    if record:
        _tally.counts(boundary)["checks"] += 1
        done = load.resolve(decisions, handles)
        if all(d.verdict is Verdict.PASS for d in decisions):
            _tally.passed(boundary, list(RULE_NAMES))
        load.enforce([d for d in decisions if d.verdict is not Verdict.PASS] or decisions, once_for=owner)
        if any(d.blocking for d in decisions):
            _tally.refused(boundary)
        elif any(d.verdict is Verdict.BROKEN for d in decisions):
            _tally.broken(boundary)
        elif done:
            _tally.counts(boundary)["resolved"] += 1
        _tally.tick(boundary)
    return decisions


def static_handles(engine: str) -> dict:
    """For `entail check`: the repairs the engine's adapter carries at load, so a static decision says resolved where
    the load will resolve (as the stop-set check does for add_stops)."""
    return {f"apply_{k}": (lambda v: True) for k in table().get("carries", {}).get(engine, {})}


def consumer_of(engine: str) -> Optional[str]:
    """The table's consumer name for an engine: transformers and diffusers hand the file to PEFT."""
    if engine in ("transformers", "diffusers"):
        return "peft"
    return engine if engine in table()["consumers"] else None


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
