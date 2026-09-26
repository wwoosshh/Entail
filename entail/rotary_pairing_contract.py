"""rotary_pairing_contract: how a rotary embedding pairs the dimensions it rotates, from the declaration to the
layer and the kernel (ROADMAP M17.4; testbed/results/m15/PAIRING_CANDIDATE.md; data/rotary_pairing.json).

Two conventions exist: 'split' rotates dimension i with i + d/2 (rotate_half; Llama, Qwen, Gemma) and 'interleaved'
rotates 2i with 2i+1 (GPT-J; GLM, Cohere, Ernie 4.5). A checkpoint rarely declares which in config.json; its
architecture's reference implementation does, in code, and a few configs carry a key (rope_interleave,
is_neox_style). An engine that builds the layer with the other convention, or a kernel that pairs split-wise
whatever the layer says, rotates every position wrongly and the model still runs (vllm#42016: the Triton MRoPE
kernel until 0.27.0; vllm#49290; vllm#53063: a draft model that did not inherit the target's pairing).

The rules, in the core:

  rotary_pairing_mismatch   a rotary layer of the built model pairs differently from the declaration. Resolved where
                            the adapter can set the layer's convention (vLLM's RotaryEmbedding.is_neox_style, read at
                            every forward), broken otherwise.
  rotary_pairing_ignored    the engine's kernel path pairs split-wise regardless of the layer (the table's `kernels`,
                            by version) and the model declares interleaved: broken (nothing to set).

A model_type the table does not know, without a key, declares nothing: no decision (a convention entail cannot
invent). Nothing here reads a device; the adapter runs this under load.safely.
"""
import json
import os
from typing import Dict, List, Optional, Tuple

from . import tally as _tally

RULE_NAMES = ("rotary_pairing_mismatch", "rotary_pairing_ignored")
_TABLE = None


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        with open(os.path.join(os.path.dirname(__file__), "data", "rotary_pairing.json"), encoding="utf-8") as f:
            _TABLE = json.load(f)
    return _TABLE


def _version_tuple(v: Optional[str]) -> Tuple[int, ...]:
    out = []
    for part in str(v or "").split("."):
        digits = ""
        for ch in part:                 # the leading digits only: "0rc1" is 0, not 01
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def declared(config: Optional[dict], model_type: Optional[str] = None, facts=None) -> Tuple[Optional[str], str]:
    """(pairing, where) as the files and the architecture declare it: a Rotary fact that carries `pairing` (v8),
    else a config key (config.json, then its text_config), else the architecture table by model_type. (None, why)
    when nothing declares it."""
    t = table()
    for f in (facts.get("Rotary") if facts is not None and hasattr(facts, "get") else []) or []:
        p = getattr(getattr(f, "value", None), "pairing", None)
        if p in t["values"]:
            return p, f.source.where
    cfgs = []
    if isinstance(config, dict):
        cfgs.append(("config.json", config))
        tc = config.get("text_config")
        if isinstance(tc, dict):
            cfgs.append(("config.json#text_config", tc))
    for where, cfg in cfgs:
        for key, spec in t["keys"].items():
            if key in cfg and isinstance(cfg[key], bool):
                return spec["true" if cfg[key] else "false"], f"{where}#{key}"
    types = [model_type] if model_type else []
    for where, cfg in cfgs:
        mt = cfg.get("model_type")
        if isinstance(mt, str):
            types.append(mt)
    for mt in types:
        row = t["architectures"].get(mt)
        if row:
            return row["pairing"], f"the reference implementation of {mt} ({row['ref']})"
    return None, f"no key and no known architecture ({', '.join(types) or 'no model_type'})"


def kernel_ignores(engine: str, version: Optional[str], mrope: bool) -> Optional[dict]:
    """The table's kernel row that ignores the pairing on this engine version for this model, or None."""
    for name, row in table().get("kernels", {}).items():
        if not name.startswith(engine + "."):
            continue
        if row.get("applies_to", "").startswith("models with mrope") and not mrope:
            continue
        before = row.get("ignores_pairing_before")
        if before and version and _version_tuple(version) < _version_tuple(before):
            return dict(row, name=name)
    return None


def check(boundary: str, consumer: str, engine: str, pairing: Optional[str], where: str, held: Dict[str, str],
          held_where: str, version: Optional[str] = None, mrope: bool = False, handles: Optional[dict] = None,
          owner=None, policy=None, record: bool = True) -> list:
    """Decide the built model's rotary layers (`held`: layer name -> 'split' | 'interleaved') and the engine's kernel
    path (by version) against the declared pairing. Nothing is decided without a declaration."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Resolution, Verdict, decide, unrepaired
    from .facts import Certainty, Fact, Source

    if pairing is None:
        return []
    policy = policy or policies.current()
    handles = handles or {}
    contract = Contract(boundary, consumer, ("Rotary",), ("Rotary",))
    decisions: List[Decision] = []
    # facts for the ledger: the declared convention and the layers' convention, carried as Coverage-free strings
    declared_fact = Fact("Rotary", None, Source("config", f"{where}: pairing {pairing}"), Certainty.UNKNOWN)
    row = kernel_ignores(engine, version, mrope)
    if row is not None:
        if pairing == "interleaved":
            verdict, blocking = unrepaired(policy, "Rotary")
            decisions.append(Decision(contract, "Rotary", verdict, RULES["rotary_pairing_ignored"], blocking=blocking,
                                      note=(f"{where} declares interleaved pairing; {row['name']} ({engine} {version}) "
                                            f"pairs split-wise regardless of the layer ({row['ref']})")))
        else:
            decisions.append(Decision(contract, "Rotary", Verdict.PASS, RULES["match"],
                                      note=f"{row['name']} pairs split-wise, as declared"))
    wrong = {name: style for name, style in (held or {}).items() if style != pairing}
    if wrong:
        chosen = Fact("Rotary", None, Source("engine", f"{held_where}: {len(wrong)} layer(s) pair "
                                                       f"{sorted(set(wrong.values()))[0]}"), Certainty.UNKNOWN)
        res = None
        if "set_pairing" in handles:
            res = Resolution(f"set the layers' pairing to {pairing}", "set_pairing", target=lambda d, c, p=pairing: p)
        # decide() compares fact values; here the values are the conventions themselves
        from .coverage import Coverage
        d_fact = Fact("Coverage", Coverage(1, 1, ()), Source("config", f"{where}: pairing {pairing}"),
                      Certainty.DECLARED)
        c_fact = Fact("Coverage", Coverage(1, 0, ("pairing",)), chosen.source, Certainty.VERIFIED)
        out = decide(Contract(boundary, consumer, ("Coverage",), ("Coverage",)), {"Coverage": d_fact},
                     {"Coverage": c_fact}, policy, resolutions={"Coverage": [res]} if res else None)
        from dataclasses import replace
        first = sorted(wrong)[0]
        note = (f"{where} declares {pairing} pairing; {len(wrong)} rotary layer(s) of the built model pair "
                f"{wrong[first]} (first: {first})")
        decisions += [replace(d, name="Rotary", rule=RULES["rotary_pairing_mismatch"],
                              note=note + ("; " + d.note if d.note else "")) for d in out]
    elif held and row is None:
        decisions.append(Decision(contract, "Rotary", Verdict.PASS, RULES["match"],
                                  note=f"{len(held)} rotary layer(s) pair {pairing}, as declared"))
    if not decisions:
        return []
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
    _ = declared_fact
    return decisions


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
