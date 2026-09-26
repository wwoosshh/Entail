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
    """The table's kernel row that ignores the pairing on this engine version for this model, or None. `mrope`:
    the model's MRoPE module dispatches to that kernel (the adapter reads the dispatched forward; under vLLM's
    default compiled CUDA path the native forward runs and honours the layer, so the row does not apply)."""
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
    """Decide the built model's rotary modules (`held`: module name -> 'split' | 'interleaved', the part of the
    model the declaration is about) and the engine's kernel path (by version; `mrope` = an MRoPE module dispatches
    to that kernel) against the declared pairing. Nothing is decided without a declaration. Modules that pair both
    ways in the same part (a DSA indexer with its own key beside the main rotary) are said unknown and left alone:
    which of them consume the declaration cannot be told from the model, and a flip would move the ones that pair
    by their own key (M17.4 review, finding 10)."""
    from dataclasses import replace

    from . import load, policies
    from .contracts import RULES, Contract, Decision, Resolution, Verdict, decide, unrepaired
    from .facts import Certainty, Fact, Rotary, Source

    if pairing is None:
        return []
    policy = policy or policies.current()
    handles = handles or {}
    held = held or {}
    contract = Contract(boundary, consumer, ("Rotary",), ("Rotary",))
    decisions: List[Decision] = []
    declared_fact = Fact("Rotary", Rotary(pairing=pairing), Source("config", where), Certainty.DECLARED)
    row = kernel_ignores(engine, version, mrope)
    if row is not None:
        kernel_fact = Fact("Rotary", Rotary(pairing="split"), Source("engine", f"{row['name']} ({engine} {version})"),
                           Certainty.VERIFIED)
        if pairing == "interleaved":
            verdict, blocking = unrepaired(policy, "Rotary")
            decisions.append(Decision(contract, "Rotary", verdict, RULES["rotary_pairing_ignored"], blocking=blocking,
                                      declared=declared_fact, chosen=kernel_fact,
                                      note=(f"{where} declares interleaved pairing; {row['name']} ({engine} {version}) "
                                            f"pairs split-wise regardless of the layer ({row['ref']})")))
        else:
            decisions.append(Decision(contract, "Rotary", Verdict.PASS, RULES["match"], declared=declared_fact,
                                      chosen=kernel_fact, note=f"{row['name']} pairs split-wise, as declared"))
    styles = sorted(set(held.values()))
    wrong = {name: style for name, style in held.items() if style != pairing}
    if len(styles) > 1:
        counts = ", ".join(f"{sum(1 for s in held.values() if s == st)} {st}" for st in styles)
        decisions.append(load.cannot_check(boundary, consumer, "Rotary",
                                           (f"{held_where}: the rotary modules pair both ways ({counts}); which of "
                                            f"them consume the declaration ({where}: {pairing}) cannot be told from "
                                            f"the model, so nothing is set"), policy))
    elif wrong:
        first = sorted(wrong)[0]
        chosen = Fact("Rotary", Rotary(pairing=styles[0]),
                      Source("engine", f"{held_where}: {len(wrong)} rotary module(s) pair {styles[0]}"),
                      Certainty.VERIFIED)
        res = None
        if "set_pairing" in handles:
            res = Resolution(f"set the modules' pairing to {pairing}", "set_pairing",
                             target=lambda d, c, p=pairing: p)
        out = decide(contract, {"Rotary": declared_fact}, {"Rotary": chosen}, policy,
                     resolutions={"Rotary": [res]} if res else None)
        note = (f"{where} declares {pairing} pairing; {len(wrong)} rotary module(s) of the built model pair "
                f"{wrong[first]} (first: {first})")
        decisions += [replace(d, rule=RULES["rotary_pairing_mismatch"], note=note + ("; " + d.note if d.note else ""))
                      for d in out]
    elif held and row is None:
        decisions.append(Decision(contract, "Rotary", Verdict.PASS, RULES["match"], declared=declared_fact,
                                  chosen=Fact("Rotary", Rotary(pairing=pairing), Source("engine", held_where),
                                              Certainty.VERIFIED),
                                  note=f"{len(held)} rotary module(s) pair {pairing}, as declared"))
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
    return decisions


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
