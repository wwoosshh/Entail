"""cache_key_contract: a store's key must cover every field that shaped the item; a permutation must move every
state (ROADMAP M17.2; LIBRARY_DESIGN.md 4.6 Identity; data/cache_key_fields.json).

vLLM's prefix cache keys a request's blocks by the tokens plus extra keys (LoRA name, multimodal hashes, cache_salt,
a digest of prompt embeddings). A request field that shapes the model's input and is not in the key (0.30.0:
prompt_is_token_ids, the mask that says which positions take the embeddings) lets a later request with the same
tokens and embeddings but another mask be served the first one's KV (vllm#56655). transformers' beam search reorders
the cache under the names it knows; a model whose cache lives under another name keeps its state unmoved across
beams (transformers#46612 on 5.12.1).

The rule is one, in the core, as a declaration-consumption table like the config keys' (adapter_config_contract):

  cache_key_incomplete  a declared field (a request field that shapes the input; a model's cache name) that the
                        consumer's key or permutation does not cover. Resolved where the adapter can extend the key
                        (vLLM: an extra key per block from the mask, the hashes remade); broken otherwise (reported,
                        the run goes on; M5.4), or refused where the policy stops.

Nothing here reads a device; the adapters run this under load.safely.
"""
import json
import os
from typing import Dict, List, Optional

from . import tally as _tally

RULE_NAMES = ("cache_key_incomplete",)
_TABLE = None


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        with open(os.path.join(os.path.dirname(__file__), "data", "cache_key_fields.json"), encoding="utf-8") as f:
            _TABLE = json.load(f)
    return _TABLE


def consumer_row(consumer: str, version: Optional[str] = None) -> Optional[dict]:
    """The consumer's row for the installed version when the table has one, else its base row."""
    rows = table()["consumers"]
    return (rows.get(f"{consumer}@{version}") if version else None) or rows.get(consumer)


def check(boundary: str, consumer: str, engine: str, present: Dict[str, bool], where: str,
          handles: Optional[dict] = None, owner=None, policy=None, record: bool = True,
          version: Optional[str] = None) -> list:
    """Decide one consumer's key against the fields present on the item. `present`: field name -> whether the item
    declares it (a request field set; a model's cache name in its forward). `handles`: extend_key_<field> repairs.
    Returns the decisions; with `record` they go through load.enforce (once per set of present fields, `owner`)."""
    from dataclasses import replace

    from . import load, policies
    from .contracts import RULES, Contract, Decision, Resolution, Verdict, decide
    from .coverage import Coverage
    from .facts import Certainty, Fact, Source

    row = consumer_row(consumer, version)
    if row is None:
        raise ValueError(f"cache_key_fields.json names no consumer {consumer!r}")
    fields = table()["fields"][row["fields"]]["names"]
    declared_names = [f for f in fields if present.get(f)]
    if not declared_names:
        return []
    policy = policy or policies.current()
    handles = handles or {}
    reads = set(row.get("reads", ()))
    missing = [f for f in declared_names if f not in reads]
    contract = Contract(boundary, consumer, ("Coverage",), ("Coverage",))
    decisions: List[Decision] = []
    for f in missing:
        declared_fact = Fact("Coverage", Coverage(1, 1, ()), Source("user", f"{where}#{f}"), Certainty.DECLARED)
        chosen = Fact("Coverage", Coverage(1, 0, (f,)),
                      Source("engine", f"{consumer} {row.get('version', '')} keys by {', '.join(sorted(reads))}"),
                      Certainty.VERIFIED)
        res = None
        if f"extend_key_{f}" in handles:
            res = Resolution(f"extend the key by {f}", f"extend_key_{f}", target=lambda d, c, f=f: f)
        out = decide(contract, {"Coverage": declared_fact}, {"Coverage": chosen}, policy,
                     resolutions={"Coverage": [res]} if res else None)
        note = (f"{f} shapes the item and the key does not cover it: another item with the same key can be served "
                f"under it")
        decisions += [replace(d, rule=RULES["cache_key_incomplete"], note=note + ("; " + d.note if d.note else ""))
                      for d in out]
    if not missing:
        fact = Fact("Coverage", Coverage(len(declared_names), len(declared_names), ()), Source("user", where),
                    Certainty.DECLARED)
        decisions.append(Decision(contract, "Coverage", Verdict.PASS, RULES["match"], declared=fact, chosen=fact))
    if record:
        _tally.counts(boundary)["checks"] += 1
        done = load.resolve(decisions, handles)
        if all(d.verdict is Verdict.PASS for d in decisions):
            _tally.passed(boundary, list(RULE_NAMES))
        load.enforce([d for d in decisions if d.verdict is not Verdict.PASS] or decisions,
                     once_for=owner if owner is not None else tuple(declared_names))
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
