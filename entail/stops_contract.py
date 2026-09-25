"""stops_contract: the ids at which a generation ends, in the core (LIBRARY_DESIGN.md 4.6; ROADMAP M15.8).

Where a generation ends is a meaning the model's files declare, and they declare it in up to three places:
generation_config.json (`eos_token_id`, an id or a list), config.json (`eos_token_id`) and the tokenizer (its
`eos_token`, an id once the tokenizer is built). Each engine builds its stop set from a different subset: transformers
5.17 from generation_config.json alone (config.json only when that file is absent; the tokenizer's eos is not
consulted), vLLM 0.30 from the tokenizer's eos plus generation_config.json, SGLang 0.5.20 from config.json plus
generation_config.json (data/stops_sources.json names the code). An id one file declares and the engine's subset
misses is an end the engine does not see: the model emits it and the run goes on to max_tokens (Llama 3, April 2024:
config.json named 128001, the model ended its answers with 128009, and the engines that read only config.json never
stopped; the fix was to declare both in generation_config.json).

The fact is Stops (vocabulary v7): the eos ids one source states, and its bos and pad. The contract takes the UNION
over the sources - every id any file calls an end is an end - and the rules are here, once:

  stop_dropped          the consumer's stop set lacks an id some source declares as an end. The repair is to add the
                        missing ids to the consumer's set (the adapter's `add_stops` handle: transformers'
                        generation_config, vLLM's generation-config fields, SGLang's model config all carry a list);
                        with it the decision is resolved, without it broken (reported, the run goes on; M5.4), or
                        refused where the policy stops.
  stop_id_out_of_range  a declared eos, bos or pad id is past the tokenizer's highest id: no token, so no stop.

A declared eos id the tokenizer does not mark as special is noted on the decision (deepseek-coder's <|EOT|>): the
stop still matches by id. A consumer whose set covers every declared id passes. Nothing here reads a device; an
error inside entail never breaks the engine (the adapters run this under load.safely).
"""
from typing import Dict, List, Optional, Sequence, Set, Tuple

from . import tally as _tally

RULE_NAMES = ("stop_dropped", "stop_id_out_of_range")


def declared_stops(facts) -> Tuple[Dict[int, List[str]], List[object]]:
    """eos id -> the sources that declare it, over every Stops fact in `facts` (a load.Declared, or a list of
    Facts); and the facts themselves."""
    stops = [f for f in (facts.get("Stops") if hasattr(facts, "get") else facts) if f.value is not None]
    ids: Dict[int, List[str]] = {}
    for f in stops:
        for i in f.value.eos:
            ids.setdefault(int(i), []).append(f.source.where)
    return ids, stops


def held_by(engine: str, facts, tokenizer_eos: Optional[int] = None, table=None) -> Optional[Set[int]]:
    """The stop set an engine would build from these declarations, by data/stops_sources.json (the static check;
    at run time the adapters read the set the engine holds). None when the table does not name the engine."""
    table = table or sources_table()
    entry = table.get(engine)
    if not isinstance(entry, dict):
        return None
    ids, stops = declared_stops(facts)
    by_kind: Dict[str, Set[int]] = {}
    for f in stops:
        kind = "generation_config" if "generation_config.json" in f.source.where else \
            "tokenizer" if "tokenizer" in f.source.where.lower() else "config"
        by_kind.setdefault(kind, set()).update(int(i) for i in f.value.eos)
    if tokenizer_eos is not None:
        by_kind.setdefault("tokenizer", set()).add(int(tokenizer_eos))
    held: Set[int] = set()
    reads = list(entry.get("reads", ()))
    if entry.get("fallback") and "generation_config" in reads and "generation_config" not in by_kind:
        reads.append(entry["fallback"])
    for kind in reads:
        held |= by_kind.get(kind, set())
    return held


_TABLE = None


def sources_table() -> dict:
    global _TABLE
    if _TABLE is None:
        import json
        import os

        with open(os.path.join(os.path.dirname(__file__), "data", "stops_sources.json"), encoding="utf-8") as f:
            _TABLE = json.load(f)
    return _TABLE


def check(boundary: str, consumer: str, facts, held: Optional[Set[int]], held_where: str, add_stops=None,
          tokenizer_size: Optional[int] = None, special_ids: Optional[Sequence[int]] = None, owner=None,
          policy=None, record: bool = True) -> list:
    """Decide one consumer's stop set against what the files declare. `facts`: load.Declared (its Stops facts: the
    files', and the tokenizer's when the caller built one). `held`: the eos ids the consumer's set holds (None: not
    read - only the ids themselves are checked). `add_stops(ids)`: the one repair, offered by the adapter.
    `tokenizer_size`: the tokenizer's highest id + 1, for stop_id_out_of_range. Returns the decisions; with `record`
    they are recorded through load.enforce (the run goes on, or stops where the policy says)."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Resolution, Verdict, decide, unrepaired
    from .facts import Certainty, Fact, Source, Stops

    policy = policy or policies.current()
    ids, stops = declared_stops(facts)
    if not stops:
        return []          # no file says where a generation ends: nothing to hold the consumer to
    contract = Contract(boundary, consumer, ("Stops",))
    decisions: List[Decision] = []
    union = tuple(sorted(ids))
    wheres = sorted({w for ws in ids.values() for w in ws})
    declared = Fact("Stops", Stops(eos=union), Source("config", "the union of " + "; ".join(wheres)),
                    Certainty.DECLARED)
    if held is not None:
        chosen = Fact("Stops", Stops(eos=tuple(sorted(int(i) for i in held))), Source("engine", held_where),
                      Certainty.VERIFIED)
    else:
        chosen = Fact("Stops", None, Source("engine", f"{held_where}: the stop set was not read"), Certainty.UNKNOWN)

    # the ids themselves: an id past the tokenizer is no token (bos and pad too)
    if tokenizer_size is not None:
        for f in stops:
            named = [("eos", i) for i in f.value.eos] + [(n, getattr(f.value, n)) for n in ("bos", "pad")
                                                          if getattr(f.value, n) is not None]
            bad = [(n, i) for n, i in named if int(i) >= int(tokenizer_size)]
            if bad:
                verdict, blocking = unrepaired(policy, "Stops")
                decisions.append(Decision(contract, "Stops", verdict, RULES["stop_id_out_of_range"], declared=f,
                                          chosen=chosen, blocking=blocking,
                                          note=", ".join(f"{n} id {i}" for n, i in bad)
                                          + f" ({f.source.where}) is past the tokenizer's highest id "
                                          f"{int(tokenizer_size) - 1}"))
    # the consumer's set against the union
    if held is not None:
        missing = tuple(i for i in union if i not in held)
        if missing:
            note = "the consumer's stop set lacks " + ", ".join(
                f"{i} (declared by {'; '.join(ids[i])})" for i in missing)
            add = Resolution("add the declared ids to the consumer's stop set", "add_stops",
                             target=lambda d, c, m=missing: m) if add_stops is not None else None
            out = decide(contract, {"Stops": declared}, {"Stops": chosen}, policy,
                         resolutions={"Stops": [add]} if add else None)
            from dataclasses import replace
            # the ladder gives the verdict (resolved with the repair, else broken or refused); the rule is this one
            decisions += [replace(d, rule=RULES["stop_dropped"], note=(note + ("; " + d.note if d.note else "")))
                          for d in out]
    if not any(d.verdict is not Verdict.PASS for d in decisions):
        note = ""
        if special_ids is not None:
            plain = [i for i in union if i not in set(int(s) for s in special_ids)]
            if plain:
                note = (f"eos id(s) {', '.join(str(i) for i in plain)} are not special tokens of the tokenizer "
                        f"(a stop still matches by id; encoding text may split them)")
        if not decisions:
            decisions.append(Decision(contract, "Stops", Verdict.PASS, RULES["match"], declared=declared,
                                      chosen=chosen if held is not None else declared, note=note))
    if record:
        _tally.counts(boundary)["checks"] += 1
        done = load.resolve(decisions, {"add_stops": add_stops} if add_stops is not None else {})
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
