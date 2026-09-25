"""contracts: compare what was declared with what a consumer uses, and decide (LIBRARY_DESIGN.md 4.5, 7; ROADMAP M1.2).

For every vocabulary name a contract needs, `decide` looks at up to three facts:
  declared  what the sources say (several candidates are allowed; the precedence picks one, a disagreement is kept)
  chosen    what the consumer will actually use, read by an adapter (the source "user" marks an explicit user choice)
  observed  what the data itself shows (bytes, strides, keys), when a check looked

and gives one Decision with a verdict, in this order:
  1. the sources disagree and the policy says stop                     -> REFUSED
  2. the data contradicts a declaration                                -> BROKEN, or the data's value is used (policy)
  3. nothing declares it (unknown, or only inferred or defaulted)      -> UNKNOWN; blocking under `require`/`stop`
                                                                          (ENTAIL_UNKNOWN; the default reports)
  4. what the consumer uses is unknown                                 -> UNKNOWN; blocking only in debug mode
  5. the consumer uses the declared value                              -> PASS
  6. it differs, and it was the user's explicit choice                 -> BROKEN (never overridden, never silent)
  7. it differs, the policy repairs nothing                            -> BROKEN
  8. it differs, a registered resolution applies                       -> RESOLVED (the adapter's handle carries it out)
  9. it differs, nothing can repair it                                 -> BROKEN

BROKEN is reported and the run goes on; where the policy stops (ENTAIL_ON_BROKEN=stop, a per-fact "Name=stop",
debug mode) the same outcome is REFUSED and blocking, before anything is produced (ROADMAP M5.4: the researcher's
decision that entail reports what it cannot repair instead of adding failures a user sees). `unrepaired` gives the
pair for a fact; every module that decides without `decide` uses it too.

An inferred fact is never the basis for a change (principle 5): with nothing declared the verdict is UNKNOWN even
when a probe has an opinion. The rules live here only; adapters supply `chosen` and the handles (principle 8).
"""
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

from . import sources as _sources
from .facts import VOCABULARY, Certainty, Fact
from .policies import Policy


class Verdict(str, Enum):
    PASS = "pass"
    RESOLVED = "resolved"
    BROKEN = "broken"       # not repaired: reported, the run goes on (M5.4)
    REFUSED = "refused"     # not repaired, and the policy stops: before anything is produced
    UNKNOWN = "unknown"


def unrepaired(policy: Policy, name: str) -> Tuple["Verdict", bool]:
    """(verdict, blocking) for a mismatch of the fact `name` that nothing repairs: (BROKEN, False) under the default
    policy, (REFUSED, True) where the policy stops (policies.Policy.stops)."""
    return (Verdict.REFUSED, True) if policy.stops(name) else (Verdict.BROKEN, False)


# Fixed wording: the ledger prints it and the tests match it.
RULES = {
    "match": "the consumer uses the declared value",
    "resolved": "the consumer differs from the declaration; a registered resolution repairs it",
    "no_resolution": "the consumer differs from the declaration and no resolution is registered",
    "policy_refuses": "the consumer differs from the declaration and the policy repairs nothing",
    "user_choice": "the user's explicit choice contradicts the declaration; it is not overridden",
    "consumer_unknown": "what the consumer uses is unknown (not read, or not in the capability table)",
    "consumer_inferred": "what the consumer uses is inferred from its code, not measured; nothing is switched on it",
    "declared_unread": "declared, but taken by nothing entail knows: the config class does not take it and no "
                       "reader of it is registered",
    "undeclared": "nothing declares it",
    "inferred_only": "only inferred, never declared",
    "defaulted_only": "only a default, never declared",
    "sources_disagree": "the sources disagree",
    "false_declaration": "the declaration contradicts the data",
    "data_used": "the declaration contradicts the data; the data's value is used",
    "cannot_check": "this boundary could not be checked",
    # code boundaries (boundaries.py, M4.1): checks about the call itself rather than a fact's value
    "positional": "a declared argument must be passed by keyword at this boundary (the keyword is its role)",
    "invalidated": "what it carried was made untrue on the way, and nothing says what it holds now",
    "predicate": "the value it carries does not satisfy what this boundary takes",
    "write_missing": "the boundary declares it writes this argument, but it was not written",
    "write_undeclared": "the boundary wrote into an argument it does not declare",
    "disagree": "arguments that must carry the same fact carry different ones",
    # container boundaries (kv_contract.py, M5.1): what a cache holds for a sequence against what the sequence has
    "kv_written": "the slots reserved and the slots written disagree",
    "kv_needed": "the slots held are not the slots the tokens need",
    "kv_shrank": "the slots held shrank since the last check, and nothing said so",
    "kv_layers": "the layers of one cache hold different lengths",
    "kv_request": "after the request, the cache does not hold the tokens the request wrote",
    # TIME and SPECIALIZATION on the host side (epochs.py, M5.2)
    "epoch_stale": "the value reads a buffer that was written after the value was made",
    "epoch_live": "a reader that reads later was handed a buffer that is written in place",
    "assumed_changed": "the artifact is reused under conditions it was not made for",
    # identity of a stored item against the identity its contents give now (identity_contract.py, M14)
    "identity_stale": "the identity a store holds no longer stands for what the item holds now",
    # a kernel's tile against the block the values are quantized in (tile_contract.py, M15.2)
    "tile_over_block": "the kernel steps a dimension in a tile that is not a divisor of the quantization block",
    # the tokenizer the engine holds against the model's vocabulary (vocab_contract.py, M15.3)
    "vocab_out_of_range": "the tokenizer can produce ids the model's embedding has no row for",
    "stop_dropped": "the consumer's stop set lacks an id the model's files declare as the end of a generation",
    "stop_id_out_of_range": "a declared eos, bos or pad id is past the tokenizer's highest id: no token, no stop",
    "vocab_not_the_models": "the folder declares two vocabularies and the engine loaded the one that is not the "
                            "model's",
}


@dataclass(frozen=True)
class Contract:
    """What one boundary needs.

    boundary          where, e.g. "load:sglang.attention_backend" or "container:vllm.allocate_slots"
    consumer          who uses the facts there, e.g. "sglang.attention.flashinfer" or "comfyui.sampler"
    needs             vocabulary names the consumer must honour
    meaning_changing  the subset whose absence may not be filled by a default (policy `require`)
    """
    boundary: str
    consumer: str
    needs: Tuple[str, ...]
    meaning_changing: Tuple[str, ...] = ()

    def __post_init__(self):
        for label, text in (("boundary", self.boundary), ("consumer", self.consumer)):
            if not isinstance(text, str) or not text:
                raise ValueError(f"Contract.{label}: expected a name, got {text!r}")
        if not isinstance(self.needs, tuple) or not self.needs:
            raise ValueError(f"Contract.needs: expected a non-empty tuple of vocabulary names, got {self.needs!r}")
        for name in self.needs + tuple(self.meaning_changing):
            if name not in VOCABULARY:
                raise ValueError(f"Contract: unknown fact name {name!r}; vocabulary has {sorted(VOCABULARY)}")
        extra = set(self.meaning_changing) - set(self.needs)
        if extra:
            raise ValueError(f"Contract.meaning_changing: {sorted(extra)} are not in needs")


@dataclass(frozen=True)
class Resolution:
    """One way to repair a mismatch for one vocabulary name. `handle` names the adapter operation that carries it
    out; `when` limits it to the mismatches it can repair (None: any). `target`, when given, says what the handle is
    given (e.g. the backend to switch to); a resolution whose target is None does not apply (M3.2)."""
    name: str
    handle: str
    when: Optional[Callable[[Fact, Fact], bool]] = None
    target: Optional[Callable[[Fact, Fact], object]] = None

    def applies(self, declared: Fact, chosen: Fact) -> bool:
        if self.when is not None and not self.when(declared, chosen):
            return False
        return self.target is None or self.target(declared, chosen) is not None

    def describe(self, declared: Fact, chosen: Fact, target=None) -> str:
        if target is not None:
            return f"{self.name} (to {target})"
        return f"{self.name} ({_value(chosen)} -> {_value(declared)})"


RESOLUTIONS: Dict[str, List[Resolution]] = {}


def register(name: str, resolution: Resolution) -> None:
    if name not in VOCABULARY:
        raise ValueError(f"register: unknown fact name {name!r}; vocabulary has {sorted(VOCABULARY)}")
    RESOLUTIONS.setdefault(name, []).append(resolution)


@dataclass(frozen=True)
class Decision:
    """One verdict, with what the ledger needs to explain it: which fact, the declared value and where it came from,
    the consumer and its choice, the rule, the verdict, and what was changed (LIBRARY_DESIGN.md 7)."""
    contract: Contract
    name: str
    verdict: Verdict
    rule: str
    declared: Optional[Fact] = None
    chosen: Optional[Fact] = None
    observed: Optional[Fact] = None
    resolution: Optional[str] = None
    handle: Optional[str] = None
    blocking: bool = False
    conflict: Tuple[Fact, ...] = ()
    target: Optional[object] = None   # what the handle is given, when the resolution names it
    note: str = ""                    # why a boundary could not be checked, or what else the ledger should say
    lost_by: Optional[str] = None     # the operation that made the declared fact untrue on the way (diagnosis, M7.1)


def agrees(declared_value, chosen_value) -> bool:
    """The chosen value honours the declared one: same class, and every field the declaration fills is equal.
    A field the declaration leaves open (None) is not compared."""
    if type(declared_value) is not type(chosen_value):
        return False
    return all(getattr(chosen_value, f.name) == getattr(declared_value, f.name)
               for f in fields(declared_value) if getattr(declared_value, f.name) is not None)


def _fill(declared_value, observed_value):
    """The declared value with the fields it leaves open taken from what the data shows."""
    return type(declared_value)(**{f.name: getattr(declared_value, f.name) if getattr(declared_value, f.name)
                                   is not None else getattr(observed_value, f.name) for f in fields(declared_value)})


def _value(fact):
    return "unknown" if fact is None or fact.value is None else str(fact.value)


def _check(fact, name, role):
    if fact is not None and (not isinstance(fact, Fact) or fact.name != name):
        raise ValueError(f"decide: {role} for {name!r} must be a Fact named {name!r}, got {fact!r}")


def decide(contract: Contract, declared: Dict[str, object], chosen: Dict[str, Fact], policy: Optional[Policy] = None,
           observed: Optional[Dict[str, Fact]] = None, resolutions: Optional[Dict[str, List[Resolution]]] = None
           ) -> List[Decision]:
    """Apply the verdict order above to every name the contract needs. `declared[name]` is a Fact or a tuple of
    candidate Facts from different sources. `resolutions` adds repairs that only this call can offer (e.g. routing
    to a backend of this engine); they are tried before the registered ones."""
    policy = policy or Policy()
    observed = observed or {}
    resolutions = resolutions or {}
    out = []
    for name in contract.needs:
        candidates = declared.get(name, ())
        candidates = (candidates,) if isinstance(candidates, Fact) else tuple(candidates)
        for fact in candidates:
            _check(fact, name, "a declared fact")
        c, o = chosen.get(name), observed.get(name)
        _check(c, name, "the chosen fact")
        _check(o, name, "the observed fact")
        d, conflict = _sources.pick(candidates)

        def decision(verdict, rule, **kw):
            base = dict(declared=d, chosen=c, observed=o, conflict=conflict)
            base.update(kw)
            return Decision(contract, name, verdict, rule, **base)

        if conflict and policy.on_source_conflict == "stop":
            out.append(decision(Verdict.REFUSED, RULES["sources_disagree"], blocking=True))
            continue

        note = None
        if o is not None and o.certainty is not Certainty.UNKNOWN:
            if d is not None and d.certainty in (Certainty.DECLARED, Certainty.VERIFIED):
                # the data contradicts a declaration only where both say something (an observation may cover a
                # few fields); confirmed, the declaration takes what the data adds to its open fields (M3.2)
                if _sources.compatible(d.value, o.value):
                    d = replace(d, value=_fill(d.value, o.value), certainty=Certainty.VERIFIED)
                elif policy.on_false_declaration == "refuse":
                    verdict, blocking = unrepaired(policy, name)
                    out.append(decision(verdict, RULES["false_declaration"], blocking=blocking))
                    continue
                else:
                    d, note = o, RULES["data_used"]
            else:
                d = o   # nothing reliable was declared; what the data shows is known, not guessed

        if d is None or d.certainty in (Certainty.UNKNOWN, Certainty.INFERRED, Certainty.DEFAULTED):
            rule = {Certainty.INFERRED: RULES["inferred_only"], Certainty.DEFAULTED: RULES["defaulted_only"]}.get(
                None if d is None else d.certainty, RULES["undeclared"])
            out.append(decision(Verdict.UNKNOWN, rule, declared=d,
                                blocking=policy.stops_unknown(name, name in contract.meaning_changing)))
            continue
        if c is None or c.certainty is Certainty.UNKNOWN:
            out.append(decision(Verdict.UNKNOWN, RULES["consumer_unknown"], declared=d,
                                blocking=policy.mode == "debug"))
            continue
        if agrees(d.value, c.value):
            out.append(decision(Verdict.PASS, note or RULES["match"], declared=d))
            continue
        verdict, blocking = unrepaired(policy, name)
        if c.source.kind == "user":
            out.append(decision(verdict, RULES["user_choice"], declared=d, blocking=blocking))
            continue
        if policy.mismatch_setting(name) == "refuse":
            out.append(decision(verdict, RULES["policy_refuses"], declared=d, blocking=blocking))
            continue
        offered = list(resolutions.get(name, ())) + RESOLUTIONS.get(name, [])
        fix = next((r for r in offered if r.applies(d, c)), None)
        if fix is None:
            out.append(decision(verdict, RULES["no_resolution"], declared=d, blocking=blocking))
        else:
            target = fix.target(d, c) if fix.target is not None else None
            out.append(decision(Verdict.RESOLVED, RULES["resolved"], declared=d, resolution=fix.describe(d, c, target),
                                handle=fix.handle, target=target))
    return out
