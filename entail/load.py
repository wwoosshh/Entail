"""load: the load-time contracts (LIBRARY_DESIGN.md 4.5, 4.7; ROADMAP M3.2).

When an engine loads a model it has the model's declarations in hand and makes its choices: which attention backend
runs, whether the head is tied to the embedding, which config keys its config class takes, what an old RoPE name
written after the config was built turns into, which kernel reads the stored weights. Each function below takes what
an adapter read - the engine's choice - and returns Decisions (contracts.decide). The rules are here; an adapter only
reads the choice and carries out a resolution with its handle (principle 8). `enforce` records the decisions, prints
them and stops the run on a blocking one.

  declared(...)        the declarations for one model: its files (readers), manifests, the config object the engine
                       holds (fields no file states count as defaults), and the user's explicit settings
  attention(...)       ModelProps (softcap, sliding window) against the backend's capabilities (caps.json);
                       resolution: route to a backend measured to honour them
  tie(...)             ModelProps.tie_word_embeddings: the declaration against the checkpoint bytes and the loader
  config_keys(...)     Coverage: every key of config.json is taken by the engine's config class, or its value
                       survives in a field the class knows (the rule measured in audits/W_MORE_FACTS.md)
  rotary_write(...)    Rotary: a value written under an old RoPE name after the config was built means what
                       config.json means by it; resolution: write it where the model reads it
  rotary_held(...)     Rotary: the RoPE the engine's config holds when the model is built, against the files and what
                       the user wrote on the object (declare_on); a base lost on the way shows here
  layout(...)          Layout: the declared storage format against what the consumer reads and what the data shows
  weights_taken(...)   Coverage: the checkpoint values the loader was given against the ones that landed (fd-shift)
  weights_written(...) Layout and Coverage at a boundary entail wraps in engine code (M4.2): what a step declares
                       it leaves in each weight (data/signatures.json) against what the tensor shows, and the values
                       sampled before it (sample_weights) against where it declares it moves them; a confirmed
                       layout goes onto the weight as a fact, so it travels with the value
  model_contracts(...) tie, rotary_held and layout together, for an engine's hook once the model is built
  cannot_check(...)    a boundary the adapter could not check, reported as such (never a silent pass)
  safely(...)          runs an adapter's reading and deciding so that an error inside entail never breaks the run
"""
import json
import os
import sys
import time
import weakref
from dataclasses import dataclass, field, fields, replace
from typing import Dict, List, Optional, Sequence

from . import caps as _caps
from . import observe as _observe
from . import readers as _readers
from . import record as _record
from . import signatures as _signatures
from . import sources as _sources
from .contracts import RULES, Contract, Decision, Resolution, Verdict, agrees, decide
from .facts import Certainty, Fact, ModelProps, Source
from .policies import Policy

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "config_keys.json"), encoding="utf-8") as _f:
    CONFIG_KEYS = json.load(_f)


# --- what is declared ------------------------------------------------------------------------------------------

@dataclass
class Declared:
    facts: Dict[str, List[Fact]] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    def get(self, name) -> List[Fact]:
        return list(self.facts.get(name, ()))


def _only(value, names):
    """`value` with every field outside `names` set to None (required fields kept); None if nothing is left."""
    wanted = tuple(f"{type(value).__name__}.{n}" for n in names)
    return _caps.project(value, wanted)


def _held(obj_facts, file_facts, label):
    """Facts from the config object an engine holds, next to the files: a field no file states came from a default
    of the engine's config class (DEFAULTED); a field the object holds differently from the files was changed after
    they were read (an override), and is kept as the engine's value so that the disagreement is recorded."""
    out = []
    for f in obj_facts:
        stated = [g.value for g in file_facts if g.name == f.name]
        names = [x.name for x in fields(f.value) if getattr(f.value, x.name) is not None]
        unstated = [n for n in names if all(getattr(s, n) is None for s in stated)]
        differs = [n for n in names if n not in unstated and all(getattr(s, n) != getattr(f.value, n)
                                                                  for s in stated if getattr(s, n) is not None)]
        if unstated:
            v = _only(f.value, unstated)
            if v is not None:
                out.append(Fact(f.name, v, Source("default", f"{label}: {', '.join(unstated)} not in the model's "
                                                             f"files, so a default of the config class"),
                                Certainty.DEFAULTED))
        if differs:
            v = _only(f.value, differs)
            if v is not None:
                out.append(Fact(f.name, v, Source("engine", f"{label}: {', '.join(differs)} differ from the files"),
                                Certainty.DECLARED))
    return out


class ByObject:
    """A side table keyed by an object's identity and emptied when the object is collected. Config objects are
    dataclasses without a hash, so a WeakKeyDictionary cannot hold them."""

    def __init__(self):
        self._d = {}

    def get(self, obj, default=None):
        entry = self._d.get(id(obj))
        return default if entry is None or entry[0]() is not obj else entry[1]

    def set(self, obj, value) -> bool:
        key = id(obj)
        try:
            ref = weakref.ref(obj, lambda _r, k=key, d=self._d: d.pop(k, None))
        except TypeError:   # an object that takes no weak reference is not remembered
            return False
        self._d[key] = (ref, value)
        return True

    def clear(self):
        self._d.clear()


_USER = ByObject()   # config object -> facts the user declared on it after it was built
# The same facts by model folder, in the environment: an engine builds its config in one process and runs the model
# in children that get the config pickled (vLLM's engine core), where the table above is empty. A child started
# after the declaration inherits the variable.
ENV_DECLARED = "ENTAIL_DECLARED"


def _folder(path):
    if not path or not os.path.exists(os.path.expanduser(str(path))):
        return None
    return os.path.realpath(os.path.expanduser(str(path)))


def declare_on(config, fact: Fact, path: Optional[str] = None) -> None:
    """Remember a fact the user stated on a config object after it was built (an old RoPE name written at launch):
    later contracts on the same object - in this process, and by model folder in processes started afterwards -
    read it as the user's declaration, which outranks the files."""
    from .manifest import value_to_json

    _USER.set(config, [f for f in _USER.get(config, []) if f.name != fact.name] + [fact])
    key = _folder(path or getattr(config, "_name_or_path", None))
    if key is None:
        return
    try:
        data = json.loads(os.environ.get(ENV_DECLARED) or "{}")
    except ValueError:
        data = {}
    data[key] = [e for e in data.get(key, []) if e.get("name") != fact.name] + \
        [{"name": fact.name, "value": value_to_json(fact.value), "where": fact.source.where}]
    os.environ[ENV_DECLARED] = json.dumps(data)


def _declared_in_env(path) -> List[Fact]:
    from .manifest import value_from_json

    key = _folder(path)
    if key is None or not os.environ.get(ENV_DECLARED):
        return []
    try:
        entries = json.loads(os.environ[ENV_DECLARED]).get(key, [])
    except ValueError:
        return []
    out = []
    for e in entries:
        try:
            out.append(Fact(e["name"], value_from_json(e["name"], e["value"]), Source("user", e["where"]),
                            Certainty.DECLARED))
        except (KeyError, ValueError, TypeError):   # an entry this vocabulary cannot read is not a declaration
            continue
    return out


def declared(model_path: Optional[str] = None, config=None, manifest_dirs: Sequence[str] = (),
             user: Sequence[Fact] = ()) -> Declared:
    """Everything declared about one model at load. With files, the files (and a pinned manifest) are the
    declaration and the engine's config object only adds what they leave open, as defaults. Without files (a config
    built in code), the config object is the declaration."""
    out = Declared()
    file_facts, have_files = [], False
    if model_path and os.path.exists(os.path.expanduser(model_path)):
        r = _sources.read_all(os.path.expanduser(model_path), manifest_dirs)
        file_facts, out.problems, have_files = list(r.facts), list(r.problems), True
    held = []
    if config is not None:
        label = f"{type(config).__name__} held by the engine"
        r = _readers.read_hf_dict(_readers.config_dict(config), label, from_object=True)
        if have_files:   # a file that is silent about a field leaves it to the class default
            held = _held(r.facts, file_facts, label)
        else:
            held = r.facts
            out.problems += r.problems
        user = list(user) + list(_USER.get(config, []))
    for f in _declared_in_env(model_path) + (_declared_in_env(getattr(config, "_name_or_path", None))
                                             if config is not None and _folder(getattr(config, "_name_or_path", None))
                                             != _folder(model_path) else []):
        if not any(u.name == f.name and u.value == f.value for u in user):
            user = list(user) + [f]
    for f in file_facts + held + list(user):
        out.facts.setdefault(f.name, []).append(f)
    return out


def _projected(facts: Declared, name: str, wanted: Sequence[str]) -> List[Fact]:
    out = []
    for f in facts.get(name):
        v = None if f.value is None else _caps.project(f.value, tuple(wanted))
        if v is not None:
            out.append(replace(f, value=v))
    return out


# --- the contracts ---------------------------------------------------------------------------------------------

def attention(engine: str, backend: str, facts: Declared, table=None, policy: Optional[Policy] = None,
              role: str = "attention", can_switch: bool = True) -> List[Decision]:
    """The attention backend an engine chose against the ModelProps the model declares (rolebench 06, 08, 17;
    fd-softcap). `role` names the group of backends that can stand in for one another ("paged_attention" for
    transformers' continuous batching). `can_switch` False: the backend is already built, so routing is not on offer
    and a mismatch is refused. Nothing is decided when the model declares no property they act on."""
    table = table or _caps.default_table()
    group = f"{engine}.{role}"
    consumer = f"{group}.{backend}"
    candidates = _projected(facts, "ModelProps", _caps.consumed(table, group))
    if not candidates:
        return []
    best, _ = _sources.pick(candidates)
    chosen = _caps.chosen_fact(table, consumer, best)
    route = Resolution("route to a backend measured to honour it", "switch_attention_backend",
                       target=lambda d, c: _caps.route(table, group, d.value, exclude=(backend,)))
    contract = Contract(f"load:{group}", consumer, ("ModelProps",), ("ModelProps",))
    return decide(contract, {"ModelProps": tuple(candidates)}, {"ModelProps": chosen}, policy,
                  resolutions={"ModelProps": [route] if can_switch else []})


def tie(engine: str, facts: Declared, model_path: Optional[str] = None, loader_ties: Optional[bool] = None,
        policy: Optional[Policy] = None, observed: Optional[Fact] = None) -> List[Decision]:
    """tie_word_embeddings: the declaration against the checkpoint (observe.tie) and against what the loader does
    (it ties the head when the config it holds says so; `loader_ties` is that value, None when unknown).
    rolebench 07: the config declares a tie and the checkpoint ships its own head -> refused."""
    candidates = _projected(facts, "ModelProps", ("ModelProps.tie_word_embeddings",))
    if observed is None and model_path:
        observed = _observe.tie(os.path.expanduser(model_path))
    if not candidates and observed is None:
        return []
    consumer = f"{engine}.loader"
    if loader_ties is None:
        chosen = Fact("ModelProps", None, Source("engine", consumer), Certainty.UNKNOWN)
    else:
        chosen = Fact("ModelProps", ModelProps(tie_word_embeddings=loader_ties),
                      Source("engine", f"{consumer} ties the head to the embedding exactly when the config it holds "
                                       f"says tie_word_embeddings"), Certainty.VERIFIED)
    contract = Contract(f"load:{consumer}", consumer, ("ModelProps",), ("ModelProps",))
    return decide(contract, {"ModelProps": tuple(candidates)}, {"ModelProps": chosen}, policy,
                  observed={"ModelProps": observed} if observed is not None else None)


def _scalars(obj, out=None):
    """Every scalar anywhere in a nested value, so a key that was renamed can still be found by its value."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            _scalars(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scalars(v, out)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out.add((type(obj).__name__, obj))
    return out


def keys_taken(raw: dict, known: set, resolved: dict):
    """Which keys of a config dict the consumer's config class took (the rule of audits/W_MORE_FACTS.md: unknown to
    the class AND its value in no known field -> not taken; 0 false positives on healthy models, catches rolebench
    15). Keys listed in data/config_keys.json as read elsewhere are notes, not losses.
    Returns (taken, left, notes)."""
    survived = _scalars({k: v for k, v in resolved.items() if k in known})
    elsewhere = CONFIG_KEYS["read_elsewhere"]
    suffixes = tuple(CONFIG_KEYS["provenance_suffixes"])
    taken, left, notes = [], [], []
    for k, v in raw.items():
        leaves = _scalars(v) - {("NoneType", None)} if isinstance(v, (dict, list)) else None
        if k == "text_config" or k in known:
            taken.append(k)
        elif leaves is None and (type(v).__name__, v) in survived:
            taken.append(k)          # renamed on the way in, but the value landed in a field the class knows
        elif leaves and leaves <= survived:
            taken.append(k)          # a dict moved into another field (rope_scaling into rope_parameters)
        elif k in elsewhere:
            notes.append(f"{k} ({elsewhere[k]})")
            taken.append(k)
        elif k.endswith(suffixes):
            notes.append(f"{k} (provenance of the tool that wrote the checkpoint)")
            taken.append(k)
        else:
            left.append(k)
    return taken, left, notes


def config_keys(engine: str, scopes: Sequence[tuple], where: str, policy: Optional[Policy] = None
                ) -> List[Decision]:
    """Coverage of config.json by the engine's config class. `scopes` is a list of (prefix, raw dict, the fields the
    class knows, the class's resolved dict) - the top level and a nested text_config. rolebench 15 -> refused."""
    from .coverage import Coverage

    given, left, notes = 0, [], []
    for prefix, raw, known, resolved in scopes:
        t, lft, n = keys_taken(raw, set(known), resolved)
        given += len(t) + len(lft)
        left += [prefix + k for k in lft]
        notes += [prefix + x for x in n]
    if given == 0:
        return []
    declared_fact = Fact("Coverage", Coverage(given, given, ()), Source("config", f"{where} (every key it gives)"),
                         Certainty.DECLARED)
    chosen = Fact("Coverage", Coverage(given, given - len(left), tuple(sorted(left))),
                  Source("engine", f"{engine} config class: a key is taken if the class knows it or its value landed "
                                   f"in a field it knows"), Certainty.VERIFIED)
    contract = Contract(f"load:{engine}.config", f"{engine}.config", ("Coverage",), ("Coverage",))
    out = decide(contract, {"Coverage": declared_fact}, {"Coverage": chosen}, policy)
    if notes:
        out = [replace(d, note=f"read elsewhere: {'; '.join(notes)}") for d in out]
    return out


def rotary_write(engine: str, owner: str, key: str, meant, as_engine, scope: str = "",
                 policy: Optional[Policy] = None, config=None) -> List[Decision]:
    """An old RoPE name (rope_theta, rope_scaling) written on a config after it was built. `meant` is the Rotary the
    write means - what config.json means by the same key - and `as_engine` the Rotary the model will read after the
    engine performs the write as it does. fd-rope: rope_scaling given at launch drops rope_theta -> resolved by
    writing it where the model reads it, as config.json would. With `config`, the meaning is remembered on that
    object as the user's declaration (declare_on), so the load contracts later compare against it, not the files."""
    at = f"[{scope}]" if scope else ""
    declared_fact = Fact("Rotary", meant, Source("user", f"{owner}.{key} written after the config was built, read as "
                                                         f"config.json reads it"), Certainty.DECLARED)
    if config is not None and not scope:
        declare_on(config, declared_fact)
    chosen = Fact("Rotary", as_engine, Source("engine", f"{owner}.rope_parameters{at} after the write as {engine} "
                                                        f"performs it"), Certainty.VERIFIED) if as_engine is not None \
        else Fact("Rotary", None, Source("engine", f"{owner}.rope_parameters{at}"), Certainty.UNKNOWN)
    convert = Resolution("write the old name where the model reads it, as config.json would", "rope_write_as_file")
    contract = Contract(f"load:{engine}.config.{key}{at}", f"{engine}.rotary_embedding", ("Rotary",), ("Rotary",))
    return decide(contract, {"Rotary": declared_fact}, {"Rotary": chosen}, policy, resolutions={"Rotary": [convert]})


def layout(consumer: str, facts: Declared, table=None, observed: Optional[Fact] = None,
           policy: Optional[Policy] = None) -> List[Decision]:
    """The declared storage format against what `consumer` (engine.role.name) reads, and against the data. rolebench
    02: fp32 block scales, a kernel that reads every scale as ue8m0 -> refused (no requantization is registered)."""
    table = table or _caps.default_table()
    group = _caps.group_of(consumer)
    wanted = _caps.consumed(table, group)
    candidates = [f for f in facts.get("Layout") if f.value is not None]
    if wanted:   # only the fields this group of consumers acts on
        candidates = _projected(Declared({"Layout": candidates}), "Layout", wanted)
        if observed is not None:
            v = _caps.project(observed.value, wanted)
            observed = None if v is None else replace(observed, value=v)
    if not candidates and observed is None:
        return []
    best, _ = _sources.pick(candidates) if candidates else (observed, ())
    effective = best
    if observed is not None and best is not observed and _sources.compatible(best.value, observed.value):
        effective = replace(best, value=type(best.value)(**{
            x.name: getattr(best.value, x.name) if getattr(best.value, x.name) is not None
            else getattr(observed.value, x.name) for x in fields(best.value)}))
    if wanted:
        chosen = _caps.chosen_fact(table, consumer, effective)
    else:        # the table says nothing about this consumer: the declaration is still checked against the data
        chosen = Fact("Layout", None, Source("engine", f"{consumer} [not in the capability table]"),
                      Certainty.UNKNOWN)
    contract = Contract(f"load:{group}", consumer, ("Layout",), ("Layout",))
    return decide(contract, {"Layout": tuple(candidates)}, {"Layout": chosen}, policy,
                  observed={"Layout": observed} if observed is not None else None)


def weights_taken(engine: str, where: str, compared: int, left: Sequence[str], policy: Optional[Policy] = None
                  ) -> List[Decision]:
    """The checkpoint weights the loader was given (`compared`, each checked on a sample of elements) against the
    ones whose sampled values did not land where its mapping puts them (`left`). fd-shift: rows shifted while
    loading -> refused. Nothing compared, nothing decided (the adapter reports that with cannot_check)."""
    from .coverage import Coverage

    if compared == 0:
        return []
    declared_fact = Fact("Coverage", Coverage(compared, compared, ()),
                         Source("file", f"{where} (every weight, sampled, where the loader's mapping puts it)"),
                         Certainty.DECLARED)
    chosen = Fact("Coverage", Coverage(compared, compared - len(left), tuple(sorted(left))),
                  Source("engine", f"{engine} loaded weights, sampled"), Certainty.VERIFIED)
    contract = Contract(f"load:{engine}.weights", f"{engine}.loader", ("Coverage",), ("Coverage",))
    return decide(contract, {"Coverage": declared_fact}, {"Coverage": chosen}, policy)


@dataclass(frozen=True)
class Weight:
    """One weight a boundary entail wraps has written, as an adapter read it (weights_written)."""
    layer: str                          # e.g. "model.layers.0.self_attn.qkv_proj"
    producer: str                       # engine.role.name of the step that wrote it, as in data/signatures.json
    tensor: object = None               # the weight; None when the layer holds none
    in_features: Optional[int] = None   # the layer's own sizes, the reference for the orientation
    out_features: Optional[int] = None
    scale: object = None                # the layer's scale tensor; None when it has none


def sample_weights(weights: Sequence[Weight], table=None) -> Dict[str, dict]:
    """Before a signed step runs: a few values of every weight whose producer declares where the step moves them
    (layer -> sample), for weights_written to compare afterwards."""
    table = table or _signatures.default_table()
    out = {}
    for w in weights:
        sig = table.lookup(w.producer)
        if sig is not None and sig.moves and w.tensor is not None:
            s = _observe.sample_values(w.tensor)
            if s is not None:
                out[w.layer] = s
    return out


def weights_written(boundary: str, weights: Sequence[Weight], before: Optional[Dict[str, dict]] = None,
                    policy: Optional[Policy] = None, table=None) -> List[Decision]:
    """What a step entail wraps left in each weight, against the step's signature (data/signatures.json; M4.2).

    Per weight, with the reading step (the producer's kernel) as the consumer:
      Layout    the signature against what the tensor shows (observe.weight_layout): the data contradicting it is
                refused (policy on_false_declaration); with use_data, a strided weight the kernel reads packed is
                made contiguous (handle "layout.contiguous", given every layer it repairs)
      Coverage  the values sampled before the step (sample_weights) against where the signature says it moves them
    A confirmed layout goes onto the weight as a fact whose source is the signature, filled with what the data adds,
    so it travels with the value. Passes are counted, one decision per producer and fact (a model has hundreds of
    weights); anything else is one decision per producer, fact, verdict and rule, naming the first weights and how
    many there are (a resolution's target is every weight it repairs). A producer with no signature, a weight that
    cannot be read, or one with no sample is reported (cannot_check), never passed; producers the table lists under
    not_values are left out, with their reason in the table."""
    from .boundaries import CONVERTERS   # the value converters of code boundaries, offered per call (M4.1 (3))
    from .core import tag
    from .coverage import Coverage

    table = table or _signatures.default_table()
    policy = policy or Policy()
    before = before or {}
    found, passed, unchecked = [], {}, {}   # (layer, decision); (producer, name) -> passes; (..., why) -> layers

    def not_checked(producer, name, why, layer):
        unchecked.setdefault((producer, name, why), []).append(layer)

    for w in weights:
        sig = table.lookup(w.producer)
        if sig is None:
            if table.why_not(w.producer) is None and w.tensor is not None:
                not_checked(w.producer, "Layout", f"{w.producer} has no signature in data/signatures.json, so what "
                                                  f"its step wrote was not checked", w.layer)
            continue
        if w.tensor is None:
            not_checked(w.producer, "Layout", f"the layer holds no {sig.value}", w.layer)
            continue
        where = f"{w.layer}.{sig.value}"
        decisions = []
        declared, taken = sig.declared("Layout"), sig.taken("Layout")
        if declared is not None:
            seen, problem = _observe.weight_layout(w.tensor, w.in_features, w.out_features, w.scale, where)
            contract = Contract(boundary, w.producer, ("Layout",))
            if seen is None:
                not_checked(w.producer, "Layout", problem, w.layer)
            elif problem and w.in_features and w.out_features and declared.value.orientation is not None:
                # the shape fits neither orientation: nothing the declaration could mean, and nothing to convert
                decisions.append(Decision(contract, "Layout", Verdict.REFUSED, RULES["false_declaration"],
                                          declared=declared, chosen=taken, observed=seen, blocking=True,
                                          note=f"{w.layer}: {problem}"))
            else:
                if problem:
                    not_checked(w.producer, "Layout", problem, w.layer)
                offered = [Resolution(r.name, r.handle, r.when, target=lambda d, c, layer=w.layer: layer)
                           for r, _ in CONVERTERS.get("Layout", [])]
                # the kernel takes what the signature states, and whatever the weight holds where it states nothing
                # (the model's dtype, for an unquantised weight): as in `layout`, the consumer's side is filled too
                taken = replace(taken, value=type(taken.value)(**{
                    f.name: getattr(taken.value, f.name) if getattr(taken.value, f.name) is not None
                    else getattr(seen.value, f.name) for f in fields(taken.value)}))
                (d,) = decide(contract, {"Layout": declared}, {"Layout": taken}, policy, observed={"Layout": seen},
                              resolutions={"Layout": offered})
                decisions.append(d)
                if d.verdict is Verdict.PASS:
                    tag(w.tensor, d.declared)
                elif d.verdict is Verdict.RESOLVED:
                    tag(w.tensor, replace(taken, source=Source("boundary", f"{taken.source.where} ({d.resolution})")))
        if sig.moves:
            sample = before.get(w.layer)
            if sample is None:
                not_checked(w.producer, "Coverage", f"no values were sampled before {sig.writes}, so where it moved "
                                                    f"them was not checked", w.layer)
            else:
                k, kept, left = _observe.moved(w.tensor, sample, sig.moves)
                promised = Fact("Coverage", Coverage(k, k, ()),
                                Source("boundary", f"{w.producer}.{sig.writes}.moves {sig.moves}"), Certainty.DECLARED)
                wanted = Fact("Coverage", Coverage(k, k, ()),
                              Source("boundary", f"{w.producer}.{sig.reads}.takes.{sig.value}"), Certainty.DECLARED)
                seen = Fact("Coverage", Coverage(k, kept, tuple(left)),
                            Source("data", f"{where}: {k} values sampled before {sig.writes}"), Certainty.VERIFIED)
                (d,) = decide(Contract(boundary, w.producer, ("Coverage",)), {"Coverage": promised},
                              {"Coverage": wanted}, policy, observed={"Coverage": seen})
                decisions.append(d)
        for d in decisions:
            if d.verdict is Verdict.PASS:
                passed[(w.producer, d.name)] = passed.get((w.producer, d.name), 0) + 1
            else:
                found.append((w.layer, d))
    summary = []
    for (producer, name), n in sorted(passed.items()):
        sig = table.lookup(producer)
        if name == "Layout":
            declared, note = sig.declared("Layout"), f"{n} weight(s), each checked against its data"
        else:
            declared = None
            note = f"{n} weight(s): the values sampled before {sig.writes} are where '{sig.moves}' puts them"
        summary.append(Decision(Contract(boundary, producer, (name,)), name, Verdict.PASS, RULES["match"],
                                declared=declared, note=note))
    for (producer, name, why), layers in sorted(unchecked.items()):
        summary.append(cannot_check(boundary, producer, name, f"{len(layers)} weight(s) (first: {layers[0]}): {why}",
                                    policy))
    groups = {}
    for layer, d in found:
        groups.setdefault((d.contract.consumer, d.name, d.verdict, d.rule, d.handle, d.blocking), []).append((layer, d))
    for members in groups.values():
        layers, first = [m[0] for m in members], members[0][1]
        shown = "; ".join(f"{layer}: {_written_detail(d)}" for layer, d in members[:3])
        more = f"; and {len(members) - 3} more" if len(members) > 3 else ""
        kw = dict(note=f"{len(members)} weight(s): {shown}{more}")
        if first.verdict is Verdict.RESOLVED:   # one repair, carried out on every weight it names
            name = next(r.name for r, _ in CONVERTERS.get(first.name, []) if r.handle == first.handle)
            kw.update(target=tuple(layers), resolution=f"{name} ({len(layers)} weight(s))")
        summary.append(replace(first, **kw))
    return summary


def _written_detail(d: Decision) -> str:
    """What one weight showed, for the note of a grouped decision: the fields where the data and the signature
    differ, the values that moved, or the reason the decision already gives."""
    o = d.observed.value if d.observed is not None else None
    if d.name == "Coverage" and o is not None:
        return f"{o.total - o.taken} of {o.total} sampled values moved (first: {o.left[0] if o.left else '?'})"
    if d.note:
        return d.note.split(": ", 1)[-1]
    ref = d.chosen if d.verdict is Verdict.RESOLVED else d.declared
    if o is None or ref is None or ref.value is None:
        return d.rule
    diff = [f"{f.name} {getattr(ref.value, f.name)} -> {getattr(o, f.name)}" for f in fields(o)
            if getattr(ref.value, f.name) is not None and getattr(o, f.name) is not None
            and getattr(ref.value, f.name) != getattr(o, f.name)]
    return ", ".join(diff) or d.rule

def rotary_held(engine: str, facts: Declared, config, policy: Optional[Policy] = None) -> List[Decision]:
    """The RoPE the model will be built with - the config object the engine holds - against the declared one: the
    files, and what the user wrote on the object after it was built (declare_on). A base the engine's config lost
    falls back to whatever the model file defaults to (fd-rope: 10,000); that shows here as a refusal."""
    declared_rot = [f for f in facts.get("Rotary") if f.source.kind not in ("default", "engine")]
    r = _readers.read_hf_dict(_readers.config_dict(config), f"{type(config).__name__} held by {engine}",
                              source_kind="engine", from_object=True)
    held = [f for f in r.facts if f.name == "Rotary"]
    boundary, consumer = f"load:{engine}.config.rope_parameters", f"{engine}.rotary_embedding"
    if not held:
        if not declared_rot:
            return []
        why = "; ".join(p for p in r.problems if "RoPE" in p or "rope" in p.lower()) or \
            "the config the engine holds has no RoPE parameters vocabulary v1 can read"
        return [cannot_check(boundary, consumer, "Rotary", why, policy)]
    contract = Contract(boundary, consumer, ("Rotary",), ("Rotary",))
    return decide(contract, {"Rotary": tuple(declared_rot)},
                  {"Rotary": replace(held[0], certainty=Certainty.VERIFIED)}, policy)


def model_contracts(engine: str, model_path: Optional[str], config, loader_ties: Optional[bool],
                    policy: Optional[Policy] = None) -> List[Decision]:
    """The contracts about the model itself, for an engine's load hook once the model is built: tie against the
    checkpoint, the RoPE the engine holds, and a stored layout against its data (the kernel that reads it is not in
    the capability table yet, so that one is reported as unknown unless the data contradicts the declaration)."""
    facts = declared(model_path, config)
    out = tie(engine, facts, model_path, loader_ties, policy)
    if config is not None:
        out += rotary_held(engine, facts, config, policy)
    if facts.get("Layout") and model_path:
        out += layout(f"{engine}.linear.unknown", facts, observed=_observe.scale_format(os.path.expanduser(model_path)),
                      policy=policy)
    return out


def cannot_check(boundary: str, consumer: str, name: str, why: str, policy: Optional[Policy] = None,
                 meaning_changing: bool = False) -> Decision:
    """A boundary an adapter could not check: reported (principle 11). It stops in debug mode, and for a
    meaning-changing fact when the policy requires such facts to be known (principle 4; default `require`)."""
    policy = policy or Policy()
    setting = policy.unknown_setting(name, meaning_changing)
    blocking = policy.mode == "debug" or (meaning_changing and setting in ("require", "stop"))
    contract = Contract(boundary, consumer, (name,), (name,) if meaning_changing else ())
    return Decision(contract, name, Verdict.UNKNOWN, RULES["cannot_check"], blocking=blocking, note=why)


# --- carrying out, recording and stopping ----------------------------------------------------------------------

def safely(boundary: str, consumer: str, name: str, work, default=None):
    """Run an adapter's reading and deciding for one boundary. An error inside entail never breaks the run it
    watches (principle 12): it becomes a cannot_check decision, printed and recorded - which stops the run only in
    debug mode. A RoleError (a decision to stop) passes through. Returns work()'s result, or `default`."""
    from .core import RoleError
    from .policies import current

    t0 = time.perf_counter()
    try:
        return work()
    except RoleError:
        raise
    except Exception as e:  # noqa: BLE001
        enforce([cannot_check(boundary, consumer, name, f"entail failed here: {type(e).__name__}: {e}", current())])
        return default
    finally:
        _write({"pid": os.getpid(), "timing": boundary, "ms": round((time.perf_counter() - t0) * 1e3, 3)})


def _write(obj) -> None:
    """One JSON line to the ENTAIL_RECORD file, if one is set; recording never breaks the run (principle 12)."""
    path = os.environ.get("ENTAIL_RECORD")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[entail] could not write the record to {path}: {e}", file=sys.stderr, flush=True)

def resolve(decisions: Sequence[Decision], handles: Dict[str, object]) -> Dict[str, object]:
    """Carry out every resolved decision with the adapter's handle of that name, given the decision's target.
    Returns handle name -> what the handle returned. A resolution whose handle the adapter does not have cannot be
    carried out; that is an adapter defect and it raises, rather than reporting a repair that did not happen."""
    done = {}
    for d in decisions:
        if d.verdict is Verdict.RESOLVED and d.handle:
            if d.handle not in handles:
                raise KeyError(f"entail: the adapter has no handle {d.handle!r} for {d.contract.boundary}")
            done[d.handle] = handles[d.handle](d.target)
    return done


LEDGER = _record.Ledger()   # every decision this process made at its boundaries


def enforce(decisions: Sequence[Decision], quiet_pass: Optional[bool] = None) -> List[Decision]:
    """Record the decisions, print them, and stop on a blocking one.

    Every decision goes to the process ledger (LEDGER) and, with ENTAIL_RECORD=<file>, as one JSON line to that file
    (engines run their model in child processes; the file collects all of them). Anything but a pass is printed as
    one line; passes too with ENTAIL_VERBOSE. A blocking decision raises RoleError before any output is produced."""
    from .core import RoleError

    decisions = list(decisions)
    LEDGER.extend(decisions)
    for d in decisions:
        _write({"pid": os.getpid(), **_record.decision_json(d)})
    verbose = bool(os.environ.get("ENTAIL_VERBOSE")) if quiet_pass is None else not quiet_pass
    for d in decisions:
        if d.verdict is not Verdict.PASS or verbose:
            print(_record.line(d), flush=True)
    stops = [d for d in decisions if d.blocking]
    if stops:
        raise RoleError("\n".join(_record.line(d) for d in stops))
    return decisions
