"""identity_contract: the contract of a store keyed by identity, in the core (LIBRARY_DESIGN.md 4.6, 4.7; ROADMAP
M14; realworld/CODEBOOK_v2.md I).

A store keyed by identity holds one key per item and serves the item's contents under that key. vLLM's prefix cache
keys each hash block of a request by a chained hash of its tokens (Request.block_hashes) and serves that block's KV
to any request whose prefix hashes the same. The key is a claim: "this block holds these tokens". When the tokens a
key stands for change but the key does not, the claim is false, and a later request that matches the OLD tokens is
served the NEW tokens' KV - silently wrong output (vllm#49377, #49449; the hashes are appended, never truncated, so
a hash chained over a discarded token survives a session rebuild).

The fact is one: for every stored item, the identity the store holds equals the identity the item's contents give
NOW. An adapter says where its engine keeps the stored keys and how to read the fresh ones (both as hex strings);
the rule is here, once:

  identity_stale  a stored identity is not the identity its item's contents give now, or a stored identity has no
                  contents left to stand for. The repair is to forget the stale identities and let the engine remake
                  them from the current contents (the adapter's `recompute` handle); with it the decision is resolved,
                  the false keys gone before the next lookup. Without a repair it is broken: reported, and the run
                  goes on (M5.4), or refused where the policy stops.

A store that still stands is counted (tally.PASSES). A stale one is a Decision recorded through load.enforce.
Nothing here reads a device: the keys are the host-side hashes the engine already computed. An error inside entail
never breaks the engine (the adapter runs this under load.safely / tally.guarded).
"""
from typing import List, Optional, Sequence

from . import tally as _tally


def first_stale(stored: Sequence[str], fresh: Sequence[str]) -> Optional[int]:
    """The first index whose stored identity is not the identity its contents give now, or the first stored index
    past what the contents can back; None if every stored identity still stands."""
    for i, (a, b) in enumerate(zip(stored, fresh)):
        if a != b:
            return i
    if len(stored) > len(fresh):
        return len(fresh)
    return None


def check(boundary: str, consumer: str, where: str, stored: Sequence[str], fresh: Sequence[str],
          recompute, covers: Optional[int] = None, owner=None) -> List[object]:
    """Decide one store's identities. `stored[i]` is the key the engine holds for item i; `fresh[i]` is the key the
    item's contents give now (both hex). `recompute(i)` forgets the stored keys from i on and lets the engine remake
    them - the one repair, offered by the adapter. Returns the decisions (also recorded and, if resolved, carried
    out). A store that stands passes; a stale one is resolved by recompute, or broken where no repair is given."""
    from . import load, policies
    from .contracts import Contract, Resolution, Verdict, decide
    from .facts import Certainty, Fact, Identity, Source

    _tally.counts(boundary)["checks"] += 1
    i = first_stale(stored, fresh)
    if i is None:
        _tally.passed(boundary, ["identity_stale"])
        _tally.tick(boundary)
        return []

    gone = i >= len(fresh)
    truth = Fact("Identity",
                 Identity(of="kv_block", index=i, key=("(no contents)" if gone else fresh[i]), covers=covers),
                 Source("data", f"{where}: the identity item {i}'s contents give now"), Certainty.VERIFIED)
    held = Fact("Identity", Identity(of="kv_block", index=i, key=stored[i], covers=covers),
                Source("engine", f"{where}: the identity the store holds for item {i}"), Certainty.VERIFIED)
    forget = Resolution("forget the stale identities and let the store remake them", "identity_recompute",
                        target=lambda d, c, _i=i: _i)
    contract = Contract(boundary, consumer, ("Identity",))
    decisions = decide(contract, {"Identity": truth}, {"Identity": held}, policies.current(),
                       resolutions={"Identity": [forget]})

    done = load.resolve(decisions, {"identity_recompute": recompute})
    load.enforce(decisions, once_for=owner)
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
