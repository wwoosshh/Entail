"""epochs: TIME and SPECIALIZATION on the host side, in the core (LIBRARY_DESIGN.md 4.6, 4.7; ROADMAP M5.2).

A buffer written in place - a position counter, a cache, a beam's per-row state - has an epoch: a count of its writes
kept on the host and advanced where its container is written (advance). Nothing is read from the device.

A value that reads a buffer later, not now - a mask function that reads the position counter when attention runs, a
graph replayed over the buffer - records the epoch it was made at (live; carried from one value to the next when one
is built from the other). Where it is read, that epoch is compared with the buffer's epoch now:

  epoch_stale  the value reads its buffer as it is now, and the buffer was written after the value was made

The resolution is at the hand-over: bind the value to the buffer's value at that moment - a snapshot - so it no longer
reads the buffer later (rolebench 10: "bind to the value at call time"; the fix of transformers' flex offset). An
adapter offers the snapshot; under a policy that repairs nothing, nothing is bound, and a stale read is reported
where it happens (broken; refused where the policy stops).

An artifact made for some conditions - a CUDA graph captured for a valid length, a graph compiled on an empty input -
records them as an Assumed fact (assume). Where it is reused the conditions now are compared (reuse):

  assumed_changed  the artifact is reused under conditions it was not made for. A caller that can remake it
                   (recapture, recompile) passes that as the resolution; otherwise it is broken: reported, and the
                   artifact is reused as it is (refused where the policy stops)

Rules that hold are counted (tally). A broken one is a Decision recorded through load.enforce; the run goes on, and
the same break again for the same owner or artifact is counted, not recorded again (tally.first); where the policy
stops it is refused and raised (M5.4). A repeated resolution is recorded once per boundary and counted afterwards.
Nothing runs inside a captured region (tally.inside_capture) and an error inside entail never breaks the engine
(guarded).
"""
import weakref
from typing import Optional

from . import tally as _tally

inside_capture = _tally.inside_capture
_EPOCHS = None       # owner -> {buffer: epoch}
_LIVE = None         # value -> ((weakref to owner, buffer, epoch it was made at), ...)
_ASSUMED = {}        # (boundary, key) -> Assumed
_RECORDED = set()    # (boundary, rule) of resolutions already recorded once


def _table(name):
    from .load import ByObject

    g = globals()
    if g[name] is None:
        g[name] = ByObject()
    return g[name]


def _owner_name(owner, buffer):
    return f"{type(owner).__name__}.{buffer}"


# --- buffers and the values that read them later -------------------------------------------------------------------

def advance(owner, buffer: str) -> int:
    """`owner`'s buffer `buffer` was written in place: its epoch goes up by one. Returns the new epoch."""
    epochs = _table("_EPOCHS").get(owner)
    if epochs is None:
        epochs = {}
        if not _EPOCHS.set(owner, epochs):
            return 0   # an owner that takes no weak reference cannot be followed
    epochs[buffer] = epochs.get(buffer, 0) + 1
    return epochs[buffer]


def epoch(owner, buffer: str) -> int:
    epochs = _table("_EPOCHS").get(owner)
    return 0 if epochs is None else epochs.get(buffer, 0)


def live(value, owner, buffer: str) -> None:
    """`value` reads `owner`'s buffer when it is used, not now (it is the buffer, or follows it): what it reads is
    checked against the buffer's epoch at this moment."""
    try:
        ref = weakref.ref(owner)
    except TypeError:
        return
    _table("_LIVE").set(value, ((ref, buffer, epoch(owner, buffer)),))


def carried(value, from_value) -> None:
    """`value` reads, later, whatever `from_value` reads (a function that closes over it)."""
    records = _table("_LIVE").get(from_value)
    if records:
        _LIVE.set(value, tuple(_LIVE.get(value) or ()) + records)


def reads(value) -> tuple:
    """((owner, buffer, epoch made at), ...) for the buffers `value` reads later, owners still alive."""
    out = []
    for ref, buffer, made in _table("_LIVE").get(value) or ():
        owner = ref()
        if owner is not None:
            out.append((owner, buffer, made))
    return tuple(out)


def _fact(name, value, where, certainty="verified"):
    from .facts import Certainty, Fact, Source

    return Fact(name, value, Source("engine", where), Certainty(certainty))


def _decide(boundary, consumer, name, verdict, rule, declared, chosen, note, resolution=None, handle=None,
            owner=None, key=None):
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired

    contract = Contract(boundary, consumer, (name,))
    if verdict == "unrepaired":
        v, blocking = unrepaired(policies.current(), name)
        if blocking:
            _tally.refused(boundary)
        else:
            _tally.broken(boundary)
            if (owner is not None or key is not None) and not _tally.first(boundary, rule, owner=owner, key=key):
                return
        load.enforce([Decision(contract, name, v, RULES[rule], declared=declared, chosen=chosen,
                               blocking=blocking, note=note)])
        return
    _tally.counts(boundary)["resolved"] += 1
    if (boundary, rule) in _RECORDED:
        return
    _RECORDED.add((boundary, rule))
    load.enforce([Decision(contract, name, Verdict.RESOLVED, RULES[rule], declared=declared, chosen=chosen,
                           resolution=resolution, handle=handle, note=note)])


def read(boundary: str, consumer: str, where: str, value) -> None:
    """`value` is being read now: every buffer it reads later must still be at the epoch the value was made at."""
    records = reads(value)
    if not records:
        return
    from .facts import Epoch

    _tally.counts(boundary)["checks"] += 1
    for owner, buffer, made in records:
        now = epoch(owner, buffer)
        if now != made:
            name = _owner_name(owner, buffer)
            _decide(boundary, consumer, "Epoch", "unrepaired", "epoch_stale",
                    _fact("Epoch", Epoch(made, owner=name), f"{where}: the epoch of {name} when the value was made"),
                    _fact("Epoch", Epoch(now, owner=name), f"{where}: the epoch of {name} when the value is read"),
                    f"{where}: made from {name} at epoch {made}, read at epoch {now}: {now - made} write(s) in between "
                    f"change what it reads, and nothing says so", owner=owner)
    _tally.passed(boundary, ["epoch_stale"])
    _tally.tick(boundary)


def bind(boundary: str, consumer: str, where: str, value, snapshot):
    """`value`, which reads a buffer later, is being handed to a reader: resolution first, bind it to the buffer's
    value now. `snapshot()` returns the copy the reader is given instead (it reads nothing later). Under a policy that
    repairs nothing, nothing is bound, and a stale read is reported where it happens (refused where the policy
    stops). Returns what to hand over."""
    records = reads(value)
    if not records:
        return value
    from . import policies
    from .facts import Epoch

    if policies.current().mismatch_setting("Epoch") == "refuse":
        return value
    copy = snapshot()
    owner, buffer, made = records[0]
    name = _owner_name(owner, buffer)
    _decide(boundary, consumer, "Epoch", "resolved", "epoch_live",
            _fact("Epoch", Epoch(made, owner=name), f"{where}: the epoch of {name} when it was handed over"),
            _fact("Epoch", None, f"{where}: what the reader would read: the buffer as it is when it reads it",
                  "unknown"),
            f"{where}: {name} was handed to a reader that reads it later; it is bound to its value now",
            resolution="bind to the value at hand-over (a snapshot)", handle="epochs.bind")
    return copy


# --- artifacts and the conditions they were made for ---------------------------------------------------------------

def _assumed(conditions):
    from .facts import Assumed

    return Assumed(tuple(sorted(conditions.items())))


def assume(boundary: str, key, **conditions) -> None:
    """The artifact `key` (a captured graph, a compiled function) was made at `boundary` under `conditions`."""
    _ASSUMED[(boundary, key)] = _assumed(conditions)


def reuse(boundary: str, consumer: str, where: str, key, remake=None, **conditions) -> str:
    """The artifact `key` is about to be reused under `conditions`. Returns "as_is" when it was made for them (or
    nothing was recorded about it), "remade" when they changed and `remake()` was called - the caller makes it again
    and assume()s the new conditions. With nothing to remake it, or a policy that repairs nothing, it is "broken":
    reported, and the caller reuses the artifact as it is (refused and raised where the policy stops)."""
    made = _ASSUMED.get((boundary, key))
    if made is None:
        _tally.counts(boundary)["skipped"] += 1
        return "as_is"
    now = _assumed(conditions)
    _tally.counts(boundary)["checks"] += 1
    if now == made:
        _tally.passed(boundary, ["assumed_changed"])
        _tally.tick(boundary)
        return "as_is"
    from . import policies

    changed = ", ".join(f"{k}: {a} -> {b}" for (k, a), (_, b) in zip(made.conditions, now.conditions) if a != b) \
        if [k for k, _ in made.conditions] == [k for k, _ in now.conditions] else f"{made.conditions} -> {now.conditions}"
    declared = _fact("Assumed", made, f"{where}: the conditions the artifact was made for")
    chosen = _fact("Assumed", now, f"{where}: the conditions it is reused under")
    if remake is None or policies.current().mismatch_setting("Assumed") == "refuse":
        _decide(boundary, consumer, "Assumed", "unrepaired", "assumed_changed", declared, chosen,
                f"{where}: reused under other conditions ({changed}), and nothing can remake it here", key=key)
        return "broken"   # reported and reused as it is; where the policy stops, the decision has raised
    remake()
    _ASSUMED.pop((boundary, key), None)
    _decide(boundary, consumer, "Assumed", "resolved", "assumed_changed", declared, chosen,
            f"{where}: reused under other conditions ({changed}); made again for them",
            resolution="remake the artifact for the conditions now", handle="epochs.remake")
    return "remade"


def guarded(boundary: str, consumer: str, work, *args, **kwargs):
    """tally.guarded for these rules: an error inside entail never breaks the engine (principle 12)."""
    return _tally.guarded(boundary, consumer, "Epoch", work, *args, **kwargs)


def stats(boundary: Optional[str] = None) -> dict:
    return _tally.stats(boundary)


def reset(boundary: Optional[str] = None) -> None:
    """Forget the counts (of one boundary, or all), every buffer's epoch and every recorded value and artifact."""
    global _EPOCHS, _LIVE
    _tally.reset(boundary)
    _EPOCHS = _LIVE = None
    for k in [k for k in _ASSUMED if boundary is None or k[0] == boundary]:
        del _ASSUMED[k]
    for k in [k for k in _RECORDED if boundary is None or k[0] == boundary]:
        _RECORDED.discard(k)


__all__ = ["advance", "epoch", "live", "carried", "reads", "read", "bind", "assume", "reuse", "inside_capture",
           "guarded", "stats", "reset"]
