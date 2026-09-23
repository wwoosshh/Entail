"""kv_contract: the contract of the containers that hold a sequence's KV, in the core (LIBRARY_DESIGN.md 4.6, 4.7;
ROADMAP M5.1).

A cache replaces its tensors as it grows, so a fact tagged on a tensor goes stale while the live cache carries
nothing (audits/PROPAGATE.md). The fact belongs to the container, and it is checked at the container's own boundary:
where an engine updates or allocates a sequence's KV. Three engines keep three sets of books - a length per cache
layer (transformers), blocks per request (vLLM), two counters per request (SGLang) - but the fact is one: how much
of this sequence is actually held, and does it match what the sequence has. An adapter only says where its engine
keeps the numbers, as a KvExtent; the rules are here, once:

  kv_written  the slots reserved and the slots written agree (an engine that counts both)
  kv_needed   the slots held are the slots the tokens need: equal, or within one allocation unit when the engine
              allocates in units (vLLM blocks). A sliding window is allowed to hold less
  kv_shrank   nothing held shrank since the last check of the same sequence, unless a window caps it
  kv_layers   the layers of one cache agree on their length within a round of updates (a window may be shorter)
  kv_request  after a request, every layer of its cache holds the tokens the request wrote (checked once, outside
              any captured region: on a compiled path that is the only place a check may live)

A rule that holds is counted (tally.PASSES): a container boundary runs per layer per step, thousands of times a
request. A broken one is a refused, blocking Decision recorded and raised through load.enforce. Nothing here reads
the device on the hot path: lengths kept in device tensors are compared on the device and read once, by flush().
Nothing runs while torch is compiling or a CUDA graph is being captured, and an error inside entail never breaks the
engine (tally.inside_capture, tally.guarded).
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import tally as _tally
from .core import RoleError
from .facts import _number
from .tally import PASSES, STATS  # noqa: F401 - (boundary, rule) -> passes; boundary -> counts


@dataclass(frozen=True)
class KvExtent:
    """What one engine says about one sequence's KV, in words that do not belong to any engine.

    held      slots that hold KV for this sequence (the engine's own number)
    needed    tokens the sequence has, so the KV it needs
    written   slots actually written, when the engine tracks it separately from what it reserved
    window    a sliding window, when the extent is allowed to be shorter than the sequence
    previous  what `held` should at least be, from the last time this sequence was checked
    granularity  the unit the engine allocates in (vLLM hands out blocks, not tokens), so `held` may exceed
                 `needed` by less than one unit. Without it, `held` must equal `needed` exactly
    """
    held: int
    needed: Optional[int] = None
    written: Optional[int] = None
    window: Optional[int] = None
    previous: Optional[int] = None
    granularity: Optional[int] = None

    def __post_init__(self):
        if self.held is None:
            raise ValueError("KvExtent.held: required")
        for name in ("held", "needed", "written", "previous"):
            _number("KvExtent", name, getattr(self, name), 0, integer=True)
        for name in ("window", "granularity"):
            _number("KvExtent", name, getattr(self, name), 0, integer=True, strict=True)


# rule -> (what the declared side is, what the engine's side is), for the two facts of a refused decision
_SIDES = {"kv_written": ("the slots written", "the slots reserved"),
          "kv_needed": ("the slots its tokens need", "the slots it holds"),
          "kv_shrank": ("what it held at the last check, with what was added since", "what it holds now"),
          "kv_layers": ("the length of this layer", "the lengths of the other layers"),
          "kv_request": ("the tokens the request wrote", "what its cache holds")}


def _rules(held, needed=None, written=None, window=None, previous=None, granularity=None):
    """The rules, on plain numbers (the one place they are written): ([(rule, note, the number the declared side
    stands for)] for the broken ones, [the rules that were checked]). Text is made only for a broken rule."""
    broken, checked = [], []
    capped = window is not None and (needed or 0) > window
    if written is not None:
        checked.append("kv_written")
        if held != written:
            broken.append(("kv_written", f"{held} KV slots reserved but {written} written. Two numbers about the same "
                                         f"sequence, and nothing compares them.", written))
    if needed is not None and not capped:
        checked.append("kv_needed")
        if granularity:
            if not (needed <= held < needed + granularity):   # enough, and less than one unit more
                broken.append(("kv_needed", f"holds {held} KV slots for {needed} tokens, which is not one allocation "
                                            f"unit ({granularity}) of the right size. The slots a sequence holds and the "
                                            f"tokens it has are two numbers nobody compares.", needed))
        elif held != needed:
            broken.append(("kv_needed", f"holds {held} KV slots but the sequence has {needed} tokens. The length nobody "
                                        f"compared is the one that drifts.", needed))
    if previous is not None and not capped:
        checked.append("kv_shrank")
        if held < previous:
            broken.append(("kv_shrank", f"was {previous} slots when last checked, {held} now. Something dropped "
                                        f"{previous - held} token(s) in between and said nothing.", previous))
    return broken, checked


def _evaluate(e: KvExtent):
    return _rules(e.held, e.needed, e.written, e.window, e.previous, e.granularity)


_stats, _tick, _passed = _tally.counts, _tally.tick, _tally.passed


def _side(where, text, value):
    from .facts import Certainty, Fact, Source

    if value is None:   # a number that was never read (kept on the device): said to be unknown, not made up
        return Fact("KvExtent", None, Source("engine", f"{where}: {text}"), Certainty.UNKNOWN)
    return Fact("KvExtent", value, Source("engine", f"{where}: {text}"), Certainty.VERIFIED)


def _refuse(boundary, consumer, where, extent, broken):
    """One refused, blocking Decision per broken rule, recorded and raised by load.enforce. `extent` is the engine's
    side; each broken rule gives the number the other side stands for (None when it was not read)."""
    from . import load
    from .contracts import RULES, Contract, Decision, Verdict

    _tally.refused(boundary)   # counted, and the summary written now: the process may not live to exit normally
    contract = Contract(boundary, consumer, ("KvExtent",))
    decisions = []
    for rule, note, wanted in broken:
        mine, theirs = _SIDES[rule]
        decisions.append(Decision(contract, "KvExtent", Verdict.REFUSED, RULES[rule],
                                  declared=_side(where, mine, None if wanted is None else KvExtent(held=wanted)),
                                  chosen=_side(where, theirs, extent), blocking=True, note=f"{where}: {note}"))
    load.enforce(decisions)


def check(boundary: str, consumer: str, where: str, extent: KvExtent) -> int:
    """Decide one extent at a container boundary. The rules that hold are counted; a broken one is refused and stops
    the run before the step produces anything. Returns the number of rules the extent was checked on."""
    broken, checked = _evaluate(extent)
    _stats(boundary)["checks"] += 1
    _passed(boundary, [r for r in checked if r not in {b[0] for b in broken}])
    _tick(boundary)
    if broken:
        _refuse(boundary, consumer, where, extent, broken)
    return len(checked)


def skipped(boundary: str, n: int = 1) -> None:
    """Sequences an adapter did not hand over because they are meant to hold less (a sliding window), counted."""
    _stats(boundary)["skipped"] += n
    _tick(boundary)


def check_extent(extent: KvExtent, where: str) -> int:
    """The 0.3.0 form: raise RoleError naming what does not add up, without a ledger. Returns the number of
    comparisons made. New code goes through check()."""
    broken, checked = _evaluate(extent)
    if broken:
        raise RoleError("; ".join(f"{where}: {note}" for _, note, _ in broken))
    return len(checked)


# --- the books of one cache: its layers' last lengths, for kv_shrank and kv_layers ------------------------------

_BOOKS = None


def _books(owner):
    """The books of one cache object, forgotten when the cache is collected."""
    global _BOOKS
    if _BOOKS is None:
        from .load import ByObject

        _BOOKS = ByObject()
    books = _BOOKS.get(owner)
    if books is None:
        books = {"lengths": {}, "counts": {}, "windowed": set(), "device": {}}
        if not _BOOKS.set(owner, books):
            return None   # an owner that takes no weak reference: nothing is remembered for it
    return books


def _is_tensor(x):
    return hasattr(x, "device") and hasattr(x, "dtype")


def grew(boundary: str, consumer: str, owner, where: str, layer: int, before, after, added: int,
         window: Optional[int] = None) -> None:
    """A layer of the cache `owner` was given `added` tokens: it held `before` and holds `after`.

    kv_needed: after == before + added (a window may hold less); kv_shrank: nothing was dropped since this layer's
    last update (its length then, plus what was added now); kv_layers: the layers of this cache - not of any other -
    hold {after, after - added} in the middle of a round. Lengths kept in device tensors are compared on the device
    and read by flush(). This runs per layer per step: on the way that holds, no text and no fact is made."""
    books = _books(owner)
    if _is_tensor(before) or _is_tensor(after):
        _defer(boundary, books, layer, before, after, added)
        return
    after, needed = int(after), int(before) + added
    last = None if books is None else books["lengths"].get(layer)
    previous = None if last is None else last + added
    broken, checked = _rules(after, needed, window=window, previous=previous)
    _stats(boundary)["checks"] += 1
    _passed(boundary, [r for r in checked if r not in {b[0] for b in broken}] if broken else checked)
    _tick(boundary)
    if broken:
        _refuse(boundary, consumer, f"{where} {type(owner).__name__} layer {layer}",
                KvExtent(held=after, needed=needed, window=window, previous=previous), broken)
    if books is None:
        return
    books["lengths"][layer] = after
    if window is not None:
        books["windowed"].add(layer)
        return   # a window caps on purpose; it takes no part in the agreement
    counts = books["counts"]   # length -> how many of this cache's layers hold it
    if last is not None:
        if counts.get(last, 0) > 1:
            counts[last] -= 1
        else:
            counts.pop(last, None)
    counts[after] = counts.get(after, 0) + 1
    if all(n == after or n == after - added for n in counts):
        _passed(boundary, ["kv_layers"])
        return
    odd = {i: n for i, n in books["lengths"].items()
           if i not in books["windowed"] and n != after and n != after - added}
    shown = dict(sorted(odd.items())[:4])
    _first, n = sorted(odd.items())[0]
    _refuse(boundary, consumer, f"{where} {type(owner).__name__} layer {layer}", KvExtent(held=n), [(
        "kv_layers", f"holds {int(after)} tokens, but {len(odd)} other layer(s) of the same cache hold a different "
                     f"length: {shown}. The layers of one cache disagree and nothing compares them.", int(after))])


def _defer(boundary, books, layer, before, after, added):
    """A length kept in a device tensor: compared on the device, no synchronisation; flush() reads the result."""
    _stats(boundary)["deferred"] += 1
    _tick(boundary)
    if books is None:
        return
    flag = books["device"].get(layer)
    if flag is None:
        import torch

        # our own tensor: a value computed inside a captured graph must not be held across steps
        flag = books["device"][layer] = torch.zeros((), dtype=torch.bool, device=after.device)
    flag.logical_or_((after != (before + added)).reshape(()))


def flush(boundary: str, consumer: str, where: str = "kv cache") -> int:
    """Read the device-side comparisons once - where a synchronisation happens anyway, after a generate or at the end
    of a request - and refuse if a layer's length did not add up. Returns the number of layers read."""
    if _BOOKS is None:
        return 0
    read, bad = 0, []
    for _ref, books in list(_BOOKS._d.values()):
        for layer, flag in list(books["device"].items()):
            read += 1
            if bool(flag):
                bad.append(layer)
            del books["device"][layer]
    if bad:
        _refuse(boundary, consumer, where, None, [(
            "kv_needed", f"layers {sorted(bad)}: the length they report is not the length they were given. A static "
                         f"cache keeps that number on the device, so nothing compared it.", None)])
    elif read:
        _passed(boundary, ["kv_needed"])
    return read


def request(boundary: str, consumer: str, where: str, lengths: Dict[int, Tuple[int, Optional[int]]],
            expected: int) -> int:
    """After a request: every layer holds the tokens the request wrote; a windowed layer holds at most its window.
    `lengths` is {layer: (length, window)} as the adapter read them, once, outside any captured region."""
    _stats(boundary)["checks"] += 1
    bad = {}
    for layer, (n, window) in lengths.items():
        want = min(expected, window) if window else expected
        if int(n) != want:
            bad[layer] = (int(n), want)
    if not bad:
        _passed(boundary, ["kv_request"])
        return len(lengths)
    first, (got, want) = sorted(bad.items())[0]
    shown = {i: f"{g} tokens, expected {w}" for i, (g, w) in sorted(bad.items())[:4]}
    _refuse(boundary, consumer, where, KvExtent(held=got), [(
        "kv_request", f"{len(bad)} layer(s) do not hold the number of tokens the request wrote: {shown}", want)])
    return len(lengths)


# --- counts, summaries and guards: shared with the other per-step boundaries (tally.py) ---------------------------

inside_capture = _tally.inside_capture


def guarded(boundary: str, consumer: str, work, *args, **kwargs):
    """tally.guarded for this contract: an error inside entail never breaks the engine (principle 12)."""
    return _tally.guarded(boundary, consumer, "KvExtent", work, *args, **kwargs)


def stats(boundary: Optional[str] = None) -> dict:
    """Counts for one boundary (checks, refused, skipped, deferred, and the passes per rule), or all of them."""
    return _tally.stats(boundary)


def reset(boundary: Optional[str] = None) -> None:
    """Forget the counts (of one boundary, or all) and every cache's books."""
    global _BOOKS
    _tally.reset(boundary)
    _BOOKS = None


__all__ = ["KvExtent", "check", "check_extent", "grew", "flush", "request", "skipped", "inside_capture", "guarded",
           "stats", "reset", "PASSES", "STATS"]
