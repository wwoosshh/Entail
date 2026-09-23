"""One RANGE contract for the KV cache, written once, translated by each engine's adapter.

Three engines were checked separately first, and each one keeps its own books:

  transformers  a length per cache layer (`get_seq_length()`), grown by `update()`
  vLLM          blocks per request in a pool, handed out by `allocate_slots`
  SGLang        `kv_allocated_len` and `kv_committed_len` on the request itself

Different names, one fact: **how much of this sequence is actually held, and does it match how much the
sequence has?** So the rule lives here once, and each adapter only says where its engine keeps the numbers.
That translation is the whole point of a role vocabulary (ROADMAP 2.1): the fact is the stable thing, the
bookkeeping is not.

The rule, in the order it is checked:
  1. reserved vs written   if an engine tracks both, they must agree
  2. held vs needed        what is held must be what the tokens need
  3. never shrinks         if the previous extent is known, nothing may have quietly taken tokens away
A windowed extent is exempt from 2 and 3: it is meant to hold less than the sequence has.
"""
from dataclasses import dataclass
from typing import Optional

from .core import RoleError


@dataclass(frozen=True)
class KvExtent:
    """What one engine says about one sequence's KV, in words that do not belong to any engine.

    held      slots that hold KV for this sequence (the engine's own number)
    needed    tokens the sequence has, so the KV it needs
    written   slots actually written, when the engine tracks it separately from what it reserved
    window    a sliding window, when the extent is allowed to be shorter than the sequence
    previous  what `held` was the last time this extent was checked, when the adapter keeps a history
    granularity  the unit the engine allocates in (vLLM hands out blocks, not tokens), so `held` may exceed
                 `needed` by less than one unit. Without it, `held` must equal `needed` exactly
    """
    held: int
    needed: Optional[int] = None
    written: Optional[int] = None
    window: Optional[int] = None
    previous: Optional[int] = None
    granularity: Optional[int] = None


def check_extent(extent, where):
    """Raise RoleError if this extent does not add up. Returns the number of comparisons actually made."""
    made = 0
    capped = extent.window is not None and (extent.needed or 0) > extent.window

    if extent.written is not None:
        made += 1
        if extent.held != extent.written:
            raise RoleError(f"{where}: {extent.held} KV slots reserved but {extent.written} written. "
                            f"Two numbers about the same sequence, and nothing compares them.")
    if extent.needed is not None and not capped:
        made += 1
        if extent.granularity:
            # allocation is granular: enough, and not more than one unit more than enough
            if not (extent.needed <= extent.held < extent.needed + extent.granularity):
                raise RoleError(f"{where}: holds {extent.held} KV slots for {extent.needed} tokens, which is "
                                f"not one allocation unit ({extent.granularity}) of the right size. The slots "
                                f"a sequence holds and the tokens it has are two numbers nobody compares.")
        elif extent.held != extent.needed:
            raise RoleError(f"{where}: holds {extent.held} KV slots but the sequence has {extent.needed} "
                            f"tokens. The length nobody compared is the one that drifts.")
    if extent.previous is not None and not capped:
        made += 1
        if extent.held < extent.previous:
            raise RoleError(f"{where}: was {extent.previous} slots when last checked, {extent.held} now. "
                            f"Something dropped {extent.previous - extent.held} token(s) in between and said "
                            f"nothing.")
    return made
