"""boundaries: signatures on code boundaries, and contracts on containers (LIBRARY_DESIGN.md 4.6).

A boundary states what it takes, what it returns and what it writes in place, with keyword-only role markers:

    @boundary(buf=INTERLEAVED, writes={"buf": SPLIT})
    def reorder(*, buf): ...

Results and written arguments carry the declared meaning to the next boundary, so nobody re-tags by hand. Declared
writes are checked against what really happened (tensor version counters), and torch custom ops against their
schema (`mutates_args`). Containers that replace their tensors (a KV cache, a buffer pool) get a contract on their
update and read boundaries instead, because a fact attached to a tensor stays behind on the replaced tensor.

The implementation still lives in core.py (the 2.2 boundary work) and kv_contract.py. M4.1 moves it here; until
then this module is the new name for the same objects. (It is `boundaries`, not `boundary`: the package
already exports the function `entail.boundary`, and a submodule of that name would replace it on import.)
"""
from .core import RoleError, boundary, carry, facts_of, tag
from .kv_contract import KvExtent, check_extent

__all__ = ["RoleError", "boundary", "carry", "facts_of", "tag", "KvExtent", "check_extent"]
