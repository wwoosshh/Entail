"""Adapter: make transformers itself keep the model's declared attention properties when a backend is chosen.

`install()` wraps PreTrainedModel._check_and_adjust_attn_implementation. transformers 5.17 calls it in two
places (modeling_utils.py:1263 at model __init__, and inside set_attn_implementation), so the decision is made
before the weights are read, and again when continuous batching switches to `paged|...`.

What happens when the requested implementation would drop a declared property depends on the policy
(core.policy()):
  resolve  (default) switch to an implementation measured to honour it - eager - and say so in one line.
           The model then runs, and runs as declared.
  refuse   raise RoleError, as before.
The paged implementations have no alternative that honours softcap, so there is nothing to switch to; that
case is refused under either policy, with a message that says what would work instead.

With the mode off (the default) the wrapper returns immediately, and `uninstall()` restores the original method.
"""
from .. import core
from . import _shared

_ORIG = None
# eager first: it is the reference implementation and it honoured every fact in the sweep; flex_attention
# honours softcap too but failed to compile for some fact combinations (sweep/RESULTS.md, void rows)
PREFERENCE = ["eager", "flex_attention"]


def verdict(config, impl):
    """(kind, message) or None. Kept as its own name because the tests and the pilot use it."""
    return _shared.verdict(config, impl, "transformers")


def install():
    """Wrap the loader's attention check. Returns 1, or 0 if already installed."""
    global _ORIG
    from transformers import PreTrainedModel

    if _ORIG is not None:
        return 0
    _ORIG = PreTrainedModel._check_and_adjust_attn_implementation

    def wrapped(self, attn_implementation, *a, **kw):
        impl = _ORIG(self, attn_implementation, *a, **kw)
        if core.mode() not in ("load", "debug"):
            return impl
        said = verdict(self.config, impl)
        if said is None:
            return impl
        if said[0] == "violation" and core.policy() == "resolve":
            paged = isinstance(impl, str) and impl.startswith("paged|")
            alt = None if paged else _shared.choose(self.config, "transformers", PREFERENCE)
            if alt is not None:
                _shared.note_resolution("transformers", "attention implementation", impl, alt, self.config)
                return _ORIG(self, alt, *a, **kw)
            if paged:
                kind, msg = said
                said = (kind, msg + "\n  no paged implementation honours it; generate() instead of "
                                    "generate_batch() keeps the declared behaviour")
        _shared.report(said)
        return impl

    PreTrainedModel._check_and_adjust_attn_implementation = wrapped
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from transformers import PreTrainedModel

    PreTrainedModel._check_and_adjust_attn_implementation = _ORIG
    _ORIG = None
    return 1
