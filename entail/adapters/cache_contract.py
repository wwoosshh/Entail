"""RANGE contract for the KV cache: the length a cache reports must be the length it actually grew to.

Measuring fact propagation showed why this cannot be done by tagging tensors (audits/PROPAGATE.md): a cache
replaces its buffers as it grows, so a fact attached to a tensor goes stale while the live cache carries
nothing. The fact belongs to the container, so it is recorded and checked at the container's own boundary,
`CacheLayerMixin.update` in transformers 5.17 (cache_utils.py:57).

Two contracts, both checked on every update:
  growth     a layer that held n tokens and is given k more must report n + k afterwards. A restored session
             that lost a token, a rewound counter, a double write all break this.
  agreement  the layers of one cache must agree on their length within an update round; a sliding-window layer
             may be shorter, never longer.

Errors name the layer, the number the cache reported and the number it should have reported. Active in debug
and load modes; off by default.
"""
from .. import core
from ..core import RoleError
from ..kv_contract import KvExtent, check_extent

_ORIG = None
_STATE = {}   # id(layer) -> {"length": int, "index": int}
_ROUND = {}   # id(cache-ish owner) -> first length seen in the current round
STATS = {"updates": 0, "checked": 0, "complaints": 0}


def _raw_length(layer):
    """The layer's length as it keeps it: an int for a dynamic layer, a device tensor for a static one."""
    try:
        return layer.get_seq_length()
    except Exception:
        keys = getattr(layer, "keys", None)
        return int(keys.shape[-2]) if keys is not None and keys.numel() else 0


def _length(layer, copy=False):
    """`copy=True` for the reading taken before the update: a static layer increments its counter in place, so
    without a clone the 'before' and 'after' readings are the same tensor and every comparison fails."""
    v = _raw_length(layer)
    if not hasattr(v, "device"):
        return int(v)
    return v.clone() if copy else v


def _window(layer):
    w = getattr(layer, "sliding_window", None)
    return int(w) if isinstance(w, int) and w > 0 else None


def _concrete_layers():
    """Every cache layer class that defines its own update.

    The mixin's `update` is abstract and each layer class overrides it, so wrapping the mixin wraps nothing;
    the first measurement recorded 0 updates because of exactly that.
    """
    from transformers.cache_utils import CacheLayerMixin

    seen, out = set(), []

    def walk(cls):
        for sub in cls.__subclasses__():
            if id(sub) in seen:
                continue
            seen.add(id(sub))
            if "update" in vars(sub):
                out.append(sub)
            walk(sub)

    walk(CacheLayerMixin)
    return out


def install():
    """Wrap the cache layers' update boundary. Returns the number of classes wrapped."""
    global _ORIG

    if _ORIG is not None:
        return 0
    _ORIG = {}

    def wrapped(self, key_states, value_states, *args, **kwargs):
        active = core.mode() in ("load", "debug")
        before = _length(self, copy=True) if active else None
        out = _ORIG[type(self)](self, key_states, value_states, *args, **kwargs)
        if not active:
            return out
        STATS["updates"] += 1
        new_tokens = int(key_states.shape[-2])
        after = _length(self)

        # A static cache keeps its length in a device tensor. Reading it from Python costs a synchronisation
        # per layer per step - measured at 8x on this card - and it perturbed a graph-captured run (the output
        # changed). So for those layers the comparison stays on the device and is read once, in flush().
        if hasattr(after, "device"):
            state = _STATE.setdefault(id(self), {"length": None, "index": len(_STATE), "window": _window(self),
                                                 "bad": None})
            import torch

            flag = state["bad"]
            if flag is None:
                # our own tensor: a CUDA-graph output must not be held across steps, the graph overwrites it
                flag = state["bad"] = torch.zeros((), dtype=torch.bool, device=after.device)
            flag.logical_or_((after != (before + new_tokens)).reshape(()))
            STATS["checked"] += 1
            return out
        window = _window(self)
        state = _STATE.setdefault(id(self), {"length": before, "index": len(_STATE), "window": _window(self)})

        # The rule is kv_contract.check_extent, shared with the vLLM and SGLang adapters. This adapter only
        # translates: what the layer holds now, what it should hold, and what it held last time.
        expected = before + new_tokens
        capped = window is not None and expected > window
        try:
            check_extent(KvExtent(held=after, needed=expected, window=window,
                                  previous=(state["length"] + new_tokens) if state["length"] is not None
                                  else None),
                         f"kv cache layer {state['index']}")
        except RoleError:
            STATS["complaints"] += 1
            raise
        state["length"] = after

        # agreement: inside one round some layers have been written and some have not, so the lengths seen
        # across layers must be exactly {after} or {after, after - new_tokens}. A third number means a layer
        # is carrying a different history: a cache restored unevenly, a layer that was skipped, a counter that
        # drifted. Sliding-window layers are left out; they cap on purpose.
        allowed = {after, after - new_tokens}
        odd = {s["index"]: s["length"] for s in _STATE.values()
               if s["length"] is not None and s["window"] is None and s["length"] not in allowed}
        if odd and not capped and window is None:
            STATS["complaints"] += 1
            raise RoleError(
                f"kv cache layer {state['index']} reports {after} tokens, but {len(odd)} other layer(s) hold "
                f"a different length: {dict(sorted(odd.items())[:4])}. The layers of one cache disagree and "
                f"nothing compares them.")
        STATS["checked"] += 1
        return out

    classes = _concrete_layers()
    for cls in classes:
        _ORIG[cls] = cls.update
        cls.update = wrapped
    return len(classes)


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    n = len(_ORIG)
    for cls, fn in _ORIG.items():
        cls.update = fn
    _ORIG = None
    _STATE.clear()
    _ROUND.clear()
    return n


def check_cache(cache, expected_length, where="after the request"):
    """The same contract from outside: every layer of this cache must report expected_length.

    Measured reason for this second form: on the compiled static-cache path, checking inside the update
    boundary costs 48x and changes the output, because the check lands inside the captured region
    (audits/CACHE_CONTRACT.md). One check per request, outside that region, costs nothing.
    """
    bad = {}
    for i, layer in enumerate(getattr(cache, "layers", []) or []):
        n = _length(layer)
        n = int(n) if not hasattr(n, "device") else int(n.item())
        window = _window(layer)
        want = min(expected_length, window) if window else expected_length
        if n != want:
            bad[i] = (n, want)
    if bad:
        STATS["complaints"] += 1
        shown = {i: f"{got} tokens, expected {want}" for i, (got, want) in sorted(bad.items())[:4]}
        raise RoleError(f"kv cache {where}: {len(bad)} layer(s) do not hold the number of tokens the request "
                        f"wrote: {shown}")
    return len(getattr(cache, "layers", []) or [])


def flush():
    """Read the device-side results once, and raise if any static layer's length did not add up.

    Call it where a synchronisation is already happening anyway: after a generate, at the end of a request.
    Returns the number of layers that were checked this way.
    """
    checked, bad = 0, []
    for state in _STATE.values():
        if state.get("bad") is None:
            continue
        checked += 1
        if bool(state["bad"]):
            bad.append(state["index"])
        state["bad"] = None
    if bad:
        STATS["complaints"] += 1
        raise RoleError(f"kv cache layers {bad}: the length they report is not the length they were given. "
                        f"A static cache keeps that number on the device, so nothing compared it.")
    return checked


def stats():
    return dict(STATS)


def reset():
    _STATE.clear()
    _ROUND.clear()
    for k in STATS:
        STATS[k] = 0
