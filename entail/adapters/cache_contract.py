"""Adapter v2 for transformers' KV cache: where a cache keeps the length of each layer (LIBRARY_DESIGN.md 4.6, 4.8;
ROADMAP M5.1; audits/CACHE_CONTRACT.md).

  hook         transformers.cache_utils.Cache.update: the container's own boundary. Models call
               past_key_values.update(key, value, layer_idx), every cache class goes through it, and it knows which
               cache and which layer, so the books are kept per cache (the version before M5.1 wrapped the layer
               classes and compared the layers of every cache the process had seen: a second generate was refused).
  read_choice  one layer's length as the layer keeps it - an int for a dynamic layer, a device tensor for a static
               one, copied before the update because the layer increments it in place - and its sliding window.
  handles      none: a cache that does not add up cannot be repaired here.
kv_contract decides: grew (kv_needed, kv_shrank, kv_layers) at every update, request (kv_request) once after a
request with check_cache, flush for the lengths kept on the device. Nothing runs while torch compiles or a CUDA graph
is captured (kv_contract.inside_capture): a check there changes the output, so a compiled static cache is checked
once, after the request.

Not installed automatically (sitecustomize): its cost on the compiled static path is measured in M5.4 first.
"""
from .. import core, kv_contract
from .base import Hook

engine = "transformers"
versions = "5.17.0"
BOUNDARY = "container:transformers.cache_update"
AFTER_REQUEST = "container:transformers.after_request"
CONSUMER = "transformers.kv_cache"
_ORIG = None


def hooks():
    return [Hook("transformers.cache_utils.Cache.update", "container")]


def read_choice(cache, layer_idx, copy=False):
    """(length, window) of one layer as the cache keeps it; (0, None) for a layer the cache has not made yet."""
    layers = getattr(cache, "layers", None) or []
    if layer_idx >= len(layers):
        return 0, None
    layer = layers[layer_idx]
    try:
        n = layer.get_seq_length()
    except Exception:  # noqa: BLE001 - a layer that cannot say its length: its keys can
        keys = getattr(layer, "keys", None)
        n = int(keys.shape[-2]) if keys is not None and keys.numel() else 0
    if hasattr(n, "device") and copy:
        n = n.clone()
    w = getattr(layer, "sliding_window", None)
    return n, (int(w) if isinstance(w, int) and w > 0 else None)


def handles():
    return {}


def _decide(cache, layer_idx, before, added):
    after, window = read_choice(cache, layer_idx)
    kv_contract.grew(BOUNDARY, CONSUMER, cache, "transformers", layer_idx, before, after, added, window)


def install():
    """Wrap Cache.update. Returns 1, or 0 if already installed."""
    global _ORIG
    from transformers.cache_utils import Cache

    if _ORIG is not None:
        return 0
    _ORIG = Cache.update

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if core.mode() not in ("load", "debug") or kv_contract.inside_capture():
            return _ORIG(self, key_states, value_states, layer_idx, *args, **kwargs)
        before = kv_contract.guarded(BOUNDARY, CONSUMER, read_choice, self, layer_idx, True)
        out = _ORIG(self, key_states, value_states, layer_idx, *args, **kwargs)
        if before is not None:
            kv_contract.guarded(BOUNDARY, CONSUMER, _decide, self, layer_idx, before[0], int(key_states.shape[-2]))
        return out

    Cache.update = update
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from transformers.cache_utils import Cache

    Cache.update = _ORIG
    _ORIG = None
    return 1


def check_cache(cache, expected_length, where="after the request"):
    """The contract from outside, once per request: every layer holds the tokens the request wrote. On a compiled
    static-cache path this is the only place a check may live (audits/CACHE_CONTRACT.md: 1.007x there)."""
    lengths = {}
    for i in range(len(getattr(cache, "layers", None) or [])):
        n, w = read_choice(cache, i)
        lengths[i] = (int(n.item()) if hasattr(n, "device") else int(n), w)
    return kv_contract.request(AFTER_REQUEST, CONSUMER, f"transformers {type(cache).__name__} {where}", lengths,
                               int(expected_length))


def flush():
    """Read the comparisons kept on the device once (static layers updated outside a captured region)."""
    return kv_contract.flush(BOUNDARY, CONSUMER, "transformers static cache")


def stats():
    return kv_contract.stats(BOUNDARY)


def reset():
    kv_contract.reset()
