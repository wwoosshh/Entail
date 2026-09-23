"""Adapter v2 for transformers' KV cache: where a cache keeps the length of each layer, and where that length is
handed out and read later (LIBRARY_DESIGN.md 4.6, 4.8; ROADMAP M5.1, M5.2; audits/CACHE_CONTRACT.md).

  hooks        transformers.cache_utils.Cache.update: the container's own boundary. Models call
               past_key_values.update(key, value, layer_idx), every cache class goes through it, and it knows which
               cache and which layer, so the books are kept per cache (the version before M5.1 wrapped the layer
               classes and compared the layers of every cache the process had seen: a second generate was refused).
               A static layer keeps its length in a device tensor it increments in place: that write advances the
               counter's epoch (M5.2).
               Cache.get_query_offset: a static cache hands out that counter itself, not its value.
               masking_utils.add_offsets_to_mask_function: the flex mask closes over the offset and reads it when
               attention runs - a reader that reads later.
               the "flex_attention" attention function: where that mask is read.
  read_choice  one layer's length as the layer keeps it - an int for a dynamic layer, a device tensor for a static
               one, copied before the update because the layer increments it in place - and its sliding window; and
               whether an offset handed out is the layer's live counter.
  handles      a copy of the offset (a snapshot), for the resolution that binds the mask to the value at hand-over.
kv_contract decides the KV rules: grew at every update, request once after a request (check_cache), flush for the
lengths kept on the device. epochs decides TIME: bind at the hand-over (resolution first), read where attention runs
(rolebench 10: every layer's update writes the counter before its attention reads the mask). Nothing runs while torch
compiles or a CUDA graph is captured (inside_capture): a check there changes the output, so a compiled static cache is
checked once, after the request, and its mask is not bound.

Not installed automatically (sitecustomize): its cost on the compiled static path is measured in M5.4 first.
"""
from .. import core, epochs, kv_contract
from .base import Hook

engine = "transformers"
versions = "5.17.0"
BOUNDARY = "container:transformers.cache_update"
AFTER_REQUEST = "container:transformers.after_request"
MASK_BUILDER = "container:transformers.mask_offset"
FLEX = "container:transformers.flex_attention"
CONSUMER = "transformers.kv_cache"
_ORIG = None


def hooks():
    return [Hook("transformers.cache_utils.Cache.update", "container"),
            Hook("transformers.cache_utils.Cache.get_query_offset", "container"),
            Hook("transformers.masking_utils.add_offsets_to_mask_function", "container"),
            Hook("transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS['flex_attention']", "container")]


def _counter(cache, layer_idx):
    """The buffer name of a layer's position counter, when the layer keeps it as a live tensor."""
    return f"layer {layer_idx} length"


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


def live_offset(cache, layer_idx, offset):
    """True when the offset a cache handed out is the layer's own counter tensor (it follows every later write)."""
    layers = getattr(cache, "layers", None) or []
    return layer_idx < len(layers) and offset is getattr(layers[layer_idx], "cumulative_length", None) \
        and hasattr(offset, "device")


def handles(offset=None):
    return {"epochs.bind": lambda: offset.clone()}


def _decide(cache, layer_idx, before, added):
    after, window = read_choice(cache, layer_idx)
    kv_contract.grew(BOUNDARY, CONSUMER, cache, "transformers", layer_idx, before, after, added, window)
    if hasattr(after, "device"):   # a counter kept in a tensor was incremented in place
        epochs.advance(cache, _counter(cache, layer_idx))


def _handed_out(cache, layer_idx, offset):
    if live_offset(cache, layer_idx, offset):
        epochs.live(offset, cache, _counter(cache, layer_idx))


def _to_mask(offset):
    """The offset handed to a mask that reads it later: bound to its value now (resolution first)."""
    return epochs.bind(MASK_BUILDER, "transformers.flex_mask", "transformers flex mask", offset,
                       handles(offset)["epochs.bind"])


def _active():
    return core.mode() in ("load", "debug") and not kv_contract.inside_capture()


def install():
    """Wrap the four places. Returns 1, or 0 if already installed."""
    global _ORIG
    from transformers import masking_utils
    from transformers.cache_utils import Cache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if _ORIG is not None:
        return 0
    _ORIG = {"update": Cache.update, "get_query_offset": Cache.get_query_offset,
             "add_offsets": masking_utils.add_offsets_to_mask_function,
             "flex": ALL_ATTENTION_FUNCTIONS.get("flex_attention")}

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if not _active():
            return _ORIG["update"](self, key_states, value_states, layer_idx, *args, **kwargs)
        before = kv_contract.guarded(BOUNDARY, CONSUMER, read_choice, self, layer_idx, True)
        out = _ORIG["update"](self, key_states, value_states, layer_idx, *args, **kwargs)
        if before is not None:
            kv_contract.guarded(BOUNDARY, CONSUMER, _decide, self, layer_idx, before[0], int(key_states.shape[-2]))
        return out

    def get_query_offset(self, layer_idx=0):
        offset = _ORIG["get_query_offset"](self, layer_idx)
        if _active():
            epochs.guarded(MASK_BUILDER, CONSUMER, _handed_out, self, layer_idx, offset)
        return offset

    def add_offsets_to_mask_function(mask_function, q_offset, kv_offset):
        if not _active():
            return _ORIG["add_offsets"](mask_function, q_offset, kv_offset)
        bound = epochs.guarded(MASK_BUILDER, CONSUMER, _to_mask, q_offset)
        q_offset = q_offset if bound is None else bound
        fn = _ORIG["add_offsets"](mask_function, q_offset, kv_offset)
        epochs.guarded(MASK_BUILDER, CONSUMER, epochs.carried, fn, q_offset)
        return fn

    def flex_attention(module, query, key, value, attention_mask, *args, **kwargs):
        if _active():
            mask_mod = getattr(attention_mask, "mask_mod", None)
            if mask_mod is not None:
                epochs.guarded(FLEX, "transformers.flex_attention", epochs.read, FLEX, "transformers.flex_attention",
                               f"transformers flex attention (layer {getattr(module, 'layer_idx', '?')})", mask_mod)
        return _ORIG["flex"](module, query, key, value, attention_mask, *args, **kwargs)

    Cache.update = update
    Cache.get_query_offset = get_query_offset
    masking_utils.add_offsets_to_mask_function = add_offsets_to_mask_function
    if _ORIG["flex"] is not None:
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from transformers import masking_utils
    from transformers.cache_utils import Cache
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    Cache.update = _ORIG["update"]
    Cache.get_query_offset = _ORIG["get_query_offset"]
    masking_utils.add_offsets_to_mask_function = _ORIG["add_offsets"]
    if _ORIG["flex"] is not None:
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = _ORIG["flex"]
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
    epochs.reset()
