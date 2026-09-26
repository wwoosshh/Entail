"""Adapter v2 for what vLLM's prefix-cache key covers of a request (LIBRARY_DESIGN.md 4.6, 4.8; ROADMAP M17.2;
cache_key_contract.py; vllm#56655).

  hook         vllm.v1.request.Request.__init__: the request is built with its input fields and computes its first
               block hashes there (v1/request.py L224 update_block_hashes). The hash's extra keys come from
               kv_cache_utils.generate_block_hash_extra_keys (LoRA name, multimodal hashes, cache_salt, a digest of
               prompt_embeds); prompt_is_token_ids - which positions take the embeddings - is not among them.
  read_choice  which of the request's input fields are set.
  handles      extend_key_prompt_is_token_ids: wrap generate_block_hash_extra_keys once so that every block's extra
               keys carry a digest of the block's mask, then remake this request's block hashes. Two requests with
               the same tokens and embeddings and different masks then key differently; the same request keys the
               same as before for itself (its own repeats still hit).
cache_key_contract decides (cache_key_incomplete); a request without a mask passes, said once per field set.
"""
import hashlib

from .. import cache_key_contract, core
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "container:vllm.request.block_hashes"
CONSUMER = "vllm.block_hashes"
_ORIG = None
_EXTENDED = []      # (module, the function it had before the extension); installed once per process


def hooks():
    return [Hook("vllm.v1.request.Request.__init__", "container")]


def read_choice(request):
    """Which input fields the request sets (a list or tensor counts when it is not empty); the fields are the
    table's (data/cache_key_fields.json, vllm.request)."""
    present = {}
    for f in cache_key_contract.table()["fields"]["vllm.request"]["names"]:
        v = getattr(request, f, None)
        try:
            present[f] = v is not None and (len(v) > 0 if hasattr(v, "__len__") else True)
        except TypeError:
            present[f] = v is not None
    return present


def _mask_digest(mask, start, end):
    return hashlib.sha256(bytes(1 if m else 0 for m in mask[start:end])).digest()


def extended(orig):
    """generate_block_hash_extra_keys with the block's mask digest appended when the request has a mask."""
    def generate_block_hash_extra_keys(request, start_token_idx, end_token_idx, start_mm_idx):
        keys, nxt = orig(request, start_token_idx, end_token_idx, start_mm_idx)
        mask = getattr(request, "prompt_is_token_ids", None)
        if mask:
            keys = tuple(keys or ()) + (("prompt_is_token_ids", _mask_digest(mask, start_token_idx, end_token_idx)),)
        return keys, nxt

    generate_block_hash_extra_keys.__entail_extended__ = True
    return generate_block_hash_extra_keys


def install_extension(module):
    """Extend the module's generate_block_hash_extra_keys once (the request hasher looks the name up at call time)."""
    fn = getattr(module, "generate_block_hash_extra_keys", None)
    if fn is None or getattr(fn, "__entail_extended__", False):
        return False
    _EXTENDED.append((module, fn))
    module.generate_block_hash_extra_keys = extended(fn)
    return True


def handles(request):
    def extend_key_prompt_is_token_ids(field):
        from vllm.v1.core import kv_cache_utils

        install_extension(kv_cache_utils)
        hashes = getattr(request, "block_hashes", None)
        if isinstance(hashes, list):
            del hashes[:]
            request.update_block_hashes()
        return True

    return {"extend_key_prompt_is_token_ids": extend_key_prompt_is_token_ids}


def _decide(request):
    present = read_choice(request)
    cache_key_contract.check(BOUNDARY, CONSUMER, engine, present, f"vllm Request {request.request_id}",
                             handles(request), owner=tuple(sorted(k for k, v in present.items() if v)))


def install():
    global _ORIG
    try:
        from vllm.v1.request import Request
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = Request.__init__

    def __init__(self, *args, **kwargs):
        _ORIG(self, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Coverage", lambda: _decide(self))

    Request.__init__ = __init__
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.v1.request import Request

    Request.__init__ = _ORIG
    _ORIG = None
    for module, fn in _EXTENDED:
        module.generate_block_hash_extra_keys = fn
    del _EXTENDED[:]
    return 1


def stats():
    return cache_key_contract.stats(BOUNDARY)


def reset():
    cache_key_contract.reset(BOUNDARY)
