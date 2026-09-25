"""Adapter v2 for vLLM's scoring (cross-encoder /score, /rerank) input: the token type the server gives the padding
(LIBRARY_DESIGN.md 4.8; ROADMAP M15.1; realworld/CODEBOOK_v2.md G; vllm#58138).

  hook         vllm.entrypoints.pooling.scoring.io_processor._apply_post_tokenization_to_token_type_ids: where the
               scoring processor truncates and pads the per-position segment ids parallel to the prompt tokens.
               vLLM 0.30 pads them with the LAST real position's id (the document's, 1), not the tokenizer's
               pad_token_type_id (0), so the padding is read as document and the scores move.
  read_choice  the id the tokenizer declares for padding (pad_token_type_id) and the id the padding was given -
               the positions past the truncated real length - or None when nothing was padded.
  handles      set_pad_type: write the declared id over the padding positions, in place, before the ids are
               compressed for the engine.
request_contract.pad_type decides (TokenType); the run goes on either way.
"""
from .. import core, request_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "request:vllm.scoring.token_type_ids"
CONSUMER = "vllm.scoring.io_processor"
_ORIG = None


def hooks():
    return [Hook("vllm.entrypoints.pooling.scoring.io_processor._apply_post_tokenization_to_token_type_ids",
                 "request")]


def _real_length(tokenizer, tok_params, n):
    """How many of the n ids survive the truncation the function applies before padding (its own rule, read from
    its parameters), so the padding starts after them."""
    max_length = getattr(tok_params, "truncate_prompt_tokens", None)
    if max_length is not None and max_length < 0:
        max_length = getattr(tok_params, "max_input_tokens", None)
    if max_length is not None and max_length < n:
        return max(int(max_length), 0)
    return n


def read_choice(tokenizer, tok_params, given, out):
    """(declared pad type or None, the id the padding got or None when nothing was padded, first padding index)."""
    real = _real_length(tokenizer, tok_params, len(given))
    if len(out) <= real:
        return None, None, real
    declared = getattr(tokenizer, "pad_token_type_id", None)
    used = out[real]   # the id the padding was given (the engine pads with one value)
    return (None if declared is None else int(declared)), int(used), real


def handles(out, real):
    def set_pad_type(type_id):
        for i in range(real, len(out)):
            out[i] = int(type_id)
        return type_id
    return {"set_pad_type": set_pad_type}


def _decide(tokenizer, tok_params, given, out):
    from .. import load

    declared, used, real = read_choice(tokenizer, tok_params, given, out)
    if used is None:
        return
    decisions = request_contract.pad_type(BOUNDARY, CONSUMER, declared, used,
                                          f"vllm scoring input, {len(out) - real} padding position(s)",
                                          declared_by=type(tokenizer).__name__)
    load.resolve(decisions, handles(out, real))


def install():
    """Wrap the module function; the call site resolves the module global at call time. Returns 1 or 0."""
    global _ORIG
    from vllm.entrypoints.pooling.scoring import io_processor as io

    if _ORIG is not None:
        return 0
    _ORIG = io._apply_post_tokenization_to_token_type_ids

    def _apply_post_tokenization_to_token_type_ids(tokenizer, tok_params, token_type_ids):
        given = list(token_type_ids)
        out = _ORIG(tokenizer, tok_params, token_type_ids)
        if core.mode() in ("load", "debug") and isinstance(out, list):
            request_contract.guarded(BOUNDARY, CONSUMER, _decide, tokenizer, tok_params, given, out)
        return out

    io._apply_post_tokenization_to_token_type_ids = _apply_post_tokenization_to_token_type_ids
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.entrypoints.pooling.scoring import io_processor as io

    io._apply_post_tokenization_to_token_type_ids = _ORIG
    _ORIG = None
    return 1


def stats():
    return request_contract.stats(BOUNDARY)


def reset():
    request_contract.reset(BOUNDARY)
