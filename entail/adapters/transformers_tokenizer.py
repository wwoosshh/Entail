"""Adapter v2 for the tokenizer transformers builds - and, through it, the tokenizer vLLM and SGLang hold, since
both call AutoTokenizer (LIBRARY_DESIGN.md 4.8; ROADMAP M15.3; realworld/CODEBOOK_v2.md G; transformers#48967).

  hook         transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained: every tokenizer class
               builds through it; the tokenizer is decided once it is built, before the first id it produces.
  read_choice  the tokenizer's base vocabulary (vocab_size) and its length (added tokens included).
  handles      none: a tokenizer that is not the model's is reported with the source that is (no repair measured).
vocab_contract reads the folder (tokenizer files, config vocab_size, embedding rows) and decides. A hub id is
resolved to its local cache folder (load.local_folder); nothing is downloaded here.
"""
from .. import core, vocab_contract
from .base import Hook

engine = "transformers"
versions = "5.17.0"
BOUNDARY = "load:transformers.tokenizer"
CONSUMER = "transformers.tokenizer"
_ORIG = None


def hooks():
    return [Hook("transformers.tokenization_utils_base.PreTrainedTokenizerBase.from_pretrained", "load")]


def read_choice(tokenizer):
    """(base vocabulary size or None, highest id + 1 or None) of a built tokenizer. The highest id is the larger of
    the base size and the ids of the added tokens (they can sit past len()); the added-token table is small, where
    get_vocab() builds the whole vocabulary as a dict at every load (M15.7 review: a visible load cost). A tokenizer
    without the table falls back to len()."""
    size = getattr(tokenizer, "vocab_size", None)
    size = int(size) if isinstance(size, int) and not isinstance(size, bool) else None
    n = None
    try:
        added = getattr(tokenizer, "added_tokens_decoder", None)
        if isinstance(added, dict) and size is not None:
            n = max([size] + [int(i) + 1 for i in added if isinstance(i, int)])
    except Exception:  # noqa: BLE001 - a table that cannot be read
        n = None
    if n is None:
        try:
            n = len(tokenizer)
        except Exception:  # noqa: BLE001 - a tokenizer without __len__
            n = None
    return size, (int(n) if isinstance(n, int) else None)


def handles():
    return {}


def _decide(name, kwargs, tokenizer):
    from .. import load

    folder = load.local_folder(name, kwargs.get("revision"), kwargs.get("cache_dir"))
    where = f"{type(tokenizer).__name__} built from {name}"
    if folder is None:
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "Vocab", f"{where}: no local folder to read")])
        return
    size, n = read_choice(tokenizer)
    vocab_contract.check(BOUNDARY, CONSUMER, folder, size, n, where, owner=folder)


def install():
    """Wrap the classmethod on the base class (every tokenizer class inherits it). Returns 1, or 0 if installed."""
    global _ORIG
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = PreTrainedTokenizerBase.__dict__["from_pretrained"].__func__

    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        tokenizer = _ORIG(cls, pretrained_model_name_or_path, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Vocab", lambda: _decide(pretrained_model_name_or_path, kwargs, tokenizer))
        return tokenizer

    PreTrainedTokenizerBase.from_pretrained = classmethod(from_pretrained)
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    PreTrainedTokenizerBase.from_pretrained = classmethod(_ORIG)
    _ORIG = None
    return 1


def stats():
    return vocab_contract.stats(BOUNDARY)


def reset():
    vocab_contract.reset(BOUNDARY)
