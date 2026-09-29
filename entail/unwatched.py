"""unwatched: say which inference engines of a run entail does not watch (field test, entail#38).

WhisperX transcribes with CTranslate2 and aligns with a wav2vec2 model through transformers; MinerU runs its document
model with llama.cpp and its layout detector through transformers. entail saw only the side models, and the page
said the checked points were fine without a word about the engines that did the main work. The engines named in
data/unwatched_engines.json are now noted when they finish importing (the start-up hook watches their modules):
one unknown per engine per process, at engine:<module>.unwatched, which names the engine and its version. It is a
statement of coverage, not a check of the engine: nothing in it compares a value, and a run whose only record is this
line is still listed on the page, as a run in which something was not checked.
"""
import json
import os
import sys

BOUNDARY = "engine:{module}.unwatched"
_NOTED = set()      # modules noted in this process
_TABLE = None


def engines() -> dict:
    """{import name: what users call it} from data/unwatched_engines.json, read once."""
    global _TABLE
    if _TABLE is None:
        with open(os.path.join(os.path.dirname(__file__), "data", "unwatched_engines.json"), encoding="utf-8") as f:
            _TABLE = json.load(f)["engines"]
    return _TABLE


def noted() -> int:
    """Record the unwatched engines imported in this process that are not noted yet. Returns how many were noted
    now. Called by the start-up hook when one of their modules has finished importing (and when entail is turned
    on after it was imported), so it looks at every engine of the table, not only the one just imported."""
    from dataclasses import replace

    from . import load, policies

    decisions = []
    for module, label in sorted(engines().items()):
        mod = sys.modules.get(module)
        if mod is None or module in _NOTED:
            continue
        _NOTED.add(module)
        version = getattr(mod, "__version__", None)
        what = f"{label} {version}" if isinstance(version, str) else label
        d = load.cannot_check(BOUNDARY.format(module=module), module, "Coverage",
                              f"this process loaded {what} ({module}), which entail does not watch: what it "
                              f"computes is not among these results", policies.current())
        # never blocking, not even in debug mode: this runs inside the program's import, where a stop would break
        # the import instead of saying more (the hook would report it as an adapter that could not install)
        decisions.append(replace(d, blocking=False))
    if decisions:
        load.enforce(decisions)
    return len(decisions)


def install() -> int:
    """For the start-up hook's table: note what is imported now."""
    return noted()


def reset() -> None:
    _NOTED.clear()
