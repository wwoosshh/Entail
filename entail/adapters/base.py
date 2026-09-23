"""The adapter interface (LIBRARY_DESIGN.md 4.8, principle 8).

An adapter gives the core three things about one engine, and nothing else:
  hooks        where the engine makes a decision that consumes a fact (a backend choice, a slot allocation ...)
  read_choice  what the engine actually chose there, as facts
  handles      the operations that carry out a resolution (switch a backend, move a setting to its current name)

No rule logic: comparing, deciding and recording belong to contracts.py and record.py. M3.3 adds a static check
that fails when an adapter file contains rules. If an engine version moves a hook, the adapter reports
"hook not found" instead of turning itself off quietly.

The adapters that exist today (transformers_adapter, sglang_adapter, vllm_*, comfyui, diffusers_adapter,
rope_alias, *_cache_contract) predate this interface. They are rewritten to it in M3.3, M5.1 and M6.2.
"""
from typing import Any, Callable, Dict, List, NamedTuple, Protocol


class Hook(NamedTuple):
    target: str   # dotted path of the engine function, e.g. "sglang.srt....ModelRunner.init_attention_backends"
    site: str     # one of sites.SITES


class Adapter(Protocol):
    engine: str
    versions: str   # engine versions this adapter was checked against

    def hooks(self) -> List[Hook]:
        ...

    def read_choice(self, hook: Hook, *args: Any, **kwargs: Any) -> Dict[str, object]:
        """Vocabulary name -> Fact, for what the engine chose at this hook."""
        ...

    def handles(self) -> Dict[str, Callable]:
        """Resolution handle name -> the function that performs it."""
        ...
