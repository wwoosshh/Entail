"""Adapter v2 for what SGLang reads from a LoRA adapter's config (LIBRARY_DESIGN.md 4.8; ROADMAP M17.1;
adapter_config_contract.py).

  hook         sglang.srt.lora.lora.LoRAAdapter.__init__: the adapter object is built from the LoRAConfig the manager
               read (srt/lora/lora_manager.py load_lora_weights), and sets scaling = lora_alpha / r (lora.py L71) -
               the one float SGLang multiplies every LoRA weight by. The config reads target_modules, r, lora_alpha
               and use_dora (lora_config.py L42-45) and nothing else of adapter_config.json.
  read_choice  the scaling the adapter holds and the keys SGLang reads (the table).
  handles      apply_use_rslora: scaling = lora_alpha / sqrt(r), what PEFT does for an rsLoRA adapter (sglang#40835:
               served 4x too weak at r=16, 8x at r=64 without it).
Other dropped keys (rank_pattern, alpha_pattern, lora_bias, modules_to_save, ...) have no carrier here: reported.
"""
from .. import adapter_config_contract, core
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "load:sglang.adapter_config"
CONSUMER = "sglang"
_ORIG = None


def hooks():
    return [Hook("sglang.srt.lora.lora.LoRAAdapter.__init__", "load")]


def read_choice(adapter):
    """What SGLang holds of the adapter's declaration: its scaling and the keys it reads."""
    return {"scaling": getattr(adapter, "scaling", None),
            "reads": list(adapter_config_contract.table()["consumers"]["sglang"]["reads"])}


def handles(adapter):
    def apply_use_rslora(scaling):
        """The core computes the scaling (adapter_config_contract.carried_value); this only assigns it."""
        adapter.scaling = float(scaling)
        return True

    return {"apply_use_rslora": apply_use_rslora}


def _decide(adapter):
    cfg = getattr(adapter, "config", None)
    declared = dict(getattr(cfg, "hf_config", None) or {})
    path = getattr(cfg, "path", None)
    where = f"{path}/adapter_config.json" if path else "adapter_config.json (dict given to SGLang)"
    adapter_config_contract.check(BOUNDARY, CONSUMER, engine, declared, where, handles(adapter), owner=path)


def install():
    global _ORIG
    try:
        from sglang.srt.lora.lora import LoRAAdapter
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = LoRAAdapter.__init__

    def __init__(self, *args, **kwargs):
        _ORIG(self, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Coverage", lambda: _decide(self))

    LoRAAdapter.__init__ = __init__
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from sglang.srt.lora.lora import LoRAAdapter

    LoRAAdapter.__init__ = _ORIG
    _ORIG = None
    return 1


def stats():
    return adapter_config_contract.stats(BOUNDARY)


def reset():
    adapter_config_contract.reset(BOUNDARY)
