"""Adapter v2 for what vLLM reads from a LoRA adapter's config (LIBRARY_DESIGN.md 4.8; ROADMAP M17.1;
adapter_config_contract.py).

  hook         vllm.lora.peft_helper.PEFTHelper.from_local_dir: the worker reads adapter_config.json here
               (lora/worker_manager.py L121) into the dataclass whose fields are the only keys vLLM keeps
               (peft_helper.py L28-40; from_dict filters every other key out at L86-88). vLLM reads use_rslora into
               its scaling (L56-58) and refuses use_dora, modules_to_save and bias loudly (validate_legal).
  read_choice  the scaling factor the helper holds and the keys vLLM reads (the table).
  handles      none: what vLLM drops (rank_pattern, alpha_pattern, lora_bias, ...) it has no place to carry;
               those are reported.
"""
from .. import adapter_config_contract, core
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "load:vllm.adapter_config"
CONSUMER = "vllm"
_ORIG = None


def hooks():
    return [Hook("vllm.lora.peft_helper.PEFTHelper.from_local_dir", "load")]


def read_choice(helper):
    """What vLLM holds of the adapter's declaration: its scaling factor and the keys it reads."""
    return {"scaling": getattr(helper, "vllm_lora_scaling_factor", None),
            "reads": list(adapter_config_contract.table()["consumers"]["vllm"]["reads"])}


def handles(helper):
    return {}


def _decide(lora_path):
    declared = adapter_config_contract.read(lora_path)
    where = f"{lora_path}/adapter_config.json"
    adapter_config_contract.check(BOUNDARY, CONSUMER, engine, declared, where, {}, owner=str(lora_path))


def install():
    global _ORIG
    try:
        from vllm.lora.peft_helper import PEFTHelper
    except ImportError:
        return 0
    if _ORIG is not None:
        return 0
    _ORIG = PEFTHelper.__dict__["from_local_dir"].__func__

    def from_local_dir(cls, lora_path, *args, **kwargs):
        helper = _ORIG(cls, lora_path, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "Coverage", lambda: _decide(lora_path))
        return helper

    PEFTHelper.from_local_dir = classmethod(from_local_dir)
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.lora.peft_helper import PEFTHelper

    PEFTHelper.from_local_dir = classmethod(_ORIG)
    _ORIG = None
    return 1


def stats():
    return adapter_config_contract.stats(BOUNDARY)


def reset():
    adapter_config_contract.reset(BOUNDARY)
