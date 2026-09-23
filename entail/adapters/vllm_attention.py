"""Adapter v2 for vLLM's attention backend choice (LIBRARY_DESIGN.md 4.8; ROADMAP M3.3).

  hook         vllm.v1.attention.selector.get_attn_backend: every attention layer asks it for its backend class,
               inside the engine-core process (install through the start-up hook). It is patched as soon as the
               selector module has run, before attention.py imports the name.
  read_choice  the backend's name (the class's get_name(), e.g. FLASH_ATTN), and the model's config object and path
               from get_current_vllm_config().model_config.
  handle       switch_attention_backend: select again with the named backend (attention_config.backend set for
               the call).
Every layer asks again, so a backend is decided once per process and model, and a switch is repeated for every
layer that asks for the same backend. The decisions are load.attention's; load.enforce records and stops.
"""
from .. import core, load, policies
from .base import Hook

engine = "vllm"
versions = "0.30.0"
_ORIG = None
_DONE = {}   # (model path, backend name) -> the backend class to return (switched or not)


def hooks():
    return [Hook("vllm.v1.attention.selector.get_attn_backend", "load")]


def read_choice(backend_cls):
    """The backend's name, and the config object and model path of the model being built."""
    from vllm.config import get_current_vllm_config

    mc = getattr(get_current_vllm_config(), "model_config", None)
    config = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
    return backend_cls.get_name(), config, getattr(mc, "model", None)


def handles(args, kwargs):
    def switch(target):
        from vllm.config import get_current_vllm_config
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        ac = get_current_vllm_config().attention_config
        saved = ac.backend
        ac.backend = AttentionBackendEnum[target]
        try:
            return _ORIG(*args, **kwargs)
        finally:
            ac.backend = saved
    return {"switch_attention_backend": switch}


def install():
    global _ORIG
    from vllm.v1.attention import selector

    if _ORIG is not None:
        return 0
    _ORIG = selector.get_attn_backend

    def get_attn_backend(*a, **kw):
        cls = _ORIG(*a, **kw)
        if core.mode() not in ("load", "debug"):
            return cls
        name, config, path = read_choice(cls)
        key = (path, name)
        if key not in _DONE:
            _DONE[key] = cls

            def decide():
                decisions = load.attention(engine, name, load.declared(path, config), policy=policies.current())
                done = load.resolve(decisions, handles(a, kw))
                load.enforce(decisions)
                return done.get("switch_attention_backend", cls)

            if config is not None:
                _DONE[key] = load.safely(f"load:{engine}.attention", f"{engine}.attention.{name}", "ModelProps",
                                         decide, cls)
        return _DONE[key]

    selector.get_attn_backend = get_attn_backend
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    from vllm.v1.attention import selector

    selector.get_attn_backend = _ORIG
    _ORIG = None
    _DONE.clear()
    return 1
