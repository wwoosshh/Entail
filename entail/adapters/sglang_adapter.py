"""Adapter: make SGLang keep the model's declared attention properties when it picks a backend.

SGLang picks the backend in the scheduler process, not in the process that constructs Engine(...), and the name
can come from the device rather than from a server argument (flashinfer is the default on this card). Two hooks:

  resolve_attention_backend_strs  (model_runner.py:1027 calls it; attention_backend_setup.py:178 defines it)
      The moment the (prefill, decode) names are chosen. build_attention_backends then reads the names back off
      the runner, so changing them here changes what is actually built. Under the `resolve` policy a backend
      that would drop a declared property is replaced by one measured to honour it (triton), with one line
      saying so.
  ModelRunner.init_attention_backends
      After the names are stamped: check them, and under `refuse` (or when no replacement exists) raise. With
      ENTAIL_VERBOSE it also prints the class that was really built, so a resolution can be seen to have
      taken effect rather than only to have been announced.

Both hooks run in the child process, so install through the sitecustomize shim (autoinstall/sitecustomize.py).
"""
import os

from .. import core
from . import _shared

_ORIG = None
_ORIG_RESOLVE = None
PREFERENCE = ["triton"]  # measured to honour softcap, final logit cap, window and rope (sweep/RESULTS.md)


def verdict(config, backend):
    return _shared.verdict(config, backend, "sglang")


def _install_resolver():
    """Replace a backend that would drop a declared property at the moment the names are chosen."""
    global _ORIG_RESOLVE
    import msgspec
    from sglang.srt.model_executor import model_runner as mr

    if _ORIG_RESOLVE is not None:
        return
    _ORIG_RESOLVE = mr.resolve_attention_backend_strs

    def resolve(*a, model_runner, **kw):
        resolved = _ORIG_RESOLVE(*a, model_runner=model_runner, **kw)
        if core.mode() not in ("load", "debug") or core.policy() != "resolve":
            return resolved
        config = getattr(model_runner.model_config, "hf_config", None)
        if config is None:
            return resolved
        changes = {}
        for phase in ("prefill", "decode"):
            name = getattr(resolved, phase, None)
            said = verdict(config, name)
            if said is not None and said[0] == "violation":
                alt = _shared.choose(config, "sglang", PREFERENCE)
                if alt is not None:
                    _shared.note_resolution("sglang", f"{phase} attention backend", name, alt, config)
                    changes[phase] = alt
        return msgspec.structs.replace(resolved, **changes) if changes else resolved

    mr.resolve_attention_backend_strs = resolve


def install():
    """Wrap backend choice and ModelRunner.init_attention_backends. Returns 1, or 0 if already installed."""
    global _ORIG
    from sglang.srt.model_executor.model_runner import ModelRunner

    if _ORIG is not None:
        return 0
    _install_resolver()
    _ORIG = ModelRunner.init_attention_backends

    def wrapped(self, *a, **kw):
        out = _ORIG(self, *a, **kw)
        if core.mode() in ("load", "debug"):
            config = getattr(self.model_config, "hf_config", None)
            if os.environ.get("ENTAIL_VERBOSE"):
                built = type(getattr(self, "attn_backend", None)).__name__
                print(f"[entail] sglang built prefill={self.prefill_attention_backend_str} "
                      f"decode={self.decode_attention_backend_str} class={built}", flush=True)
            if config is not None:
                seen = set()
                for phase in ("prefill", "decode"):
                    name = getattr(self, f"{phase}_attention_backend_str", None)
                    if name in seen:
                        continue
                    seen.add(name)
                    _shared.report(verdict(config, name), where=f"sglang {phase}")
        return out

    ModelRunner.init_attention_backends = wrapped
    return 1


def uninstall():
    global _ORIG, _ORIG_RESOLVE
    if _ORIG is None:
        return 0
    from sglang.srt.model_executor import model_runner as mr
    from sglang.srt.model_executor.model_runner import ModelRunner

    ModelRunner.init_attention_backends = _ORIG
    if _ORIG_RESOLVE is not None:
        mr.resolve_attention_backend_strs = _ORIG_RESOLVE
    _ORIG = _ORIG_RESOLVE = None
    return 1
