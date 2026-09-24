"""Adapter v2 for SGLang's attention backends (LIBRARY_DESIGN.md 4.8; ROADMAP M3.3).

SGLang picks the backends in the scheduler process, not in the process that constructs Engine(...), and the name can
come from the device rather than from a server argument (flashinfer is the default on this card). So the hooks run in
the child process: install through the start-up hook (autoinstall/sitecustomize.py).

  hooks        resolve_attention_backend_strs (model_runner.py:1027 calls it; attention_backend_setup.py:178 defines
               it): the moment the (prefill, decode) names are chosen; build_attention_backends reads them back off
               the runner, so a change here is what gets built.
               ModelRunner.init_attention_backends: after the names are stamped. It decides only if the first hook
               did not run for this runner (an SGLang that chose the names another way); with ENTAIL_VERBOSE it
               prints the class that was really built, so a resolution can be seen to have taken effect.
  read_choice  the (prefill, decode) names, the model's config object and its path.
  handle       switch_attention_backend: the same resolved names with one phase replaced.
The decisions are load.attention's; load.enforce records them and stops on a blocking one.
"""
import os

from .. import core, load, policies
from .base import Hook

engine = "sglang"
versions = "0.5.20"
_ORIG = None
_ORIG_RESOLVE = None
_ORIG_LOAD = None
_DECIDED = "_entail_decided"


# SGLang 0.5.20 ties when the config says so and skips a shipped lm_head.weight (model_loader/loader.py), so it
# never compares the two (load.tie, M11.2)
compares_head = False


def hooks():
    return [Hook("sglang.srt.model_executor.model_runner.resolve_attention_backend_strs", "load"),
            Hook("sglang.srt.model_executor.model_runner.ModelRunner.init_attention_backends", "load"),
            Hook("sglang.srt.model_executor.model_runner.ModelRunner.load_model", "load")]


def read_choice(model_runner, resolved=None):
    """{phase: backend name}, the config object and the model path for one runner."""
    names = {p: getattr(resolved, p, None) if resolved is not None else
             getattr(model_runner, f"{p}_attention_backend_str", None) for p in ("prefill", "decode")}
    mc = getattr(model_runner, "model_config", None)
    return names, getattr(mc, "hf_config", None), getattr(mc, "model_path", None)


def handles(resolved, phase):
    import msgspec

    return {"switch_attention_backend": lambda target: msgspec.structs.replace(resolved, **{phase: target})}


def _decide(model_runner, resolved=None):
    """load.attention for each distinct phase backend; returns the (possibly switched) resolved names."""
    names, config, path = read_choice(model_runner, resolved)
    if config is None:
        return resolved
    facts = load.declared(path, config)
    seen = {}
    for phase, name in names.items():
        if not isinstance(name, str):
            continue
        if name in seen:    # prefill and decode on one backend: one decision, and the same repair
            if seen[name] is not None and resolved is not None:
                resolved = handles(resolved, phase)["switch_attention_backend"](seen[name])
            continue
        # after the build (resolved is None) a backend can no longer be switched, so nothing is offered as a repair
        decisions = load.attention(engine, name, facts, policy=policies.current(), can_switch=resolved is not None)
        done = load.resolve(decisions, handles(resolved, phase)) if resolved is not None else {}
        load.enforce(decisions)
        seen[name] = None
        if "switch_attention_backend" in done:
            resolved = done["switch_attention_backend"]
            seen[name] = getattr(resolved, phase)
    try:
        setattr(model_runner, _DECIDED, True)
    except Exception:  # noqa: BLE001 - a runner that takes no attribute is decided again later, never skipped
        pass
    return resolved


def _install_resolver():
    global _ORIG_RESOLVE
    from sglang.srt.model_executor import model_runner as mr

    if _ORIG_RESOLVE is not None:
        return
    _ORIG_RESOLVE = mr.resolve_attention_backend_strs

    def resolve(*a, model_runner, **kw):
        resolved = _ORIG_RESOLVE(*a, model_runner=model_runner, **kw)
        if core.mode() not in ("load", "debug"):
            return resolved
        return load.safely(f"load:{engine}.attention", f"{engine}.attention", "ModelProps",
                           lambda: _decide(model_runner, resolved), resolved)

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
            if os.environ.get("ENTAIL_VERBOSE"):
                built = type(getattr(self, "attn_backend", None)).__name__
                print(f"[entail] sglang built prefill={self.prefill_attention_backend_str} "
                      f"decode={self.decode_attention_backend_str} class={built}", flush=True)
            if not getattr(self, _DECIDED, False):
                load.safely(f"load:{engine}.attention", f"{engine}.attention", "ModelProps", lambda: _decide(self))
        return out

    ModelRunner.init_attention_backends = wrapped
    _install_load(ModelRunner)
    return 1


def _install_load(ModelRunner):
    """After the weights are loaded: the contracts about the model itself (load.model_contracts)."""
    global _ORIG_LOAD
    _ORIG_LOAD = ModelRunner.load_model

    def load_model(self, *a, **kw):
        out = _ORIG_LOAD(self, *a, **kw)
        if core.mode() in ("load", "debug"):
            _, config, path = read_choice(self)
            tie = getattr(config, "tie_word_embeddings", None)

            def decide():
                load.enforce(load.model_contracts(engine, path, config, tie if isinstance(tie, bool) else None,
                                                  policies.current(), compares_head=compares_head))

            if config is not None:
                load.safely(f"load:{engine}.loader", f"{engine}.loader", "ModelProps", decide)
        return out

    ModelRunner.load_model = load_model


def uninstall():
    global _ORIG, _ORIG_RESOLVE, _ORIG_LOAD
    if _ORIG is None:
        return 0
    from sglang.srt.model_executor import model_runner as mr
    from sglang.srt.model_executor.model_runner import ModelRunner

    ModelRunner.init_attention_backends = _ORIG
    if _ORIG_LOAD is not None:
        ModelRunner.load_model = _ORIG_LOAD
    if _ORIG_RESOLVE is not None:
        mr.resolve_attention_backend_strs = _ORIG_RESOLVE
    _ORIG = _ORIG_RESOLVE = _ORIG_LOAD = None
    return 1
