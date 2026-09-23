"""entail: keep what a value means intact across LLM inference-stack boundaries.

Most users never import this: `pip install` puts a start-up hook in the environment, and `ENTAIL=load` turns it
on for a process and every process it spawns (see README). `enable()` does the same from inside a script.
"""
import os

# core first: it re-exports boundary and carry from boundaries, which imports core
from .core import (RoleError, boundary, carry, check_config_keys, check_props, check_tied, envelopes_of, facts_of,
                   mode, policy, require, set_mode, set_policy, tag)
from .boundaries import advance  # noqa: E402
from .facts import (LAYOUT_KINDS, Assumed, Epoch, KernelCaps, LatentScale, Layout, ModelProps, Origin, Positions,
                    Prediction, Quantized, Reduction, Rotary, Template, Valid)

__version__ = "0.3.0"


def enable(mode="load", policy="resolve"):
    """Turn entail on in this process, and in any process it starts from now on.

    Sets ENTAIL / ENTAIL_POLICY (children inherit the environment), sets the mode and policy here, and installs the
    adapters: at once for engines already imported, and as each one finishes importing otherwise."""
    os.environ["ENTAIL"] = mode
    os.environ["ENTAIL_POLICY"] = policy
    set_mode(mode)
    set_policy(policy)
    from .adapters.autoinstall import sitecustomize as _hook

    _hook.activate()


__all__ = ["RoleError", "advance", "boundary", "carry", "check_config_keys", "check_props", "check_tied", "enable",
           "envelopes_of", "facts_of", "mode", "policy", "require", "set_mode", "set_policy", "tag", "LAYOUT_KINDS",
           "Assumed", "Epoch", "KernelCaps", "LatentScale", "Layout", "ModelProps", "Origin", "Positions",
           "Prediction", "Quantized", "Reduction", "Rotary", "Template", "Valid", "__version__"]
