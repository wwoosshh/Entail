"""The safety modes' table against the engines installed here (ROADMAP product track P3, its external evaluation: the
table is tied to engine internals, so each version is checked, not trusted). Skips an engine that is not installed.
Run in each engine's environment: python tests/test_safe_mode_engines.py

  vLLM    every option of data/safe_mode.json is an engine argument (EngineArgs, and the configs it holds), and every
          IR op of this vLLM - the ops that choose their kernel apart from custom_ops (S9) - is one the table puts on
          its definition: a new IR op in a later version fails here instead of staying on its kernel in the explicit
          safe mode
  SGLang  every option is a ServerArgs field
"""
import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import safe_mode  # noqa: E402
from entail.adapters import sglang_safe, vllm_safe  # noqa: E402


def _installed(name):
    try:
        __import__(name)
        return True
    except Exception:  # noqa: BLE001 - not installed, or not importable here
        return False


def test_vllm_has_every_option_and_no_ir_op_the_table_misses():
    if not _installed("vllm"):
        print("skip: vllm is not installed here")
        return
    import vllm
    from vllm.engine.arg_utils import EngineArgs

    args = EngineArgs(model="/nonexistent/model")          # built, not run: nothing is loaded
    have, lack = vllm_safe.read_options(args)
    assert not lack, f"vLLM {vllm.__version__} lacks {lack}: data/safe_mode.json needs this version's options"
    assert set(have) == set(safe_mode.features("vllm")), have
    ir = [o.split(".", 1)[1] for o, _ in safe_mode.features("vllm")["custom_kernels"] if o.startswith("ir_op_priority.")]
    from vllm.config.kernel import IrOpPriorityConfig
    from vllm.ir.op import IrOp
    import vllm.kernels.vllm_c  # noqa: F401 - registers the vllm_c kernels

    fields = {f.name for f in dataclasses.fields(IrOpPriorityConfig)}
    assert fields == set(ir), (f"vLLM {vllm.__version__} chooses the kernels of {sorted(fields)} by IR op priority; "
                               f"the table puts {sorted(ir)} on their definition")
    for op in ir:
        assert "native" in IrOp.registry[op].impls, op   # the definition the safe mode sends it to
    # the safe mode's handle on real engine arguments: the definition first, the engine's own after it
    for feature, options in have.items():
        vllm_safe.handles(args)["safe_mode"](options)
    assert args.enforce_eager is True and args.enable_prefix_caching is False and args.speculative_config is None
    assert [getattr(args.ir_op_priority, op)[0] for op in ir] == ["native"] * len(ir)
    assert list(args.compilation_config.custom_ops) == ["none"], args.compilation_config.custom_ops
    assert not any(vllm_safe.read_choice(args)[1].values()), vllm_safe.read_choice(args)


def test_sglang_has_every_option():
    if not _installed("sglang"):
        print("skip: sglang is not installed here")
        return
    import sglang
    from sglang.srt.server_args import ServerArgs

    if dataclasses.is_dataclass(ServerArgs):
        fields = {f.name for f in dataclasses.fields(ServerArgs)}
    else:                                                   # 0.5.20: a msgspec Struct
        fields = set(getattr(ServerArgs, "__struct_fields__", ()))
    assert fields, "no fields found on ServerArgs"
    stand_in = type("Args", (), {name: None for name in fields})()
    have, lack = sglang_safe.read_options(stand_in)
    assert not lack, f"SGLang {sglang.__version__} lacks {lack}: data/safe_mode.json needs this version's options"
    assert set(have) == set(safe_mode.features("sglang")), have


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
