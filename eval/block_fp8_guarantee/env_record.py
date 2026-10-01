"""The environment record before the freeze: python env_record.py <out.json>

GPU, driver, Python, torch, CUDA, Triton, vLLM (with the sha256 of each installed distribution's RECORD), precision
settings, the entail commit, and which block FP8 kernel vLLM chooses on this GPU for the scenarios' shape - unforced,
and with Marlin and Humming turned off (VLLM_DISABLED_KERNELS, the engine runs' setting).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROBE = r"""
import sys, json
sys.path.insert(0, %r)
import torch, common
with common.vllm_context():
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key
    k = init_fp8_linear_kernel(activation_quant_key=create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128)),
                               weight_quant_key=create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128)),
                               input_dtype=torch.bfloat16, out_dtype=torch.bfloat16, weight_shape=(1536, 2560))
    print("CHOSEN", type(k).__name__)
"""


def main(out):
    import common

    env = common.environment()
    choice = {}
    for label, disabled in (("unforced", None),
                            ("marlin_humming_disabled", "MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel")):
        e = dict(os.environ)
        e.pop("VLLM_DISABLED_KERNELS", None)
        if disabled:
            e["VLLM_DISABLED_KERNELS"] = disabled
        p = subprocess.run([sys.executable, "-c", PROBE % HERE], env=e, capture_output=True, text=True)
        got = [ln.split(" ", 1)[1] for ln in p.stdout.splitlines() if ln.startswith("CHOSEN ")]
        choice[label] = got[0] if got else f"error: {p.stderr[-800:]}"
    env["block_fp8_kernel_choice"] = choice
    common.write_json(out, env)
    print(choice)


if __name__ == "__main__":
    main(sys.argv[1])
