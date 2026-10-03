"""Write the freeze manifest: python freeze.py <calibration.json> <env.json> <out manifest.json> [--cost-budget F ...]

The manifest fixes, before the holdout: the plans (supported envelope, tolerance constants, budgets) - "plan" for the
guarantee (ENTAIL=guarantee, check "output") and, from v3 (L5.4c), "structure_plan" for the structural check
experiment (ENTAIL=structure, check "static"); it is also the file ENTAIL_GUARANTEE_PLAN reads, each mode its own
plan - the oracle's constant, the environment (its fingerprint), the sha256 of every file of the profile and of the
harness, the entail commit, and the holdout rule. The holdout's seeds and free parameters are drawn from the sha256 of
this file, so they exist only once it is written.
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402

CODE = ("entail/guarantee.py", "entail/kernel_ir.py", "entail/adapters/vllm_block_fp8_guarantee.py",
        "entail/adapters/autoinstall/sitecustomize.py", "entail/core.py", "entail/policies.py", "entail/record.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration")
    ap.add_argument("env")
    ap.add_argument("out")
    ap.add_argument("--cost-budget", type=float, default=None,
                    help="largest accepted ratio of the guarantee's normal repeat time to off's (median, eager)")
    ap.add_argument("--engine-budget", type=float, default=None,
                    help="largest accepted ratio of the structural experiment's steady generation time to off's in the "
                         "engine's CUDA graph configuration (seeded and unseeded runs, engine_smoke.py)")
    ap.add_argument("--engine-gpu-mem", type=float, default=0.55,
                    help="vLLM's gpu_memory_utilization for every engine run (off and the modes alike)")
    ap.add_argument("--engine-configs", default="eager,graphs,default")
    ap.add_argument("--engine-modes", default="guarantee,structure")
    ap.add_argument("--engine-rounds", default="",
                    help="rounds per configuration, e.g. graphs=3 (default 1); a round is off, each mode, off")
    ap.add_argument("--what", default="M19 L5.4d (v4): the guarantee (check output) and, apart, the structural check "
                                      "experiment (check static): implementation, tolerances and harness frozen "
                                      "before the holdout")
    a = ap.parse_args()
    configs = a.engine_configs.split(",")
    rounds = {c: 1 for c in configs}
    for item in filter(None, a.engine_rounds.split(",")):
        k, _, v = item.partition("=")
        rounds[k.strip()] = int(v)
    with open(a.calibration, encoding="utf-8") as f:
        cal = json.load(f)
    with open(a.env, encoding="utf-8") as f:
        env = json.load(f)
    assert cal["decode_all_256_agree"] and cal["exact_case"]["equal"], "the oracle's own checks failed"
    from entail import guarantee as g

    plan = g.Plan(c_acc=cal["proposed"]["c_acc"], check="output", integrity="checksum", records="all").to_json()
    structure_plan = g.Plan(name="vllm-0.30-triton-dense-block-fp8-structure", c_acc=cal["proposed"]["c_acc"],
                            check="static", integrity="epoch", records="changes").to_json()
    code = {p: common.sha256_file(os.path.join(common.ENTAIL_ROOT, p)) for p in CODE}
    for p in sorted(glob.glob(os.path.join(HERE, "*.py"))):
        code[os.path.relpath(p, common.ENTAIL_ROOT).replace(os.sep, "/")] = common.sha256_file(p)
    manifest = {
        "frozen": True,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "what": a.what,
        "plan": plan,
        "structure_plan": structure_plan,
        "oracle": {"c": cal["proposed"]["oracle_c"], "form": "|out - truth| <= ulp(out dtype, |truth|) + c * "
                                                             "(|a| @ |b|^T), float64 on the CPU, bit-field fp8 decode",
                   "calibration_observed_max": cal["oracle_c_observed_max"]},
        "calibration": {"entail_c_acc_observed_max": cal["entail_c_acc_observed_max"],
                        "rule": cal["proposed"]["rule"], "exact_case": cal["exact_case"]},
        "supported": {"dtypes": "float8_e4m3fn values, float32 scales", "block": [128, 128],
                      "out_dtypes": ["bfloat16", "float16"], "K": "a multiple of 128", "A": "contiguous",
                      "scales": "positive, finite; activation scales row- or column-major as the quantizer issued",
                      "nan_inf": "a non-finite reference value or scale refuses the call (eager) or, in a graph, "
                                 "stops the device at the gate (the graph's next operation does not run) and refuses "
                                 "the replay",
                      "graphs": "CUDA graphs captured from eager code (torch.cuda.graph); torch.compile refused",
                      "not": ["UE8M0 scales", "expert parallel", "multi-GPU", "fnuz fp8"]},
        "cases": {"file_sha256": common.sha256_file(os.path.join(HERE, "cases.py")),
                  "normal": 8, "defect": 12, "admission": 4, "integrity": 2,
                  "holdout_rule": "seeds and free parameters = sha256(f'{sha256 of this manifest file}:{case id}')"},
        "cost_budget": {"guarantee_normal_repeat_ratio_max": a.cost_budget,
                        "structure_engine_graphs_steady_ratio_max": a.engine_budget,
                        "engine_rule": "per round, the mode's seeded and unseeded generate times over the mean of the "
                                       "round's two off runs, averaged; the budget reads the median over the rounds",
                        "note": "the conservation verdict and the cost verdict are separate; the check is never "
                                "narrowed to meet this"},
        "engine": {"configs": configs, "modes": a.engine_modes.split(","), "gpu_memory_utilization": a.engine_gpu_mem,
                   "rounds": rounds, "model": "~/models/Qwen3-4B-FP8",
                   "kernels_disabled": "MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel"},
        "guarantee_floor": {k: v for k, v in g.GUARANTEE_FLOOR.items()},
        "before_the_holdout": "freeze_check.py compares every file under code, the entail commit and the "
                              "environment record with this manifest; run_suite.py does not start the holdout on a "
                              "mismatch, cost.py and engine_smoke.py neither (unless marked post-freeze)",
        "env": env,
        "code": code,
        "entail_git": common.git_head(common.ENTAIL_ROOT),
    }
    common.write_json(a.out, manifest)
    print(a.out, common.sha256_file(a.out))


if __name__ == "__main__":
    main()
