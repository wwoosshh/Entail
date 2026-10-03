"""Cost, apart from conservation: python cost.py <out_root> [pairs]   (one measurement:  python cost.py --one ...)

Each measurement is a fresh process (off, guarantee or - from v3 - structure, BFG_COST_MODES), in rounds whose order
rotates (off first, then the next mode first, ...), `pairs` rounds (default 5). In each process, on the real vLLM path (TritonFp8BlockScaledMMKernel,
N=6144, K=2560 and N=2560, K=9728: Qwen3-4B's fused QKV-like and down-projection shapes), it times:
  start         process start -> the kernel object and the weights processed (issued, under the profile)
  first         the first call of each shape (Triton compiles the kernel in both modes)
  repeat        median of 30 calls after 5 warm-ups, at M = 1, 16, 256 (decode-like to prefill-like)
  bypass        the same with a consumer kernel that reads the neighbour block's scale (repaired with the reference
                under the profile: the cost of the reference path; off hands the wrong values on, for the time only)
  (repeat runs vLLM's own kernel, bypass the misreading one; the plan comes from BFG_FREEZE, so the same script
  measures check "output" and check "static")
  memory        peak allocated bytes, and the extra transient of one checked call
Nothing in the check is skipped or sampled to save time.
"""
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
T_START = time.perf_counter()

SHAPES = ((6144, 2560), (2560, 9728))
MS = (1, 16, 256)


def one(mode, out_path):
    import torch

    import common
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    r = {"mode": mode, "shapes": {}}
    with common.vllm_context():
        lau = common.install_launcher(None)     # vLLM's own kernel; the bypass below swaps in a misreading one
        launches = 0
        kernels = {}
        for N, K in SHAPES:
            k, _ = common.make_kernel(N, K, torch.bfloat16)
            w, s = common.make_weight(N, K, N + K, spread=1.0)
            layer = common.Layer(w, s, f"cost{N}x{K}")
            k.process_weights_after_loading(layer)
            kernels[(N, K)] = (k, layer)
        torch.cuda.synchronize()
        r["start_s"] = time.perf_counter() - T_START
        for (N, K), (k, layer) in kernels.items():
            sh = r["shapes"][f"{N}x{K}"] = {}
            x = common.make_input(16, K, 7)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            k.apply_weights(layer, x)
            torch.cuda.synchronize()
            sh["first_ms"] = (time.perf_counter() - t0) * 1e3
            for label in ("repeat", "bypass"):
                launches += lau.launches
                lau = common.install_launcher(None if label == "repeat" else "neighbor")
                lau.flag.fill_(1)
                for M in MS:
                    x = common.make_input(M, K, 100 + M)
                    for _ in range(5):
                        k.apply_weights(layer, x)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    times = []
                    for _ in range(30):
                        t0 = time.perf_counter()
                        k.apply_weights(layer, x)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - t0) * 1e3)
                    sh[f"{label}_M{M}"] = {"median_ms": statistics.median(times), "min_ms": min(times),
                                           "max_ms": max(times),
                                           "transient_bytes": int(torch.cuda.max_memory_allocated() - base)}
            launches += lau.launches
            lau = common.install_launcher(None)
        r["peak_bytes"] = int(torch.cuda.max_memory_allocated())
        r["launches"] = launches + lau.launches
        if mode in ("guarantee", "structure"):
            from entail.adapters import vllm_block_fp8_guarantee as ad
            r["entail"] = ad.stats()
    common.write_json(out_path, r)


def main(out_root, pairs=5):
    import common

    os.makedirs(out_root, exist_ok=True)
    if os.environ.get("BFG_FREEZE"):           # bound to a freeze (v4): this checkout and environment must match it
        import freeze_check

        freeze_check.require(os.environ["BFG_FREEZE"], out_root, "the cost measurement")
    auto = os.path.join(common.ENTAIL_ROOT, "entail", "adapters", "autoinstall")
    modes = ["off"] + os.environ.get("BFG_COST_MODES", "guarantee,structure").split(",")
    order = []
    for i in range(pairs):
        rot = modes[i % len(modes):] + modes[:i % len(modes)]
        order += [(m, i) for m in rot]
    runs = []
    for mode, i in order:
        env = dict(os.environ, ENTAIL=mode, ENTAIL_QUIET="start",
                   PYTHONPATH=os.pathsep.join([common.ENTAIL_ROOT, auto, HERE]),
                   ENTAIL_LOG_DIR=os.path.join(out_root, f"logs_{mode}_{i}"),
                   ENTAIL_GUARANTEE_RECORD="off")
        if os.environ.get("BFG_FREEZE"):
            env["ENTAIL_GUARANTEE_PLAN"] = os.environ["BFG_FREEZE"]
        path = os.path.join(out_root, f"{mode}_{i}.json")
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, "--one", mode, path], env=env, capture_output=True, text=True)
        runs.append({"mode": mode, "pair": i, "exit": p.returncode, "wall_s": time.time() - t0, "path": path,
                     "stderr_tail": p.stderr[-1500:]})
        print(mode, i, p.returncode, round(time.time() - t0, 1), flush=True)
    summary = {"order": [f"{m}{i}" for m, i in order], "runs": runs}
    got = {}
    for run in runs:
        if run["exit"] == 0:
            with open(run["path"], encoding="utf-8") as f:
                got.setdefault(run["mode"], []).append(json.load(f))
    table = {}
    for mode, rs in got.items():
        t = table[mode] = {"start_s": [x["start_s"] for x in rs], "peak_bytes": [x["peak_bytes"] for x in rs]}
        for shape in rs[0]["shapes"]:
            for key in rs[0]["shapes"][shape]:
                vals = [x["shapes"][shape][key] for x in rs]
                if isinstance(vals[0], dict):
                    t[f"{shape}.{key}.median_ms"] = [v["median_ms"] for v in vals]
                    t[f"{shape}.{key}.transient_bytes"] = [v["transient_bytes"] for v in vals]
                else:
                    t[f"{shape}.{key}"] = vals
    summary["table"] = table
    ratios = {}
    for mode in modes[1:]:
        if "off" not in table or mode not in table:
            continue
        rm = ratios[mode] = {}
        for key, gv in table[mode].items():
            ov = table["off"].get(key)
            if key.endswith("median_ms") or key.endswith("first_ms") or key == "start_s":
                if ov:
                    rm[key] = {"off_median": statistics.median(ov), "mode_median": statistics.median(gv),
                               "ratio": statistics.median(gv) / max(statistics.median(ov), 1e-9),
                               "mode_range": [min(gv), max(gv)], "off_range": [min(ov), max(ov)]}
    summary["ratios"] = ratios
    common.write_json(os.path.join(out_root, "cost_summary.json"), summary)
    for mode, rm in ratios.items():
        for k, v in rm.items():
            print(f"{mode:9s} {k:40s} off {v['off_median']:9.3f}  {mode} {v['mode_median']:9.3f}  x{v['ratio']:.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
