"""One scenario in one mode, in this (fresh) process: python run_case.py <case.json> <mode> <out_dir>

mode: off | load | guarantee. The suite (run_suite.py) sets ENTAIL to match and puts entail's start-up hook on the
path, so entail installs itself the way it does for a user (autoinstall), before this script touches vLLM. The
harness then puts the case's consumer defect in (below entail's wrapper: the kernel object the launcher calls, or
above it: the call site's scale cache) and runs the steps through vLLM's own kernel object.

Written to out_dir: result.json (per step: what was delivered, what the next operation read, the oracle's verdict on
it, the consumer kernel's launches, entail's decisions, RNG digests, timings) and tensors.npz (inputs as the
producers made them, the weights and scales, every observed output, the oracle's truth for each).
"""
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(case_path, mode, out_dir):
    t_start = time.perf_counter()
    with open(case_path, encoding="utf-8") as f:
        case = json.load(f)
    freeze = {}
    if os.environ.get("BFG_FREEZE"):
        with open(os.environ["BFG_FREEZE"], encoding="utf-8") as f:
            freeze = json.load(f)
    c_oracle = float(freeze.get("oracle", {}).get("c", 0) or 0)
    import torch

    import common
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    os.makedirs(out_dir, exist_ok=True)
    res = {"case": case, "mode": mode, "pid": os.getpid(), "env_ENTAIL": os.environ.get("ENTAIL"), "steps": [],
           "freeze_sha256": os.environ.get("BFG_FREEZE_SHA256")}
    g = ad = None
    if mode == "guarantee":
        from entail import guarantee as g
        from entail.adapters import vllm_block_fp8_guarantee as ad
        if case.get("plan"):
            import dataclasses
            g.set_plan(dataclasses.replace(g.plan(), **case["plan"]))
        res["plan"] = g.plan().to_json()
        res["plan_fp"] = g.plan().fingerprint()
    fr = None
    if mode == "load":
        from entail.adapters import function_reference as fr
    out_dtype = getattr(torch, case["out"])
    tensors = {}

    with common.vllm_context():
        # the case's consumer defect, put in after entail installed itself
        lau = common.install_launcher(case.get("mutant") if case.get("mutant") in common.MUTANTS else None)
        lau.rows_from = int(case.get("rows_from", 64))
        cache = common.ScaleCache().install() if case.get("mechanism", "").startswith("consumer-side") else None
        if case.get("tile_k256"):
            res["tile_table"] = common.tile_config_case()
        res["mutant_source_sha256"] = getattr(lau, "source_sha256", None)
        # the producers' outputs, as the next operation after them sees them (observer, outermost)
        produced = []
        q_inner = fp8_utils.per_token_group_quant_fp8

        def observe_quant(*a, **k):
            out = q_inner(*a, **k)
            produced.append(out)
            return out

        fp8_utils.per_token_group_quant_fp8 = observe_quant
        if mode == "guarantee" and case.get("skip_hook") == "weights":
            from vllm.model_executor.kernels.linear.scaled_mm import BlockScaledMMLinearKernel as bk
            orig = ad._WRAPPED.get((bk.Fp8BlockScaledMMLinearKernel, "process_weights_after_loading"))
            if orig is not None:
                bk.Fp8BlockScaledMMLinearKernel.process_weights_after_loading = orig
            g.uninstalled(g.WEIGHT_PRODUCER)
            res["skipped_hook"] = g.WEIGHT_PRODUCER

        N, K = case["N"], case["K"]
        kernel, unforced = common.make_kernel(N, K, out_dtype)
        res["kernel"] = {"used": type(kernel).__name__, "vllm_would_choose": unforced}
        layers, wraw = [], []
        for i in range(case["layers"]):
            w, s = common.make_weight(N, K, case["wseed"][i])
            layer = common.Layer(w, s, f"layer{i}")
            kernel.process_weights_after_loading(layer)
            layers.append(layer)
            wraw.append((common.to_np(layer.weight), common.to_np(layer.weight_scale_inv)))
        if case["layers"] > 1:
            lau.alt = layers[1].weight_scale_inv          # the other weight's scale, for alt_weight in the kernel
        W2 = torch.randn((N, 64), generator=torch.Generator().manual_seed(5), dtype=torch.float32).cuda()
        graph = None

        def record_step(step, y_obs, delivered, exc, A, As, layer_i, extra):
            entry = dict(step)
            entry.update(extra)
            entry["delivered"] = delivered
            entry["exception"] = exc
            if A is not None:
                A_u8, As_np = common.to_np(A), common.to_np(As).astype(np.float64)
                B_u8, Bs_np = wraw[layer_i]
                truth, bound = common.oracle(A_u8, As_np, B_u8, Bs_np.astype(np.float64))
                k = f"s{len(res['steps'])}"
                tensors[f"{k}_A"], tensors[f"{k}_As"] = A_u8, As_np
                tensors[f"{k}_truth"] = truth
                entry["truth_key"] = k
                if y_obs is not None:
                    arr = common.out_f64(y_obs, case["out"]).reshape(truth.shape)
                    tensors[f"{k}_out"] = y_obs
                    entry["poisoned"] = bool(np.isnan(arr).all())
                    ok, viol, ratio, nonfin = common.judge(arr, truth, bound, case["out"],
                                                          c_oracle or common.C_ORACLE_DEFAULT)
                    # the calibration's own numbers: |out - truth| beyond one ulp, over the bound
                    excess = np.maximum(np.abs(arr - truth) - common.ulp_out(truth, case["out"]), 0.0)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cal = np.where(bound > 0, excess / bound, np.where(excess > 0, np.inf, 0.0))
                    entry["oracle"] = {"ok": ok, "violations": viol, "elements": int(truth.size),
                                       "worst_ratio": ratio, "nonfinite": nonfin,
                                       "c_used": c_oracle or common.C_ORACLE_DEFAULT,
                                       "calibration_max": float(np.nanmax(cal)) if cal.size else 0.0}
            res["steps"].append(entry)

        def stats():
            if mode == "guarantee":
                return ad.stats()
            if mode == "load":
                return fr.stats()
            return {}

        def delta(a, b):
            return {k: b[k] - a.get(k, 0) for k in b if isinstance(b[k], (int, float)) and b[k] != a.get(k, 0)}

        for step in case["steps"]:
            op = step["op"]
            li = step.get("layer", 0)
            lau.flag.fill_(int(step.get("mut", 0)))
            if cache is not None:
                cache.armed = bool(step.get("cache", 0))
                if cache.armed:
                    cache.store.setdefault(tuple(layers[li].weight_scale_inv.shape), layers[li].weight_scale_inv)
            torch.cuda.synchronize()
            rng0 = common.rng_state()
            st0, l0, h0 = stats(), lau.launches, (cache.hits if cache else 0)
            produced.clear()
            y = A = As = None
            delivered, exc = False, None
            t0 = time.perf_counter()
            if op == "reload":
                w, s = common.make_weight(N, K, step["seed"])
                layers[li] = common.Layer(w, s, f"layer{li}-reloaded")
                kernel.process_weights_after_loading(layers[li])
                wraw[li] = (common.to_np(layers[li].weight), common.to_np(layers[li].weight_scale_inv))
                record_step(step, None, None, None, None, None, li, {"rng_same": rng0 == common.rng_state()})
                continue
            x = common.make_input(step["M"], K, step["seed"], step.get("dist", "normal"), dtype=torch.bfloat16
                                  if case["out"] != "float32" else torch.float32)
            try:
                if op == "call":
                    y = kernel.apply_weights(layers[li], x)
                    A, As = produced[-1]
                elif op == "direct":
                    if step["quant"] == "native":
                        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
                        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
                        A, As = QuantFP8(False, GroupShape(1, 128)).forward_native(x)
                    else:
                        A, As = fp8_utils.per_token_group_quant_fp8(x, 128, column_major_scales=True)
                    step = dict(step, scale_stride=list(As.stride()))
                    y = fp8_utils.w8a8_triton_block_scaled_mm(A, layers[li].weight, As, layers[li].weight_scale_inv,
                                                              [128, 128], out_dtype)
                elif op == "capture":
                    graph = {"x": x.clone(), "layer": li}
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        for _ in range(2):
                            kernel.apply_weights(layers[li], graph["x"])
                    torch.cuda.current_stream().wait_stream(s)
                    produced.clear()
                    graph["g"] = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph["g"]):
                        yy = kernel.apply_weights(layers[li], graph["x"])
                        graph["obs"] = torch.empty_like(yy)
                        graph["obs"].copy_(yy)                     # the graph's next operation reads yy here
                        graph["z"] = graph["obs"].float() @ W2
                    graph["A"], graph["As"] = produced[-1]
                    torch.cuda.synchronize()
                    y = None
                elif op == "replay":
                    li = graph["layer"]
                    graph["x"].copy_(x)
                    try:
                        graph["g"].replay()
                    finally:
                        torch.cuda.synchronize()
                    y = graph["obs"]
                    A, As = graph["A"], graph["As"]
                delivered = op != "capture"
                if y is not None:
                    _z = y.float() @ W2                            # the next operation consumes the value
            except Exception as e:  # noqa: BLE001 - refusal, or an execution error: both are observations
                exc = {"type": type(e).__name__, "kind": getattr(e, "kind", None), "message": str(e)[:2000],
                       "trace": traceback.format_exc()[-3000:]}
                if op == "replay":
                    y, A, As = graph["obs"], graph["A"], graph["As"]     # what the graph's next operation read
                    delivered = False
                elif op in ("call", "direct") and produced:
                    A, As = produced[-1]
                    y = None
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            st1 = stats()
            extra = {"ms": round(ms, 3), "rng_same": rng0 == common.rng_state(),
                     "kernel_launches": lau.launches - l0, "cache_hits": (cache.hits if cache else 0) - h0,
                     "entail": delta(st0, st1)}
            y_obs = common.to_np(y) if y is not None else None
            record_step(step, y_obs, delivered, exc, A, As, li, extra)
            if exc and op == "capture":
                break
        res["entail_final"] = stats()
        res["launcher_total"] = lau.launches
    res["seconds"] = round(time.perf_counter() - t_start, 3)
    res["peak_mem"] = int(torch.cuda.max_memory_allocated())
    np.savez_compressed(os.path.join(out_dir, "tensors.npz"), **tensors)
    common.write_json(os.path.join(out_dir, "result.json"), res)
    print(json.dumps({"case": case["id"], "mode": mode, "steps": len(res["steps"]),
                      "delivered": [s.get("delivered") for s in res["steps"]],
                      "oracle_ok": [s.get("oracle", {}).get("ok") for s in res["steps"]]}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
