"""Normal calibration and the oracle's own checks (before the freeze): python calibrate.py <out.json>

  1. fp8 decoding: the oracle's bit-field decoding of all 256 codes against torch's conversion (NaN codes apart).
  2. exact case: power-of-two values and scales, where every product and sum is exact in float32; the kernel's
     output must equal the oracle's truth rounded to the output dtype.
  3. normal calibration (seeds of its own, none of the scenarios'): vLLM's unmodified kernel on normal inputs of
     several sizes and distributions; for each value, |kernel - truth| beyond one ulp over |a| @ |b|^T (the
     oracle's constant) and |kernel - entail's reference| beyond one ulp over sum_blk |a_blk| |b_blk| (the
     profile's constant). The largest of each, and the constants proposed from them (x4, up to a power of two).
  4. the mutants' distance: each consumer misread switched on, on the same calibration inputs: the share of values
     beyond each proposed tolerance (a misread that the tolerance would admit is not an effective defect).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(out_path):
    import torch

    import common
    from entail import guarantee as g
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    res = {}
    # 1. decoding
    codes = np.arange(256, dtype=np.uint8)
    mine = common.fp8_e4m3fn(codes)
    theirs = torch.from_numpy(codes).view(torch.float8_e4m3fn).to(torch.float32).numpy().astype(np.float64)
    same = (mine == theirs) | (np.isnan(mine) & np.isnan(theirs))
    res["decode_all_256_agree"] = bool(same.all())
    res["decode_nan_codes"] = [int(c) for c in codes[np.isnan(mine)]]

    with common.vllm_context():
        lau = common.install_launcher(None)
        # 2. exact case
        K, N, M = 512, 256, 32
        gen = torch.Generator().manual_seed(3)
        pick = torch.tensor([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        A = pick[torch.randint(0, 6, (M, K), generator=gen)].to(torch.float8_e4m3fn).cuda()
        B = pick[torch.randint(0, 6, (N, K), generator=gen)].to(torch.float8_e4m3fn).cuda()
        As = (2.0 ** torch.randint(-3, 3, (M, K // 128), generator=gen).float()).cuda()
        Bs = (2.0 ** torch.randint(-3, 3, (N // 128, K // 128), generator=gen).float()).cuda()
        y = fp8_utils.w8a8_triton_block_scaled_mm(A, B, As, Bs, [128, 128], torch.bfloat16)
        t, _b = common.oracle(common.to_np(A), common.to_np(As), common.to_np(B), common.to_np(Bs))
        t_bf16 = torch.from_numpy(t).to(torch.bfloat16).float().numpy()
        out = common.out_f64(common.to_np(y), "bfloat16")
        res["exact_case"] = {"equal": bool(np.array_equal(out, t_bf16.astype(np.float64))),
                             "max_abs_diff": float(np.abs(out - t_bf16).max()), "elements": int(t.size)}

        # 3. normal calibration
        kernel, unforced = common.make_kernel(1536, 2560, torch.bfloat16)
        res["kernel"] = {"used": type(kernel).__name__, "vllm_would_choose": unforced}
        layers = []
        for ws in (101, 102):
            w, s = common.make_weight(1536, 2560, ws)
            layer = common.Layer(w, s, f"cal{ws}")
            kernel.process_weights_after_loading(layer)
            layers.append(layer)
        produced = []
        q = fp8_utils.per_token_group_quant_fp8

        def observe(*a, **k):
            o = q(*a, **k)
            produced.append(o)
            return o

        fp8_utils.per_token_group_quant_fp8 = observe
        runs = []
        ora_max = ent_max = 0.0
        samples = []
        for li, layer in enumerate(layers):
            for M in (1, 7, 16, 64, 100, 256, 512):
                for dist in ("normal", "heavy", "small"):
                    seed = 1000 * li + 10 * M + {"normal": 1, "heavy": 2, "small": 3}[dist]
                    x = common.make_input(M, 2560, seed, dist)
                    produced.clear()
                    y = kernel.apply_weights(layer, x)
                    A, As = produced[-1]
                    t, bound = common.oracle(common.to_np(A), common.to_np(As).astype(np.float64),
                                             common.to_np(layer.weight), common.to_np(layer.weight_scale_inv))
                    out = common.out_f64(common.to_np(y), "bfloat16").reshape(t.shape)
                    ex = np.maximum(np.abs(out - t) - common.ulp_out(t, "bfloat16"), 0)
                    o = float(np.max(np.where(bound > 0, ex / np.maximum(bound, 1e-300), 0)))
                    r, S = g.reference(A, layer.weight, As, layer.weight_scale_inv, (128, 128))
                    rf = r.cpu().numpy().astype(np.float64)
                    Sf = S.cpu().numpy().astype(np.float64)
                    ulp_r = g.ulp(r, torch.bfloat16).cpu().numpy().astype(np.float64)
                    ex2 = np.maximum(np.abs(out - rf) - ulp_r, 0)
                    e = float(np.max(np.where(Sf > 0, ex2 / np.maximum(Sf, 1e-300), 0)))
                    ref_vs_truth = float(np.max(np.abs(rf - t) / np.maximum(bound, 1e-300)))
                    runs.append({"layer": li, "M": M, "dist": dist, "oracle_excess_over_bound": o,
                                 "entail_excess_over_S": e, "reference_vs_truth_over_bound": ref_vs_truth})
                    ora_max, ent_max = max(ora_max, o), max(ent_max, e)
                    samples.append((layer, x))
        res["normal_runs"] = runs

        def up(v):
            return 2.0 ** math.ceil(math.log2(max(v * 4, 2.0 ** -40)))

        res["oracle_c_observed_max"] = ora_max
        res["entail_c_acc_observed_max"] = ent_max
        res["proposed"] = {"oracle_c": up(ora_max), "c_acc": up(ent_max), "rule": "4 x observed max, up to a power "
                                                                                  "of two"}
        # 4. the mutants' distance on the same inputs
        dist_rows = []
        for kind in common.MUTANTS:
            ml = common.install_launcher(kind)
            ml.alt = layers[1].weight_scale_inv
            ml.rows_from = 64
            ml.flag.fill_(1)
            for (layer, x) in samples[::5]:
                produced.clear()
                y = kernel.apply_weights(layer, x)
                A, As = produced[-1]
                t, bound = common.oracle(common.to_np(A), common.to_np(As).astype(np.float64),
                                         common.to_np(layer.weight), common.to_np(layer.weight_scale_inv))
                out = common.out_f64(common.to_np(y), "bfloat16").reshape(t.shape)
                ok, viol, ratio, _ = common.judge(out, t, bound, "bfloat16", res["proposed"]["oracle_c"])
                dist_rows.append({"mutant": kind, "M": int(x.shape[0]), "layer_is_alt": layer is layers[1],
                                  "beyond_oracle": viol, "elements": int(t.size), "worst_ratio": ratio})
            ml.flag.fill_(0)
        common.install_launcher(None)
        res["mutant_distance"] = dist_rows
        res["launches"] = lau.launches
    common.write_json(out_path, res)
    print(json.dumps({k: res[k] for k in ("decode_all_256_agree", "exact_case", "oracle_c_observed_max",
                                         "entail_c_acc_observed_max", "proposed")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
