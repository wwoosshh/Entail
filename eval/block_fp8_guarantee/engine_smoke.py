"""The real engine: one FP8 model in vLLM 0.30, off against the guarantee profile and (from v3, L5.4c) against the
structural check experiment (BFG_ENGINE_MODES, default "guarantee,structure"), each in a fresh process.
  python engine_smoke.py <out_root> [model]          (one run:  python engine_smoke.py --one <mode> <config> <out>)

Configurations (BFG_ENGINE_CONFIGS, default "eager,default"): eager (enforce_eager), graphs (no torch.compile,
CUDA graphs of decode steps captured from eager code: compilation mode 0, FULL_DECODE_ONLY) and default
(torch.compile + CUDA graphs: the profile does not support it; the run records what happens - a refusal is the
expected outcome, not a failure of the smoke). VLLM_DISABLED_KERNELS turns Marlin and Humming off so the Triton block kernel is chosen on this GPU (the
choice is read from vLLM's log). The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so the
profile's counts and the RNG states are readable here.

Recorded: the kernel vLLM selected; the producers' issues per layer (every block FP8 linear must have its weight
issued and every consumer call its activation); the profile's outcomes (normal, repaired, blocked); greedy outputs
twice (cold and warm prefix cache), seeded sampling, unseeded sampling (recorded, not compared), token ids and top-5
log-probabilities; the Python, NumPy, torch and CUDA RNG digests before and after each generate; time and memory.

Bound to a freeze (BFG_FREEZE, from v4 / L5.4d): the run starts only when this checkout and environment match the
manifest (freeze_check.py), and its settings - configurations, modes, gpu_memory_utilization, rounds per
configuration - are the manifest's "engine" entry, not the environment's. BFG_POST_FREEZE=1 runs anyway with the
environment's settings, as a measurement after the freeze. A round is off, each mode, off; a mode's timing ratio in a
round is to that round's two off runs, and the frozen budget rule reads the median over rounds.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROMPTS = ["The capital of France is", "def fibonacci(n):", "In 1905, Albert Einstein published",
           "List three prime numbers greater than 100:", "Water boils at a temperature of",
           "The quick brown fox jumps over the lazy dog. Summarize this sentence in five words:"]
DISABLED = "MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel"


def one(mode, config, out_path, model):
    import random

    import numpy as np
    import torch

    import common

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    from vllm import LLM, SamplingParams

    r = {"mode": mode, "config": config, "model": model, "round": int(os.environ.get("BFG_ENGINE_ROUND", "0")),
         "gpu_memory_utilization": float(os.environ.get("BFG_ENGINE_GPU_MEM", "0.55"))}
    t0 = time.perf_counter()
    # BFG_ENGINE_GPU_MEM (default 0.55, the frozen setting): added after the v3 freeze, when the guarantee's graph
    # configuration could not start at 0.55 (its in-graph check left no memory for the KV cache); a run with another
    # value is a supplementary measurement, reported apart from the frozen one
    kw = dict(model=model, gpu_memory_utilization=float(os.environ.get("BFG_ENGINE_GPU_MEM", "0.55")),
              max_model_len=2048, seed=0, enable_prefix_caching=True)
    if config == "eager":
        kw["enforce_eager"] = True
    elif config == "graphs":            # no torch.compile, CUDA graphs of whole decode steps captured from eager code
        kw["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
    try:
        llm = LLM(**kw)
    except Exception as e:  # noqa: BLE001
        r["start_error"] = {"type": type(e).__name__, "kind": getattr(e, "kind", None), "message": str(e)[:3000]}
        r["start_s"] = time.perf_counter() - t0
        common.write_json(out_path, r)
        return
    r["start_s"] = time.perf_counter() - t0
    runs = []

    def gen(label, sp):
        before = common.rng_state()
        t = time.perf_counter()
        try:
            outs = llm.generate(PROMPTS, sp, use_tqdm=False)
            res = [{"ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text,
                    "logprobs": [{str(k): v.logprob for k, v in (lp or {}).items()}
                                 for lp in (o.outputs[0].logprobs or [])]} for o in outs]
            err = None
        except Exception as e:  # noqa: BLE001
            res, err = None, {"type": type(e).__name__, "kind": getattr(e, "kind", None), "message": str(e)[:3000]}
        runs.append({"label": label, "s": time.perf_counter() - t, "rng_before": before,
                     "rng_after": common.rng_state(), "outputs": res, "error": err})

    gen("greedy_cold", SamplingParams(temperature=0.0, max_tokens=24, logprobs=5))
    gen("greedy_warm", SamplingParams(temperature=0.0, max_tokens=24, logprobs=5))
    gen("seeded", SamplingParams(temperature=0.8, top_p=0.95, max_tokens=24, seed=1234, logprobs=5))
    gen("unseeded", SamplingParams(temperature=0.8, top_p=0.95, max_tokens=24, logprobs=5))
    r["runs"] = runs
    r["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    if mode != "off":
        from entail.adapters import vllm_block_fp8_guarantee as ad
        r["entail"] = ad.stats()
    common.write_json(out_path, r)


def reach(record_path):
    """From the profile's records: weight issues per layer, consumer calls and their outcomes."""
    out = {"lines": 0, "outcomes": {}, "blocked_kinds": {}, "layers_consumed": set(), "weights_issued": set()}
    if not os.path.exists(record_path):
        return {"missing": record_path}
    with open(record_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out["lines"] += 1
            o = d.get("outcome")
            out["outcomes"][o] = out["outcomes"].get(o, 0) + 1
            if o == "blocked":
                out["blocked_kinds"][d.get("blocked_kind")] = out["blocked_kinds"].get(d.get("blocked_kind"), 0) + 1
            b = (d.get("issues") or {}).get("B")
            if b:
                out["layers_consumed"].add(b["serial"])
    out["layers_consumed"] = len(out["layers_consumed"])
    out["weights_issued"] = None
    return out


def settings():
    """(configs, modes, gpu_memory_utilization, rounds per configuration, where they came from)."""
    fz = os.environ.get("BFG_FREEZE")
    if fz and os.environ.get("BFG_POST_FREEZE") != "1":
        with open(fz, encoding="utf-8") as f:
            eng = json.load(f).get("engine")
        if eng:
            return (eng["configs"], eng["modes"], float(eng["gpu_memory_utilization"]),
                    {k: int(v) for k, v in eng["rounds"].items()}, "freeze manifest")
    configs = os.environ.get("BFG_ENGINE_CONFIGS", "eager,default").split(",")
    modes = os.environ.get("BFG_ENGINE_MODES", "guarantee,structure").split(",")
    mem = float(os.environ.get("BFG_ENGINE_GPU_MEM", "0.55"))
    rounds = {}
    for item in filter(None, os.environ.get("BFG_ENGINE_ROUNDS", "").split(",")):
        k, _, v = item.partition("=")
        rounds[k.strip()] = int(v)
    return configs, modes, mem, rounds, "environment"


def main(out_root, model):
    import common

    os.makedirs(out_root, exist_ok=True)
    if os.environ.get("BFG_FREEZE"):
        import freeze_check

        freeze_check.require(os.environ["BFG_FREEZE"], out_root, "the engine run")
    configs, modes, mem, rounds, source = settings()
    common.write_json(os.path.join(out_root, "settings.json"), {
        "configs": configs, "modes": modes, "gpu_memory_utilization": mem, "rounds": rounds, "source": source,
        "post_freeze": os.environ.get("BFG_POST_FREEZE") == "1"})
    auto = os.path.join(common.ENTAIL_ROOT, "entail", "adapters", "autoinstall")
    plan = []
    for config in configs:
        for rnd in range(rounds.get(config, 1)):
            for mode in ["off"] + modes + ["off"]:
                plan.append((config, mode, rnd))
    results = []
    for i, (config, mode, rnd) in enumerate(plan):
        d = os.path.join(out_root, f"{i:02d}_{config}_{mode}")
        os.makedirs(d, exist_ok=True)
        env = dict(os.environ, ENTAIL=mode, ENTAIL_QUIET="start", VLLM_ENABLE_V1_MULTIPROCESSING="0",
                   VLLM_DISABLED_KERNELS=DISABLED, PYTHONPATH=os.pathsep.join([common.ENTAIL_ROOT, auto, HERE]),
                   ENTAIL_LOG_DIR=os.path.join(d, "entail_logs"),
                   ENTAIL_GUARANTEE_RECORD=os.path.join(d, f"{mode}.jsonl"),
                   BFG_ENGINE_ROUND=str(rnd), BFG_ENGINE_GPU_MEM=str(mem))
        if os.environ.get("BFG_FREEZE"):
            env["ENTAIL_GUARANTEE_PLAN"] = os.environ["BFG_FREEZE"]
        t0 = time.time()
        p = subprocess.run([sys.executable, __file__, "--one", mode, config, os.path.join(d, "result.json"), model],
                           env=env, capture_output=True, text=True, timeout=7200)
        for name, text in (("stdout.txt", p.stdout), ("stderr.txt", p.stderr)):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(text)
        selected = sorted({ln.split("Selected ", 1)[1].split(" for ")[0] for ln in (p.stdout + p.stderr).splitlines()
                           if "Selected " in ln and " for " in ln})
        row = {"dir": d, "config": config, "mode": mode, "round": rnd, "exit": p.returncode,
               "wall_s": time.time() - t0, "selected_kernels": selected}
        if mode != "off":
            row["reach"] = reach(os.path.join(d, f"{mode}.jsonl"))
        results.append(row)
        print(json.dumps(row, default=str), flush=True)
    common.write_json(os.path.join(out_root, "smoke_runs.json"), results)
    compare(out_root)


def compare(out_root):
    """Off against each mode, per configuration and round: greedy (cold and warm cache) and seeded outputs equal
    token for token and in their top-5 log-probabilities; the RNG digests before and after each generate equal; the
    warm greedy run equal to the cold one (the prefix cache). Unseeded sampling is recorded, not compared. Timing: in
    each round the mode's generate time over the mean of that round's off runs; the steady ratio is the mean of the
    seeded and unseeded ratios; over rounds, its median (the frozen budget rule from v4)."""
    import glob
    import statistics

    import common

    rows = {}
    for d in sorted(glob.glob(os.path.join(out_root, "*_*_*"))):
        p = os.path.join(d, "result.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        rows.setdefault(r["config"], []).append((os.path.basename(d), r))
    out = {}
    for config, runs in rows.items():
        c = out[config] = {"runs": [n for n, _ in runs]}
        offs_all = [r for n, r in runs if r["mode"] == "off" and "runs" in r]
        if len(offs_all) >= 2:
            c["offA_vs_offB"] = _cmp(offs_all[0], offs_all[1])
        for mode in sorted({r["mode"] for _n, r in runs if r["mode"] != "off"}):
            gs_all = [r for n, r in runs if r["mode"] == mode]
            for r in gs_all:
                if "start_error" in r:
                    c[f"{mode}_start_error"] = r["start_error"]
            per_round = []
            for rnd in sorted({r.get("round", 0) for r in gs_all}):
                offs = [r for r in offs_all if r.get("round", 0) == rnd]
                gs = [r for r in gs_all if r.get("round", 0) == rnd and "runs" in r]
                if not (offs and gs):
                    continue
                g = gs[0]
                t = {"round": rnd, "start_s": {"off": [r["start_s"] for r in offs], mode: g["start_s"]},
                     "outputs": _cmp(offs[0], g), "warm_equals_cold": _same_outputs(g, "greedy_cold", "greedy_warm")}
                for label in ("greedy_cold", "greedy_warm", "seeded", "unseeded"):
                    o = [_run(r, label)["s"] for r in offs if _run(r, label)]
                    gv = _run(g, label)
                    if o and gv:
                        t[label] = {"off": o, mode: gv["s"], "ratio": gv["s"] / (sum(o) / len(o))}
                steady = [t[k]["ratio"] for k in ("seeded", "unseeded") if k in t]
                t["steady_ratio"] = sum(steady) / len(steady) if steady else None
                t["entail"] = {k: v for k, v in g.get("entail", {}).items() if not isinstance(v, (list,))}
                per_round.append(t)
            if not per_round:
                continue
            c[f"{mode}_rounds"] = per_round
            c[f"off_vs_{mode}"] = per_round[0]["outputs"]                  # the first round, as before v4
            c[f"{mode}_timing"] = {k: v for k, v in per_round[0].items() if k not in ("outputs", "entail")}
            c[f"{mode}_warm_equals_cold"] = all(t["warm_equals_cold"] for t in per_round)
            c[f"{mode}_entail"] = per_round[0]["entail"]
            ratios = [t["steady_ratio"] for t in per_round if t["steady_ratio"] is not None]
            c[f"{mode}_steady_ratio_rounds"] = ratios
            c[f"{mode}_steady_ratio_median"] = statistics.median(ratios) if ratios else None
            c[f"{mode}_outputs_equal_all_rounds"] = all(
                all(v.get("same_ids") == "6/6" and v.get("max_logprob_diff") == 0.0 and v.get("rng_before_same") and
                    v.get("rng_after_same") for k, v in t["outputs"].items() if k != "unseeded")
                for t in per_round)
    common.write_json(os.path.join(out_root, "compare.json"), out)
    print(json.dumps(out, indent=1, default=str)[:4000])


def _run(r, label):
    return next((x for x in r["runs"] if x["label"] == label), None)


def _same_outputs(r, a, b):
    x, y = _run(r, a), _run(r, b)
    return bool(x and y and x["outputs"] and y["outputs"] and [o["ids"] for o in x["outputs"]] ==
                [o["ids"] for o in y["outputs"]])


def _cmp(r1, r2):
    res = {}
    for label in ("greedy_cold", "greedy_warm", "seeded", "unseeded"):
        x, y = _run(r1, label), _run(r2, label)
        if not x or not y or not x["outputs"] or not y["outputs"]:
            res[label] = {"missing": True, "errors": [x and x["error"], y and y["error"]]}
            continue
        ids = sum(1 for a, b in zip(x["outputs"], y["outputs"]) if a["ids"] == b["ids"])
        dmax = 0.0
        for a, b in zip(x["outputs"], y["outputs"]):
            for la, lb in zip(a["logprobs"], b["logprobs"]):
                for k in set(la) & set(lb):
                    dmax = max(dmax, abs(la[k] - lb[k]))
        res[label] = {"same_ids": f"{ids}/{len(x['outputs'])}", "max_logprob_diff": dmax,
                      "rng_before_same": x["rng_before"] == y["rng_before"],
                      "rng_after_same": x["rng_after"] == y["rng_after"],
                      "rng_untouched_in_run": [x["rng_before"] == x["rng_after"], y["rng_before"] == y["rng_after"]]}
    return res


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2])
        sys.exit(0)
    if sys.argv[1] == "--one":
        one(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/models/Qwen3-4B-FP8"))
