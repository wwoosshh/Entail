"""The scenarios (docs/semantic-guarantee-gpu-protocol.ko.md "첫 시나리오 묶음"): normal 8, consumer defects 12
(four misreads x three timings), admission and unsupported 4. `cases(split, key)` gives the dev set (fixed seeds, used
for development and calibration) or the holdout (seeds and the free parameters drawn from a key that exists only
after the freeze: the freeze manifest's hash). The holdout varies inputs and timings inside the four published
misread classes; it is not a measure of new, unseen bug types, and it is not blind (no separate evaluator).

A step is one of:
  call      {"op": "call", "layer": i, "M": rows, "seed": s, "dist": d, "mut": 0|1, "cache": 0|1}
  reload    {"op": "reload", "layer": i, "seed": s}        new weight values, processed again by the producer
  capture   {"op": "capture", "layer": i, "M": rows, "seed": s}   warm up, then capture quant + consumer + observer
  replay    {"op": "replay", "seed": s, "mut": 0|1}       new input into the captured buffer, then replay
  direct    {"op": "direct", "layer": i, "M": rows, "seed": s, "quant": "native"}   consumer called with an
            activation another quantizer made (vLLM's QuantFP8.forward_native)
"""
import hashlib
import random

N, K = 1536, 2560        # 12 x 20 blocks of 128
TIMINGS = ("T1", "T2", "T3")
MUTANTS = ("interval", "neighbor", "alt_weight", "rows")


def _rng(split, key, cid):
    if split == "dev":
        return random.Random(f"dev-{cid}")
    return random.Random(int(hashlib.sha256(f"{key}:{cid}".encode()).hexdigest()[:16], 16))


def _seed(r):
    return r.randrange(1, 2 ** 31)


def normal_cases(split, key):
    out = []

    def case(cid, steps, **kw):
        c = {"id": cid, "group": "normal", "N": N, "K": K, "out": "bfloat16", "layers": 1, "steps": steps}
        c.update(kw)
        out.append(c)

    r = _rng(split, key, "N1")
    case("N1-small-cold", [{"op": "call", "layer": 0, "M": r.choice([1, 4, 16]), "seed": _seed(r), "dist": "normal"}],
         wseed=[_seed(r)])
    r = _rng(split, key, "N2")
    case("N2-large", [{"op": "call", "layer": 0, "M": r.choice([512, 640, 1024]), "seed": _seed(r),
                       "dist": "normal"}], wseed=[_seed(r)])
    r = _rng(split, key, "N3")
    case("N3-distributions", [{"op": "call", "layer": 0, "M": r.choice([64, 96, 200]), "seed": _seed(r), "dist": d}
                              for d in ("heavy", "small", "normal")], wseed=[_seed(r)])
    r = _rng(split, key, "N4")
    m, s = r.choice([32, 128, 256]), _seed(r)
    case("N4-cold-warm", [{"op": "call", "layer": 0, "M": m, "seed": s, "dist": "normal"} for _ in range(3)],
         wseed=[_seed(r)])
    r = _rng(split, key, "N5")
    case("N5-new-weights", [{"op": "call", "layer": 0, "M": 64, "seed": _seed(r), "dist": "normal"},
                            {"op": "reload", "layer": 0, "seed": _seed(r)},
                            {"op": "call", "layer": 0, "M": 64, "seed": _seed(r), "dist": "normal"},
                            {"op": "call", "layer": 1, "M": 96, "seed": _seed(r), "dist": "normal"}],
         layers=2, wseed=[_seed(r), _seed(r)])
    r = _rng(split, key, "N6")
    case("N6-colmajor-scales", [{"op": "direct", "layer": 0, "M": r.choice([48, 160]), "seed": _seed(r),
                                 "quant": "colmajor"}], wseed=[_seed(r)])
    r = _rng(split, key, "N7")
    case("N7-repeated", [{"op": "call", "layer": r.randrange(2), "M": r.choice([1, 3, 17, 64, 65, 128, 300]),
                          "seed": _seed(r), "dist": "normal"} for _ in range(12)], layers=2,
         wseed=[_seed(r), _seed(r)])
    r = _rng(split, key, "N8")
    case("N8-graph", [{"op": "capture", "layer": 0, "M": r.choice([64, 128, 256]), "seed": _seed(r)}] +
         [{"op": "replay", "seed": _seed(r), "mut": 0} for _ in range(3)], wseed=[_seed(r)])
    return out


def defect_cases(split, key):
    out = []
    for kind in MUTANTS:
        for timing in TIMINGS:
            cid = f"D-{kind}-{timing}"
            r = _rng(split, key, cid)
            ws = [_seed(r), _seed(r)]
            rows_from = r.choice([64, 80, 128, 200])
            c = {"id": cid, "group": "defect", "mutant": kind, "timing": timing, "N": N, "K": K, "out": "bfloat16",
                 "layers": 2, "wseed": ws, "rows_from": rows_from}
            python_cache = kind == "alt_weight" and timing != "T3"
            c["mechanism"] = ("consumer-side scale cache at the call site (Python)" if python_cache else
                              "the consumer kernel's own read, switched by a device flag")
            if timing == "T1":
                M = r.choice([max(rows_from + 16, 96), 256])
                steps = []
                if python_cache:          # the cache already holds layer 1's scale under the shared shape
                    steps.append({"op": "call", "layer": 1, "M": 16, "seed": _seed(r), "dist": "normal", "mut": 0,
                                  "cache": 1, "role": "fills the cache"})
                steps += [{"op": "call", "layer": 0, "M": M, "seed": _seed(r), "dist": "normal", "mut": 1,
                           "cache": 1, "role": "first call"},
                          {"op": "call", "layer": 0, "M": M, "seed": _seed(r), "dist": "normal", "mut": 1, "cache": 1,
                           "role": "second call"},
                          {"op": "call", "layer": 0, "M": r.choice([M, 300]), "seed": _seed(r), "dist": "normal",
                           "mut": 1, "cache": 1, "role": "third call"}]
            elif timing == "T2":
                M2 = r.choice([256, 300, 384])
                steps = [{"op": "call", "layer": 0, "M": 16, "seed": _seed(r), "dist": "normal", "mut": 0,
                          "cache": 0, "role": "normal first call (16 rows)"}]
                if python_cache:
                    steps.append({"op": "call", "layer": 1, "M": 16, "seed": _seed(r), "dist": "normal", "mut": 0,
                                  "cache": 1, "role": "another weight of the same shape fills the cache"})
                steps += [{"op": "call", "layer": 0, "M": M2, "seed": _seed(r), "dist": "normal", "mut": 1,
                           "cache": 1, "role": "new batch and input"},
                          {"op": "call", "layer": 0, "M": M2, "seed": _seed(r), "dist": "normal", "mut": 1,
                           "cache": 1, "role": "second call"},
                          {"op": "call", "layer": 0, "M": r.choice([M2, 512]), "seed": _seed(r), "dist": "normal",
                           "mut": 1, "cache": 1, "role": "third call"}]
            else:
                M = r.choice([max(rows_from + 32, 128), 256])
                steps = [{"op": "capture", "layer": 0, "M": M, "seed": _seed(r), "role": "normal warm-up + capture"},
                         {"op": "replay", "seed": _seed(r), "mut": 0, "role": "normal replay"},
                         {"op": "replay", "seed": _seed(r), "mut": 1, "role": "defect comes on at replay"},
                         {"op": "replay", "seed": _seed(r), "mut": 1, "role": "second defective replay"},
                         {"op": "replay", "seed": _seed(r), "mut": 1, "role": "third defective replay"}]
            c["steps"] = steps
            out.append(c)
    return out


def admission_cases(split, key):
    r = _rng(split, key, "A")
    base = {"group": "admission", "N": N, "K": K, "out": "bfloat16", "layers": 1, "wseed": [_seed(r)]}
    call = {"op": "call", "layer": 0, "M": 64, "seed": _seed(r), "dist": "normal"}
    return [
        dict(base, id="A1-declaration-missing", expect="declaration",
             steps=[{"op": "direct", "layer": 0, "M": 64, "seed": _seed(r), "quant": "native"}]),
        dict(base, id="A2-hook-missing", expect="hook", skip_hook="weights", steps=[dict(call)]),
        dict(base, id="A3-unsupported-output-dtype", expect="unsupported", out="float32", steps=[dict(call)]),
        dict(base, id="A4-budget", expect="budget", plan={"budget_bytes": 1 << 20}, steps=[dict(call)]),
    ]


def known_dev_cases():
    """Development cases outside the 24: the known tile defect (a tuned table gives BLOCK_SIZE_K 256 at larger
    batches), which must be shown repaired and delivered."""
    return [{"id": "dev-tile-k256", "group": "defect", "mutant": "tile_k256", "timing": "T2", "N": N, "K": K,
             "out": "bfloat16", "layers": 1, "wseed": [11], "tile_k256": True,
             "steps": [{"op": "call", "layer": 0, "M": 16, "seed": 21, "dist": "normal", "mut": 0},
                       {"op": "call", "layer": 0, "M": 256, "seed": 22, "dist": "normal", "mut": 0},
                       {"op": "call", "layer": 0, "M": 300, "seed": 23, "dist": "normal", "mut": 0}]}]


def cases(split="dev", key=""):
    out = normal_cases(split, key) + defect_cases(split, key) + admission_cases(split, key)
    assert len(out) == 24, len(out)
    if split == "dev":
        out += known_dev_cases()
    for c in out:
        c["split"] = split
    return out
