"""Tests for graph_types (ROADMAP M19 L6, step 3): the one rule over a compiled graph, on a graph built by hand the
way Dynamo builds vLLM's (placeholders for the activation, the Marlin-packed weight, its scales; the C++ matmul;
elementwise and view operations). Nothing about one model is written here: the facts attached to the inputs and the
operations' signatures decide. torch (CPU) only.
Run: python tests/test_graph_types.py
"""
import os
import sys

import torch
import torch.fx as fx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import graph_types as GT  # noqa: E402

K, N, M = 2560, 128, 4


def marlin_gemm(*a, **k):
    raise RuntimeError("not run")


marlin_gemm.__entail_op__ = "_C.marlin_gemm"


def rms_norm(*a, **k):
    raise RuntimeError("not run")


rms_norm.__entail_op__ = "vllm_ir.rms_norm.default"


def build(a_dtype=torch.bfloat16, s_dtype=None, size_k=K, size_n=N, a_transposed=False, scale_kind="scale",
          packed_shape=None, with_norm=True):
    """A graph: a -> (rms_norm) -> marlin_gemm(a, w, s) -> silu -> * 2 -> output; facts on the three inputs."""
    g = fx.Graph()
    a = g.placeholder("a")
    w = g.placeholder("w")
    s = g.placeholder("s")
    nw = g.placeholder("nw")
    a_t = torch.zeros((K, M) if a_transposed else (M, K), dtype=a_dtype)
    w_t = torch.zeros(packed_shape or (K // 16, N * 4), dtype=torch.int32)
    s_t = torch.zeros((K // 128, N), dtype=s_dtype or a_dtype)
    nw_t = torch.zeros((K,), dtype=a_dtype)
    for node, t in ((a, a_t), (w, w_t), (s, s_t), (nw, nw_t)):
        node.meta["example_value"] = t
    x = a
    if with_norm:
        x = g.call_function(rms_norm, (a, nw, 1e-6, None))
        x.meta["example_value"] = a_t.clone()
    out = g.call_function(marlin_gemm, (x, None, w, None, s, None, None, None, None, 1, M, size_n, size_k, False,
                                        True, False))
    out.meta["example_value"] = torch.zeros((M, N), dtype=a_dtype)
    act = g.call_function(torch.nn.functional.silu, (out,))
    act.meta["example_value"] = out.meta["example_value"]
    two = g.call_function(torch.mul, (act, 2.0))
    two.meta["example_value"] = out.meta["example_value"]
    g.output(two)
    facts = {
        "a": {"names": ["hidden", "token"] if a_transposed else ["token", "hidden"], "kind": "value", "serial": 0,
              "pair": 0, "groups": [1, 1]},
        "w": {"names": [None, None], "kind": "value", "serial": 1, "pair": 2, "groups": [1, 1],
              "packed": {"form": "marlin_fp8", "size_k": K, "size_n": N, "group": 128}},
        "s": {"names": ["hidden", "feature"], "kind": scale_kind, "serial": 2, "pair": 1, "groups": [128, 1],
              "packed": {"form": "marlin_permuted", "size_k": K, "size_n": N, "group": 128}},
        "nw": {"names": ["hidden"], "kind": "value", "serial": 0, "pair": 0, "groups": [1]},
    }
    inputs = [a_t, w_t, s_t, nw_t]
    by_ptr = {t.data_ptr(): facts[k] for k, t in zip(("a", "w", "s", "nw"), inputs)}
    return fx.GraphModule(torch.nn.Module(), g), inputs, lambda t: by_ptr.get(t.data_ptr())


def check(**kw):
    gm, inputs, fact_of = build(**kw)
    return GT.check_graph(gm, inputs, fact_of)


def main():
    v = check()
    assert v["verdict"] == "checked" and v["sites"]["_C.marlin_gemm"] == {"proven": 1} and \
        v["sites"]["vllm_ir.rms_norm"] == {"proven": 1} and v["inputs_with_facts"] == 4, v
    print("ok a Marlin matmul handed its own packed weight and scales, after an RMS norm: proven,", v["checks"],
          "pairings,", v["nodes"], "nodes")

    v = check(size_k=K + 128)
    assert v["verdict"] == "violation" and "size_k" in v["violations"][0]["why"], v
    print("ok size_k not the packed weight's: violation:", v["violations"][0]["why"])

    v = check(s_dtype=torch.float32)
    assert v["verdict"] == "violation" and "scales" in v["violations"][0]["why"], v
    print("ok scales in float32 for a bf16 activation: violation:", v["violations"][0]["why"])

    v = check(a_transposed=True)
    assert v["verdict"] == "violation" and "token" in v["violations"][0]["why"], v
    print("ok a transposed activation: violation:", v["violations"][0]["why"])

    gm, inputs, fact_of = build()
    other = dict(fact_of(inputs[2]))
    other["pair"], other["serial"] = 9, 10
    v = GT.check_graph(gm, inputs, lambda t: other if t.data_ptr() == inputs[2].data_ptr() else fact_of(t))
    assert v["verdict"] == "violation" and "issue 10" in v["violations"][0]["why"], v
    print("ok another layer's scales handed with the weight: violation:", v["violations"][0]["why"])

    v = GT.check_graph(gm, inputs, lambda t: None if t.data_ptr() == inputs[1].data_ptr() else fact_of(t))
    assert v["verdict"] == "checked" and v["sites"]["_C.marlin_gemm"] == {"unproven": 1} and \
        "not given" in v["unproven_sites"][0]["why"], v
    print("ok a weight without a meaning: the site is unproven, not a violation:", v["unproven_sites"][0]["why"])

    v = check(packed_shape=(K // 16, N * 4 // 2))
    assert v["verdict"] == "violation" and "size_n" in v["violations"][0]["why"], v
    print("ok a packed weight holding half the features: violation:", v["violations"][0]["why"])

    # elementwise pairing and the scale applied twice
    g = fx.Graph()
    x, sc = g.placeholder("x"), g.placeholder("sc")
    xt, st = torch.zeros((M, K), dtype=torch.bfloat16), torch.zeros((M, 1), dtype=torch.bfloat16)
    x.meta["example_value"], sc.meta["example_value"] = xt, st
    m1 = g.call_function(torch.mul, (x, sc))
    m1.meta["example_value"] = xt
    m2 = g.call_function(torch.mul, (m1, sc))
    m2.meta["example_value"] = xt
    g.output(m2)
    f = {xt.data_ptr(): {"names": ["token", "hidden"], "kind": "value", "serial": 3, "pair": 4, "groups": [1, 1]},
         st.data_ptr(): {"names": ["token", None], "kind": "scale", "serial": 4, "pair": 3, "groups": [1, 1]}}
    v = GT.check_graph(fx.GraphModule(torch.nn.Module(), g), [xt, st], lambda t: f.get(t.data_ptr()))
    assert v["verdict"] == "violation" and "second time" in v["violations"][0]["why"], v
    print("ok a scale applied twice in torch operations: violation:", v["violations"][0]["why"])

    g = fx.Graph()
    x, y = g.placeholder("x"), g.placeholder("y")
    xt, yt = torch.zeros((M, K), dtype=torch.bfloat16), torch.zeros((K, M), dtype=torch.bfloat16)
    x.meta["example_value"], y.meta["example_value"] = xt, yt
    yt_ = g.call_method("transpose", (y, 0, 1))
    yt_.meta["example_value"] = xt
    s_ = g.call_function(torch.add, (x, yt_))
    s_.meta["example_value"] = xt
    g.output(s_)
    f = {xt.data_ptr(): {"names": ["token", "hidden"], "kind": "value", "serial": 0, "pair": 0, "groups": [1, 1]},
         yt.data_ptr(): {"names": ["hidden", "token"], "kind": "value", "serial": 0, "pair": 0, "groups": [1, 1]}}
    v = GT.check_graph(fx.GraphModule(torch.nn.Module(), g), [xt, yt], lambda t: f.get(t.data_ptr()))
    assert v["verdict"] == "checked" and v["violations"] == [] and v["checks"] == 1, v
    f[yt.data_ptr()]["names"] = ["token", "hidden"]
    v = GT.check_graph(fx.GraphModule(torch.nn.Module(), g), [xt, yt], lambda t: f.get(t.data_ptr()))
    assert v["verdict"] == "violation" and "disagree" in v["violations"][0]["why"], v
    print("ok a transpose followed by an addition: axes pair after the transpose, disagree without it:",
          v["violations"][0]["why"])


if __name__ == "__main__":
    main()
    print("all ok")
