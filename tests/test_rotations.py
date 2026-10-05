"""Tests for the rotation pairing (ROADMAP M22.4; entail/declarations.py rotations, kernel_check's layer scopes): a
rotary layer says, while it runs, what the query and key it is handed mean - tokens, heads, and in each head the
rotation pairs its own is_neox_style declares - and the one rule (kernel_types, unchanged) holds a kernel to that.
The kernels are vLLM's Triton MRoPE kernel as three versions compiled it (their TTIR, recorded on the GPU):
  0.22.0  pairs feature j with j + rotary_dim/2 whatever the layer declares (vllm#42016)
  0.30.0  pairs as the layer declares (is_neox_style a constexpr; this IR: interleaved, through tt.split)
  0.19.0  Qwen3.5's shapes: split-wise pairs over a rotary dimension of 64 in heads of 256
CPU only. Run: python tests/test_rotations.py
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import declarations as D  # noqa: E402
from entail import kernel_check as KC  # noqa: E402
from entail import kernel_types as KT  # noqa: E402

DATA = os.path.join(os.path.dirname(HERE), "entail", "data")
IR = os.path.join(HERE, "data", "kernel_ir")


def table(name):
    with open(os.path.join(DATA, f"{name}_declarations.json"), encoding="utf-8") as f:
        return json.load(f)


def mrope(version, T, n_q, n_k, head_size, rotary_dim, neox):
    """One launch of the MRoPE kernel: q and k as a rotary layer with this pairing declares them (rotated in place:
    the kernel's outputs), cos and sin as the launch alone tells (no meaning)."""
    with open(os.path.join(IR, f"vllm{version}_triton_mrope_forward.ttir"), encoding="utf-8") as f:
        ttir = f.read()
    meanings = {}
    for arg, n in (("q_ptr", n_q), ("k_ptr", n_k)):
        shape, stride = (T, n * head_size), (n * head_size, 1)
        names, vshape, vstride = D.rotation_view(shape, stride, head_size, rotary_dim, neox)
        m = KC._meaning_from(shape, stride, KC.scoped_fact(names, vshape, vstride))
        m.kind = "output"
        meanings[arg] = m
    half = rotary_dim // 2
    for arg in ("cos", "sin"):
        meanings[arg] = KC._meaning_from((3, T, half), (T * half, half, 1), None)
    return KT.check_launch(ttir, meanings, {"num_tokens": T}, (T,))


def main():
    # the view: a head's features as its rotation pairs
    names, shape, stride = D.rotation_view((5, 2048), (2048, 1), 128, 128, True)
    assert names == ["token", "head", None, "freq"] and shape == (5, 16, 2, 64) and stride == (2048, 128, 64, 1)
    names, shape, stride = D.rotation_view((5, 2048), (2048, 1), 128, 128, False)
    assert names == ["token", "head", "freq", None] and shape == (5, 16, 64, 2) and stride == (2048, 128, 2, 1)
    _n, shape, stride = D.rotation_view((5, 16, 128), (4096, 128, 1), 128, 128, True)
    assert shape == (5, 16, 2, 64) and stride == (4096, 128, 64, 1), (shape, stride)
    _n, shape, stride = D.rotation_view((5, 4096), (4096, 1), 256, 64, True)
    assert shape == (5, 16, 8, 32) and stride == (4096, 256, 32, 1), (shape, stride)
    assert D.rotation_view((5, 2000), (2000, 1), 128, 128, True) is None
    print("ok a query's features as heads and rotation pairs: split-wise (j, j + rotary_dim/2) or interleaved "
          "(2i, 2i+1); a partial rotary dimension keeps the rest of the head unrotated")

    m = KC._meaning_from((5, 2048), (2048, 1), KC.scoped_fact(["token", "head", "freq", None], (5, 16, 64, 2),
                                                              (2048, 128, 2, 1)))
    assert m.shape == (5, 16, 64, 2) and m.names() == ["token", "head", "freq", None]
    a = KC._sig_of(KC.scoped_fact(["token", "head", None, "freq"], (5, 16, 2, 64), (2048, 128, 64, 1)))
    b = KC._sig_of(KC.scoped_fact(["token", "head", None, "freq"], (5, 16, 4, 32), (2048, 128, 32, 1)))
    assert a != b, "the view is part of the launch key"
    print("ok the meaning is the view's; the view is part of a launch's key")

    # vllm#42016: the 0.22.0 kernel pairs split-wise; GLM-OCR's text model declares interleaved pairs
    v = mrope("0220", 16, 16, 8, 128, 128, neox=False)
    assert v.verdict == "violation" and "'freq'" in v.why, v
    print(f"ok vLLM 0.22.0's MRoPE kernel with a layer that declares interleaved pairs (GLM-OCR): {v.verdict}: "
          f"{v.why}")
    v = mrope("0220", 16, 16, 8, 128, 128, neox=True)
    assert v.verdict != "violation", v
    print(f"ok the same kernel with a layer that declares split-wise pairs (its own convention): no violation "
          f"({v.verdict}: {v.why[:60]})")
    v = mrope("0300", 16, 16, 8, 128, 128, neox=False)
    assert v.verdict != "violation", v
    print(f"ok vLLM 0.30.0's kernel, compiled for interleaved pairs: no violation ({v.verdict}: {v.why[:60]})")
    v = mrope("0190", 16, 16, 4, 256, 64, neox=True)
    assert v.verdict != "violation", v
    print(f"ok vLLM 0.19.0's kernel with Qwen3.5's split-wise pairs over 64 of 256 features: no violation "
          f"({v.verdict}: {v.why[:60]})")
    v = mrope("0190", 16, 16, 4, 256, 64, neox=False)
    assert v.verdict == "violation" and "'freq'" in v.why, v
    print("ok the same kernel with a layer that declares interleaved pairs: violation")

    # the layer says it while it runs: hooks on the table's rotary kinds
    T = 6
    seen = []

    def kernel(*tensors):        # what the launch path asks for each tensor it is handed
        for t in tensors:
            seen.append(KC._scope_fact(tuple(t.shape), tuple(t.stride()), t.dtype))

    def rotary_class(name):
        class Rotary(torch.nn.Module):
            def __init__(self, neox):
                super().__init__()
                self.head_size, self.rotary_dim, self.is_neox_style = 128, 128, neox

            def forward(self, positions, query, key):
                q, k = query.contiguous(), key.contiguous()          # copies, as triton_mrope makes
                cos = torch.zeros(3, T, 64, dtype=q.dtype)
                kernel(q, k, cos, positions)
                return query, key
        Rotary.__name__ = name
        return Rotary

    class Attention(torch.nn.Module):
        def __init__(self, cls, neox):
            super().__init__()
            self.rotary_emb = cls(neox)

        def forward(self, positions, hidden):
            q, k, _v = hidden.split([2048, 1024, 1024], dim=-1)
            return self.rotary_emb(positions, q, k)

    for engine, cls in (("vllm", "RotaryEmbeddingBase"), ("sglang", "RotaryEmbedding")):
        r = D.Declarations(table(engine), cuda_only=False)
        model = Attention(rotary_class(cls), neox=False)
        assert r.rotations(model) == 1 and r.rotations(model) == 0, "hooked once"
        seen.clear()
        model(torch.zeros(3, T, dtype=torch.long), torch.zeros(T, 4096, dtype=torch.bfloat16))
        fq, fk, fcos, fpos = seen
        assert fq["names"] == ["token", "head", "freq", None] and fq["view"]["shape"] == [T, 16, 64, 2], fq
        assert fk["view"]["shape"] == [T, 8, 64, 2], fk
        assert fcos is None and fpos is None, "cos and the positions are not a query or key"
        assert not KC._SCOPES and KC._scope_fact((T, 2048), (2048, 1), torch.bfloat16) is None, "only while it runs"
        print(f"ok {engine}'s table: a {cls} says, while it runs, that the query and key copies a kernel is handed are "
              f"tokens x heads x interleaved pairs (its is_neox_style False); cos, the positions, and anything after "
              f"it get nothing")

    model = Attention(rotary_class("RotaryEmbeddingBase"), neox=True)
    D.Declarations(table("vllm"), cuda_only=False).rotations(model)
    seen.clear()
    model(torch.zeros(3, T, dtype=torch.long), torch.zeros(T, 4096, dtype=torch.bfloat16))
    assert seen[0]["names"] == ["token", "head", None, "freq"] and seen[0]["view"]["shape"] == [T, 16, 2, 64]
    print("ok a layer that declares split-wise pairs: tokens x heads x halves x pairs")

    class Failing(rotary_class("RotaryEmbeddingBase")):
        def forward(self, positions, query, key):
            raise RuntimeError("the engine's own error")

    model = Attention(Failing, neox=True)
    D.Declarations(table("vllm"), cuda_only=False).rotations(model)
    try:
        model(torch.zeros(3, T, dtype=torch.long), torch.zeros(T, 4096, dtype=torch.bfloat16))
    except RuntimeError:
        pass
    assert not KC._SCOPES, "a layer that raises still ends its scope"
    print("ok a layer that raises ends its scope (the engine's error is its own)")


if __name__ == "__main__":
    main()
