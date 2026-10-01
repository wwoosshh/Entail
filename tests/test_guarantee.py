"""Tests for the guarantee profile (ROADMAP M19 L5.4a; entail/guarantee.py, adapters/vllm_block_fp8_guarantee.py) on
stand-in vLLM modules (no vLLM, no GPU): the producers issue, the gate admits, repairs before dispatch, checks the
whole output and hands on the kernel's or the reference's values, or refuses. CPU torch.
Run: python tests/test_guarantee.py"""
import json
import os
import sys
import tempfile
import types
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402

from entail import guarantee as g  # noqa: E402
from entail.adapters import vllm_block_fp8_guarantee as ad  # noqa: E402

FP8 = torch.float8_e4m3fn
BN = BK = 128


def quant(x, group_size, eps=1e-10, dtype=None, column_major_scales=False, tma_aligned_scales=False, out_q=None,
          use_ue8m0=None):
    """A stand-in for vLLM's per_token_group_quant_fp8 (same signature, same meaning)."""
    M, K = x.shape
    xg = x.float().reshape(M, K // group_size, group_size)
    s = (xg.abs().amax(dim=2) / 448.0).clamp_min(eps)
    q = (xg / s[:, :, None]).clamp(-448, 448).reshape(M, K).to(FP8)
    return q, s.contiguous()


def kernel_factory(mode="faithful", rows_from=64, alt=None):
    """A stand-in for the Triton kernel: the same block loop; `mode` misreads one thing the way a consumer can."""
    def kernel(A, B, As, Bs, block_size, output_dtype=torch.float16):
        gn, gk = block_size
        M, K = A.shape
        N = B.shape[0]
        acc = torch.zeros((M, N), dtype=torch.float32)
        for kb in range(K // gk):
            a = A[:, kb * gk:(kb + 1) * gk].float()
            b = B[:, kb * gk:(kb + 1) * gk].float()
            sa = As[:, kb]
            idx = torch.arange(N) // gn
            if mode == "neighbor":
                idx = (idx + 1).clamp_max(Bs.shape[0] - 1)
            sb = (alt if mode == "alt" else Bs)[idx, kb]
            part = (a @ b.T) * sa[:, None] * sb[None, :]
            if mode == "rows":
                wrong = (a @ b.T) * sa[:, None] * Bs[(idx + 1).clamp_max(Bs.shape[0] - 1), kb][None, :]
                part[rows_from:] = wrong[rows_from:]
            acc += part
        return acc.to(output_dtype)
    return kernel


def stand_ins(mode="faithful", **kw):
    """Fake vLLM modules under the real names, wrapped by the adapter."""
    for name in (ad.FP8_UTILS, ad.BLOCK_KERNEL):
        sys.modules.pop(name, None)
    fu = types.ModuleType(ad.FP8_UTILS)
    fu.per_token_group_quant_fp8 = quant
    fu.w8a8_triton_block_scaled_mm = kernel_factory(mode, **kw)
    bk = types.ModuleType(ad.BLOCK_KERNEL)

    class Fp8BlockScaledMMLinearKernel:
        def __init__(self, block=(BN, BK)):
            self.weight_group_shape = SimpleNamespace(row=block[0], col=block[1])

        def _get_layer_params(self, layer):
            return SimpleNamespace(weight=layer.weight, block_scale=layer.weight_scale_inv,
                                   block_scale_attr="weight_scale_inv")

        def process_weights_after_loading(self, layer):
            pass

    bk.Fp8BlockScaledMMLinearKernel = Fp8BlockScaledMMLinearKernel
    sys.modules[ad.FP8_UTILS], sys.modules[ad.BLOCK_KERNEL] = fu, bk
    ad.uninstall()
    g.reset()
    ad.install_fp8_utils()
    ad.install_weights()
    return fu, bk


def weight(N, K, seed, spread=True, bn=BN, bk=BK):
    """fp8 weight [N, K] with a block scale per bn x bk block; blocks of very different size when `spread`."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn((N, K), generator=gen)
    if spread:
        w = w * torch.exp(torch.randn((N // BN, K // BK), generator=gen) * 1.5).repeat_interleave(BN, 0) \
            .repeat_interleave(BK, 1)
    wb = w.reshape(N // bn, bn, K // bk, bk)
    s = (wb.abs().amax(dim=(1, 3)) / 448.0).clamp_min(1e-10)
    q = (wb / s[:, None, :, None]).clamp(-448, 448).reshape(N, K).to(FP8)
    return q, s.contiguous()


def layer_of(bk, N=256, K=512, seed=1, block=(BN, BK), made=(BN, BK)):
    q, s = weight(N, K, seed, bn=made[0], bk=made[1])
    layer = SimpleNamespace(weight=q, weight_scale_inv=s, prefix=f"layer{seed}")
    bk.Fp8BlockScaledMMLinearKernel(block).process_weights_after_loading(layer)
    return layer


def act(M=96, K=512, seed=7):
    return torch.randn((M, K), generator=torch.Generator().manual_seed(seed))


def call(fu, layer, x, Bs=None, block=(BN, BK), out=torch.bfloat16):
    A, As = fu.per_token_group_quant_fp8(x, BK)
    return fu.w8a8_triton_block_scaled_mm(A, layer.weight, As, layer.weight_scale_inv if Bs is None else Bs,
                                          list(block), out), A, As


def truth(A, As, B, Bs):
    a = A.double() * As.double().repeat_interleave(BK, 1)
    b = B.double() * Bs.double().repeat_interleave(BN, 0).repeat_interleave(BK, 1)
    return a @ b.T


def refused(fn, kind):
    try:
        fn()
    except g.Refused as e:
        assert e.kind == kind, (e.kind, e.why)
        return e
    raise AssertionError(f"not refused (wanted {kind})")


def lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    tmp = tempfile.mkdtemp()
    rec = os.path.join(tmp, "g.jsonl")
    os.environ["ENTAIL_GUARANTEE_RECORD"] = rec
    g.set_plan(None)

    # 1. normal: the kernel's own output is handed on
    fu, bk = stand_ins("faithful")
    layer = layer_of(bk)
    y, A, As = call(fu, layer, act())
    assert y.dtype == torch.bfloat16 and tuple(y.shape) == (96, 256)
    last = lines(rec)[-1]
    assert last["outcome"] == "normal_delivered" and last["delivered"] and last["permit"] == last["call"], last
    assert last["checks"]["elements"] == 96 * 256 and last["path_after"] == "kernel"
    assert g.permit_of(y)["permit"] == last["call"]
    t = truth(A, As, layer.weight, layer.weight_scale_inv)
    assert float(((y.double() - t).abs() / (t.abs() + 1)).max()) < 0.02
    print("ok normal call handed on with a permit")

    # 2. a consumer reading the neighbour block's scale: the whole output is wrong, the reference is handed on
    fu, bk = stand_ins("neighbor")
    layer = layer_of(bk)
    y, A, As = call(fu, layer, act())
    last = lines(rec)[-1]
    assert last["outcome"] == "repaired_delivered" and last["path_after"] == "reference", last
    assert last["checks"]["beyond_tolerance"] > 0
    t = truth(A, As, layer.weight, layer.weight_scale_inv)
    assert float(((y.double() - t).abs() / (t.abs() + 1)).max()) < 0.02
    print("ok neighbour-scale read repaired with the reference")

    # 3. wrong only on rows past 64 (outside the old 64-row slice): still the whole output is checked
    fu, bk = stand_ins("rows", rows_from=80)
    layer = layer_of(bk)
    y, A, As = call(fu, layer, act(M=96))
    last = lines(rec)[-1]
    assert last["outcome"] == "repaired_delivered" and 0 < last["checks"]["beyond_tolerance"] <= 16 * 256, last
    print("ok rows past 64 caught")

    # 4. the scale of another weight of the same shape: repaired before dispatch by the producer's pairing
    fu, bk = stand_ins("faithful")
    l1, l2 = layer_of(bk, seed=1), layer_of(bk, seed=2)
    y, A, As = call(fu, l1, act(), Bs=l2.weight_scale_inv)
    last = lines(rec)[-1]
    assert last["outcome"] == "repaired_delivered" and last["path_after"] == "kernel", last
    assert [r["handle"] for r in last["repairs"]] == ["pair_Bs"], last["repairs"]
    t = truth(A, As, l1.weight, l1.weight_scale_inv)
    assert float(((y.double() - t).abs() / (t.abs() + 1)).max()) < 0.02
    print("ok another weight's scale replaced by the paired one")

    # 5. a block size the consumer was handed wrong: the declared one is used
    y, A, As = call(fu, l1, act(), block=(128, 64))
    last = lines(rec)[-1]
    assert [r["handle"] for r in last["repairs"]] == ["block"] and last["block"] == [128, 128], last
    print("ok block size repaired from the declaration")

    # 6. an activation the hooked producer did not make (shape-guessing would pass it): refused
    A2 = act().to(FP8)
    As2 = torch.ones((96, 4))
    e = refused(lambda: fu.w8a8_triton_block_scaled_mm(A2, l1.weight, As2, l1.weight_scale_inv, [128, 128],
                                                       torch.bfloat16), "declaration")
    assert lines(rec)[-1]["outcome"] == "blocked" and not lines(rec)[-1]["delivered"]
    print("ok unissued activation refused:", e.why[:60])

    # 7. a required hook missing: refused
    g.uninstalled(g.WEIGHT_PRODUCER)
    refused(lambda: call(fu, l1, act()), "hook")
    g.installed(g.WEIGHT_PRODUCER)
    print("ok missing hook refused")

    # 8. a declaration that does not hold (scales made for 128-blocks, declared 64) and a block outside the plan
    small = layer_of(bk, seed=3, block=(64, 64))
    A8, As8 = fu.per_token_group_quant_fp8(act(), 64)
    e = refused(lambda: fu.w8a8_triton_block_scaled_mm(A8, small.weight, As8, small.weight_scale_inv, [64, 64],
                                                       torch.bfloat16), "declaration")
    print("ok inconsistent declaration refused:", e.why[:70])
    small = layer_of(bk, seed=3, block=(64, 64), made=(64, 64))
    A8, As8 = fu.per_token_group_quant_fp8(act(), 64)
    refused(lambda: fu.w8a8_triton_block_scaled_mm(A8, small.weight, As8, small.weight_scale_inv, [64, 64],
                                                   torch.bfloat16), "unsupported")
    print("ok block outside the plan refused")

    # 9. the budget
    g.set_plan(g.Plan(budget_bytes=1000))
    refused(lambda: call(fu, l1, act()), "budget")
    g.set_plan(g.Plan(budget_ms=0.0))
    refused(lambda: call(fu, l1, act()), "budget")
    g.set_plan(None)
    print("ok budgets refuse")

    # 10. writes after the issue: through PyTorch (version counter) and behind its back (.data: bytes)
    l3 = layer_of(bk, seed=4)
    l3.weight_scale_inv.mul_(1.0)
    refused(lambda: call(fu, l3, act()), "epoch")
    l4 = layer_of(bk, seed=5)
    l4.weight_scale_inv.data[0, 0] *= 2
    e = refused(lambda: call(fu, l4, act()), "integrity")
    assert "Bs" in e.why, e.why
    print("ok in-place write refused (epoch), write behind the counter refused (integrity)")

    # 11. the activation's storage issued again before the consumer read it
    A, As = fu.per_token_group_quant_fp8(act(), BK)
    g.issue_activation(A[:], As[:], BK)        # another value issued in the same storage (a reused buffer)
    refused(lambda: fu.w8a8_triton_block_scaled_mm(A, l1.weight, As, l1.weight_scale_inv, [128, 128],
                                                   torch.bfloat16), "epoch")
    print("ok re-issued storage refused")

    # 12. an unsupported output dtype and UE8M0-like scale dtypes
    refused(lambda: call(fu, l1, act(), out=torch.float32), "unsupported")
    print("ok unsupported output dtype refused")

    # 13. the checksum sees a swap of two words
    x = torch.arange(4096, dtype=torch.int32)
    c0 = int(g.checksum(x))
    x[[5, 2000]] = x[[2000, 5]]
    assert int(g.checksum(x)) != c0
    print("ok checksum sees a swap")

    # 14. every decision is a line with the plan, environment and outcome
    L = lines(rec)
    assert all(k in L[0] for k in ("plan_fp", "env_fp", "contract", "consumer", "outcome", "delivered"))
    counts = {}
    for line in L:
        counts[line["outcome"]] = counts.get(line["outcome"], 0) + 1
    assert counts.get("blocked", 0) >= 8 and counts.get("repaired_delivered", 0) >= 4, counts
    print("ok record:", counts)

    # 15. the autoinstall table for ENTAIL=guarantee holds only the profile's entries
    src = open(os.path.join(os.path.dirname(HERE), "entail", "adapters", "autoinstall", "sitecustomize.py"),
               encoding="utf-8").read()
    assert "GUARANTEE_TARGETS" in src and 'TARGETS = dict(GUARANTEE_TARGETS)' in src
    print("ok autoinstall table")
    ad.uninstall()
    for name in (ad.FP8_UTILS, ad.BLOCK_KERNEL):
        sys.modules.pop(name, None)


if __name__ == "__main__":
    main()
