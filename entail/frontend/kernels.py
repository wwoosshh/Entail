"""frontend.kernels: the hand kernel the attention lowering "triton" runs (ROADMAP M8.1).

Flash-decoding attention for one query token over a static KV cache, from the research's week-2 kernel
(phase0/week2/triton_decode_attn.py): a valid length per row (no mask is ever built) and heads shared by index (no
copy of K/V), with fixed shapes whatever the valid length, so a CUDA graph can hold it. Two kernels - split-K over
the cache, then combine - with float32 accumulation, bfloat16/float16 in and out; the head dim must be a power of two.
It is registered as the custom op entail::decode_attention so torch.compile and CUDA graphs treat it as one opaque
node, as the week-2 and week-4 measurements did. Needs triton and a CUDA device; imported only by that lowering.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_kernel(Q, K, V, VALID, MP, LP, OP,
                  sqb, sqh, skb, skh, skn, svb, svh, svn, smb, smh, sms, sob, soh, sos,
                  sm_scale, L_MAX,
                  GROUP: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)
    kvh = h // GROUP
    valid = tl.load(VALID + b)
    per_split = (L_MAX + SPLIT - 1) // SPLIT
    start = s * per_split
    end = tl.minimum(start + per_split, valid)

    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + h * sqh + offs_d).to(tl.float32)
    m_i = tl.full([], float("-inf"), tl.float32)
    l_i = tl.zeros([], tl.float32)
    acc = tl.zeros([D], tl.float32)

    for n0 in range(start, start + per_split, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < end
        k = tl.load(K + b * skb + kvh * skh + offs_n[:, None] * skn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * sm_scale
        scores = tl.where(mask_n, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(scores - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        v = tl.load(V + b * svb + kvh * svh + offs_n[:, None] * svn + offs_d[None, :],
                    mask=mask_n[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    tl.store(MP + b * smb + h * smh + s * sms, m_i)
    tl.store(LP + b * smb + h * smh + s * sms, l_i)
    tl.store(OP + b * sob + h * soh + s * sos + offs_d, acc)


@triton.jit
def _combine_kernel(MP, LP, OP, OUT, smb, smh, sms, sob, soh, sos, soutb, south,
                    D: tl.constexpr, SPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SPLIT)
    m = tl.load(MP + b * smb + h * smh + offs_s * sms)
    l = tl.load(LP + b * smb + h * smh + offs_s * sms)
    m_max = tl.max(m, axis=0)
    w = tl.exp(m - m_max)
    l_tot = tl.sum(l * w, axis=0)
    offs_d = tl.arange(0, D)
    o = tl.load(OP + b * sob + h * soh + offs_s[:, None] * sos + offs_d[None, :])
    out = tl.sum(o * w[:, None], axis=0) / l_tot
    tl.store(OUT + b * soutb + h * south + offs_d, out.to(OUT.dtype.element_ty))


def decode_attention(q, k_cache, v_cache, valid_len, sm_scale, split=8, block_n=64):
    """q [B, Hq, 1, D]; k_cache, v_cache [B, Hkv, Lmax, D] (D contiguous); valid_len [B] int32 on the device: the
    keys 0..valid_len-1 are read, the rest never loaded. Returns [B, Hq, 1, D] in q's dtype."""
    B, Hq, one, D = q.shape
    Hkv, Lmax = k_cache.shape[1], k_cache.shape[2]
    q2 = q.reshape(B, Hq, D)
    mp = torch.empty(B, Hq, split, device=q.device, dtype=torch.float32)
    lp = torch.empty_like(mp)
    op = torch.empty(B, Hq, split, D, device=q.device, dtype=torch.float32)
    out = torch.empty(B, Hq, D, device=q.device, dtype=q.dtype)
    _split_kernel[(B, Hq, split)](
        q2, k_cache, v_cache, valid_len, mp, lp, op,
        q2.stride(0), q2.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        mp.stride(0), mp.stride(1), mp.stride(2), op.stride(0), op.stride(1), op.stride(2),
        sm_scale, Lmax,
        GROUP=Hq // Hkv, D=D, BLOCK_N=block_n, SPLIT=split, num_warps=4)
    _combine_kernel[(B, Hq)](
        mp, lp, op, out, mp.stride(0), mp.stride(1), mp.stride(2),
        op.stride(0), op.stride(1), op.stride(2), out.stride(0), out.stride(1),
        D=D, SPLIT=split, num_warps=4)
    return out.reshape(B, Hq, 1, D)


@torch.library.custom_op("entail::decode_attention", mutates_args=())
def decode_attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, valid: torch.Tensor,
                        scale: float) -> torch.Tensor:
    return decode_attention(q, k, v, valid, scale)


@decode_attention_op.register_fake
def _(q, k, v, valid, scale):
    return torch.empty_like(q)
