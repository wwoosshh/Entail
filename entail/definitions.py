"""definitions: what engine functions compute, for the ones that carry no definition of their own (M19 L3).

vLLM's custom ops carry a native forward (their definition) beside the kernel, and kernel_reference_contract holds the
kernel against it. Many engine kernels are plain functions with no such twin: the fused MoE, the block-quantized FP8
matmul, SGLang's Gated DeltaNet gate. This module writes each one's definition in plain PyTorch, from the declared
arguments (the quantization config, the scales, the block size) - the same parameters, the same outputs in shape and
dtype - so adapters/function_reference.py can hold the kernel against it on the first real call and, when they
differ, send the function to it. A definition is the function's meaning, not a rule about a defect: which input,
configuration or layout shows a defect is nowhere here.

Every definition takes `_dtype`, the dtype its arithmetic runs in: None is the inputs' own (so a run in the inputs'
reduced dtype gives the definition's own rounding noise), float32 the reference. A kernel that works in float32 on
inputs that carry no reduced dtype (fp8 codes with float32 scales) names `noise` bfloat16: its float32 reference
would round like the kernel and leave no floor, so its noise is taken where it shows (a conservative floor). A call
a definition does not cover (another quantization scheme, an expert-parallel map) raises NotImplementedError, and
the call is not compared. `capturable` says whether the definition can run inside a CUDA graph capture (the fused
MoE's loops over the experts present synchronise with the host, so it cannot).
"""
from typing import Callable, NamedTuple, Optional, Tuple


class Definition(NamedTuple):
    target: str            # "module:function", where the function is defined
    engine: str            # "vllm", "sglang"
    rows: Tuple[str, ...]  # parameters that hold the token dimension (cut along dimension 0 for the comparison)
    fn: Callable           # the definition, the function's parameters plus _dtype
    noise: Optional[str]   # dtype of the noise run when the inputs carry no reduced dtype of their own
    source: str            # what the definition follows
    capturable: bool = False   # runs inside a CUDA graph capture (no host synchronisation)
    probe: Optional[Callable] = None   # (cut arguments, generator) -> the values a warm-up probe gives the arguments
    #                                    whose range only the function knows (M19 L3.3a); floating token arguments
    #                                    get seeded normal values without it
    # M19 L3.3d, for functions that write their result into buffers and index one another's rows (the model runner's
    # bookkeeping: positions, slot mappings):
    writes: Tuple[str, ...] = ()       # arguments the function writes: compared after the call, with its return
    whole: bool = False                # compared on copies of the whole call (the rows index each other: not cut)
    fresh: Optional[Callable] = None   # (bound arguments) -> fresh buffers for the outputs the function would write
    #                                    into the engine's own (a persistent slot-mapping buffer): each run gets its own
    exact: bool = False                # integer outputs (indices, positions) compared exactly


def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, activation=None, apply_router_weight_on_input=False,
                  global_num_experts=-1, expert_map=None, quant_config=None, _dtype=None):
    """vLLM 0.30's fused_experts: for each token and each of its top-k experts, down(silu(gate) * up) of the token,
    weighted by the routing weight and summed. w1 is [E, 2I, H] (gate rows, then up rows), w2 [E, H, I].
    Unquantized, or INT8 W8A8 (not block-quantized) as the config and the scales declare it: a weight scale holds
    one value per expert or one per output channel, by its own shape; activations are quantized per token when the
    config says so, else per tensor, with the static scale when one is given and dynamically otherwise (over every
    token for the input, over every token and expert for the intermediate)."""
    import torch

    act = str(getattr(activation, "value", activation) or "silu").lower()
    if act != "silu":
        raise NotImplementedError(f"activation {act}")
    if expert_map is not None:
        raise NotImplementedError("an expert-parallel map")
    qc = quant_config
    int8 = bool(qc is not None and getattr(qc, "use_int8_w8a8", False))
    if qc is not None and not int8 and (getattr(qc, "quant_dtype", None) is not None
                                        or getattr(qc, "weight_quant_dtype", None) is not None):
        raise NotImplementedError(f"quantization {getattr(qc, 'quant_dtype', None)} / "
                                  f"{getattr(qc, 'weight_quant_dtype', None)}")
    if qc is not None and any(getattr(qc, k, None) is not None for k in ("w1_bias", "w2_bias", "w1_zp", "w2_zp")):
        raise NotImplementedError("expert biases or zero points")
    if int8 and getattr(qc, "block_shape", None):
        raise NotImplementedError("block-quantized INT8")
    if int8 and apply_router_weight_on_input:
        raise NotImplementedError("INT8 with the routing weight applied to the input")
    dt = _dtype or hidden_states.dtype
    inter = w1.shape[1] // 2
    per_token = bool(int8 and getattr(qc, "per_act_token_quant", False))

    def weight(w, s, e):
        we = w[e].to(dt)
        if not int8:
            return we
        s = s[e].reshape(-1).to(dt)
        if s.numel() == 1:
            return we * s
        if s.numel() == we.shape[0]:
            return we * s.view(-1, 1)
        raise NotImplementedError(f"weight scale of shape {tuple(s.shape)}")

    def scale_of(vs, s):
        """The activation scale for the values vs (a list of tensors quantized together)."""
        if per_token:
            return None
        if s is not None:
            if s.numel() != 1:
                raise NotImplementedError(f"static activation scale of shape {tuple(s.shape)}")
            return s.reshape(()).to(dt)
        return torch.stack([v.abs().amax() for v in vs if v.numel()]).amax().clamp_min(1e-10) / 127

    def quant(v, s):
        if not int8:
            return v
        if s is None:
            s = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127
        return torch.clamp(torch.round(v / s), -128, 127) * s

    a1, a2 = (qc.a1_scale, qc.a2_scale) if int8 else (None, None)
    w1s, w2s = (qc.w1_scale, qc.w2_scale) if int8 else (None, None)
    x = hidden_states.to(dt)
    tw = topk_weights.to(dt)
    ids = topk_ids.long()
    s1 = scale_of([x], a1) if int8 else None
    pairs = []
    for e in torch.unique(ids).tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        xe = x[tok] * tw[tok, slot].unsqueeze(-1) if apply_router_weight_on_input else x[tok]
        h = quant(xe, s1) @ weight(w1, w1s, e).T
        pairs.append((e, tok, slot, torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]))
    s2 = scale_of([p[3] for p in pairs], a2) if int8 and pairs else None
    out = torch.zeros(x.shape[0], w2.shape[1], dtype=dt, device=x.device)
    for e, tok, slot, h in pairs:
        y = quant(h, s2) @ weight(w2, w2s, e).T
        if not apply_router_weight_on_input:
            y = y * tw[tok, slot].unsqueeze(-1)
        out.index_add_(0, tok, y)
    return out.to(hidden_states.dtype)


def w8a8_triton_block_scaled_mm(A, B, As, Bs, block_size, output_dtype=None, _dtype=None):
    """vLLM 0.30's w8a8_triton_block_scaled_mm: A (activations, fp8) with a scale per token and per K group, B
    (weights [N, K], fp8) with a scale per [block_n, block_k] block; the product of the dequantized operands."""
    import torch

    if "e8m0" in str(As.dtype) or "e8m0" in str(Bs.dtype):
        raise NotImplementedError("exponent-only (UE8M0) scales")
    dt = _dtype or torch.float32
    gn, gk = block_size
    K = A.shape[-1]
    a = A.reshape(-1, K).to(dt) * As.reshape(-1, As.shape[-1]).to(dt).repeat_interleave(gk, dim=1)[:, :K]
    b = B.to(dt) * Bs.to(dt).repeat_interleave(gn, dim=0)[: B.shape[0]].repeat_interleave(gk, dim=1)[:, :K]
    return (a @ b.T).reshape(*A.shape[:-1], B.shape[0]).to(output_dtype or torch.float16)


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0, _dtype=None):
    """SGLang 0.5.20's fused_gdn_gating: the Gated DeltaNet gate g = -exp(A_log) * softplus(a + dt_bias) (softplus
    with its beta and linear above the threshold) and beta = sigmoid(b), each [1, tokens, heads] in float32."""
    import torch

    dt = _dtype or torch.float32
    x = beta * (a.to(dt) + dt_bias.to(dt))
    sp = torch.where(x <= threshold, torch.log1p(torch.exp(x)) / beta, x / beta)
    g = -torch.exp(A_log.to(dt)) * sp
    return g.to(torch.float32).unsqueeze(0), torch.sigmoid(b.to(dt)).to(torch.float32).unsqueeze(0)


def prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens, _dtype=None):
    """vLLM 0.30's prepare_pos_seq_lens (the model runner's inputs): batch row r is request state idx_mapping[r],
    its query tokens are [query_start_loc[r], query_start_loc[r+1]); its sequence length is its computed tokens plus
    its query length, and each query token's position is its computed tokens plus its offset in the query. The
    seq_lens rows past the batch are 0 (a full CUDA graph reads them); positions outside the batch's tokens are left."""
    import torch

    n = int(idx_mapping.shape[0])
    qsl = query_start_loc[: n + 1].long()
    q = qsl[1:] - qsl[:-1]
    nct = num_computed_tokens[idx_mapping.long()].long()
    seq_lens[:n] = (nct + q).to(seq_lens.dtype)
    seq_lens[n:] = 0
    start, end = int(qsl[0]), int(qsl[-1])
    if end > start:
        req = torch.repeat_interleave(torch.arange(n, device=pos.device), q)
        tok = torch.arange(start, end, device=pos.device)
        pos[tok] = (nct[req] + (tok - qsl[req])).to(pos.dtype)
    return None


def compute_slot_mappings(self, idx_mapping, query_start_loc, positions, num_tokens_padded, out=None, _dtype=None):
    """vLLM 0.30's BlockTables.compute_slot_mappings: where each token's KV goes, per KV-cache group. A token at
    position p of the request in state row s goes to slot block_table[s, p // kernel_block_size] * kernel_block_size
    + p % kernel_block_size; a group whose slot mapping is disabled gets the pad id for every token, and every slot
    from the batch's last token to the end of the buffer is the pad id (a CUDA graph reads them). Context parallelism
    is not covered."""
    import torch

    pad = -1
    slots = self.slot_mappings if out is None else out
    if self.num_kv_cache_groups == 0:
        return slots[:, :num_tokens_padded]
    if int(getattr(self, "cp_size", 1)) != 1:
        raise NotImplementedError("context parallelism")
    n = int(idx_mapping.shape[0])
    qsl = query_start_loc[: n + 1].long()
    start, end = int(qsl[0]), int(qsl[-1])
    slots[:, end:] = pad
    if end > start:
        req = torch.repeat_interleave(torch.arange(n, device=slots.device), qsl[1:] - qsl[:-1])
        tok = torch.arange(start, end, device=slots.device)
        p = positions[tok].long()
        state = idx_mapping.long()[req]
        for g in range(self.num_kv_cache_groups):
            if not self._slot_mapping_enabled[g]:
                slots[g, tok] = pad
                continue
            kbs = int(self.kernel_block_sizes[g])
            blocks = self.block_tables[g].gpu[state, p // kbs].long()
            slots[g, tok] = (blocks * kbs + p % kbs).to(slots.dtype)
    return slots[:, :num_tokens_padded]


def fresh_slot_mappings(bound):
    """Each comparison run writes its slot mappings into a buffer of its own, a copy of the engine's (the rows before
    the batch keep what they held), never into the engine's persistent buffer."""
    target = bound.get("out")
    if target is None:
        target = bound["self"].slot_mappings
    return {"out": target.clone()}


def probe_experts(args, gen):
    """A warm-up probe's routing: each token's top-k experts drawn without repeats from all the layer's experts
    (w1's first dimension) and weights that sum to one. A warm-up call routes every token alike, to the same k."""
    import torch

    ids, w = args["topk_ids"], args["topk_weights"]
    T, K = int(ids.shape[0]), int(ids.shape[1])
    E = int(args["w1"].shape[0])
    if K > E:
        return {}
    order = torch.rand((T, E), generator=gen, device=ids.device).argsort(dim=1)[:, :K]
    weights = torch.softmax(torch.randn((T, K), generator=gen, device=w.device), dim=-1)
    return {"topk_ids": order.to(ids.dtype), "topk_weights": weights.to(w.dtype)}


def probe_block_scales(args, gen):
    """A warm-up probe's activation scales for the block FP8 matmul: positive, one per token and K group."""
    import torch

    As = args["As"]
    if not As.is_floating_point():
        return {}
    vals = torch.rand(tuple(As.shape), generator=gen, device=As.device, dtype=torch.float32) + 0.5
    return {"As": vals.to(As.dtype)}


DEFINITIONS = (
    Definition("vllm.model_executor.layers.fused_moe.fused_moe:fused_experts", "vllm",
               ("hidden_states", "topk_weights", "topk_ids"), fused_experts, None,
               "per-expert gate/up, SiLU-and-mul, down, routing-weighted sum; INT8 W8A8 by the quant config",
               probe=probe_experts),
    Definition("vllm.model_executor.layers.quantization.utils.fp8_utils:w8a8_triton_block_scaled_mm", "vllm",
               ("A", "As"), w8a8_triton_block_scaled_mm, "bfloat16",
               "the product of the block-dequantized FP8 operands", True, probe=probe_block_scales),
    Definition("sglang.kernels.ops.attention.fla.fused_gdn_gating:fused_gdn_gating", "sglang",
               ("a", "b"), fused_gdn_gating, "bfloat16",
               "g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)", True),
    # M19 L3.3d: chosen by how often normal runs reach them (every model, every step of vLLM 0.30's GPU runner;
    # lowlevel/l2/results/census), and by the kind of meaning L1 found lost most often among these (index and
    # position bases)
    Definition("vllm.v1.worker.gpu.input_batch:prepare_pos_seq_lens", "vllm", ("idx_mapping",),
               prepare_pos_seq_lens, None,
               "positions = computed tokens + offset in the query; seq_lens = computed + query length, 0 past the batch",
               writes=("pos", "seq_lens"), whole=True, exact=True),
    Definition("vllm.v1.worker.gpu.block_table:BlockTables.compute_slot_mappings", "vllm", ("idx_mapping",),
               compute_slot_mappings, None,
               "slot = block_table[state, p // kernel_block] * kernel_block + p % kernel_block; pad after the batch",
               whole=True, fresh=fresh_slot_mappings, exact=True),
)
