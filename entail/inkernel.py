"""inkernel: the consumer kernel itself checks, at the moment it reads them, that its operands are the bytes their
producers issued (ROADMAP M19 L5.4e, approach B: inside the operation; ENTAIL=structure_inkernel).

The structural check (kernel_ir) proves what the consumer kernel computes from its operands; it cannot see whether
the operands still hold what the producers made when the kernel reads them (the integrity cases of L5.4c-L5.4e: a
write between the producer and the consumer). Here the check is put into the consumer kernel: entail rewrites the
kernel's own source so that in each step of its K loop, beside the tiles of A and B it loads for the dot (and the
scales As and Bs), it also reads the same addresses again as 32-bit words and adds them up, each weighted by its
position, and at the end atomically adds its sums into a small buffer. The same sums are made right after the
producers ran (`issue_sums`: the activation's at every issue - vLLM's quantizer is a CUDA C++ op whose code cannot be
rewritten - the weight's once, when it is issued at load). A small kernel compares the two on the device; on a
difference the device stops (torch._assert_async) before the next operation, which comes after it in stream order.

Why a second read: the tiles the dot reads are copied asynchronously into shared memory (Triton's software pipeline);
using them for anything else takes them out of that path and slows the matmul by up to 1.9x (measured, 2026-10-03).
The second read is of the same bytes in the same step of the loop, mostly from the cache the first one filled.

The sum of a tensor in its logical [R, W] order of 32-bit words (an fp8 [R, C] tensor is [R, C / 4] words) is
    S = sum over (r, w) of word(r, w) * hr(r) * hc(w)   modulo 2^32
hr(x) = mix(x, 0x9E3779B1, 0x85EBCA77) | 1, hc(x) = mix(x, 0x7FEB352D, 0x846CA68B) | 1, mix(x, p, q) = (x * p) ^
((x * q) >> 16) on 32 bits. Every weight is odd, so a change of one word always changes S; the weights are mixed, so
changes at several places do not cancel by their arrangement (one in 2^32 by chance). In the kernel a tile's sum is
hr(row) * (sum over w of word * hc(w)); a row or column the program does not count gets the weight 0, and only one
program per row of A (pid_n = 0) and per column of B (pid_m = 0) counts. Inside the kernel every derived value starts
from a number loaded from the buffer (`_ez`, zero at run time): to kernel_ir it is data, so the check's arithmetic is
not evaluated with the kernel's addresses, and the only write it adds is the atomic add into the buffer, which
kernel_ir allows for a tensor bound as "integrity_sums".

The rewrite is anchored on the lines of vLLM 0.30's _w8a8_triton_block_scaled_mm (its signature's BLOCK_SIZE_M, the
program ids, the line that adds the scaled dot into the accumulator, the conversion after the loop); the harness's
misreading variants keep those lines. A kernel without the anchors is not rewritten: the gate then hands on the
reference's output, as for a launch that is not proven.
"""
import hashlib
import importlib.util
import inspect
import os
import tempfile
import threading

HR = (0x9E3779B1, 0x85EBCA77)
HC = (0x7FEB352D, 0x846CA68B)
SLOTS = 8                  # int32 per check buffer: 0 A, 1 B, 2 As, 3 Bs as the kernel read them; 4 zero (the seed)
ISSUE_SLOTS = 2            # int32 per issue: 0 values, 1 scales
EXTRA = ("EntailChk", "EntailAw", "EntailBw")   # the rewritten kernel's added parameters

_LOCK = threading.Lock()
_REWRITTEN = {}            # id(JITFunction) -> (the JITFunction, the rewritten JITFunction or None, why not)
_FOLDER = None

_SIG = "    BLOCK_SIZE_M: tl.constexpr,\n"
_PIDS = "    pid_n = (pid % num_pid_in_group) // group_size_m\n"
_TERM = "        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]\n"
_AFTER = "    if C.dtype.element_ty == tl.bfloat16:\n"


def _mix(x, h):
    return f"((({x}) * {h[0]}) ^ ((({x}) * {h[1]}) >> 16)) | 1"


_SETUP = f"""\
    _ez = tl.load(EntailChk + 4)
    _kw = K // 4
    _er = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    _ec = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    _xr = (_er + _ez).to(tl.uint32)
    _xc = (_ec + _ez).to(tl.uint32)
    _hra = tl.where(_er < M, {_mix("_xr", HR)}, 0)
    _hrb = tl.where(_ec < N, {_mix("_xc", HR)}, 0)
    _xn = ((_ec + _ez) // group_n).to(tl.uint32)
    _hrbs = tl.where((_ec < N) & (_ec % group_n == 0), {_mix("_xn", HR)}, 0)
    _vsa = _hra * 0
    _vsb = _hrb * 0
    _ssas = (_ez * 0).to(tl.uint32)
    _ssbs = (_ez * 0).to(tl.uint32)
"""

_LOADS = f"""\
        _wk = k * (BLOCK_SIZE_K // 4) + tl.arange(0, BLOCK_SIZE_K // 4)
        _xk = (_wk + _ez).to(tl.uint32)
        _hk = tl.where(_wk < _kw, {_mix("_xk", HC)}, 0)
        _xs = (offs_ks + _ez).to(tl.uint32)
        _hks = tl.where((k * BLOCK_SIZE_K) % group_k == 0, {_mix("_xs", HC)}, 0)
        if pid_n + _ez == 0:
            _aw = tl.load(EntailAw + (_er[:, None] * _kw + _wk[None, :]),
                          mask=(_er[:, None] + _ez < M) & (_wk[None, :] < _kw), other=0)
            _vsa += tl.sum(_aw.to(tl.uint32, bitcast=True) * _hk[None, :], axis=1)
            _ssas += tl.sum(a_s.to(tl.uint32, bitcast=True) * _hra, axis=0) * _hks
        if pid_m + _ez == 0:
            _bw = tl.load(EntailBw + (_ec[None, :] * _kw + _wk[:, None]),
                          mask=(_wk[:, None] + _ez < _kw) & (_ec[None, :] < N), other=0)
            _vsb += tl.sum(_bw.to(tl.uint32, bitcast=True) * _hk[:, None], axis=0)
            _ssbs += tl.sum(b_s.to(tl.uint32, bitcast=True) * _hrbs, axis=0) * _hks
"""

_STORE = """\
    tl.atomic_add(EntailChk + 0, tl.sum(_vsa * _hra, axis=0).to(tl.int32, bitcast=True), sem="relaxed")
    tl.atomic_add(EntailChk + 1, tl.sum(_vsb * _hrb, axis=0).to(tl.int32, bitcast=True), sem="relaxed")
    tl.atomic_add(EntailChk + 2, _ssas.to(tl.int32, bitcast=True), sem="relaxed")
    tl.atomic_add(EntailChk + 3, _ssbs.to(tl.int32, bitcast=True), sem="relaxed")
"""


def rewrite_source(src: str):
    """The kernel's source with the check put in, or (None, why) when an anchor is missing."""
    src = src[src.index("@triton.jit"):] if "@triton.jit" in src else src
    for anchor, what in ((_SIG, "the signature's BLOCK_SIZE_M"), (_PIDS, "the program ids"),
                         (_TERM, "the scaled dot added to the accumulator"), (_AFTER, "the conversion after the loop")):
        if src.count(anchor) != 1:
            return None, f"{what} is not where vLLM 0.30's kernel has it ({src.count(anchor)} matches)"
    out = src.replace(_SIG, "".join(f"    {p},\n" for p in EXTRA) + _SIG, 1)
    out = out.replace(_PIDS, _PIDS + _SETUP, 1)
    out = out.replace(_TERM, _TERM + _LOADS, 1)     # after the dot: its loads are issued first
    out = out.replace(_AFTER, _STORE + _AFTER, 1)
    return out, None


def rewritten(fn):
    """(the rewritten JITFunction, None) for a Triton kernel `fn`, or (None, why). Cached per kernel object."""
    global _FOLDER
    hit = _REWRITTEN.get(id(fn))
    if hit is not None and hit[0] is fn:
        return hit[1], hit[2]
    with _LOCK:
        try:
            src = inspect.getsource(fn.fn)
        except (OSError, TypeError, AttributeError) as e:
            _REWRITTEN[id(fn)] = (fn, None, f"the kernel's source could not be read ({type(e).__name__})")
            return None, _REWRITTEN[id(fn)][2]
        new, why = rewrite_source(src)
        if new is None:
            _REWRITTEN[id(fn)] = (fn, None, why)
            return None, why
        text = "import triton\nimport triton.language as tl\n\n\n" + new
        digest = hashlib.sha256(text.encode()).hexdigest()
        name = f"entail_inkernel_{digest[:16]}"
        if _FOLDER is None:
            _FOLDER = tempfile.mkdtemp(prefix="entail_inkernel_")
        path = os.path.join(_FOLDER, f"{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        k = getattr(mod, fn.fn.__name__, None)
        if k is None:
            _REWRITTEN[id(fn)] = (fn, None, "the rewritten module has no kernel of that name")
            return None, _REWRITTEN[id(fn)][2]
        k.entail_source_sha256 = digest
        _REWRITTEN[id(fn)] = (fn, k, None)
        return k, None


def words(t):
    """t's bytes as int32 words in its logical row order ([R, C] fp8 -> [R, C / 4]); t must be contiguous with a last
    dimension that is a multiple of 4 bytes."""
    x = t.reshape(-1, t.shape[-1]) if t.dim() != 2 else t
    return x.view(__import__("torch").int32)


# --- the sums made outside the consumer, and the comparison ---------------------------------------------------------

_KERNELS = {}


def _kernels():
    if _KERNELS:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def entail_issue_sums(X, RX, WX, sx0, Y, RY, CY, sy0, sy1, Out, NPX, BLOCK: tl.constexpr):
        # programs [0, NPX) add up X (int32 words, rows of WX) into Out[0]; the others Y (float32) into Out[1]
        pid = tl.program_id(0)
        if pid < NPX:
            i = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            m = i < RX * WX
            r, c = i // WX, i % WX
            v = tl.load(X + r * sx0 + c, mask=m, other=0).to(tl.uint32, bitcast=True)
        else:
            i = (pid - NPX).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            m = i < RY * CY
            r, c = i // CY, i % CY
            v = tl.load(Y + r * sy0 + c * sy1, mask=m, other=0.0).to(tl.uint32, bitcast=True)
        ru, cu = r.to(tl.uint32), c.to(tl.uint32)
        hr = ((ru * 0x9E3779B1) ^ ((ru * 0x85EBCA77) >> 16)) | 1
        hc = ((cu * 0x7FEB352D) ^ ((cu * 0x846CA68B) >> 16)) | 1
        s = tl.sum(tl.where(m, v * hr * hc, 0), axis=0).to(tl.int32, bitcast=True)
        tl.atomic_add(Out + (pid >= NPX).to(tl.int32), s, sem="relaxed")

    @triton.jit
    def entail_compare(Seen, Act, W, Flags, Host):
        # Flags[0]: a bit per operand whose sums differ (1 A, 2 As, 4 B, 8 Bs); Flags[1]: 1 when none differs. The
        # same two numbers go to Host, pinned host memory written from the device (readable after a stop too)
        bits = (tl.load(Seen + 0) != tl.load(Act + 0)).to(tl.int32) \
            | ((tl.load(Seen + 2) != tl.load(Act + 1)).to(tl.int32) << 1) \
            | ((tl.load(Seen + 1) != tl.load(W + 0)).to(tl.int32) << 2) \
            | ((tl.load(Seen + 3) != tl.load(W + 1)).to(tl.int32) << 3)
        ok = (bits == 0).to(tl.int32)
        tl.store(Flags + 0, bits)
        tl.store(Flags + 1, ok)
        tl.store(Host + 0, bits)
        tl.store(Host + 1, ok)

    _KERNELS["issue"] = entail_issue_sums
    _KERNELS["compare"] = entail_compare
    return _KERNELS


BLOCK = 4096


def buffer(device):
    """One zeroed int32 buffer for an activation's issue and its consumer: [0, SLOTS) the consumer kernel's sums (slot 4
    stays zero: the seed), [SLOTS, SLOTS + ISSUE_SLOTS) the issue's sums. One fill for both."""
    import torch

    return torch.zeros(SLOTS + ISSUE_SLOTS, dtype=torch.int32, device=device)


def issue_sums(values, scales, out=None):
    """S of a value tensor (fp8, contiguous, read as int32 words in its logical row order) and of its scale tensor
    (float32, 2-D, any strides), into a new int32 [ISSUE_SLOTS] device tensor (or `out`, which must hold zeros).
    No synchronisation; can be captured."""
    import torch
    import triton

    k = _kernels()["issue"]
    x = words(values)
    y = scales.reshape(-1, scales.shape[-1]) if scales.dim() != 2 else scales
    if out is None:
        out = torch.zeros(ISSUE_SLOTS, dtype=torch.int32, device=values.device)
    npx = triton.cdiv(x.numel(), BLOCK)
    npy = triton.cdiv(y.numel(), BLOCK)
    k[(npx + npy,)](x, x.shape[0], x.shape[1], x.stride(0), y, y.shape[0], y.shape[1], y.stride(0), y.stride(1),
                    out, npx, BLOCK=BLOCK)
    return out


def compare(seen, act, w, flags, host):
    """Compare on the device; the flags go to `flags` (device) and `host` (pinned host memory, written directly)."""
    _kernels()["compare"][(1,)](seen, act, w, flags, host)
    return flags
