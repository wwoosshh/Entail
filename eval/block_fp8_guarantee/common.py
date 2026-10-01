"""Shared pieces of the block FP8 guarantee evaluation (ROADMAP M19 L5.4a; docs/semantic-guarantee-gpu-protocol.ko.md).

  oracle       the judge: an independent float64 implementation on the CPU (numpy). It decodes the fp8 bytes by their
               bit fields, applies the producer's block scales and multiplies - no entail, no vLLM, no torch
               arithmetic. Its tolerance: one ulp of the output dtype at the true value plus C_ORACLE x (|a| @ |b|^T),
               C_ORACLE fixed by the normal calibration before the freeze.
  harness      the real vLLM path: a TritonFp8BlockScaledMMKernel made by vLLM's own init_fp8_linear_kernel, a layer
               holding a checkpoint-style fp8 weight and weight_scale_inv, process_weights_after_loading (the weight
               producer), apply_weights (QuantFP8 -> per_token_group_quant_fp8, the activation producer; the custom
               op -> w8a8_triton_block_scaled_mm, the consumer).
  mutations    consumer defects put into the consumer itself: a launcher object takes the place of vLLM's Triton
               kernel object (fp8_utils._w8a8_triton_block_scaled_mm, looked up by the launcher at each call), so
               vLLM's launcher, its config choice and entail's wrapper all stay as they are, and it launches vLLM's
               own kernel source with one misread patched in, switched on by a flag in device memory (so a defect
               can come on at a CUDA graph replay). The other weight's scale reused by a consumer cache is a Python
               change of the call site (apply_block_scaled_mm), the way such a cache would sit.
  observer     the next operation: it reads the returned tensor (a full copy is kept, and a small matmul consumes
               it); inside a graph the copy is captured right after the consumer, so it is what the graph's next
               operation actually read.
"""
import hashlib
import importlib
import inspect
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENTAIL_ROOT = os.path.dirname(os.path.dirname(HERE))
BLOCK = (128, 128)
C_ORACLE_DEFAULT = 2.0 ** -12     # replaced by the frozen value (freeze manifest "oracle.c")
if ENTAIL_ROOT not in sys.path:
    sys.path.insert(0, ENTAIL_ROOT)


# --- the oracle (no torch arithmetic) ---------------------------------------------------------------------------------

def fp8_e4m3fn(u8):
    """float64 values of float8_e4m3fn bytes: sign 1 bit, exponent 4 (bias 7), mantissa 3; exponent 0 is subnormal
    (m/8 x 2^-6); 0x7F/0xFF are NaN; there is no infinity."""
    u = np.asarray(u8, dtype=np.uint8).astype(np.int64)
    s, e, m = (u >> 7) & 1, (u >> 3) & 0xF, u & 7
    mag = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1.0 + m / 8.0) * np.power(2.0, e - 7.0))
    mag = np.where((e == 15) & (m == 7), np.nan, mag)
    return np.where(s == 1, -mag, mag)


def bf16_to_f64(u16):
    """float64 values of bfloat16 bit patterns (the high half of a float32)."""
    u = np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).astype(np.float64)


def ulp_out(t, dtype):
    """One unit in the last place of the output dtype at |t| (float64)."""
    p, tiny = {"bfloat16": (8, 2.0 ** -133), "float16": (11, 2.0 ** -24)}[dtype]
    a = np.abs(t)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(a > 0, a, 1.0)))
    return np.where(a > 0, np.maximum(np.power(2.0, e + 1 - p), tiny), tiny)


def oracle(A_u8, As, B_u8, Bs, block=BLOCK):
    """(truth, bound) in float64: the product of the block-dequantised operands as the producers made them, and
    |a| @ |b|^T. A_u8 [M, K] and B_u8 [N, K] are fp8 bytes; As [M, K/gk] and Bs [ceil(N/gn), ceil(K/gk)] float."""
    gn, gk = block
    a = fp8_e4m3fn(A_u8)
    b = fp8_e4m3fn(B_u8)
    M, K = a.shape
    N = b.shape[0]
    sa = np.repeat(np.asarray(As, dtype=np.float64), gk, axis=1)[:, :K]
    sb = np.repeat(np.repeat(np.asarray(Bs, dtype=np.float64), gn, axis=0)[:N], gk, axis=1)[:, :K]
    a, b = a * sa, b * sb
    return a @ b.T, np.abs(a) @ np.abs(b).T


def judge(out, truth, bound, dtype, c):
    """The oracle's verdict on a delivered output (float64 array): (ok, violations, worst ratio, nonfinite)."""
    tol = ulp_out(truth, dtype) + c * bound
    d = np.abs(out - truth)
    fin = np.isfinite(out)
    bad = (~fin) | (d > tol)
    ratio = np.where(fin, d / np.maximum(tol, 1e-300), np.inf)
    return (not bool(bad.any())), int(bad.sum()), float(ratio.max()) if ratio.size else 0.0, int((~fin).sum())


# --- environment ------------------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def dist_record_hash(name):
    """sha256 of an installed distribution's RECORD (it lists every installed file with its own hash)."""
    from importlib import metadata

    try:
        d = metadata.distribution(name)
        rec = d.read_text("RECORD")
        return {"version": d.version, "record_sha256": sha256_text(rec or ""),
                "direct_url": d.read_text("direct_url.json")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def environment():
    import platform
    import subprocess

    import torch

    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update({"gpu": p.name, "cc": f"{p.major}.{p.minor}", "vram": int(p.total_memory),
                    "sm_count": p.multi_processor_count})
    try:
        env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                                        "--format=csv,noheader"], capture_output=True, text=True,
                                       timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        env["driver"] = f"unavailable: {e}"
    for name in ("vllm", "triton", "torch", "numpy", "transformers", "entail-ai"):
        env[f"dist_{name}"] = dist_record_hash(name)
    env["precision"] = {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                        "float32_matmul_precision": torch.get_float32_matmul_precision(),
                        "allow_bf16_reduced": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    env["entail_git"] = git_head(ENTAIL_ROOT)
    env["VLLM_DISABLED_KERNELS"] = os.environ.get("VLLM_DISABLED_KERNELS")
    return env


def git_head(path):
    import subprocess

    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return {"head": head, "dirty": bool(dirty)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


# --- the vLLM path ----------------------------------------------------------------------------------------------------

def vllm_context():
    """A current vLLM config in which QuantFP8 runs its CUDA path (forward_cuda -> per_token_group_quant_fp8), as in
    an eager engine run."""
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    try:
        from vllm.config import CompilationMode
        cc = CompilationConfig(mode=CompilationMode.NONE, custom_ops=["all"])
    except Exception:  # noqa: BLE001
        cc = CompilationConfig(level=0, custom_ops=["all"])
    return set_current_vllm_config(VllmConfig(compilation_config=cc))


def make_kernel(N, K, out_dtype, block=BLOCK):
    """vLLM's own choice of the block FP8 kernel, forced to the Triton one (as VLLM_DISABLED_KERNELS does in the
    engine on this GPU); returns (kernel, the class vLLM would choose unforced)."""
    import torch
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFp8BlockScaledMMKernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

    akey = create_fp8_quant_key(static=False, group_shape=GroupShape(1, block[1]))
    wkey = create_fp8_quant_key(static=True, group_shape=GroupShape(*block))
    kw = dict(activation_quant_key=akey, weight_quant_key=wkey, input_dtype=out_dtype, out_dtype=out_dtype,
              weight_shape=(N, K))
    try:
        unforced = type(init_fp8_linear_kernel(**kw)).__name__
    except Exception as e:  # noqa: BLE001
        unforced = f"error: {type(e).__name__}: {e}"
    k = init_fp8_linear_kernel(**kw, force_kernel=TritonFp8BlockScaledMMKernel)
    assert type(k).__name__ == "TritonFp8BlockScaledMMKernel", type(k)
    return k, unforced


def make_weight(N, K, seed, block=BLOCK, spread=1.5, device="cuda"):
    """A checkpoint-style block FP8 weight: values with per-block magnitudes that differ (lognormal, `spread`), the
    block scale max|w| / 448, the fp8 codes. Returns (weight fp8 [N, K], weight_scale_inv float32 [N/gn, K/gk])."""
    import torch

    gn, gk = block
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    w = torch.randn((N, K), generator=gen, dtype=torch.float32)
    nb, kb = -(-N // gn), -(-K // gk)
    mag = torch.exp(torch.randn((nb, kb), generator=gen) * spread)
    w = w * mag.repeat_interleave(gn, 0)[:N].repeat_interleave(gk, 1)[:, :K]
    s = torch.zeros((nb, kb))
    q = torch.empty((N, K), dtype=torch.float8_e4m3fn)
    for i in range(nb):
        for j in range(kb):
            blk = w[i * gn:(i + 1) * gn, j * gk:(j + 1) * gk]
            sc = float(blk.abs().max()) / 448.0 or 1e-10
            s[i, j] = sc
            q[i * gn:(i + 1) * gn, j * gk:(j + 1) * gk] = (blk / sc).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.to(device), s.to(device)


class Layer:
    """What vLLM's LinearBase leaves for the kernel: parameters `weight` and `weight_scale_inv`, a prefix."""

    def __new__(cls, weight, scale, prefix):
        import torch

        m = torch.nn.Module()
        m.weight = torch.nn.Parameter(weight, requires_grad=False)
        m.weight_scale_inv = torch.nn.Parameter(scale, requires_grad=False)
        m.input_scale = None
        m.prefix = prefix
        return m


def make_input(M, K, seed, dist="normal", dtype=None, device="cuda"):
    import torch

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    if dist == "normal":
        x = torch.randn((M, K), generator=gen)
    elif dist == "heavy":                      # heavy tails and a few outlier channels, as activations have
        x = torch.randn((M, K), generator=gen) / torch.rand((M, K), generator=gen).clamp_min(0.05).sqrt()
        ch = torch.randperm(K, generator=gen)[:max(1, K // 64)]
        x[:, ch] *= 40.0
    elif dist == "small":
        x = torch.randn((M, K), generator=gen) * 1e-3
    else:
        raise ValueError(dist)
    return x.to(dtype or torch.bfloat16).to(device)


# --- consumer mutations -----------------------------------------------------------------------------------------------

MUTANTS = ("interval", "neighbor", "alt_weight", "rows")


def _patched_source(kind):
    """vLLM's own kernel source with one misread in it, active when the device flag Mut is non-zero."""
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    src = inspect.getsource(fp8_utils._w8a8_triton_block_scaled_mm.fn)
    src = src[src.index("def "):]
    src = src.replace("def _w8a8_triton_block_scaled_mm(", f"def _mut_{kind}(", 1)
    assert "    Bs,\n" in src
    src = src.replace("    Bs,\n", "    Bs,\n    Mut,\n    BsAlt,\n    ROWS_FROM,\n", 1)
    head = "    pid = tl.program_id(axis=0)\n"
    assert head in src
    src = src.replace(head, head + "    mut = tl.load(Mut)\n", 1)
    if kind == "interval":              # the scale of K block k read at interval 2 x group_k
        old = "        offs_ks = k_start // group_k\n"
        assert old in src
        src = src.replace(old, "        offs_ks = tl.where(mut != 0, (k_start // (group_k * 2)), k_start // group_k)\n")
    elif kind == "neighbor":            # the weight scale of the next N block
        old = "    offs_bsn = offs_bn // group_n\n"
        assert old in src
        src = src.replace(old, "    offs_bsn = tl.where(mut != 0, tl.minimum(offs_bn // group_n + 1, "
                               "(N - 1) // group_n), offs_bn // group_n)\n")
    elif kind == "alt_weight":          # another weight's block scale (same shape) read in place of this one's
        old = "        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)\n"
        assert old in src
        src = src.replace(old, "        b_s = tl.where(mut != 0, tl.load(BsAlt + offs_bsn * stride_Bs_n + offs_ks * "
                               "stride_Bs_k), tl.load(Bs_ptrs + offs_ks * stride_Bs_k))\n")
    elif kind == "rows":                # rows from ROWS_FROM on read the activation scale of the next K group
        old = "        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)\n"
        assert old in src
        src = src.replace(old, "        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)\n"
                               "        a_s_next = tl.load(As_ptrs + tl.minimum(offs_ks + 1, (K - 1) // group_k) * "
                               "stride_As_k)\n"
                               "        a_s = tl.where((mut != 0) & (offs_am >= ROWS_FROM), a_s_next, a_s)\n")
    else:
        raise ValueError(kind)
    return "import triton\nimport triton.language as tl\n\n\n@triton.jit\n" + src


class Launcher:
    """Stands where fp8_utils._w8a8_triton_block_scaled_mm stands: `launcher[grid](*args, **config)`. It counts the
    consumer kernel's launches and, for a mutant, launches the patched kernel with the flag and the extra operands."""

    def __init__(self, original, kind=None, device="cuda"):
        import torch

        self.original, self.kind = original, kind
        self.flag = torch.zeros(1, dtype=torch.int32, device=device)
        self.alt = None
        self.rows_from = 64
        self.launches = 0
        self.kernel = None
        if kind is not None:
            folder = tempfile.mkdtemp(prefix="entail_mut_")
            path = os.path.join(folder, f"mut_{kind}.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write(_patched_source(kind))
            sys.path.insert(0, folder)
            mod = importlib.import_module(f"mut_{kind}")
            self.kernel = getattr(mod, f"_mut_{kind}")
            self.source_sha256 = sha256_file(path)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches += 1
            if self.kernel is None:
                return self.original[grid](*args, **kwargs)
            A, B, C, As, Bs = args[:5]
            alt = self.alt if self.alt is not None else Bs
            return self.kernel[grid](A, B, C, As, Bs, self.flag, alt, self.rows_from, *args[5:], **kwargs)
        return launch


def install_launcher(kind=None):
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    orig = fp8_utils._w8a8_triton_block_scaled_mm
    if isinstance(orig, Launcher):
        orig = orig.original
    lau = Launcher(orig, kind)
    fp8_utils._w8a8_triton_block_scaled_mm = lau
    return lau


class ScaleCache:
    """A consumer-side cache of weight scales keyed by shape, at the call site (apply_block_scaled_mm): the first
    scale stored for a shape is handed to every weight of that shape. `armed` turns it on."""

    def __init__(self):
        self.store, self.armed, self.hits = {}, False, 0

    def install(self):
        from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFp8BlockScaledMMKernel

        orig = TritonFp8BlockScaledMMKernel.apply_block_scaled_mm
        cache = self

        def apply_block_scaled_mm(self, A, B, As, Bs):
            key = tuple(Bs.shape)
            if cache.armed:
                if key in cache.store and cache.store[key] is not Bs:
                    cache.hits += 1
                    Bs = cache.store[key]
            return orig(self, A, B, As, Bs)

        TritonFp8BlockScaledMMKernel.apply_block_scaled_mm = apply_block_scaled_mm
        return self


def tile_config_case():
    """The known tile defect (M19 L3 notes, lowlevel/l2/RESULTS.md): a tuned table that gives BLOCK_SIZE_K 256 at
    larger batches, past the 128-wide scale groups the kernel reads once per K tile."""
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    table = {1: {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "num_warps": 4,
                 "num_stages": 2},
             256: {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 8, "num_warps": 4,
                   "num_stages": 2}}

    def get_w8a8_block_fp8_configs(N, K, block_n, block_k):
        return table

    fp8_utils.get_w8a8_block_fp8_configs = get_w8a8_block_fp8_configs
    return table


# --- observation ------------------------------------------------------------------------------------------------------

def rng_state():
    """Digests of the Python, NumPy, torch CPU and CUDA generators' states (read without drawing)."""
    import random

    import torch

    st = {"python": sha256_text(repr(random.getstate())),
          "numpy": sha256_text(repr(np.random.get_state()[1].tolist())),
          "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
    if torch.cuda.is_available():
        st["cuda"] = hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    return st


def to_np(t):
    """A host numpy copy that keeps the exact values: fp8 as bytes, bf16 as bit patterns, others as they are."""
    import torch

    t = t.detach().cpu()
    if t.dtype == torch.float8_e4m3fn:
        return t.view(torch.uint8).numpy().copy()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16).copy()
    return t.numpy().copy()


def out_f64(arr, dtype):
    return bf16_to_f64(arr) if dtype == "bfloat16" else np.asarray(arr, dtype=np.float64)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)


def now():
    return time.perf_counter()
