"""Adapter v2 for SGLang's block-FP8 Triton kernels: the K tile of a kernel config against the quantization block
(LIBRARY_DESIGN.md 4.8; ROADMAP M15.2; realworld/CODEBOOK_v2.md G; sglang#39626).

  hooks        sglang.kernels.ops.quantization.fp8_kernel.w8a8_block_fp8_matmul_triton (0.5.20; the older path
               sglang.srt.layers.quantization.fp8_kernel is tried too): the dense matmul that looks its config up
               (get_w8a8_block_fp8_configs, lru-cached per (N, K, block_n, block_k)) and launches with the weights'
               block_size. The engine's own sanitiser only clamps tiles that are too small.
               sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.try_get_optimal_moe_config:
               the fused-MoE config lookup (shipped files, SGLANG_MOE_CONFIG_DIR, or an override), which has no
               sanitiser at all; the shipped H100 file for E=512, N=256, block [128, 128] carries BLOCK_SIZE_K=256
               for M=64..512 (found in the M15.2 review).
  read_choice  the block along K (block_size[1]) and, for every entry of a config map, the entry's BLOCK_SIZE_K
               (and BLOCK_SIZE_N). A map (or a MoE config) is decided once, by identity; the hot path pays a dict
               lookup and runs nothing while a CUDA graph is captured or torch is compiling (the getter returns
               None then).
  handles      clamp_tile_k: write the block into the offending entry's BLOCK_SIZE_K in place - the engine's own
               cached object, so it picks the clamped config from then on.
tile_contract decides (tile_over_block).
"""
from .. import core, tile_contract
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "load:sglang.fp8_block_kernel_config"
CONSUMER = "sglang.kernel.w8a8_block_fp8_matmul"
MOE_BOUNDARY = "load:sglang.fused_moe_kernel_config"
MOE_CONSUMER = "sglang.kernel.fused_moe_triton"
_ORIG = None
_MODULE = None
_ORIG_MOE = None
_MOE_MODULE = None
_CHECKED = {}   # id(config map or config) -> the object (held, so an id is never reused by another)


def hooks():
    return [Hook("sglang.kernels.ops.quantization.fp8_kernel.w8a8_block_fp8_matmul_triton", "load"),
            Hook("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.try_get_optimal_moe_config",
                 "load")]


def _module():
    import importlib

    for name in ("sglang.kernels.ops.quantization.fp8_kernel", "sglang.srt.layers.quantization.fp8_kernel"):
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    raise ImportError("sglang: no fp8_kernel module")


def read_choice(configs, block_size):
    """(block_k, [(m_key, BLOCK_SIZE_K, BLOCK_SIZE_N)]) for a config map; None when the engine takes its default
    (tile == block, correct by construction) or the map is not a map."""
    if not isinstance(configs, dict) or not configs:
        return None
    block_k = int(block_size[1])
    rows = []
    for key, cfg in configs.items():
        if isinstance(cfg, dict) and "BLOCK_SIZE_K" in cfg:
            rows.append((key, int(cfg["BLOCK_SIZE_K"]), cfg.get("BLOCK_SIZE_N")))
    return block_k, rows


def handles(configs, key):
    def clamp_tile_k(tile):
        configs[key]["BLOCK_SIZE_K"] = int(tile)
        return tile
    return {"clamp_tile_k": clamp_tile_k}


def _decide(configs, block_size, shape):
    """Decide every entry of one dense config map (once per map)."""
    read = read_choice(configs, block_size)
    _CHECKED[id(configs)] = configs
    if read is None:
        return
    block_k, rows = read
    for key, tile_k, tile_n in rows:
        tile_contract.check(BOUNDARY, CONSUMER, f"sglang block-fp8 matmul {shape}, config for M={key}", block_k,
                            tile_k, handles(configs, key)["clamp_tile_k"], tile_n=tile_n)


def _content_key(cfg, block_k):
    """A config by its content (the engine copies the down config on every call: `down_config = dict(**...)`, so
    identity would decide it every time and hold every copy; M15.4 review). The block is part of the key."""
    return ("moe", block_k) + tuple(sorted((k, v) for k, v in cfg.items() if isinstance(v, (int, float, str, bool))))


def _decide_moe(config, down_config, block_shape, shape):
    """Decide the up and down configs the fused-MoE lookup returned for one call: a content is decided (and
    recorded) once; a later copy with the same content is clamped again without a new record, since the engine's
    copy is what reaches the kernel (recorded once, repaired every time - enforce's once_for, by content)."""
    block_k = int(block_shape[1])
    for label, cfg in (("up", config), ("down", down_config)):
        if not isinstance(cfg, dict) or "BLOCK_SIZE_K" not in cfg:
            continue
        key = _content_key(cfg, block_k)
        if key in _CHECKED:
            if _CHECKED[key] is not None:   # a content decided as broken, and how it was repaired: repair this copy
                cfg["BLOCK_SIZE_K"] = _CHECKED[key]
            continue
        if len(_CHECKED) > 4096:   # bounded: the maps are finite; a runaway of copies must not grow it
            _CHECKED.clear()
        before = int(cfg["BLOCK_SIZE_K"])
        tile_contract.check(MOE_BOUNDARY, MOE_CONSUMER, f"sglang fused-moe {shape}, {label} config", block_k,
                            before, lambda t, c=cfg: c.__setitem__("BLOCK_SIZE_K", int(t)) or t,
                            tile_n=cfg.get("BLOCK_SIZE_N"))
        after = int(cfg["BLOCK_SIZE_K"])
        _CHECKED[key] = after if after != before else None        # what a copy of this content must be given
        _CHECKED[_content_key(cfg, block_k)] = None                # the content as it is now passes


def _active():
    return core.mode() in ("load", "debug") and not tile_contract.inside_capture()


def install():
    """Wrap the dense matmul. Returns 1, or 0 if already installed."""
    global _ORIG, _MODULE
    if _ORIG is not None:
        return 0
    _MODULE = _module()
    _ORIG = _MODULE.w8a8_block_fp8_matmul_triton

    def w8a8_block_fp8_matmul_triton(A, B, As, Bs, block_size, *a, **kw):
        if _active():
            try:
                configs = _MODULE.get_w8a8_block_fp8_configs(int(B.shape[0]), int(A.shape[-1]),
                                                             int(block_size[0]), int(block_size[1]))
            except Exception:  # noqa: BLE001 - the lookup is the engine's; a failure there is not ours to raise
                configs = None
            if isinstance(configs, dict) and id(configs) not in _CHECKED:
                from .. import load

                shape = f"N={int(B.shape[0])},K={int(A.shape[-1])}"
                load.safely(BOUNDARY, CONSUMER, "KernelConfig", lambda: _decide(configs, block_size, shape))
        return _ORIG(A, B, As, Bs, block_size, *a, **kw)

    _MODULE.w8a8_block_fp8_matmul_triton = w8a8_block_fp8_matmul_triton
    return 1


def install_moe():
    """Wrap the fused-MoE config lookup; the kernel module binds the name at its import, right after this module
    has run. Returns 1, or 0 if already installed."""
    global _ORIG_MOE, _MOE_MODULE
    if _ORIG_MOE is not None:
        return 0
    import importlib

    _MOE_MODULE = importlib.import_module("sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config")
    _ORIG_MOE = _MOE_MODULE.try_get_optimal_moe_config

    def try_get_optimal_moe_config(w1_shape, w2_shape, top_k, dtype, M, *a, **kw):
        out = _ORIG_MOE(w1_shape, w2_shape, top_k, dtype, M, *a, **kw)
        block_shape = kw.get("block_shape") if "block_shape" in kw else (a[1] if len(a) > 1 else None)
        if _active() and block_shape and len(block_shape) == 2 and block_shape[1]:
            config = out[0] if isinstance(out, tuple) else out
            down = out[1][0] if isinstance(out, tuple) and isinstance(out[1], tuple) else None
            bk = int(block_shape[1])
            # a content decided as passing (None) needs nothing; one not seen, or decided as broken, goes in
            # (the engine's copy of a broken down config must be repaired each time)
            fresh = [c for c in (config, down) if isinstance(c, dict) and "BLOCK_SIZE_K" in c
                     and _CHECKED.get(_content_key(c, bk), 0) is not None]
            if fresh:
                from .. import load

                shape = f"E={w2_shape[0]},N={w2_shape[2]},M={M}"
                load.safely(MOE_BOUNDARY, MOE_CONSUMER, "KernelConfig",
                            lambda: _decide_moe(config, down, block_shape, shape))
        return out

    _MOE_MODULE.try_get_optimal_moe_config = try_get_optimal_moe_config
    return 1


def uninstall():
    global _ORIG, _ORIG_MOE
    n = 0
    if _ORIG is not None:
        _MODULE.w8a8_block_fp8_matmul_triton = _ORIG
        _ORIG = None
        n += 1
    if _ORIG_MOE is not None:
        _MOE_MODULE.try_get_optimal_moe_config = _ORIG_MOE
        _ORIG_MOE = None
        n += 1
    _CHECKED.clear()
    return n


def stats():
    return {"dense": tile_contract.stats(BOUNDARY), "moe": tile_contract.stats(MOE_BOUNDARY)}


def reset():
    _CHECKED.clear()
    tile_contract.reset(BOUNDARY)
    tile_contract.reset(MOE_BOUNDARY)
