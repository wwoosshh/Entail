"""Adapter v2 for SGLang's block-FP8 Triton matmul: the K tile of the kernel config against the quantization block
(LIBRARY_DESIGN.md 4.8; ROADMAP M15.2; realworld/CODEBOOK_v2.md G; sglang#39626).

  hook         sglang.kernels.ops.quantization.fp8_kernel.w8a8_block_fp8_matmul_triton (0.5.20; the older path
               sglang.srt.layers.quantization.fp8_kernel is tried too): the matmul that looks its config up
               (get_w8a8_block_fp8_configs, lru-cached per (N, K, block_n, block_k)) and launches the kernel with
               the weights' block_size. The shipped tuned configs all divide (1,887 entries swept on 0.5.20); a
               hand-written or externally tuned config can carry a K tile over the block, and the engine's own
               sanitiser only clamps tiles that are too small.
  read_choice  the block along K (block_size[1]) and, for every entry of the config map the matmul will pick from,
               the entry's BLOCK_SIZE_K (and BLOCK_SIZE_N). A map is checked once: the matmul's hot path then pays a
               set lookup on the map's identity.
  handles      clamp_tile_k: write the block into the offending entry's BLOCK_SIZE_K in place - the map is the
               engine's cached object, so the matmul picks the clamped config from then on.
tile_contract decides (tile_over_block).
"""
from .. import core, tile_contract
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "load:sglang.fp8_block_kernel_config"
CONSUMER = "sglang.kernel.w8a8_block_fp8_matmul"
_ORIG = None
_MODULE = None
_CHECKED = set()   # id(config map) already decided in this process


def hooks():
    return [Hook("sglang.kernels.ops.quantization.fp8_kernel.w8a8_block_fp8_matmul_triton", "load")]


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
    if id(configs) in _CHECKED:
        return
    read = read_choice(configs, block_size)
    if read is None:
        return
    block_k, rows = read
    for key, tile_k, tile_n in rows:
        tile_contract.check(BOUNDARY, CONSUMER, f"sglang block-fp8 matmul {shape}, config for M={key}", block_k,
                            tile_k, handles(configs, key)["clamp_tile_k"], tile_n=tile_n, owner=configs)
    _CHECKED.add(id(configs))


def install():
    """Wrap the matmul. Returns 1, or 0 if already installed."""
    global _ORIG, _MODULE
    if _ORIG is not None:
        return 0
    _MODULE = _module()
    _ORIG = _MODULE.w8a8_block_fp8_matmul_triton

    def w8a8_block_fp8_matmul_triton(A, B, As, Bs, block_size, *a, **kw):
        if core.mode() in ("load", "debug"):
            from .. import load

            def work():
                configs = _MODULE.get_w8a8_block_fp8_configs(int(B.shape[0]), int(A.shape[-1]), int(block_size[0]),
                                                             int(block_size[1]))
                _decide(configs, block_size, f"N={int(B.shape[0])},K={int(A.shape[-1])}")
            load.safely(BOUNDARY, CONSUMER, "KernelConfig", work)
        return _ORIG(A, B, As, Bs, block_size, *a, **kw)

    _MODULE.w8a8_block_fp8_matmul_triton = w8a8_block_fp8_matmul_triton
    return 1


def uninstall():
    global _ORIG
    if _ORIG is None:
        return 0
    _MODULE.w8a8_block_fp8_matmul_triton = _ORIG
    _ORIG = None
    _CHECKED.clear()
    return 1


def stats():
    return tile_contract.stats(BOUNDARY)


def reset():
    _CHECKED.clear()
    tile_contract.reset(BOUNDARY)
