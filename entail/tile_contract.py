"""tile_contract: a kernel's tile against the block its values are quantized in, in the core (LIBRARY_DESIGN.md
4.6, 4.7; ROADMAP M15.2; realworld/CODEBOOK_v2.md G; sglang#39626).

A block-quantized matmul carries one scale per quantization block along K. The kernel steps K in tiles and moves
its scale pointer once every (block_k // tile_k) tiles: the tile must be a divisor of the block, or the pointer
steps off the block boundaries and every product past the first block uses the wrong scale, silently (a tile of
64 over a block of 32 gave 64 where 288 was right). The engine's own default config takes tile == block; a tuned
or hand-written config can carry any tile. Along N the kernel finds each column's scale by itself, so only K is
constrained.

The fact is one: the tile the kernel will step K in divides the block the weights are quantized in. An adapter says
where its engine keeps the block (the matmul's block_size) and the tile (the config it will pick); the rule is here:

  tile_over_block  the K tile is larger than the block or does not divide it. The repair is to clamp the tile to
                   the block - the engine's own default, correct by construction (the adapter's `clamp` handle
                   writes it into the config the engine will use). Without a repair it is broken: reported, and the
                   run goes on (M5.4), or refused where the policy stops.

A config that divides is counted (tally.PASSES); the adapter remembers the configs it has checked, so the matmul's
hot path pays a set lookup. An error inside entail never breaks the engine (the adapter runs under load.safely).
"""
from typing import List, Optional

from . import tally as _tally


def divides(block_k: int, tile_k: int) -> bool:
    """Whether a K tile keeps the scale steps on the block boundaries."""
    return 0 < tile_k <= block_k and block_k % tile_k == 0


def check(boundary: str, consumer: str, where: str, block_k: int, tile_k: int, clamp, tile_n: Optional[int] = None,
          owner=None) -> List[object]:
    """Decide one kernel config. `block_k` is the quantization block along K the weights declare (the coarsest tile
    allowed); `tile_k` the tile the kernel will step in; `clamp(t)` writes t as the tile into the config the engine
    will use - the one repair. Returns the decisions (recorded and, if resolved, carried out); [] when it divides."""
    from . import load, policies
    from .contracts import Contract, Resolution, Verdict, decide
    from .facts import Certainty, Fact, KernelConfig, Source

    _tally.counts(boundary)["checks"] += 1
    if divides(int(block_k), int(tile_k)):
        _tally.passed(boundary, ["tile_over_block"])
        _tally.tick(boundary)
        return []
    declared = Fact("KernelConfig", KernelConfig(tile_k=int(block_k)),
                    Source("config", f"{where}: the weights' quantization block along K (the coarsest tile allowed)"),
                    Certainty.DECLARED)
    chosen = Fact("KernelConfig", KernelConfig(tile_k=int(tile_k), tile_n=tile_n),
                  Source("engine", f"{where}: the K tile of the kernel config the engine will use"), Certainty.VERIFIED)
    fix = Resolution("clamp the K tile to the quantization block", "clamp_tile_k", target=lambda d, c: d.value.tile_k)
    contract = Contract(boundary, consumer, ("KernelConfig",))
    decisions = decide(contract, {"KernelConfig": declared}, {"KernelConfig": chosen}, policies.current(),
                       resolutions={"KernelConfig": [fix]})
    done = load.resolve(decisions, {"clamp_tile_k": clamp})
    load.enforce(decisions, once_for=owner)
    if any(d.blocking for d in decisions):
        _tally.refused(boundary)
    elif any(d.verdict is Verdict.BROKEN for d in decisions):
        _tally.broken(boundary)
    elif done:
        _tally.counts(boundary)["resolved"] += 1
    _tally.tick(boundary)
    return decisions


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
