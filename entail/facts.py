"""Role vocabulary v0 (RESEARCH_PLAN.md section 2.2, entail/DESIGN.md section 1.1).

Facts are small frozen dataclasses. A consumer states what it accepts either as one fact (must be equal),
a tuple of facts (closed set: must be one of them) or a predicate (callable returning bool).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

LAYOUT_KINDS = frozenset({"dense", "strided", "q8_0", "fp8_block", "int4_packed"})


@dataclass(frozen=True)
class Layout:
    kind: str
    dtype: Optional[str] = None
    block: Optional[Tuple[int, ...]] = None
    packing: Optional[str] = None       # e.g. "interleaved" | "split" for q8_0
    scale_format: Optional[str] = None  # e.g. "fp32" | "ue8m0"

    def __post_init__(self):
        if self.kind not in LAYOUT_KINDS:
            raise ValueError(f"unknown layout kind {self.kind!r}; closed set is {sorted(LAYOUT_KINDS)}")


@dataclass(frozen=True)
class Positions:
    frame: str                    # "absolute" | "chunk_relative"
    offset: Optional[int] = None  # start of the chunk, for chunk_relative


@dataclass(frozen=True)
class Valid:
    length: Optional[int] = None
    window: Optional[int] = None


@dataclass(frozen=True)
class Reduction:
    state: str                  # spmd_types notation: "R", "P", "S", "I", "V"
    dim: Optional[int] = None   # for "S"
    group: Optional[str] = None


@dataclass(frozen=True)
class Quantized:
    dtype: str                    # e.g. "float8_e4m3fn"
    scale: Optional[float] = None  # per-tensor scale; None when the value is not quantized


@dataclass(frozen=True)
class ModelProps:
    softcap: Optional[float] = None
    sliding_window: Optional[int] = None
    tie_word_embeddings: Optional[bool] = None


@dataclass(frozen=True)
class KernelCaps:
    softcap: bool = False
    sliding_window: bool = False


@dataclass(frozen=True)
class Invalidated:
    """A fact that stopped being true because of an operation, kept instead of being dropped.

    Erasing a fact is what the whole study is about, so propagation never erases: when an operation changes what
    a fact means (a transpose changes which axis is which), the output carries this marker, and the next
    boundary that wants that kind of fact fails with the reason instead of finding nothing.
    """
    kind: str   # the fact class name that stopped being true, e.g. "Layout"
    why: str    # the operation that did it, e.g. "aten.transpose"
