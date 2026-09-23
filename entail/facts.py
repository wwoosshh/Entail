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


PREDICTION_KINDS = frozenset({"eps", "v", "x0", "flow", "edm"})


@dataclass(frozen=True)
class Prediction:
    """What a diffusion model's network predicts (PROPERTY), and whether its schedule reaches zero terminal SNR.

    `kind` is a closed set. `zsnr` None means the source does not say; comparisons then use the kind alone."""
    kind: str
    zsnr: Optional[bool] = None

    def __post_init__(self):
        if self.kind not in PREDICTION_KINDS:
            raise ValueError(f"unknown prediction kind {self.kind!r}; closed set is {sorted(PREDICTION_KINDS)}")

    def __str__(self):
        name = {"eps": "eps", "v": "v-prediction", "x0": "x0", "flow": "flow", "edm": "EDM"}[self.kind]
        return name + {True: " with zero terminal SNR", False: " without zero terminal SNR", None: ""}[self.zsnr]


@dataclass(frozen=True)
class Base:
    """The model family an artifact was made for: a checkpoint's architecture, the base a LoRA was trained on."""
    family: str

    def __str__(self):
        return self.family


@dataclass(frozen=True)
class Invalidated:
    """A fact that stopped being true because of an operation, kept instead of being dropped.

    Erasing a fact is what the whole study is about, so propagation never erases: when an operation changes what
    a fact means (a transpose changes which axis is which), the output carries this marker, and the next
    boundary that wants that kind of fact fails with the reason instead of finding nothing.
    """
    kind: str   # the fact class name that stopped being true, e.g. "Layout"
    why: str    # the operation that did it, e.g. "aten.transpose"
