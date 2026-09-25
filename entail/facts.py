"""Fact vocabulary v4 and the fact envelope (LIBRARY_DESIGN.md 4.1 and 6; ROADMAP M1.1, M4.2, M5.3, M9.3).

A fact class is a small frozen dataclass. Every field that names a kind of thing takes its value from a closed set,
and a value outside the set is an error (principle 1): a new kind of layout, prediction or rope type is added here,
with a vocabulary version bump, never passed through silently. Numeric fields are checked for type and range.

The classes keep the names and fields they had in v0, so existing code keeps working; what v1 added is the checks,
six new classes (Rotary, LatentScale, Template, Epoch, Assumed, Origin), and the envelope `Fact`, which says where a
fact came from and how sure the library is of it.

v2 (M4.2) adds two optional fields to Layout, for the weights a loader repacks: `orientation` (which axis holds the
output features) and `scale_granularity` (how many values one scale covers). v3 (M5.3) adds Template.tool_call_format,
the format a model writes its tool calls in, under names that do not belong to any engine (each engine's parsers are
mapped to them in data/caps.json). v4 (M9.3) adds Rotary.low_freq_factor and high_freq_factor, llama3's scaling: the
readers named them as not carried, and nothing said so while the model ran (M9.1, S1). v5 (M14) adds one class,
Identity: what a stored or cached item stands for, so a store keyed by identity (vLLM's prefix-cache block hashes)
does not serve one sequence's KV under another's key (codebook v2 I; vllm#49377, #49449). v6 (M15) adds TokenType:
the token type id a position is given by its role (padding), so a server that pads a cross-encoder's input does not
give the padding the last real token's segment (vllm#58138; codebook v2 G). v7 (M15.8) adds Stops: the ids at
which a generation ends (and begins, and is padded), declared in up to three files that each engine reads a
different subset of (Llama 3, April 2024: config.json named one end, the model emitted another). Each version only
adds optional fields or whole classes, so an older fact is a newer fact with
them open, and a fact written with an older version is still read (READABLE_VERSIONS); it may not state a field its
version did not have (ADDED_IN).

A consumer states what it accepts either as one fact (must be equal), a tuple of facts (closed set: must be one of
them) or a predicate (callable returning bool); see core.boundary.
"""
from dataclasses import dataclass, fields
from enum import Enum
from typing import Optional, Tuple

VOCAB_VERSION = 7
READABLE_VERSIONS = frozenset({1, 2, 3, 4, 5, 6, 7})   # a later version only adds optional fields or classes; fields in ADDED_IN
ADDED_IN = {("Layout", "orientation"): 2, ("Layout", "scale_granularity"): 2, ("Template", "tool_call_format"): 3,
            ("Rotary", "low_freq_factor"): 4, ("Rotary", "high_freq_factor"): 4,
            ("Rotary", "beta_fast"): 6, ("Rotary", "beta_slow"): 6, ("Rotary", "attention_factor"): 6,
            ("Rotary", "mscale"): 6, ("Rotary", "mscale_all_dim"): 6, ("Rotary", "truncate"): 6,
            ("Rotary", "long_factor_sha256"): 6, ("Rotary", "short_factor_sha256"): 6, ("Rotary", "factor_terms"): 6,
            ("Rotary", "local_theta"): 6, ("Rotary", "partial_rotary_factor"): 6, ("Rotary", "local_factor"): 6,
            ("Rotary", "mrope_section"): 6, ("Rotary", "mrope_interleaved"): 6}


def _closed(cls_name, field, value, allowed, optional=True):
    if value is None and optional:
        return
    if value not in allowed:
        raise ValueError(f"{cls_name}.{field}: unknown value {value!r}; closed set is {sorted(allowed)}")


def _number(cls_name, field, value, minimum, integer=False, strict=False):
    """None, or an int (or float unless `integer`) >= minimum (> minimum when `strict`). Booleans are not numbers."""
    if value is None:
        return
    kinds = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds):
        raise ValueError(f"{cls_name}.{field}: expected {'an int' if integer else 'a number'}, got {value!r}")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{cls_name}.{field}: expected {'>' if strict else '>='} {minimum}, got {value!r}")


# --- LAYOUT ---------------------------------------------------------------------------------------------------

LAYOUT_KINDS = frozenset({"dense", "strided", "q8_0", "fp8_block", "int4_packed"})
LAYOUT_PACKINGS = frozenset({"interleaved", "split"})
SCALE_FORMATS = frozenset({"fp32", "bf16", "fp16", "ue8m0", "e4m3"})
DTYPES = frozenset({"float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2", "int8", "uint8", "int4",
                    "uint4"})
# v2: a weight matrix is (out_features, in_features) - what torch.nn.functional.linear reads - or the transpose.
ORIENTATIONS = frozenset({"out_in", "in_out"})
# v2: how many values one scale covers. "unscaled": the stored values are used as they are.
SCALE_GRANULARITIES = frozenset({"unscaled", "per_tensor", "per_channel", "per_group", "per_block"})


@dataclass(frozen=True)
class Layout:
    """How a value is stored: its kind, packing and scale format; for a weight matrix, which axis is which and how
    many values share a scale (v2)."""
    kind: str
    dtype: Optional[str] = None
    block: Optional[Tuple[int, ...]] = None
    packing: Optional[str] = None       # "interleaved" | "split" (q8_0)
    scale_format: Optional[str] = None  # "fp32" | "ue8m0" | ...
    orientation: Optional[str] = None        # v2: "out_in" | "in_out"
    scale_granularity: Optional[str] = None  # v2: "unscaled" | "per_tensor" | "per_channel" | ...

    def __post_init__(self):
        if self.kind not in LAYOUT_KINDS:
            raise ValueError(f"unknown layout kind {self.kind!r}; closed set is {sorted(LAYOUT_KINDS)}")
        _closed("Layout", "dtype", self.dtype, DTYPES)
        _closed("Layout", "packing", self.packing, LAYOUT_PACKINGS)
        _closed("Layout", "scale_format", self.scale_format, SCALE_FORMATS)
        _closed("Layout", "orientation", self.orientation, ORIENTATIONS)
        _closed("Layout", "scale_granularity", self.scale_granularity, SCALE_GRANULARITIES)
        if self.block is not None and not (isinstance(self.block, tuple) and self.block
                                           and all(isinstance(b, int) and not isinstance(b, bool) and b > 0
                                                   for b in self.block)):
            raise ValueError(f"Layout.block: expected a tuple of positive ints, got {self.block!r}")


# --- DTYPE ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Quantized:
    """The dtype a value is stored in, and the per-tensor scale that turns it back (None when not quantized)."""
    dtype: str
    scale: Optional[float] = None

    def __post_init__(self):
        _closed("Quantized", "dtype", self.dtype, DTYPES, optional=False)
        _number("Quantized", "scale", self.scale, 0, strict=True)


# --- FRAME ----------------------------------------------------------------------------------------------------

POSITION_FRAMES = frozenset({"absolute", "chunk_relative"})
ROPE_TYPES = frozenset({"default", "linear", "dynamic", "yarn", "longrope", "llama3", "proportional"})
# proportional (v6, M15.7 sweep: the Gemma 4 family, 5 of 230): transformers derives the frequencies from the head
# dimension and partial_rotary_factor, so those (and the base) are what an engine can lose
# mrope (v6, M15.7 sweep: 27 of 230, the Qwen-VL and Qwen3.5 families) is NOT a type name here: transformers 5
# normalises the old spelling {"type": "mrope"} to rope_type "default" in the object it holds and vLLM knows no
# scaling type "mrope" (M15.7 review), so the fact of mrope is its section - the rotary dimensions split among
# time, height and width - and whether the split interleaves (Rotary.mrope_section, mrope_interleaved)


@dataclass(frozen=True)
class Positions:
    frame: str                    # "absolute" | "chunk_relative"
    offset: Optional[int] = None  # start of the chunk, for chunk_relative

    def __post_init__(self):
        _closed("Positions", "frame", self.frame, POSITION_FRAMES, optional=False)
        _number("Positions", "offset", self.offset, 0, integer=True)


@dataclass(frozen=True)
class Rotary:
    """The rotary position embedding a model was trained with: its type, base and scaling. low_freq_factor and
    high_freq_factor are llama3's (v4): the wavelengths it leaves alone and the ones it scales by `factor`."""
    rope_type: str = "default"
    theta: Optional[float] = None
    factor: Optional[float] = None
    original_max_position: Optional[int] = None
    low_freq_factor: Optional[float] = None
    high_freq_factor: Optional[float] = None
    # v6 (M15.4): yarn's tuning (gpt-oss), longrope's per-dimension factors (Phi-3.5/Phi-4, kept as a digest and
    # their count, since a list of 48-64 floats is compared, not read), and a second base for the local layers of a
    # model that alternates two RoPEs (Gemma 3's rope_local_base_freq)
    beta_fast: Optional[float] = None
    beta_slow: Optional[float] = None
    attention_factor: Optional[float] = None
    mscale: Optional[float] = None
    mscale_all_dim: Optional[float] = None
    truncate: Optional[bool] = None
    long_factor_sha256: Optional[str] = None
    short_factor_sha256: Optional[str] = None
    factor_terms: Optional[int] = None
    local_theta: Optional[float] = None
    # v6 (M15.4 review): the share of each head's dimensions that rotate (Phi-4-mini 0.75; a config top-level key),
    # and the scaling factor of the local layers when a model alternates two RoPEs (None: the local layers scale
    # nothing, which is what Gemma 3 declares; an engine that applies the global factor to them is caught)
    partial_rotary_factor: Optional[float] = None
    local_factor: Optional[float] = None
    mrope_section: Optional[Tuple[int, ...]] = None
    mrope_interleaved: Optional[bool] = None

    def __post_init__(self):
        _closed("Rotary", "rope_type", self.rope_type, ROPE_TYPES, optional=False)
        if self.mrope_section is not None:
            ok = isinstance(self.mrope_section, tuple) and self.mrope_section and all(
                isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in self.mrope_section)
            if not ok:
                raise ValueError(f"Rotary.mrope_section: expected a tuple of positive ints, got {self.mrope_section!r}")
        if self.mrope_interleaved is not None and not isinstance(self.mrope_interleaved, bool):
            raise ValueError(f"Rotary.mrope_interleaved: expected a bool, got {self.mrope_interleaved!r}")
        _number("Rotary", "theta", self.theta, 0, strict=True)
        _number("Rotary", "factor", self.factor, 0, strict=True)
        _number("Rotary", "original_max_position", self.original_max_position, 0, integer=True, strict=True)
        _number("Rotary", "low_freq_factor", self.low_freq_factor, 0, strict=True)
        _number("Rotary", "high_freq_factor", self.high_freq_factor, 0, strict=True)
        _number("Rotary", "beta_fast", self.beta_fast, 0)
        _number("Rotary", "beta_slow", self.beta_slow, 0)
        _number("Rotary", "attention_factor", self.attention_factor, 0)
        _number("Rotary", "mscale", self.mscale, 0)
        _number("Rotary", "mscale_all_dim", self.mscale_all_dim, 0)
        if self.truncate is not None and not isinstance(self.truncate, bool):
            raise ValueError(f"Rotary.truncate: expected a bool, got {self.truncate!r}")
        for name in ("long_factor_sha256", "short_factor_sha256"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, str) or len(v) != 64):
                raise ValueError(f"Rotary.{name}: expected a sha256 hex digest, got {v!r}")
        _number("Rotary", "factor_terms", self.factor_terms, 0, integer=True, strict=True)
        _number("Rotary", "local_theta", self.local_theta, 0, strict=True)
        _number("Rotary", "partial_rotary_factor", self.partial_rotary_factor, 0, strict=True)
        if self.partial_rotary_factor is not None and self.partial_rotary_factor > 1:
            raise ValueError(f"Rotary.partial_rotary_factor: expected <= 1, got {self.partial_rotary_factor!r}")
        _number("Rotary", "local_factor", self.local_factor, 0, strict=True)


# --- RANGE (KvExtent lives in kv_contract.py) -------------------------------------------------------------------

@dataclass(frozen=True)
class Valid:
    length: Optional[int] = None
    window: Optional[int] = None

    def __post_init__(self):
        _number("Valid", "length", self.length, 0, integer=True)
        _number("Valid", "window", self.window, 0, integer=True, strict=True)


# --- PROPERTY -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelProps:
    """Properties a model requires of whatever runs it."""
    softcap: Optional[float] = None
    sliding_window: Optional[int] = None
    tie_word_embeddings: Optional[bool] = None

    def __post_init__(self):
        _number("ModelProps", "softcap", self.softcap, 0, strict=True)
        _number("ModelProps", "sliding_window", self.sliding_window, 0, integer=True)
        if self.tie_word_embeddings is not None and not isinstance(self.tie_word_embeddings, bool):
            raise ValueError(f"ModelProps.tie_word_embeddings: expected a bool, got {self.tie_word_embeddings!r}")


PREDICTION_KINDS = frozenset({"eps", "v", "x0", "flow", "edm"})


@dataclass(frozen=True)
class Prediction:
    """What a diffusion model's network predicts, and whether its schedule reaches zero terminal SNR.

    `kind` is a closed set. `zsnr` None means the source does not say; comparisons then use the kind alone."""
    kind: str
    zsnr: Optional[bool] = None

    def __post_init__(self):
        if self.kind not in PREDICTION_KINDS:
            raise ValueError(f"unknown prediction kind {self.kind!r}; closed set is {sorted(PREDICTION_KINDS)}")
        if self.zsnr is not None and not isinstance(self.zsnr, bool):
            raise ValueError(f"Prediction.zsnr: expected a bool, got {self.zsnr!r}")

    def __str__(self):
        name = {"eps": "eps", "v": "v-prediction", "x0": "x0", "flow": "flow", "edm": "EDM"}[self.kind]
        return name + {True: " with zero terminal SNR", False: " without zero terminal SNR", None: ""}[self.zsnr]


@dataclass(frozen=True)
class LatentScale:
    """How a VAE's latents are scaled (and shifted) between the encoder, the sampler and the decoder."""
    scale: float
    shift: Optional[float] = None

    def __post_init__(self):
        if self.scale is None:
            raise ValueError("LatentScale.scale: required")
        _number("LatentScale", "scale", self.scale, 0, strict=True)
        if self.shift is not None and (isinstance(self.shift, bool) or not isinstance(self.shift, (int, float))):
            raise ValueError(f"LatentScale.shift: expected a number, got {self.shift!r}")


REASONING_HISTORY = frozenset({"keep", "drop"})
# v3: the formats a model writes tool calls in, named apart from any engine. Only formats whose parsers were checked
# in both vLLM and SGLang are here (data/caps.json names the parsers): "hermes" is <tool_call>{json}</tool_call>,
# "llama3_json" a JSON call after an optional <|python_tag|>, "pythonic" a list of Python calls.
TOOL_CALL_FORMATS = frozenset({"hermes", "llama3_json", "pythonic"})


@dataclass(frozen=True)
class Template:
    """What a request must look like for this model: its chat template, whether earlier reasoning is sent back, and
    (v3) the format it writes tool calls in."""
    chat_template_sha256: Optional[str] = None
    reasoning_history: Optional[str] = None
    tool_call_format: Optional[str] = None   # v3

    def __post_init__(self):
        h = self.chat_template_sha256
        if h is not None and not (isinstance(h, str) and len(h) == 64 and all(c in "0123456789abcdef" for c in h)):
            raise ValueError(f"Template.chat_template_sha256: expected 64 lowercase hex digits, got {h!r}")
        _closed("Template", "reasoning_history", self.reasoning_history, REASONING_HISTORY)
        _closed("Template", "tool_call_format", self.tool_call_format, TOOL_CALL_FORMATS)


# --- REDUCTION ------------------------------------------------------------------------------------------------

REDUCTION_STATES = frozenset({"R", "P", "S", "I", "V"})


@dataclass(frozen=True)
class Reduction:
    state: str                  # spmd_types notation: "R", "P", "S", "I", "V"
    dim: Optional[int] = None   # for "S"
    group: Optional[str] = None

    def __post_init__(self):
        _closed("Reduction", "state", self.state, REDUCTION_STATES, optional=False)
        _number("Reduction", "dim", self.dim, 0, integer=True)
        if self.state == "S" and self.dim is None:
            raise ValueError("Reduction.dim: a sharded value ('S') must say which dim it is sharded on")


# --- TIME, SPECIALIZATION, PRECEDENCE -------------------------------------------------------------------------

@dataclass(frozen=True)
class Epoch:
    """Which version of a buffer's contents a value was made from (bumped whenever the buffer is rewritten)."""
    version: int
    owner: Optional[str] = None

    def __post_init__(self):
        _number("Epoch", "version", self.version, 0, integer=True)
        if self.version is None:
            raise ValueError("Epoch.version: required")


IDENTITY_OF = frozenset({"kv_block"})


@dataclass(frozen=True)
class Identity:
    """What a stored or cached item stands for, so a later reader knows it is still the same thing (v5, M14).

    A store keyed by identity - vLLM's prefix-cache block hashes - serves KV under this key. If the tokens the key
    stands for change and the key does not, a later request whose prefix matches the OLD tokens is served the NEW
    tokens' KV, silently (vllm#49377, #49449). The fact makes the key comparable with the identity the item's
    current contents give.

    of      the kind of item whose identity this is (closed set: kv_block)
    index   which one (the block's position in the sequence)
    key     the identity the engine holds for it, as a hex string
    covers  how many tokens that identity stands for (informative; None when not known)
    """
    of: str
    index: int
    key: str
    covers: Optional[int] = None

    def __post_init__(self):
        _closed("Identity", "of", self.of, IDENTITY_OF, optional=False)
        _number("Identity", "index", self.index, 0, integer=True)
        _number("Identity", "covers", self.covers, 0, integer=True)
        if not isinstance(self.key, str) or not self.key:
            raise ValueError(f"Identity.key: expected a non-empty string, got {self.key!r}")


@dataclass(frozen=True)
class Assumed:
    """The conditions a compiled artifact was specialised for, as sorted (name, value) pairs."""
    conditions: Tuple[Tuple[str, object], ...]

    def __post_init__(self):
        ok = isinstance(self.conditions, tuple) and all(
            isinstance(c, tuple) and len(c) == 2 and isinstance(c[0], str) for c in self.conditions)
        if not ok:
            raise ValueError(f"Assumed.conditions: expected a tuple of (name, value) pairs, got {self.conditions!r}")
        if list(self.conditions) != sorted(self.conditions, key=lambda c: c[0]):
            raise ValueError("Assumed.conditions: pairs must be sorted by name, so equal assumptions compare equal")


ORIGINS = frozenset({"default", "checkpoint", "config", "manifest", "user"})


@dataclass(frozen=True)
class Origin:
    """Where the value a consumer uses for one setting came from."""
    setting: str
    came_from: str

    def __post_init__(self):
        if not isinstance(self.setting, str) or not self.setting:
            raise ValueError(f"Origin.setting: expected a setting name, got {self.setting!r}")
        _closed("Origin", "came_from", self.came_from, ORIGINS, optional=False)


@dataclass(frozen=True)
class KernelConfig:
    """The tile a kernel steps a dimension in, against the block the values are quantized in (v6, M15.2; codebook
    v2 G). A block-quantized matmul applies one scale per quantization block along K. A kernel whose K tile is not
    a divisor of that block steps its scale pointer off the block boundaries and multiplies by the wrong scale,
    silently (sglang#39626: a tile of 64 over a block of 32 gave 64 where 288 was right). The declared value is the
    block (the coarsest tile allowed); the chosen value is the kernel's tile. Along N the kernel indexes scales per
    column, so only K is constrained.
    """
    tile_k: int
    tile_n: Optional[int] = None

    def __post_init__(self):
        _number("KernelConfig", "tile_k", self.tile_k, 0, integer=True, strict=True)
        _number("KernelConfig", "tile_n", self.tile_n, 0, integer=True, strict=True)
        if self.tile_k is None:
            raise ValueError("KernelConfig.tile_k: required")


@dataclass(frozen=True)
class Vocab:
    """The vocabulary a tokenizer holds: how many base tokens (v6, M15.3; codebook v2 G).

    A model folder can carry two tokenizers (vocab.txt with 100,000 entries and a tokenizer.json with 32,000, from
    another model) and the engine picks one by its own precedence; the model's vocabulary is what its config and its
    embedding rows say. A tokenizer that is not the model's produces ids that mean other tokens, silently
    (transformers#48967). `size` is the base vocabulary (without added tokens); `added` the added tokens, when known.
    """
    size: int
    added: Optional[int] = None

    def __post_init__(self):
        _number("Vocab", "size", self.size, 0, integer=True, strict=True)
        _number("Vocab", "added", self.added, 0, integer=True)
        if self.size is None:
            raise ValueError("Vocab.size: required")


TOKEN_ROLES = frozenset({"pad"})


@dataclass(frozen=True)
class TokenType:
    """The token type (segment) id a position is given by its role (v6, M15; codebook v2 G).

    A cross-encoder reads a segment id per position: 0 for the query, 1 for the document. The tokenizer declares
    which id its padding carries (`pad_token_type_id`, 0 for BERT tokenizers). A server that pads the input itself
    and gives the padding the last real token's id moves the query/document boundary and changes the scores
    (vllm#58138). role is a closed set; type_id is the id positions of that role are given.
    """
    role: str
    type_id: int

    def __post_init__(self):
        _closed("TokenType", "role", self.role, TOKEN_ROLES, optional=False)
        _number("TokenType", "type_id", self.type_id, 0, integer=True)
        if self.type_id is None:
            raise ValueError("TokenType.type_id: required")


@dataclass(frozen=True)
class Stops:
    """Where a generation ends (v7, M15.8): the token ids a model emits to end its output (`eos`), and the ids it
    begins with (`bos`) and pads with (`pad`), as ONE source states them. The same meaning is declared in up to three
    files - generation_config.json, config.json and the tokenizer's eos_token - and every engine builds its stop set
    from a different subset (data/stops_sources.json). The contract takes the union: an id any file calls an end is
    an end, and a consumer whose set lacks it runs past the end of an answer (stops_contract.py)."""
    eos: Tuple[int, ...] = ()
    bos: Optional[int] = None
    pad: Optional[int] = None

    def __post_init__(self):
        if not isinstance(self.eos, tuple) or any(isinstance(i, bool) or not isinstance(i, int) or i < 0
                                                  for i in self.eos):
            raise ValueError(f"Stops.eos: expected a tuple of token ids (ints >= 0), got {self.eos!r}")
        _number("Stops", "bos", self.bos, 0, integer=True)
        _number("Stops", "pad", self.pad, 0, integer=True)
        # an empty eos is a value: a consumer whose set holds no end (a generation config without eos_token_id;
        # M15.8 E2: tiny-random-Llama's) - the readers emit a fact only when a file states something


# --- not in the vocabulary ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelCaps:
    """What a kernel honours. A capability, not a fact about a value; moves to caps.py in M3.1."""
    softcap: bool = False
    sliding_window: bool = False


@dataclass(frozen=True)
class Base:
    """The model family an artifact was made for. Used by the ComfyUI adapter; not in vocabulary v1 yet, because the
    family names are not a closed set (decided with the sources in M2 and the image work in M6)."""
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
    kind: str       # the fact class name that stopped being true, e.g. "Layout"
    why: str        # the operation that did it, e.g. "transpose"
    was: str = ""   # where the fact came from before it stopped being true (its source), when known (M7.3)


# --- the vocabulary and the envelope --------------------------------------------------------------------------

# The ten fact kinds (RESEARCH_PLAN.md 2.2), and which kind each vocabulary name belongs to (LIBRARY_DESIGN.md 6).
FACT_KINDS = frozenset({"LAYOUT", "DTYPE", "FRAME", "RANGE", "PROPERTY", "MAPPING", "REDUCTION", "TIME",
                        "SPECIALIZATION", "PRECEDENCE"})
VOCABULARY = {
    "Layout": "LAYOUT", "Quantized": "DTYPE", "Rotary": "FRAME", "Positions": "FRAME", "Valid": "RANGE",
    "KvExtent": "RANGE", "ModelProps": "PROPERTY", "Prediction": "PROPERTY", "LatentScale": "PROPERTY",
    "Template": "PROPERTY", "Coverage": "MAPPING", "TokenType": "MAPPING", "Reduction": "REDUCTION", "Epoch": "TIME",
    "Identity": "TIME", "Assumed": "SPECIALIZATION", "Origin": "PRECEDENCE", "KernelConfig": "LAYOUT",
    "Vocab": "MAPPING", "Stops": "MAPPING",
}
_HERE = {"Layout": Layout, "Quantized": Quantized, "Rotary": Rotary, "Positions": Positions, "Valid": Valid,
         "ModelProps": ModelProps, "Prediction": Prediction, "LatentScale": LatentScale, "Template": Template,
         "TokenType": TokenType, "Reduction": Reduction, "Epoch": Epoch, "Identity": Identity, "Assumed": Assumed,
         "Origin": Origin, "KernelConfig": KernelConfig, "Vocab": Vocab, "Stops": Stops}


def vocabulary_class(name):
    """The class for a vocabulary name. KvExtent and Coverage live next to their rules and are imported on demand."""
    if name in _HERE:
        return _HERE[name]
    if name == "KvExtent":
        from .kv_contract import KvExtent
        return KvExtent
    if name == "Coverage":
        from .coverage import Coverage
        return Coverage
    raise ValueError(f"unknown fact name {name!r}; vocabulary v{VOCAB_VERSION} has {sorted(VOCABULARY)}")


class Certainty(str, Enum):
    """How sure the library is of a fact (LIBRARY_DESIGN.md principle 3)."""
    DECLARED = "declared"    # an artifact, a pinned manifest, a code boundary or the user states it
    VERIFIED = "verified"    # declared or read from the data, and checked against the data (bytes, strides, keys)
    INFERRED = "inferred"    # derived without a declaration (a probe, a key pattern); never the only basis for a change
    DEFAULTED = "defaulted"  # a default filled it in; recorded so it is never silent
    UNKNOWN = "unknown"      # nobody says; reported, and for meaning-changing facts not replaced by a default


# "engine": what an engine chose, read by an adapter. "data": what the data itself shows (bytes, strides, keys).
SOURCE_KINDS = frozenset({"user", "manifest", "boundary", "file", "config", "probe", "default", "engine", "data"})


@dataclass(frozen=True)
class Source:
    """Where a fact came from: a kind from SOURCE_KINDS and a precise address, e.g.
    Source("file", "model.safetensors#__metadata__.modelspec.prediction_type")."""
    kind: str
    where: str

    def __post_init__(self):
        _closed("Source", "kind", self.kind, SOURCE_KINDS, optional=False)
        if not isinstance(self.where, str) or not self.where:
            raise ValueError(f"Source.where: expected an address, got {self.where!r}")

    def __str__(self):
        return f"{self.kind}: {self.where}"


@dataclass(frozen=True)
class Fact:
    """One fact about a value: which vocabulary name, its value, where it came from, and how sure it is.

    `value` is an instance of the vocabulary class called `name` (e.g. Prediction("v")), or None exactly when the
    certainty is UNKNOWN. The kind (LAYOUT, PROPERTY ...) is VOCABULARY[name].
    """
    name: str
    value: Optional[object]
    source: Source
    certainty: Certainty
    vocab_version: int = VOCAB_VERSION

    def __post_init__(self):
        if self.name not in VOCABULARY:
            raise ValueError(f"unknown fact name {self.name!r}; vocabulary v{VOCAB_VERSION} has {sorted(VOCABULARY)}")
        if self.vocab_version not in READABLE_VERSIONS:
            raise ValueError(f"fact {self.name} was written with vocabulary v{self.vocab_version}; "
                             f"this library reads v{', v'.join(str(v) for v in sorted(READABLE_VERSIONS))}")
        for (name, field), version in ADDED_IN.items():
            if name == self.name and self.vocab_version < version and getattr(self.value, field, None) is not None:
                raise ValueError(f"fact {self.name} was written with vocabulary v{self.vocab_version}, which has no "
                                 f"{name}.{field} (added in v{version})")
        if not isinstance(self.certainty, Certainty):
            raise ValueError(f"Fact.certainty: expected a Certainty, got {self.certainty!r}")
        if not isinstance(self.source, Source):
            raise ValueError(f"Fact.source: expected a Source, got {self.source!r}")
        if (self.value is None) != (self.certainty is Certainty.UNKNOWN):
            raise ValueError(f"fact {self.name}: a fact has no value exactly when its certainty is unknown "
                             f"(value {self.value!r}, certainty {self.certainty.value})")
        cls = vocabulary_class(self.name)
        if self.value is not None and not isinstance(self.value, cls):
            raise ValueError(f"fact {self.name}: holds a {type(self.value).__name__}, expected {cls.__name__}")

    @property
    def kind(self):
        return VOCABULARY[self.name]


def unconstrained_fields(value):
    """Fields of a fact value that are None, i.e. that the value does not say anything about."""
    return {f.name for f in fields(value) if getattr(value, f.name) is None}
