"""observe: what the data itself shows, read deterministically (LIBRARY_DESIGN.md principle 5).

A declaration is checked against the data before anything trusts it: tensor names, dtypes, strides and bytes, never
a behaviour probe. What is claimed is a Fact with Source kind "data" and certainty VERIFIED; when the data does not
settle the question nothing is claimed. From the files, only safetensors headers and a few element bytes are read;
no tensor is loaded and torch is not needed:

  head(path)           the checkpoint's lm_head.weight against its embedding: absent, differs, or the same in every
                       byte (a stored copy of a tied head; M11.2)
  tie(path)            ModelProps.tie_word_embeddings: no lm_head.weight means the head is the embedding; a head
                       whose shape, dtype or sampled bytes differ from the embedding is not tied
  scale_format(path)   Layout.scale_format of block-quantized weights: the dtype of their scale tensors

A value already in memory - a weight a loader has written (M4.2) - is read from the tensor it is given:
  weight_layout(...)   Layout of a weight matrix: dtype, dense or strided, which axis holds the output features (from
                       the layer's own sizes), and how many values one scale covers (from its scale tensor); with
                       what it could not read, so that is reported rather than passed
  sample_values(...)   a few values spread over a tensor, to compare after a step that declares where it moves them
  moved(...)           which of those values are not where the declared move puts them
These only index the tensor; torch is imported when a sample is taken, never at import.
"""
import json
import os
import struct

from .facts import DTYPES, Certainty, Fact, Layout, ModelProps, Source

ROWS = 16    # rows sampled when comparing the head with the embedding
COLS = 8     # elements per sampled row
_SIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "I64": 8, "I32": 4,
         "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
# safetensors dtype of a scale tensor -> Layout.scale_format
SCALE_FORMAT = {"F32": "fp32", "BF16": "bf16", "F16": "fp16", "F8_E8M0": "ue8m0", "F8_E4M3": "e4m3"}
SCALE_SUFFIXES = ("weight_scale_inv", "weight_scale")


class Checkpoint:
    """The tensors of a safetensors checkpoint (one file, or a folder with an index or with *.safetensors files),
    from their headers only."""

    def __init__(self, path):
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
            self.files = [os.path.join(path, f) for f in files]
        else:
            self.files = [path] if path.endswith(".safetensors") else []
        self.tensors = {}   # name -> (file, dtype, shape, absolute start offset)
        for f in self.files:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            header.pop("__metadata__", None)
            for name, info in header.items():
                self.tensors[name] = (f, info["dtype"], tuple(info["shape"]), 8 + n + info["data_offsets"][0])

    def find(self, suffix):
        return sorted(k for k in self.tensors if k.endswith(suffix))

    def row_bytes(self, name, row, count):
        """The raw bytes of the first `count` elements of one row of a 2-D tensor."""
        f, dtype, shape, start = self.tensors[name]
        size = _SIZE[dtype]
        with open(f, "rb") as fh:
            fh.seek(start + row * shape[1] * size)
            return fh.read(min(count, shape[1]) * size)

    def same_bytes(self, a, b, chunk=1 << 23):
        """None when two tensors of one dtype and shape hold the same bytes, else the offset of the first byte that
        differs. Read in chunks, stopping at the first difference."""
        fa, dtype, shape, oa = self.tensors[a]
        fb, _, _, ob = self.tensors[b]
        n = _SIZE[dtype]
        for s in shape:
            n *= s
        done = 0
        with open(fa, "rb") as ha, open(fb, "rb") as hb:
            ha.seek(oa)
            hb.seek(ob)
            while done < n:
                x, y = ha.read(min(chunk, n - done)), hb.read(min(chunk, n - done))
                if x != y:
                    return done + next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
                if not x:
                    break
                done += len(x)
        return None


class Head:
    """What a checkpoint shows about its output head. `kind`: "absent" (no lm_head.weight: the head can only be the
    embedding), "differs" (its dtype, shape or bytes differ from the embedding's), "same" (every byte equals the
    embedding's: a stored copy of a tied head) or "sampled" (the sampled rows equal; the rest was not read)."""

    def __init__(self, kind, where):
        self.kind, self.where = kind, where

    def __repr__(self):
        return f"Head({self.kind!r}, {self.where!r})"


def head(path, full=True):
    """The checkpoint's lm_head.weight against its embedding, or None when the checkpoint cannot say (no safetensors,
    not exactly one embedding, more than one head). Sixteen rows are sampled first; with `full`, a head whose sample
    matches is then compared byte for byte (M11.2: vLLM 0.30 and transformers 5.17 compare the two tensors before
    tying, so whether a shipped head is the embedding decides what the loader does). The full read costs the size of
    the two tensors once per load, and only for a checkpoint that ships a copy of a tied head."""
    ck = Checkpoint(path)
    if not ck.tensors:
        return None
    embeds, heads = ck.find("embed_tokens.weight"), ck.find("lm_head.weight")
    if len(embeds) != 1 or len(heads) > 1:
        return None
    where = f"{path}: {len(ck.tensors)} tensors"
    if not heads:
        return Head("absent", f"{where}, no lm_head.weight (the head can only be the embedding)")
    e, h = ck.tensors[embeds[0]], ck.tensors[heads[0]]
    if e[1:3] != h[1:3]:
        return Head("differs", f"{where}, lm_head.weight {h[1]} {list(h[2])} differs from the embedding {e[1]} "
                               f"{list(e[2])}")
    rows = e[2][0]
    for i in range(ROWS):
        r = (i * max(1, rows // ROWS)) % rows
        if ck.row_bytes(embeds[0], r, COLS) != ck.row_bytes(heads[0], r, COLS):
            return Head("differs", f"{where}, lm_head.weight row {r} holds other bytes than the embedding")
    if not full:
        return Head("sampled", f"{where}, lm_head.weight equals the embedding on {ROWS} sampled rows")
    at = ck.same_bytes(embeds[0], heads[0])
    if at is not None:
        return Head("differs", f"{where}, lm_head.weight differs from the embedding (first at byte {at})")
    return Head("same", f"{where}, lm_head.weight equals the embedding in every byte (a stored copy of the tied "
                        f"head)")


def tie_fact(h):
    """ModelProps.tie_word_embeddings as a Head shows it: absent -> tied, differs -> not tied; None for a head that
    equals the embedding (consistent with a tie and with a stored copy, so nothing is claimed) or no Head."""
    if h is None or h.kind not in ("absent", "differs"):
        return None
    return Fact("ModelProps", ModelProps(tie_word_embeddings=h.kind == "absent"), Source("data", h.where),
                Certainty.VERIFIED)


def tie(path):
    """What the checkpoint shows about tied embeddings, or None when it cannot say (no checkpoint, or sampled bytes
    that match - consistent with a tie, but a sample is not proof; head(path) reads the whole tensors)."""
    return tie_fact(head(path, full=False))


# float dtype -> (bytes, struct code, exponent bits, mantissa bits)
_FLOAT = {"F32": (4, "<I", 8, 23), "BF16": (2, "<H", 8, 7), "F16": (2, "<H", 5, 10)}


def _all_powers_of_two(ck, name):
    """Every value of a float tensor is a positive power of two (so an exponent-only format, ue8m0, holds it)."""
    f, dtype, shape, start = ck.tensors[name]
    size, code, e_bits, m_bits = _FLOAT[dtype]
    n = 1
    for s in shape:
        n *= s
    with open(f, "rb") as fh:
        fh.seek(start)
        raw = fh.read(n * size)
    top = (1 << e_bits) - 1
    for (bits,) in struct.iter_unpack(code, raw):
        exponent, mantissa, sign = (bits >> m_bits) & top, bits & ((1 << m_bits) - 1), bits >> (e_bits + m_bits)
        if sign or mantissa or exponent in (0, top):
            return False
    return True


def scale_format(path, kind="fp8_block"):
    """The format the block scales of a checkpoint are in, as a Layout of `kind`, or None when there are no scale
    tensors or they disagree among themselves. Float scales whose every value is a power of two are ue8m0 values
    whatever their storage (a scale converted to ue8m0 and saved as fp32 is still an exponent)."""
    ck = Checkpoint(path)
    scales = [n for s in SCALE_SUFFIXES for n in ck.find(s)]
    if not scales:
        return None
    dtypes = {ck.tensors[n][1] for n in scales}
    if len(dtypes) != 1 or next(iter(dtypes)) not in SCALE_FORMAT:
        return None
    dtype = next(iter(dtypes))
    fmt, how = SCALE_FORMAT[dtype], f"all {dtype}"
    if dtype in _FLOAT and all(_all_powers_of_two(ck, n) for n in scales):
        fmt, how = "ue8m0", f"all {dtype}, every value a power of two"
    return Fact("Layout", Layout(kind, scale_format=fmt),
                Source("data", f"{path}: {len(scales)} scale tensors, {how}"), Certainty.VERIFIED)


# --- values in memory (M4.2) -------------------------------------------------------------------------------------

# Element dtypes that hold one value per element, so dense and strided say all there is to say about the storage.
# Integer tensors can hold packed values (two int4 in a byte, block bytes of q8_0): the tensor does not show which.
ONE_VALUE_PER_ELEMENT = frozenset({"float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"})
SAMPLE_K = 16   # values sampled per weight: which tensor it is, where shape and dtype only say what kind


def _dtype(tensor):
    return str(getattr(tensor, "dtype", "")).replace("torch.", "")


def _granularity(scale, out_features):
    """How many values one scale covers, from the scale tensor's own shape (None when its shape does not say)."""
    if scale is None:
        return "unscaled"
    shape = tuple(scale.shape)
    n = 1
    for s in shape:
        n *= s
    if n == 1:
        return "per_tensor"
    if out_features and n == out_features and sum(1 for s in shape if s != 1) == 1:
        return "per_channel"
    if len(shape) == 2 and out_features in shape:
        return "per_group"
    if len(shape) == 2:
        return "per_block"
    return None


def weight_layout(weight, in_features=None, out_features=None, scale=None, where="weight"):
    """What a weight matrix shows about its layout: (Fact or None, what could not be read).

    The sizes are the layer's own (its in and out features); without them the orientation is not read. A square
    weight does not show its orientation either (a value sample does: see moved). A shape that is neither (out, in)
    nor (in, out) is reported, not turned into a value. `scale` is the layer's scale tensor, None when it has none."""
    dtype = _dtype(weight)
    if dtype not in DTYPES:
        return None, f"{where}: dtype {dtype} is not in the vocabulary"
    if dtype not in ONE_VALUE_PER_ELEMENT:
        return None, f"{where}: a {dtype} tensor may hold packed values; its layout cannot be read from the tensor"
    shape = tuple(weight.shape)
    if len(shape) != 2:
        return None, f"{where}: not a matrix (shape {shape})"
    orientation, problem = None, None
    if in_features and out_features:
        as_out_in, as_in_out = shape == (out_features, in_features), shape == (in_features, out_features)
        if as_out_in and not as_in_out:
            orientation = "out_in"
        elif as_in_out and not as_out_in:
            orientation = "in_out"
        elif not as_out_in:
            problem = (f"{where}: shape {shape} is neither (out, in) = {(out_features, in_features)} nor (in, out) "
                       f"= {(in_features, out_features)}")
    else:
        problem = f"{where}: the layer gives no in and out features, so the orientation was not read"
    kind = "dense" if weight.is_contiguous() else "strided"
    value = Layout(kind, dtype=dtype, orientation=orientation, scale_granularity=_granularity(scale, out_features))
    scale_shape = "no scale" if scale is None else f"scale {tuple(scale.shape)}"
    return Fact("Layout", value, Source("data", f"{where}: shape {shape}, strides {tuple(weight.stride())}, "
                                                f"{scale_shape}"), Certainty.VERIFIED), problem


def sample_values(tensor, k=SAMPLE_K):
    """k values spread evenly over a tensor, as floats, with where they were taken (flat indices), its shape and
    dtype. None for an empty tensor."""
    import torch

    n = tensor.numel()
    if n == 0:
        return None
    # integer arithmetic on purpose: linspace computes in float32, and past 2**24 elements it can round an index up
    # past the end of the tensor, which on CUDA is a device-side assert that kills the process.
    step = max(1, n // k)
    idx = (torch.arange(k, dtype=torch.long, device=tensor.device) * step).clamp_(max=n - 1)
    vals = _at(tensor, idx).to(torch.float32).tolist()
    return {"idx": idx, "vals": vals, "shape": tuple(tensor.shape), "dtype": _dtype(tensor)}


def _at(tensor, idx):
    """The values at flat (row-major) indices. A matrix is indexed by row and column, so a strided weight (fp8
    keeps the transpose) is not copied whole to take sixteen values."""
    t = tensor.detach()
    if t.dim() == 2:
        return t[idx // t.shape[1], idx % t.shape[1]]
    return t.reshape(-1).index_select(0, idx)


def moved(tensor, before, move="identity"):
    """(values sampled, values found where `move` puts them, [what moved]) for a sample taken before a step.
    "identity": the step leaves every value where it was, so shape, dtype and each sampled value are unchanged; a
    tensor whose shape or dtype changed has none of them where they were."""
    if move != "identity":
        raise ValueError(f"observe.moved: no index mapping for the move {move!r}")
    k = len(before["vals"])
    if tuple(tensor.shape) != before["shape"] or _dtype(tensor) != before["dtype"]:
        return k, 0, [f"it went from {before['shape']} {before['dtype']} to {tuple(tensor.shape)} {_dtype(tensor)}"]
    now = _at(tensor, before["idx"]).float().tolist()
    left = [f"flat index {int(before['idx'][i])}: {a:.6g} -> {b:.6g}"
            for i, (a, b) in enumerate(zip(before["vals"], now)) if a != b and not (a != a and b != b)]
    return k, k - len(left), left
