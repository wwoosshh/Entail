"""observe: what the data itself shows, read deterministically from the files (LIBRARY_DESIGN.md principle 5).

A declaration is checked against the data before anything trusts it: tensor names, dtypes and bytes, never a
behaviour probe. Every function returns a Fact with Source kind "data" and certainty VERIFIED, or None when the
data does not settle the question (then nothing is claimed). Only safetensors headers and a few element bytes are
read; no tensor is loaded and torch is not needed.

  tie(path)            ModelProps.tie_word_embeddings: no lm_head.weight means the head is the embedding; a head
                       whose shape, dtype or sampled bytes differ from the embedding is not tied
  scale_format(path)   Layout.scale_format of block-quantized weights: the dtype of their scale tensors
"""
import json
import os
import struct

from .facts import Certainty, Fact, Layout, ModelProps, Source

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


def tie(path):
    """What the checkpoint shows about tied embeddings, or None when it cannot say (no checkpoint, or sampled bytes
    that match - consistent with a tie, but a sample is not proof)."""
    ck = Checkpoint(path)
    if not ck.tensors:
        return None
    embeds, heads = ck.find("embed_tokens.weight"), ck.find("lm_head.weight")
    if len(embeds) != 1 or len(heads) > 1:
        return None
    where = f"{path}: {len(ck.tensors)} tensors"
    if not heads:
        return Fact("ModelProps", ModelProps(tie_word_embeddings=True),
                    Source("data", f"{where}, no lm_head.weight (the head can only be the embedding)"),
                    Certainty.VERIFIED)
    e, h = ck.tensors[embeds[0]], ck.tensors[heads[0]]
    if e[1:3] != h[1:3]:
        return Fact("ModelProps", ModelProps(tie_word_embeddings=False),
                    Source("data", f"{where}, lm_head.weight {h[1]} {list(h[2])} differs from the embedding "
                                   f"{e[1]} {list(e[2])}"), Certainty.VERIFIED)
    rows = e[2][0]
    for i in range(ROWS):
        r = (i * max(1, rows // ROWS)) % rows
        if ck.row_bytes(embeds[0], r, COLS) != ck.row_bytes(heads[0], r, COLS):
            return Fact("ModelProps", ModelProps(tie_word_embeddings=False),
                        Source("data", f"{where}, lm_head.weight row {r} holds other bytes than the embedding"),
                        Certainty.VERIFIED)
    return None


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
