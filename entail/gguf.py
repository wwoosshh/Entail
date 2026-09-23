"""Read the metadata of a GGUF file without loading its tensors (ROADMAP M2.1).

GGUF (llama.cpp's format) starts with the magic `GGUF`, a version, the tensor and key-value counts, then typed
key-value pairs. Only the key-value pairs are read. Large token tables are read past and replaced by their length,
so a caller never holds a vocabulary in memory.
"""
import struct

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
_BIG_ARRAY = 1024   # arrays longer than this are summarised as ("array", element type, length)


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        size = struct.calcsize(fmt)
        data = self.f.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return struct.unpack(fmt, data)[0]

    def string(self):
        n = self.take("<Q")
        if n > 1 << 30:
            raise ValueError(f"GGUF string of {n} bytes: not a GGUF header")
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF header")
        return data.decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype in _SCALARS:
            return self.take(_SCALARS[vtype])
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, n = self.take("<I"), self.take("<Q")
            if n > 1 << 32:
                raise ValueError(f"GGUF array of {n} elements: not a GGUF header")
            items = [self.value(etype) for _ in range(n)]
            return ("array", etype, n) if n > _BIG_ARRAY else items
        raise ValueError(f"unknown GGUF value type {vtype}")


def read_metadata(path):
    """The key-value metadata of a GGUF file, as a dict."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file (bad magic)")
        r = _Reader(f)
        version = r.take("<I")
        if version not in (2, 3):
            raise ValueError(f"GGUF version {version} is not supported (2 and 3 are)")
        r.take("<Q")            # tensor count
        n_kv = r.take("<Q")
        if n_kv > 1 << 20:
            raise ValueError(f"{n_kv} GGUF keys: not a GGUF header")
        meta = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.take("<I"))
        return meta


def write_minimal(path, meta):
    """Write a GGUF file holding only `meta` (no tensors). For tests: str, float, int and bool values."""
    def s(text):
        b = text.encode("utf-8")
        return struct.pack("<Q", len(b)) + b
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(meta))
    for key, value in meta.items():
        body += s(key)
        if isinstance(value, bool):
            body += struct.pack("<I?", 7, value)
        elif isinstance(value, int):
            body += struct.pack("<Iq", 11, value)
        elif isinstance(value, float):
            body += struct.pack("<If", 6, value)
        else:
            body += struct.pack("<I", _STRING) + s(str(value))
    with open(path, "wb") as f:
        f.write(body)
