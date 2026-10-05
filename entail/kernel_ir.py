"""kernel_ir: what a Triton kernel reads, read from its own intermediate representation (ROADMAP M19 L5.4b).

The guarantee profile of L5.4a checked the meaning of a block FP8 matmul at its edges - which scale goes with which
value on the way in, the whole output against a reference on the way out - and paid for the second part at every
call, because what the kernel does inside (which scale it reads for which element) could not be seen. Triton
compiles each kernel from Python to an IR (TTIR) that the compiler keeps and hands out (CompiledKernel.asm["ttir"]);
nothing in the compiler is changed here. This module reads that IR and, for one concrete launch (grid, integer
arguments, constexpr configuration), follows every integer and address the kernel computes, for every program and
every loop iteration, without any data: the values the kernel loads are never needed, only where it loads them.

The meaning it holds the kernel to is the producers' (guarantee.Issue): an activation value A[m, k] goes with the
scale As[m, k // group_k] issued with it, a weight value B[n, k] with Bs[n // block_n, k // block_k] issued with
it. The contract of the launch is the whole product: every output element C[m, n] (M x N, the output's shape) is
stored once, and what is stored is

    C[m, n] = convert( 0 + sum over every k in [0, K), once each, of  A[m, k] * B[n, k] * As[..] * Bs[..] )

with the scales the producers issued for those elements. The IR proves this when all of the following hold for
every program and every loop iteration (masked-out operand lanes load an exact zero and contribute nothing):

  terms        each addend is one `tt.dot` (accumulating into an exact zero) of a load of the issued activation
               and a load of the issued weight, multiplied by exactly one loaded activation scale and one loaded
               weight scale and nothing else (a constant factor other than 1 is a violation)
  scale reads  the activation scale multiplied into output row i is, for every k the row consumes, the one issued
               for (m, k // group_k) with that activation; the weight scale multiplied into output column j is, for
               every k, the one issued for (n // block_n, k // block_k) with that weight
  contraction  both operands are read at the same k for each contraction index, masked alike along k
  sum          what is stored is a sum of such terms from an exact zero (only float conversions between it and the
               store), all for the same output rows and columns, whose k cover [0, K) exactly once
  store        it is stored at C[m, n] for its (row of A, column of B); over the whole launch every element of C is
               stored exactly once; nothing else is written anywhere
  indices      the integer arithmetic stays inside its i32 range (the IR wraps; this module does not model that)

The verdict, per launch:
  proven       all of the above, whatever the data (the arithmetic itself - products, sums, rounding - is the
               compiler's and the device's, not checked here)
  violation    the IR does not compute the contract for this launch: an element multiplied by a scale that is not
               its own, a k missing or counted twice, an output element never stored, a write elsewhere, an extra
               factor (the first such is described)
  possible     the scale read is chosen by a value the kernel loads at run time (data the IR cannot see), and one
               of the choices is not the element's own scale
  unproven     the IR does something this module does not model (an op, a data-dependent address, a loop whose
               bounds differ between programs, a layout it cannot invert, a value stored that is not a sum of
               terms): nothing is claimed
A launch that is not proven gets no claim from here; the caller decides (the reference's output, or refuse).
(2026-10-03, L5.4c: before this, only the scale reads of the terms found were checked, so a kernel that stored
nothing, or summed one K group of twenty, was "proven".)
Integers are evaluated with numpy over all programs (in chunks); a launch is decided once and cached by its
caller, so the cost is per launch configuration and size, not per call.
"""
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

CHUNK_ELEMENTS = 1 << 22     # programs evaluated together: about this many elements in the largest tensor
FAST = True                  # keep values apart along their axes (False: element by element only; for tests)
VERDICTS = ("proven", "violation", "possible", "unproven")


class Unmodelled(Exception):
    """The IR does something this module does not model: the launch is unproven."""


# --- parsing --------------------------------------------------------------------------------------------------------

@dataclass
class Op:
    results: List[str]
    name: str
    operands: List[str]
    text: str                      # the rest of the line (attributes, predicates, types)
    rtype: str = ""                # the result type (after the last ':' or '->')
    body: List["Op"] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Func:
    name: str
    args: List[tuple]              # (name, type)
    body: List[Op]


# a location annotation: `loc(` as a word of its own, not the end of a name (a kernel called `..._loc(`)
_LOC = re.compile(r'\s*(?<![\w@%$.])loc\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)')
_VAL = re.compile(r'%[A-Za-z0-9_.$#-]+')


def _strip(line: str) -> str:
    prev = None
    while prev != line:
        prev, line = line, _LOC.sub("", line)
    return line.rstrip()


def _split_top(s: str, sep: str = ",") -> List[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "<({[":
            depth += 1
        elif ch in ">)}]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def _results(lhs: str) -> List[str]:
    lhs = lhs.strip()
    m = re.match(r'(%[A-Za-z0-9_.$-]+):(\d+)$', lhs)
    if m:
        return [f"{m.group(1)}#{i}" for i in range(int(m.group(2)))]
    return [x.strip() for x in lhs.split(",")]


def parse(ttir: str) -> Func:
    """The first public function of a TTIR module."""
    lines = [_strip(x) for x in ttir.splitlines()]
    lines = [x for x in lines if x.strip() and not x.lstrip().startswith("#loc")]
    i = 0
    while i < len(lines) and "tt.func" not in lines[i]:
        i += 1
    if i == len(lines):
        raise Unmodelled("no tt.func in the IR")
    head = lines[i]
    name = re.search(r'@([A-Za-z0-9_]+)', head).group(1)
    argtext = head[head.index("(") + 1: head.rindex(")")]
    args = []
    for a in _split_top(argtext):
        m = re.match(r'(%[A-Za-z0-9_]+)\s*:\s*([^{]+)', a)
        if m:
            args.append((m.group(1), m.group(2).strip()))
    body, _, _ = _block(lines, i + 1)
    return Func(name, args, body)


def _block(lines, i):
    """The ops of a region, up to its closing line: (ops, the next line, the closing line). A closing "} else {"
    opens the else region of the op that opened this one (scf.if); a closing "}) : (types) -> type" ends a region in
    parentheses and gives that op (a generic op such as "tt.reduce") its types."""
    ops = []
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("}"):
            return ops, i + 1, line
        op = _op(line)
        i += 1
        if line.endswith("{"):
            op.body, i, close = _block(lines, i)
            if close.startswith("} else"):
                op.extra["else"], i, close = _block(lines, i)
            if close.startswith("})"):
                tail = close[2:].strip()
                if tail.startswith(":"):
                    types = tail[1:].strip()
                    if "->" in types:
                        src, rt = types.rsplit("->", 1)
                        op.extra["src"], op.rtype = src.strip(), rt.strip()
                    else:
                        op.rtype = types
        ops.append(op)
    return ops, i, ""


def _op(line: str) -> Op:
    results = []
    if " = " in line and line.startswith("%"):
        lhs, rhs = line.split(" = ", 1)
        results = _results(lhs)
    else:
        rhs = line
    rhs = rhs.rstrip("{").strip()
    if rhs.startswith('"'):                 # the generic form: "dialect.op"(%operands) <{attributes}> ({ region })
        q = rhs.index('"', 1)
        name, rest = rhs[1:q], rhs[q + 1:]
        op = Op(results, name, [], rest)
        m = re.match(r'\(([^()]*)\)', rest)
        op.operands = _VAL.findall(m.group(1)) if m else []
        ax = re.search(r'axis = (-?\d+)', rest)
        if ax:
            op.extra["axis"] = int(ax.group(1))
        return op
    name = rhs.split()[0]
    rest = rhs[len(name):].strip()
    op = Op(results, name, [], rest)
    if name == "scf.if":
        op.operands = _VAL.findall(rest.split("->")[0])[:1]
        return op
    if name == "scf.for":
        m = re.match(r'(%\S+)\s*=\s*(%\S+)\s+to\s+(%\S+)\s+step\s+(%\S+)(.*)', rest)
        if not m:
            raise Unmodelled(f"scf.for form not understood: {rest[:80]}")
        for i, k in enumerate(("iv", "lb", "ub", "step")):
            op.extra[k] = m.group(i + 1)
        iters = re.search(r'iter_args\((.*?)\)\s*->', m.group(5))
        op.extra["iter"] = []
        if iters:
            for pair in _split_top(iters.group(1)):
                k, v = [x.strip() for x in pair.split("=")]
                op.extra["iter"].append((k, v))
        return op
    # operands: the %values before the first ':' at depth 0 (types after it)
    depth, cut = 0, len(rest)
    for idx, ch in enumerate(rest):
        if ch in "<({[":
            depth += 1
        elif ch in ">)}]":
            depth -= 1
        elif ch == ":" and depth == 0:
            cut = idx
            break
    op.operands = _VAL.findall(rest[:cut])
    op.rtype = rest[cut + 1:].strip() if cut < len(rest) else ""
    for sep in ("->", " to "):              # "source type -> result type"; a conversion: "source type to result type"
        if sep in op.rtype:
            src, op.rtype = op.rtype.rsplit(sep, 1)
            op.extra["src"], op.rtype = src.strip(), op.rtype.strip()
    if name == "arith.constant" and not op.rtype and rest.strip() in ("true", "false"):
        op.rtype = "i1"                     # a boolean constant prints without its type
    return op


def _shape(t: str):
    """The shape of a TTIR type: () for a scalar, the dimensions of a tensor."""
    m = re.match(r'tensor<([0-9x]+)x', t.strip())
    if m:
        return tuple(int(x) for x in m.group(1).split("x"))
    return ()


def _elem(t: str) -> str:
    """The element type of a TTIR type; for a pointer (what tt.load prints), the type it points to."""
    t = t.split(",")[0].strip()
    m = re.match(r'tensor<[0-9x]+x(.*)>$', t)
    t = m.group(1) if m else t
    m = re.match(r'!tt\.ptr<(.*)>$', t)
    return m.group(1) if m else t


# --- values ---------------------------------------------------------------------------------------------------------
#
# An integer tensor is kept apart where it can be: value[p, i0, i1, ...] = s[p] + v0[p, i0] + v1[p, i1] + ... (a
# scalar per program and one vector per axis), the form a kernel's index arithmetic takes (ranges, broadcasts,
# strides, offsets). Operations that keep the form keep it; an operation over one axis is done on that axis's vector;
# anything else makes the value dense (every element, for every program). So a tile of 64 x 128 addresses costs 192
# numbers per program, not 8192, and the check below works on the vectors.
#
# Integer meaning (2026-10-03, L5.4d): every integer value has its bit width w (i1 .. i64; index is 64) and every
# element holds exactly what the IR holds - the w-bit two's complement pattern, read signed (an i1 is 0 or 1). Each
# operation is computed as MLIR's arith defines it: add, sub, mul and shl wrap modulo 2^w; signed and unsigned
# readings are chosen by the operation (divsi / divui, cmpi slt / ult, extsi / extui, shrsi / shrui, minsi / minui);
# truncation keeps the low bits; extension sign- or zero-extends. Poison and undefined behaviour (nsw / nuw overflow,
# a shift by w or more, division by zero, the signed minimum over -1) and anything outside the exact range of the
# arithmetic (|value| > 2^60) are not modelled: the launch is unproven. Before, values were plain int64: unsigned
# operations read negative values as negative, extensions and truncations kept the value, only i32 results were
# range-checked.

_W = {"i1": 1, "i8": 8, "i16": 16, "i32": 32, "i64": 64, "index": 64}
_LIMIT = 1 << 58             # magnitudes computed exactly (int64 sums of a few of these cannot overflow)


def _width(t) -> Optional[int]:
    """The bit width of an integer type (i1 .. i64, index, or a tensor of them); None for anything else."""
    return _W.get(_elem(t).strip()) if t else None


def _int_lit(text):
    """The integer literal of an arith.constant (decimal, hexadecimal, true / false), or None."""
    m = re.match(r'\s*(?:dense<)?\s*(true|false|-?0x[0-9a-fA-F]+|-?\d+)(?![\d.eE])', text)
    if not m:
        return None
    v = m.group(1)
    if v in ("true", "false"):
        return int(v == "true")
    return int(v, 16) if "0x" in v else int(v)


def _fits(lo, hi, w) -> bool:
    """Whether every value in lo..hi is a w-bit signed value (an i1: 0 or 1)."""
    if w == 1:
        return lo >= 0 and hi <= 1
    return -(1 << (w - 1)) <= lo and hi < (1 << (w - 1))


class E:
    """An integer (or 0/1 boolean: b) tensor over the programs: s (P|1,), v {axis: (P|1, n)}, or dense d; w its bit
    width (None for a pointer's offset, which is address arithmetic)."""
    __slots__ = ("shape", "s", "v", "d", "b", "w")

    def __init__(self, shape, s=None, v=None, d=None, b=False, w=None):
        self.shape, self.s, self.v, self.d, self.b = tuple(shape), s, dict(v or {}), d, b
        self.w = 1 if b else w
        if self.s is None and self.d is None:
            self.s = np.zeros((1,), dtype=np.int64)

    def axes(self):
        return set(self.v)

    def scalar_only(self):
        return self.d is None and not self.v

    def along(self, pos, n=None):
        """The values along one axis, (P|1, n): for a value whose other axes add nothing."""
        n = self.shape[pos] if n is None else n
        part = self.v.get(pos)
        base = self.s.reshape(-1, 1)
        return base + part if part is not None else np.broadcast_to(base, (base.shape[0], n))

    def full(self):
        if self.d is not None:
            return self.d
        nd = len(self.shape)
        out = self.s.reshape((-1,) + (1,) * nd)
        for pos, arr in self.v.items():
            shp = [arr.shape[0]] + [1] * nd
            shp[pos + 1] = arr.shape[1]
            out = out + arr.reshape(shp)
        return np.broadcast_to(out, (out.shape[0],) + self.shape)


class Mk:
    """A boolean that is the AND of several (a mask built from conditions on different axes)."""
    __slots__ = ("shape", "f")

    def __init__(self, shape, factors):
        self.shape, self.f = tuple(shape), list(factors)

    def full(self):
        out = None
        for x in self.f:
            a = x.full().astype(bool)
            out = a if out is None else (out & a)
        return out


class T:
    """A value the kernel loads at run time, or computed from one: unknown here."""
    __slots__ = ("why",)

    def __init__(self, why):
        self.why = why


class Ptr:
    __slots__ = ("arg", "off", "taint")

    def __init__(self, arg, off, taint=None):
        self.arg, self.off, self.taint = arg, off, taint


class F:
    """A floating value, by where it comes from: kind is leaf (a load), dot, mul, sel, acc or opaque."""
    __slots__ = ("kind", "arg", "off", "mask", "taint", "parts", "cond", "stash")

    def __init__(self, kind, arg=None, off=None, mask=None, taint=None, parts=(), cond=None, stash=None):
        self.kind, self.arg, self.off, self.mask, self.taint = kind, arg, off, mask, taint
        self.parts, self.cond, self.stash = list(parts), cond, stash


def _as_bool_full(x):
    return x.full().astype(bool) if x is not None else None


def _const(x):
    """The value of a float constant (None when x is not one or the value is not readable)."""
    if isinstance(x, F) and x.kind == "opaque" and (x.stash or {}).get("const"):
        return x.stash.get("value")
    return None


def _zero(x) -> bool:
    return _const(x) == 0.0


class Acc:
    """A sum, from an exact zero, of checked scaled dot terms: the operands they read (A, B), the output rows and
    columns they belong to (the same for every term that adds something to a program), K, per program whether any
    term adds something (has (P,)), and per term and program the k it covers as one range (lo (P,), count (P,))."""
    __slots__ = ("src", "rows", "rows_valid", "cols", "cols_valid", "K", "has", "ranges")

    def __init__(self, src=None, rows=None, rows_valid=None, cols=None, cols_valid=None, K=None, has=None,
                 ranges=()):
        self.src, self.rows, self.rows_valid, self.cols, self.cols_valid = src, rows, rows_valid, cols, cols_valid
        self.K, self.has, self.ranges = K, has, tuple(ranges)

    def plus(self, other):
        if not self.ranges:
            return other
        if not other.ranges:
            return self
        if self.src != other.src or self.K != other.K:
            raise Unmodelled("a sum of terms over different operands")
        both = self.has & other.has
        if not (_same_coords(self.rows, self.rows_valid, other.rows, other.rows_valid, both)
                and _same_coords(self.cols, self.cols_valid, other.cols, other.cols_valid, both)):
            raise Unmodelled("terms of one sum belong to different output rows or columns")
        if self.has.all():               # the usual case: every program already has its rows and columns
            return Acc(self.src, self.rows, self.rows_valid, self.cols, self.cols_valid, self.K, self.has,
                       self.ranges + other.ranges)
        h = self.has[:, None]
        return Acc(self.src, np.where(h, self.rows, other.rows), np.where(h, self.rows_valid, other.rows_valid),
                   np.where(h, self.cols, other.cols), np.where(h, self.cols_valid, other.cols_valid), self.K,
                   self.has | other.has, self.ranges + other.ranges)


def _same_coords(a, va, b, vb, sel) -> bool:
    """Whether two (P, n) coordinate arrays agree, with their validity, on the programs `sel` (P,)."""
    if np.shape(va) != np.shape(vb):
        return False
    if not sel.all():
        a, va, b, vb = a[sel], va[sel], b[sel], vb[sel]
    return bool(np.array_equal(va, vb) and not np.any((a != b) & va))


# --- the binding of a launch ----------------------------------------------------------------------------------------

@dataclass
class Tensor:
    """What a pointer argument of the launch holds, as its producer issued it."""
    role: str                      # activation, activation_scale, weight, weight_scale, output, other
    serial: int = 0
    pair: int = 0
    shape: tuple = ()
    stride: tuple = ()             # in elements
    block: tuple = (1, 1)          # weight: (block_n, block_k); activation: (1, group_k)


@dataclass
class Verdict:
    verdict: str
    why: str = ""
    terms: int = 0                 # scaled dot terms checked (one per program chunk and loop iteration)
    elements: int = 0              # operand elements whose scale was checked
    programs: int = 0
    seconds: float = 0.0
    example: Optional[dict] = None
    dense: bool = False            # some value could not be kept apart and was evaluated element by element

    def to_json(self):
        return {"verdict": self.verdict, "why": self.why, "terms": self.terms, "elements": self.elements,
                "programs": self.programs, "seconds": round(self.seconds, 4), "example": self.example,
                "dense": self.dense}


class _Fail(Exception):
    def __init__(self, verdict, why, example=None):
        super().__init__(why)
        self.verdict, self.why, self.example = verdict, why, example


class _Dense(Exception):
    """The fast (kept-apart) evaluation met a value it must make dense: start again with dense evaluation."""


# --- evaluation -----------------------------------------------------------------------------------------------------

_CMP = {"eq": np.equal, "ne": np.not_equal, "slt": np.less, "sle": np.less_equal, "sgt": np.greater,
        "sge": np.greater_equal, "ult": np.less, "ule": np.less_equal, "ugt": np.greater, "uge": np.greater_equal}
_SIGNED_CMP = ("slt", "sle", "sgt", "sge")
_UNSIGNED_CMP = ("ult", "ule", "ugt", "uge")
_INT_BINARY = ("arith.addi", "arith.subi", "arith.muli", "arith.divsi", "arith.divui", "arith.remsi", "arith.remui",
               "arith.floordivsi", "arith.ceildivsi", "arith.ceildivui", "arith.minsi", "arith.maxsi",
               "arith.minui", "arith.maxui", "arith.andi", "arith.ori", "arith.xori", "arith.shli", "arith.shrsi",
               "arith.shrui")
_INT_CASTS = ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast", "arith.index_castui")
_FLOAT_CASTS = ("arith.truncf", "arith.extf", "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui")
_ELEM_BYTES = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}


def _elem_bytes(t):
    """Bytes of one element of type t (f8 kinds: 1); None when unknown."""
    t = t.strip()
    return 1 if t.startswith("f8") else _ELEM_BYTES.get(t)


class _Run:
    def __init__(self, fn: Func, binding: Dict[str, Tensor], ints: Dict[str, int], pids, dense: bool):
        self.fn, self.binding, self.ints, self.pids, self.allow_dense = fn, binding, ints, pids, dense
        self.P = int(pids[0].shape[0])
        self.terms = 0
        self.elements = 0
        self.went_dense = False
        self.stores = []           # per store to the output: (row lo, rows, column lo, columns) of the programs storing

    # -- kept-apart arithmetic --

    def _dense(self, shape, arr, b=False, w=None):
        if not self.allow_dense:
            raise _Dense()
        self.went_dense = True
        return E(shape, d=np.asarray(arr), b=b, w=w)

    def _general(self, f, args, shape, b=False, w=None):
        """f over operands (E) of one shape: on the scalars, on the one axis they vary along, or dense."""
        if any(a.d is not None for a in args) or len(set().union(*(a.axes() for a in args))) > 1:
            return self._dense(shape, np.asarray(f(*[a.full() for a in args])).astype(np.int64), b, w)
        axes = set().union(*(a.axes() for a in args))
        if not axes:
            return E(shape, s=np.asarray(f(*[a.s for a in args])).astype(np.int64), b=b, w=w)
        pos = axes.pop()
        n = shape[pos]
        vals = [a.along(pos, n) for a in args]
        return E(shape, v={pos: np.asarray(f(*vals)).astype(np.int64)}, b=b, w=w)

    def _add(self, a, b, sign=1):
        """The exact sum (no wrap; the caller holds it in its width)."""
        if a.d is not None or b.d is not None:
            return self._dense(a.shape, a.full() + sign * b.full(), w=a.w)
        v = dict(a.v)
        for pos, arr in b.v.items():
            v[pos] = v[pos] + sign * arr if pos in v else sign * arr
        return E(a.shape, s=a.s + sign * b.s, v=v, w=a.w)

    def _mul(self, a, b):
        """The exact product (no wrap; the caller holds it in its width)."""
        if a.d is None and b.d is None and (a.scalar_only() or b.scalar_only()):
            sc, other = (a, b) if a.scalar_only() else (b, a)
            x = sc.s
            return E(other.shape, s=other.s * x, v={p: arr * x.reshape(-1, 1) for p, arr in other.v.items()},
                     w=a.w)
        return self._general(np.multiply, [a, b], a.shape, w=a.w)

    def _int(self, x):
        """An E for an integer operand (a mask becomes its dense product)."""
        if isinstance(x, Mk):
            return self._dense(x.shape, x.full().astype(np.int64), True, 1)
        return x

    # -- integer meaning: widths, two's complement, signed and unsigned readings --

    def _typed(self, x, w, b=None):
        """x as a w-bit value (its elements unchanged: they already are w-bit values)."""
        return E(x.shape, s=x.s, v=x.v, d=x.d, b=(w == 1) if b is None else b, w=w)

    def _each(self, x, f, w, b=False):
        """f on every element of x, keeping the form where it can (a scalar per program, one axis), else dense."""
        if x.d is not None:
            return self._dense(x.shape, f(np.asarray(x.d)), b, w)
        if not x.v:
            return E(x.shape, s=np.asarray(f(np.asarray(x.s))), b=b, w=w)
        if len(x.v) == 1:
            pos = next(iter(x.v))
            return E(x.shape, v={pos: np.asarray(f(np.asarray(x.along(pos))))}, b=b, w=w)
        return self._dense(x.shape, f(np.asarray(x.full())), b, w)

    def _wrap(self, x, w):
        """x held in w bits as the IR holds it: the low w bits, read signed (two's complement); an i1 is its low bit.
        Exact: elements outside the range wrap one by one."""
        lo, hi = _bounds(x)
        if _fits(lo, hi, w):
            return self._typed(x, w)
        if w == 1:
            return self._each(x, lambda a: a & 1, 1, b=True)
        half, span = 1 << (w - 1), 1 << w
        return self._each(x, lambda a: ((a + half) % span) - half, w)

    def _unsigned(self, x, w):
        """The elements of x read unsigned (w bits): a negative one is itself plus 2^w."""
        if w == 1:
            return x
        lo, _hi = _bounds(x)
        if lo >= 0:
            return x
        if w >= 62:
            raise Unmodelled("a negative 64-bit value read unsigned: beyond the checker's exact integer range")
        span = 1 << w
        return self._each(x, lambda a: np.where(a < 0, a + span, a), w)

    def _signed(self, x, w):
        """The elements of x read signed: an i1 that is true is -1."""
        if w == 1:
            return self._each(x, lambda a: -a, 1)
        return x

    def _operands(self, n, args, w):
        out = []
        for x in args:
            x = self._int(x)
            if not isinstance(x, E):
                raise Unmodelled(f"{n} of a {type(x).__name__}")
            if x.w != w:
                raise Unmodelled(f"{n}: a {x.w}-bit operand in a {w}-bit operation")
            out.append(x)
        return out

    def _int_op(self, n, op, args, shape):
        """An integer operation exactly as MLIR's arith defines it. What it cannot compute exactly (poison,
        undefined behaviour, values beyond the exact range) is not modelled: the launch is unproven."""
        if n in _INT_CASTS:
            return self._cast(n, op, args[0])
        w = _width(op.rtype)
        if w is None:
            raise Unmodelled(f"{n} on {op.rtype or 'a type it does not print'}: not an integer type this module "
                             f"models")
        m = re.search(r'overflow<([^>]*)>', op.text)
        flags = {f.strip() for f in m.group(1).split(",")} if m else set()
        if n == "arith.andi" and all(isinstance(x, Mk) or (isinstance(x, E) and x.b) for x in args):
            fs = []
            for x in args:
                fs += x.f if isinstance(x, Mk) else [x]
            return Mk(shape or args[0].shape, fs) if len(fs) > 1 else fs[0]
        xs = self._operands(n, args, w)
        if w == 1 and n not in ("arith.andi", "arith.ori", "arith.xori"):
            raise Unmodelled(f"{n} on i1 values")
        if n in ("arith.andi", "arith.ori", "arith.xori"):
            # bitwise on sign-extended w-bit values gives the sign-extended w-bit result: the form is kept
            f = {"arith.andi": np.bitwise_and, "arith.ori": np.bitwise_or, "arith.xori": np.bitwise_xor}[n]
            return self._general(f, xs, xs[0].shape, b=(w == 1), w=w)
        a, b = xs
        (la, ha), (lb, hb) = _bounds(a), _bounds(b)
        if n in ("arith.addi", "arith.subi", "arith.muli"):
            if n == "arith.muli":
                if max(abs(la), abs(ha)) * max(abs(lb), abs(hb)) > _LIMIT:
                    raise Unmodelled(f"{n}: a product beyond the checker's exact integer range")
                r = self._mul(a, b)
            else:
                if max(abs(la), abs(ha)) + max(abs(lb), abs(hb)) > _LIMIT:
                    raise Unmodelled(f"{n}: a sum beyond the checker's exact integer range")
                r = self._add(a, b, 1 if n == "arith.addi" else -1)
            return self._poison_or_wrap(n, r, a, b, w, flags)
        if n in ("arith.divsi", "arith.remsi", "arith.floordivsi", "arith.ceildivsi"):
            half = 1 << (w - 1)

            def sdiv(x, y, n=n, half=half):
                if np.any(y == 0):
                    raise Unmodelled(f"{n}: a division by zero (undefined)")
                if np.any((x == -half) & (y == -1)):
                    raise Unmodelled(f"{n}: the signed minimum divided by -1 overflows (undefined)")
                q = np.abs(x) // np.abs(y)
                q = np.where((x < 0) ^ (y < 0), -q, q)          # truncated toward zero
                if n == "arith.divsi":
                    return q
                if n == "arith.remsi":
                    return x - y * q                             # the sign of the dividend
                if n == "arith.floordivsi":
                    return np.floor_divide(x, y)
                return -np.floor_divide(-x, y)                   # ceildivsi
            return self._wrap(self._general(sdiv, [a, b], a.shape, w=w), w)
        if n in ("arith.divui", "arith.remui", "arith.ceildivui"):
            ua, ub = self._unsigned(a, w), self._unsigned(b, w)

            def udiv(x, y, n=n):
                if np.any(y == 0):
                    raise Unmodelled(f"{n}: a division by zero (undefined)")
                if n == "arith.divui":
                    return x // y
                if n == "arith.remui":
                    return x % y
                return -((-x) // y)                              # ceildivui
            return self._wrap(self._general(udiv, [ua, ub], a.shape, w=w), w)
        if n in ("arith.minsi", "arith.maxsi"):
            return self._general(np.minimum if n == "arith.minsi" else np.maximum, [a, b], a.shape, w=w)
        if n in ("arith.minui", "arith.maxui"):
            ua, ub = self._unsigned(a, w), self._unsigned(b, w)
            r = self._general(np.minimum if n == "arith.minui" else np.maximum, [ua, ub], a.shape, w=w)
            return self._wrap(r, w)
        if n in ("arith.shli", "arith.shrsi", "arith.shrui"):
            ub = self._unsigned(b, w)
            hi_shift = _bounds(ub)[1]
            if hi_shift >= w:
                raise Unmodelled(f"{n} by {hi_shift} bits of a {w}-bit value: the result is poison")
            if n == "arith.shli":
                if max(abs(la), abs(ha)) * (1 << max(hi_shift, 0)) > _LIMIT:
                    raise Unmodelled(f"{n}: a shift beyond the checker's exact integer range")
                r = self._general(np.left_shift, [a, ub], a.shape, w=w)
                return self._poison_or_wrap(n, r, a, ub, w, flags)
            if n == "arith.shrsi":
                return self._general(np.right_shift, [a, ub], a.shape, w=w)     # arithmetic: the sign is kept
            return self._wrap(self._general(np.right_shift, [self._unsigned(a, w), ub], a.shape, w=w), w)
        raise Unmodelled(f"the op {n} is not modelled")

    def _poison_or_wrap(self, n, r, a, b, w, flags):
        """r (exact) as the w-bit result; with nsw / nuw, an overflow makes poison (not modelled)."""
        if "nsw" in flags and not _fits(*_bounds(r), w):
            raise Unmodelled(f"{n} with nsw overflows {w} bits: the result is poison")
        if "nuw" in flags:
            ua, ub = self._unsigned(a, w), self._unsigned(b, w)
            ru = (self._add(ua, ub, 1) if n == "arith.addi" else self._add(ua, ub, -1) if n == "arith.subi" else
                  self._mul(ua, ub) if n == "arith.muli" else
                  self._general(np.left_shift, [ua, ub], ua.shape, w=w))
            lo, hi = _bounds(ru)
            if lo < 0 or hi >= (1 << w):
                raise Unmodelled(f"{n} with nuw overflows {w} bits: the result is poison")
        return self._wrap(_exact(r, n), w)

    def _cast(self, n, op, x):
        """extsi, extui, trunci, index_cast(ui): the value as the IR converts it."""
        ws, wd = _width(op.extra.get("src", "")), _width(op.rtype)
        if ws is None or wd is None:
            raise Unmodelled(f"{n} from {op.extra.get('src')} to {op.rtype}: not integer types this module models")
        x = self._operands(n, [x], ws)[0]
        widen = wd > ws
        if n in ("arith.extsi", "arith.extui") and not widen:
            raise Unmodelled(f"{n} to as many or fewer bits")
        if n == "arith.trunci" and widen:
            raise Unmodelled(f"{n} to more bits")
        if wd == ws:
            return self._typed(x, wd)
        if widen:
            if n in ("arith.extsi", "arith.index_cast"):
                return self._typed(self._signed(x, ws), wd, b=False)      # sign extension keeps the signed value
            return self._typed(self._unsigned(x, ws), wd, b=False)       # zero extension: the unsigned value
        return self._wrap(x, wd)                                          # truncation: the low wd bits

    def run(self):
        env = {}
        for name, typ in self.fn.args:
            key = name[1:]
            if typ.startswith("!tt.ptr"):
                env[name] = Ptr(key, E(()))
            elif key in self.ints:
                w = _width(typ)
                v = int(self.ints[key])
                if w is None:
                    raise Unmodelled(f"argument {key} has the type {typ}, not an integer type this module models")
                if not _fits(v, v, w) or abs(v) > _LIMIT:
                    raise Unmodelled(f"argument {key} = {v} is not a value of its type {typ}")
                env[name] = E((), s=np.full((1,), v, dtype=np.int64), b=(w == 1), w=w)
            else:
                raise Unmodelled(f"argument {key} has no value in the launch")
        self._ops(self.fn.body, env)

    def _ops(self, ops, env):
        for op in ops:
            self._op(op, env)

    def _get(self, env, name):
        if name not in env:
            raise Unmodelled(f"{name} used before it is defined")
        return env[name]

    def _op(self, op, env):  # noqa: C901 - one branch per modelled op
        n = op.name
        args = [self._get(env, a) for a in op.operands]
        shape = _shape(op.rtype)
        r = None
        if n == "arith.constant":
            et = _elem(op.rtype)
            if et.startswith("f") or et.startswith("bf"):
                m = re.match(r'(?:dense<)?([-0-9.eE+a-z]+)>?', op.text)
                try:
                    val = float(m.group(1)) if m else None
                except ValueError:             # a hex bit pattern: not read here
                    val = None
                r = F("opaque", stash={"const": True, "value": val})
            else:
                w, v = _width(op.rtype), _int_lit(op.text)
                if w is None or v is None:
                    raise Unmodelled(f"a constant of type {op.rtype} this module does not read: {op.text[:60]}")
                # a w-bit literal is printed signed, or (up to 2^w - 1) unsigned; beyond the exact range: not modelled
                if not -(1 << (w - 1)) <= v < (1 << w) or abs(v) > max(_LIMIT, 1):
                    raise Unmodelled(f"a constant {v} of type {op.rtype} beyond the checker's exact integer range")
                r = self._wrap(E(shape, s=np.full((1,), v, dtype=np.int64), w=64), w)
        elif n == "tt.get_program_id":
            axis = {"x": 0, "y": 1, "z": 2}[op.text.split()[0]]
            r = E((), s=self.pids[axis], w=32)
        elif n == "tt.make_range":
            s = int(re.search(r'start = (-?\d+)', op.text).group(1))
            e = int(re.search(r'end = (-?\d+)', op.text).group(1))
            r = E((e - s,), v={0: np.arange(s, e, dtype=np.int64)[None, :]}, w=32)
        elif n in _INT_BINARY or n in _INT_CASTS:
            ts = [x for x in args if isinstance(x, T)]
            r = T(ts[0].why) if ts else self._int_op(n, op, args, shape)
        elif n == "arith.cmpi":
            pred = op.text.split(",")[0].strip()
            ts = [x for x in args if isinstance(x, T)]
            if ts:
                r = T(ts[0].why)
            else:
                w = _width(op.rtype)                          # cmpi prints its operands' type
                if w is None or pred not in _CMP:
                    raise Unmodelled(f"cmpi {pred} on {op.rtype}")
                a, b = self._operands(n, args, w)
                if pred in _SIGNED_CMP:
                    a, b = self._signed(a, w), self._signed(b, w)
                elif pred in _UNSIGNED_CMP:
                    a, b = self._unsigned(a, w), self._unsigned(b, w)
                r = self._general(_CMP[pred], [a, b], a.shape, b=True, w=1)
        elif n == "arith.select":
            c, a, b = args
            if isinstance(a, F) or isinstance(b, F):
                r = F("sel", parts=[a, b], cond=c)
            elif isinstance(c, T) or isinstance(a, T) or isinstance(b, T):
                r = T(c.why if isinstance(c, T) else "a selected value")
            elif isinstance(a, Ptr) or isinstance(b, Ptr):
                raise Unmodelled("a select between pointers")
            else:
                c, a, b = self._int(c), self._int(a), self._int(b)
                if c.w != 1 or a.w != b.w:
                    raise Unmodelled("a select whose condition is not i1 or whose values differ in width")
                r = self._general(lambda x, y, z: np.where(x.astype(bool), y, z), [c, a, b], a.shape,
                                  b=a.b and b.b, w=a.w)
        elif n in ("tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape"):
            r = self._shape_op(n, op, args[0], shape)
        elif n in _FLOAT_CASTS:
            x = args[0]
            if isinstance(x, F):
                if n in ("arith.truncf", "arith.extf"):
                    r = x
                elif n in ("arith.fptosi", "arith.fptoui"):
                    r = T(f"{n} of a float value")         # an integer that depends on data from here on
                else:
                    raise Unmodelled(f"{n} of a float")
            elif isinstance(x, (E, Mk)) and n in ("arith.sitofp", "arith.uitofp"):
                r = F("opaque", stash={"why": f"{n}: a float made from an integer"})
            elif isinstance(x, T):
                r = T(x.why)
            else:
                raise Unmodelled(f"{n} of a {type(x).__name__}")
        elif n == "tt.bitcast":
            x = args[0]
            src, dst = op.extra.get("src", ""), op.rtype
            if isinstance(x, Ptr):
                bs, bd = _elem_bytes(_elem(src)), _elem_bytes(_elem(dst))
                if bs is None or bs != bd:
                    raise Unmodelled(f"a pointer cast from {_elem(src)} to {_elem(dst)}: offsets would count "
                                     f"other elements")
                r = x
            elif isinstance(x, F):
                r = T(f"{n} of a float value") if _width(dst) else F("opaque", stash={"why": f"{n} of a float"})
            elif isinstance(x, T):
                r = T(x.why)
            else:
                ws, wd = _width(src), _width(dst)
                if ws is None or ws != wd:
                    raise Unmodelled(f"{n} from {src} to {dst}")
                r = self._typed(self._operands(n, [x], ws)[0], wd)
        elif n == "tt.addptr":
            p, o = args
            if not isinstance(p, Ptr):
                raise Unmodelled(f"an address computed from a pointer the checker does not follow "
                                 f"({p.why if isinstance(p, T) else type(p).__name__})")
            if isinstance(o, T) or p.taint is not None:
                r = Ptr(p.arg, p.off, taint=(o.why if isinstance(o, T) else p.taint))
            else:
                o = self._int(o)
                if not isinstance(o, E) or o.w is None or o.w == 1:
                    raise Unmodelled("a pointer offset that is not an integer")
                off = p.off
                if off.shape != o.shape:
                    off = self._shape_op("tt.splat", op, off, o.shape) if not off.shape else off
                lo, hi = _bounds(o)
                lo2, hi2 = _bounds(off)
                if max(abs(lo), abs(hi)) + max(abs(lo2), abs(hi2)) > _LIMIT:
                    raise Unmodelled("a pointer offset beyond the checker's exact integer range")
                s = self._add(off, o)                   # address arithmetic: exact, the offset read signed
                r = Ptr(p.arg, E(s.shape, s=s.s, v=s.v, d=s.d, w=None))
        elif n == "tt.load":
            p = args[0]
            mask = args[1] if len(args) > 1 else None
            et = _elem(op.rtype)
            if isinstance(mask, T) and (et.startswith("f") or et.startswith("bf")):
                raise Unmodelled("a load masked by data read at run time")
            if not isinstance(p, Ptr):
                raise Unmodelled("a load from a non-pointer")
            if et.startswith("f") or et.startswith("bf"):
                other = args[2] if len(args) > 2 else None
                # masked-out lanes hold `other` (undefined without one): they contribute nothing only when it is 0
                r = F("leaf", arg=p.arg, off=(p.off if p.taint is None else None), mask=mask, taint=p.taint,
                      stash={"masked_zero": mask is None or _zero(other)})
            else:
                r = T(f"a value loaded from {p.arg}")
        elif n == "tt.dot":
            a, b = args[0], args[1]
            if not (isinstance(a, F) and isinstance(b, F) and a.kind == "leaf" and b.kind == "leaf"):
                raise Unmodelled("a dot whose operands are not loads")
            if len(args) > 2 and not _zero(args[2]):
                raise Unmodelled("a dot that accumulates into something other than an exact zero")
            for leaf in (a, b):
                if not (leaf.stash or {}).get("masked_zero"):
                    raise _Fail("unproven", f"masked-out lanes of {leaf.arg} in the dot are not loaded as zero")
            r = F("dot", parts=[a, b])
        elif n == "arith.mulf":
            parts = []
            for x in args:
                if not isinstance(x, F):
                    raise Unmodelled("a float multiply by a non-float")
                parts += x.parts if x.kind == "mul" else [x]
            r = F("mul", parts=parts)
        elif n == "arith.addf":
            a, b = args
            sa, sb = self._acc_of(a), self._acc_of(b)
            if sa is None or sb is None:      # something else is added: not a sum of terms (stored, it is unproven)
                r = F("opaque", stash={"why": "a sum with an addend that is not a scaled dot term"})
            else:
                r = F("acc", stash=sa.plus(sb))
        elif n == "scf.for":
            self._for(op, env)
            return
        elif n == "scf.if":
            self._if(op, env, args[0])
            return
        elif n == "tt.reduce":
            x = args[0]
            if isinstance(x, T):
                r = T(x.why)
            else:
                raise Unmodelled(f"a reduction of a {type(x).__name__} value")
        elif n == "tt.atomic_rmw":
            ptr = args[0] if args else None
            if not isinstance(ptr, Ptr):
                raise Unmodelled("an atomic through a value that is not a pointer")
            info = self.binding.get(ptr.arg)
            if info is not None and info.role == "output":
                raise _Fail("violation", f"the kernel writes to its output {ptr.arg} with an atomic (each element is "
                                         f"to be stored once)")
            if info is None or info.role != "integrity_sums" or "add" not in op.text.split(",")[0]:
                raise _Fail("violation", f"the kernel writes to {ptr.arg} with an atomic other than an add into an "
                                         f"integrity sums buffer")
            r = T("the old value of an atomic add")
        elif n == "scf.yield":
            env["__yield__"] = args
            return
        elif n == "tt.store":
            self._store(args)
            return
        elif n in ("tt.return", "tt.func", "module"):
            return
        else:
            raise Unmodelled(f"the op {n} is not modelled")
        for name in op.results[:1]:
            env[name] = r

    def _shape_op(self, n, op, x, shape):
        if isinstance(x, T):
            return x
        if isinstance(x, Ptr):
            return x if x.taint is not None else Ptr(x.arg, self._shape_op(n, op, x.off, shape))
        if isinstance(x, Mk):
            return Mk(shape, [self._shape_op(n, op, f, shape) for f in x.f])
        if isinstance(x, F):
            if x.kind == "leaf":
                return F("leaf", arg=x.arg, off=None if x.off is None else self._shape_op(n, op, x.off, shape),
                         mask=None if x.mask is None else self._shape_op(n, op, x.mask, shape), taint=x.taint,
                         stash=x.stash)
            if x.kind == "opaque":
                return x
            if x.kind == "sel":
                c = x.cond if isinstance(x.cond, T) else self._shape_op(n, op, x.cond, shape)
                return F("sel", parts=[self._shape_op(n, op, p, shape) for p in x.parts], cond=c)
            raise Unmodelled(f"{n} of a {x.kind} value")
        if not isinstance(x, E):
            raise Unmodelled(f"{n} of {type(x).__name__}")
        if n == "tt.splat":
            if x.shape:
                raise Unmodelled("a splat of a tensor")
            return E(shape, s=x.s, b=x.b, w=x.w)
        if n == "tt.expand_dims":
            axis = int(re.search(r'axis = (\d+)', op.text).group(1))
            new = x.shape[:axis] + (1,) + x.shape[axis:]
            if x.d is not None:
                return E(new, d=np.expand_dims(x.d, axis + 1), b=x.b, w=x.w)
            return E(new, s=x.s, v={(p + 1 if p >= axis else p): a for p, a in x.v.items()}, b=x.b, w=x.w)
        if n == "tt.broadcast":
            if x.d is not None:
                return E(shape, d=np.broadcast_to(x.d, (x.d.shape[0],) + tuple(shape)), b=x.b, w=x.w)
            s, v = x.s, {}
            for p, a in x.v.items():
                if x.shape[p] == 1 and shape[p] > 1:
                    s = s + a[:, 0]
                else:
                    v[p] = a
            return E(shape, s=s, v=v, b=x.b, w=x.w)
        return self._dense(shape, x.full().reshape((-1,) + tuple(shape)), x.b, x.w)

    def _for(self, op, env):
        lb, ub, st = (self._get(env, op.extra[k]) for k in ("lb", "ub", "step"))
        for v in (lb, ub, st):
            if not isinstance(v, E) or not v.scalar_only() or np.unique(v.s).size != 1:
                raise Unmodelled("a loop whose bounds differ between programs or depend on data")
        if "unsignedCmp" in op.text:
            raise Unmodelled("a loop that compares its bounds unsigned")
        w = lb.w
        if w is None or w == 1 or ub.w != w or st.w != w:
            raise Unmodelled("a loop whose bounds are not integers of one width")
        lo, hi, step = int(lb.s.flat[0]), int(ub.s.flat[0]), int(st.s.flat[0])
        if step <= 0:
            raise Unmodelled("a loop with a non-positive step")
        if lo < hi and (hi - 1) + step >= (1 << (w - 1)):
            raise Unmodelled(f"a loop whose {w}-bit induction variable can overflow")
        carried = [self._get(env, v) for _k, v in op.extra["iter"]]
        names = [k for k, _v in op.extra["iter"]]
        for it in range(lo, hi, step):
            inner = dict(env)
            inner[op.extra["iv"]] = E((), s=np.full((1,), it, dtype=np.int64), w=w)
            for k, v in zip(names, carried):
                inner[k] = v
            self._ops(op.body, inner)
            carried = inner.get("__yield__", carried)
        base = op.results[0].split("#")[0] if op.results else None
        for i, v in enumerate(carried):
            if base:
                env[f"{base}#{i}"] = v
        if len(op.results) == 1 and carried:
            env[op.results[0]] = carried[0]

    _IN_DATA_BRANCH = ("tt.store", "tt.dot", "tt.atomic_rmw", "scf.for")

    def _if(self, op, env, cond):
        """scf.if. A condition read from data (the in-kernel integrity check of L5.4e): neither branch is followed;
        every result is data, and a branch may not store, dot, loop or do an atomic. A condition the launch decides
        alike for every program: that branch is followed."""
        branches = [op.body, op.extra.get("else", [])]
        if isinstance(cond, T):
            def walk(ops):
                for o in ops:
                    if o.name in self._IN_DATA_BRANCH:
                        raise Unmodelled(f"{o.name} under a condition read from data")
                    walk(o.body)
                    walk(o.extra.get("else", []))
            for b in branches:
                walk(b)
            for name in op.results:
                env[name] = T(f"a value chosen by a condition read from data ({cond.why})")
            return
        c = self._int(cond)
        if not isinstance(c, E) or not c.scalar_only() or np.unique(c.s).size != 1:
            raise Unmodelled("a branch whose condition differs between programs")
        inner = dict(env)
        self._ops(branches[0] if int(c.s.flat[0]) else branches[1], inner)
        values = inner.get("__yield__", [])
        if len(values) < len(op.results):
            raise Unmodelled("a branch that yields fewer values than the op has results")
        for name, v in zip(op.results, values):
            env[name] = v

    @staticmethod
    def _has_dot(x):
        return x.kind == "dot" or (x.kind == "mul" and any(p.kind == "dot" for p in x.parts))

    def _acc_of(self, x) -> Optional[Acc]:
        """x as a sum of checked terms from zero: an exact zero is the empty sum, a scaled dot term (checked here)
        a sum of one; None for anything else."""
        if not isinstance(x, F):
            return None
        if _zero(x):
            return Acc()
        if x.kind == "acc":
            return x.stash
        if self._has_dot(x):
            t = self._check_term(x)
            P = self.P
            kv = _bp(t["k_valid"], P, np.shape(t["k_valid"])[-1])
            lo, cnt = _ranges(_bp(t["k"], P, kv.shape[1]), kv, "a contraction tile reads one k twice",
                              "a contraction tile whose k are not one range")
            R, C = np.shape(t["rows"])[-1], np.shape(t["cols"])[-1]
            return Acc(t["src"], _bp(t["rows"], P, R), _bp(t["rows_valid"], P, R), _bp(t["cols"], P, C),
                       _bp(t["cols_valid"], P, C), t["K"], cnt > 0, ((lo, cnt),))
        return None

    # --- the meaning check -------------------------------------------------------------------------------------------

    def _check_term(self, term):
        parts = term.parts if term.kind == "mul" else [term]
        dots = [p for p in parts if p.kind == "dot"]
        if len(dots) != 1:
            raise Unmodelled("a term with more than one dot")
        a, b = dots[0].parts
        for leaf, role in ((a, "activation"), (b, "weight")):
            info = self.binding.get(leaf.arg)
            if info is None or info.role != role:
                raise _Fail("unproven", f"the dot reads {leaf.arg}, which is not issued as {role}")
            if leaf.off is None:
                raise _Fail("unproven", f"the dot reads {leaf.arg} at an address chosen by data ({leaf.taint})")
        ai, bi = self.binding[a.arg], self.binding[b.arg]
        if len(ai.shape) != 2 or len(bi.shape) != 2 or int(ai.shape[1]) != int(bi.shape[1]):
            raise Unmodelled(f"{a.arg} {ai.shape} and {b.arg} {bi.shape} do not share one contraction length")
        scales = []
        for f in parts:
            if f.kind == "dot":
                continue
            if f.kind == "opaque" and (f.stash or {}).get("const"):
                v = f.stash.get("value")
                if v == 1.0:
                    continue
                if v is None:
                    raise Unmodelled("a term multiplied by a constant this module cannot read")
                raise _Fail("violation", f"the scaled dot term is also multiplied by the constant {v}")
            scales.append(f)
        fast = _Fast(self, a, b) if FAST and not self.went_dense else None
        if fast is not None and not fast.ok:
            fast = None
            if not self.allow_dense:
                raise _Dense()
        seen = {"activation_scale": 0, "weight_scale": 0}
        for f in scales:
            alts = self._alternatives(f)
            roles = {self._scale_role(alt) for alt, _ in alts}
            if len(roles) != 1:
                raise Unmodelled("a factor chosen between an activation scale and a weight scale")
            seen[roles.pop()] += 1
            for alt, data_choice in alts:
                role = self._scale_role(alt)
                try:
                    if fast is not None and fast.scale(alt, role):
                        continue
                    if not self.allow_dense:
                        raise _Dense()
                    self.went_dense = True
                    _dense_scale(self, alt, role, a, b)
                except _Fail as e:
                    if data_choice and e.verdict == "violation":
                        raise _Fail("possible", f"a scale chosen at run time by {data_choice}: {e.why}", e.example)
                    raise
        if not seen["activation_scale"] or not seen["weight_scale"]:
            missing = [k for k, v in seen.items() if not v]
            raise _Fail("violation", f"the dot's product is not multiplied by its {' and '.join(missing)}")
        extra = {k: v for k, v in seen.items() if v > 1}
        if extra:
            raise _Fail("violation", "the dot's product is multiplied by " +
                        " and ".join(f"{v} {k.replace('_', ' ')}s" for k, v in extra.items()))
        self.terms += 1
        K = int(ai.shape[1])
        if fast is not None:
            self.elements += fast.elements
            return {"src": (a.arg, b.arg), "K": K, "rows": fast.m, "rows_valid": fast.vR, "cols": fast.n,
                    "cols_valid": fast.vC, "k": fast.k, "k_valid": fast.vT}
        rows, cols, count, k, kv = _dense_rows_cols(self, a, b)
        self.elements += count
        return {"src": (a.arg, b.arg), "K": K, "rows": rows, "rows_valid": rows >= 0, "cols": cols,
                "cols_valid": cols >= 0, "k": k, "k_valid": kv}

    def _alternatives(self, f):
        if f.kind == "leaf":
            return [(f, None)]
        if f.kind == "sel":
            c = f.cond
            if isinstance(c, T):
                return [(x, c.why) for p in f.parts for x, _ in self._alternatives(p)]
            raise Unmodelled("a scale selected by a condition computed from the launch (not modelled yet)")
        raise Unmodelled(f"a factor of kind {f.kind}")

    def _scale_role(self, leaf):
        info = self.binding.get(leaf.arg)
        if info is None or info.role not in ("activation_scale", "weight_scale"):
            raise _Fail("violation", f"the product is multiplied by {leaf.arg}, which is not issued as a scale")
        return info.role

    def _store(self, args):
        """A store: only to the output, only a complete sum of terms, each value at its own C[m, n]. The stored
        rows and columns of each program are kept (self.stores) for the check over the whole launch."""
        ptr, val = args[0], args[1]
        mask = args[2] if len(args) > 2 else None
        if not isinstance(ptr, Ptr):
            raise Unmodelled("a store through a value that is not a pointer")
        info = self.binding.get(ptr.arg)
        if info is None or info.role != "output":
            raise _Fail("violation", f"the kernel writes to {ptr.arg}, which is not its output")
        if ptr.taint is not None:
            raise _Fail("unproven", f"the output address is chosen by data ({ptr.taint})")
        if isinstance(mask, T):
            raise Unmodelled("a store masked by data")
        acc = self._acc_of(val)
        if acc is None:
            raise _Fail("unproven", "the value stored to the output is not a sum of scaled dot terms "
                                    f"({getattr(val, 'kind', type(val).__name__)})")
        if not acc.ranges:
            raise _Fail("violation", "the kernel stores an exact zero, not the product, to its output")
        if len(info.shape) != 2 or len(info.stride) != 2:
            raise Unmodelled(f"an output of shape {info.shape}")
        M, N = int(info.shape[0]), int(info.shape[1])
        s0, s1 = int(info.stride[0]), int(info.stride[1])
        if s1 != 1 or s0 < N:
            raise Unmodelled(f"an output whose rows overlap or whose columns are not contiguous (strides {s0}, {s1})")
        ai, bi = self.binding[acc.src[0]], self.binding[acc.src[1]]
        if (M, N) != (int(ai.shape[0]), int(bi.shape[0])):
            raise _Fail("violation", f"the output is {M}x{N}, the product of {acc.src[0]} and {acc.src[1]} is "
                                     f"{int(ai.shape[0])}x{int(bi.shape[0])}")
        P = self.P
        off = ptr.off
        if len(off.shape) != 2 or np.shape(acc.rows)[-1] != off.shape[0] or np.shape(acc.cols)[-1] != off.shape[1]:
            raise Unmodelled("a store whose tile is not the sum's rows x columns")
        R, C = off.shape
        rows, cols = _bp(acc.rows, P, R), _bp(acc.cols, P, C)
        rv, cv = _bp(acc.rows_valid, P, R), _bp(acc.cols_valid, P, C)
        sep = _separate_mask(mask, off.shape, P) if FAST and off.d is None and off.axes() <= {0, 1} else None
        if sep is not None:
            vS, v0, v1 = sep
            sr = v0 & vS.reshape(-1, 1)                  # stored lanes along the rows, along the columns
            sc = v1 & vS.reshape(-1, 1)
            live = sr.any(axis=1) & sc.any(axis=1)
            sr, sc = sr & live[:, None], sc & live[:, None]
            if np.any(sr & ~rv) or np.any(sc & ~cv):
                raise _Fail("violation", "the kernel stores output lanes whose sum has no row or column of the "
                                         "operands (masked-out operand lanes)")
            u = _bp(off.v.get(0, np.zeros((1, R), dtype=np.int64)), P, R) + _bp(off.s.reshape(-1, 1), P, 1) \
                - rows * s0
            w = _bp(off.v.get(1, np.zeros((1, C), dtype=np.int64)), P, C) - cols * s1
            uc, wc = _constant_over(u, sr), _constant_over(w, sc)
            if uc is None or wc is None or np.any((uc + wc)[live] != 0):
                raise _Fail("violation", "an output value is stored where another row or column belongs")
        else:
            if not self.allow_dense:
                raise _Dense()
            self.went_dense = True
            want = rows[:, :, None] * s0 + cols[:, None, :] * s1
            offd = np.broadcast_to(off.full(), np.broadcast_shapes(off.full().shape, want.shape))
            want = np.broadcast_to(want, offd.shape)
            m = np.ones(offd.shape, dtype=bool) if mask is None else \
                np.array(np.broadcast_to(_as_bool_full(mask), offd.shape))
            if np.any(m & ~(rv[:, :, None] & cv[:, None, :])):
                raise _Fail("violation", "the kernel stores output lanes whose sum has no row or column of the "
                                         "operands (masked-out operand lanes)")
            if (m & (offd != want)).any():
                raise _Fail("violation", "an output value is stored where another row or column belongs")
            sr, sc = m.any(axis=2), m.any(axis=1)
            if not np.array_equal(m, sr[:, :, None] & sc[:, None, :]):
                raise Unmodelled("a store mask that is not a set of rows times a set of columns")
            live = sr.any(axis=1) & sc.any(axis=1)
        bad = _coverage(acc, live)
        if bad.any():
            p = int(np.nonzero(bad)[0][0])
            raise _Fail("violation", _coverage_why(acc, p), {"program_chunk_index": p})
        rlo, rcnt = _ranges(rows, sr, "one output row is stored twice by one program",
                            "the rows a program stores are not one range", twice="unproven")
        clo, ccnt = _ranges(cols, sc, "one output column is stored twice by one program",
                            "the columns a program stores are not one range", twice="unproven")
        self.stores.append((rlo[live], rcnt[live], clo[live], ccnt[live]))


def _bp(x, P, n):
    """x broadcast to (P, n)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.broadcast_to(x, (P, n))


_BIG = np.iinfo(np.int64).max


def _bounds(x):
    """(least, greatest) element of an integer value over all programs, exactly (the axes of one program are
    independent, so per program the extremes are the sums of the parts' extremes)."""
    if x.d is not None:
        if x.d.size == 0:
            return 0, 0
        return int(x.d.min()), int(x.d.max())
    lo = np.asarray(x.s, dtype=np.int64).reshape(-1)
    hi = lo
    for arr in x.v.values():
        if arr.size == 0:
            continue
        lo = lo + arr.min(axis=1)
        hi = hi + arr.max(axis=1)
    return int(lo.min()), int(hi.max())


def _exact(x, what):
    """x, when every part of it is within the exact range; otherwise the computation is not modelled."""
    parts = [x.d] if x.d is not None else [x.s] + list(x.v.values())
    for p in parts:
        p = np.asarray(p)
        if p.size and (int(p.max()) > _LIMIT or int(p.min()) < -_LIMIT):
            raise Unmodelled(f"{what}: a value beyond the checker's exact integer range (2^60)")
    return x


def _ranges(vals, valid, twice_why, gaps_why, twice="violation"):
    """Per program, the values `vals` (P, n) takes where `valid` (P, n), as one range: (lo (P,), count (P,)); lo is
    0 where nothing is valid. A value taken twice raises `twice` (the verdict) with `twice_why`; values that do not
    form one range are not modelled. The usual form (valid lanes in one run, stepping by one) is decided without
    sorting."""
    vals, valid = np.asarray(vals), np.asarray(valid, dtype=bool)
    cnt = valid.sum(axis=1).astype(np.int64)
    lo = np.where(cnt > 0, np.where(valid, vals, _BIG).min(axis=1), 0)
    if vals.shape[1] < 2:
        return lo, cnt
    runs = valid[:, 0].astype(np.int64) + (valid[:, 1:] & ~valid[:, :-1]).sum(axis=1)
    pair = valid[:, 1:] & valid[:, :-1]
    if np.all(runs <= 1) and not np.any(pair & (vals[:, 1:] - vals[:, :-1] != 1)):
        return lo, cnt
    s = np.sort(np.where(valid, vals, _BIG), axis=1)
    idx = np.arange(vals.shape[1])[None, :]
    inrun = idx < cnt[:, None]
    if np.any(inrun[:, 1:] & (s[:, 1:] == s[:, :-1])):
        raise _Fail(twice, twice_why)
    if np.any(inrun & (s != lo[:, None] + idx)):
        raise Unmodelled(gaps_why)
    return lo, cnt


def _coverage(acc, live):
    """Programs of `live` (P,) whose sum does not cover k = 0 .. K-1 exactly once."""
    lo = np.stack([np.broadcast_to(r[0], live.shape) for r in acc.ranges])       # [terms, P]
    cnt = np.stack([np.broadcast_to(r[1], live.shape) for r in acc.ranges])
    key = np.where(cnt > 0, lo, _BIG)
    order = np.argsort(key, axis=0, kind="stable")
    los, cs = np.take_along_axis(key, order, 0), np.take_along_axis(cnt, order, 0)
    start = np.cumsum(cs, axis=0) - cs                # where each range must begin to follow the ones before
    ok = np.all((cs == 0) | (los == start), axis=0) & (cs.sum(axis=0) == acc.K)
    return live & ~ok


def _coverage_why(acc, p) -> str:
    seen = np.zeros(acc.K + 1, dtype=np.int64)
    outside = 0
    for lo, cnt in acc.ranges:
        lo, cnt = np.asarray(lo).reshape(-1), np.asarray(cnt).reshape(-1)
        a, c = int(lo[p if lo.size > 1 else 0]), int(cnt[p if cnt.size > 1 else 0])
        if c <= 0:
            continue
        if a < 0 or a + c > acc.K:
            outside += c
            a, c = max(a, 0), max(0, min(a + c, acc.K) - max(a, 0))
        seen[a: a + c] += 1
    seen = seen[: acc.K]
    missing, twice = int((seen == 0).sum()), int((seen > 1).sum())
    parts = []
    if missing:
        first = int(np.nonzero(seen == 0)[0][0])
        parts.append(f"{missing} of the K={acc.K} contraction indices are never added (the first k={first})")
    if twice:
        parts.append(f"{twice} are added more than once")
    if outside:
        parts.append(f"{outside} lie outside 0..K-1")
    return "the value stored is not the whole product: " + "; ".join(parts or ["its k do not tile 0..K-1"]) + \
        f" ({len(acc.ranges)} terms)"


def _tiling(stores, M, N):
    """None when the stores of a launch (row lo, rows, column lo, columns per storing program) cover the M x N
    output exactly once; otherwise (verdict, why, example)."""
    if not stores:
        return "violation", "the kernel never stores to its output", None
    r0 = np.concatenate([s[0] for s in stores])
    rc = np.concatenate([s[1] for s in stores])
    c0 = np.concatenate([s[2] for s in stores])
    cc = np.concatenate([s[3] for s in stores])
    nz = (rc > 0) & (cc > 0)
    r0, r1, c0, c1 = r0[nz], r0[nz] + rc[nz], c0[nz], c0[nz] + cc[nz]
    if r0.size == 0:
        return "violation", "the kernel never stores to its output", None
    if np.any(r0 < 0) or np.any(r1 > M) or np.any(c0 < 0) or np.any(c1 > N):
        return "violation", f"a store outside the {M}x{N} output", None
    rs = np.unique(np.concatenate([[0, M], r0, r1]))
    cs = np.unique(np.concatenate([[0, N], c0, c1]))
    if rs.size * cs.size > (1 << 26):
        return "unproven", "the stores' rows and columns are too irregular to check", None
    i0, i1, j0, j1 = (np.searchsorted(rs, r0), np.searchsorted(rs, r1), np.searchsorted(cs, c0),
                      np.searchsorted(cs, c1))
    d = np.zeros((rs.size + 1, cs.size + 1), dtype=np.int64)
    np.add.at(d, (i0, j0), 1)
    np.add.at(d, (i1, j0), -1)
    np.add.at(d, (i0, j1), -1)
    np.add.at(d, (i1, j1), 1)
    cover = d.cumsum(axis=0).cumsum(axis=1)[: rs.size - 1, : cs.size - 1]   # cell [rs[i], rs[i+1]) x [cs[j], ..)
    never = np.argwhere(cover == 0)
    if never.size:
        i, j = (int(x) for x in never[0])
        n = int(sum((rs[a + 1] - rs[a]) * (cs[b + 1] - cs[b]) for a, b in never))
        return "violation", (f"{n} of the {M * N} output elements are never stored (the first block: rows "
                             f"{rs[i]}..{rs[i + 1] - 1}, columns {cs[j]}..{cs[j + 1] - 1})"), None
    many = np.argwhere(cover > 1)
    if many.size:
        i, j = (int(x) for x in many[0])
        return "unproven", (f"output elements are stored more than once (rows {rs[i]}..{rs[i + 1] - 1}, columns "
                            f"{cs[j]}..{cs[j + 1] - 1})"), None
    return None


def _constant_over(vals, valid):
    """Per program, the one value `vals` (P, n) takes where `valid` (P, n); None when it takes more than one."""
    big = np.iinfo(np.int64).max
    lo = np.where(valid, vals, big).min(axis=1)
    hi = np.where(valid, vals, -big).max(axis=1)
    has = valid.any(axis=1)
    if np.any(has & (lo != hi)):
        return None
    return np.where(has, lo, 0)


def _separate_mask(mask, shape, P):
    """(valid per program (P,), valid along axis 0 (P, n0), valid along axis 1 (P, n1)) for a 2-D mask that is an
    AND of conditions on one axis each; None when it is not (or not 2-D)."""
    if len(shape) != 2:
        return None
    n0, n1 = shape
    vS = np.ones((P,), dtype=bool)
    v0 = np.ones((P, n0), dtype=bool)
    v1 = np.ones((P, n1), dtype=bool)
    if mask is None:
        return vS, v0, v1
    factors = mask.f if isinstance(mask, Mk) else [mask]
    for x in factors:
        if not isinstance(x, E) or x.d is not None or len(x.axes()) > 1:
            return None
        if not x.v:
            vS = vS & np.broadcast_to(x.s.astype(bool), (P,))
        elif 0 in x.v:
            v0 = v0 & _bp(x.along(0, n0).astype(bool), P, n0)
        else:
            v1 = v1 & _bp(x.along(1, n1).astype(bool), P, n1)
    return vS, v0, v1


def _coords_sep(off, info, row_axis, valid_row, valid_col, P):
    """(row coordinate (P, n_row), column coordinate (P, n_col)) of a 2-D load whose address is kept apart, in the
    tensor `info` (strides (s0, 1)), the row along `row_axis`; None when the address does not split that way."""
    if off.d is not None or not off.axes() <= {0, 1} or len(off.shape) != 2 or len(info.stride) != 2 \
            or int(info.stride[1]) != 1:
        return None
    col_axis = 1 - row_axis
    s0 = int(info.stride[0])
    rows, cols = info.shape
    n_r, n_c = off.shape[row_axis], off.shape[col_axis]
    rv = _bp(off.v.get(row_axis, np.zeros((1, n_r), dtype=np.int64)), P, n_r)
    cv = _bp(off.v.get(col_axis, np.zeros((1, n_c), dtype=np.int64)), P, n_c)
    sc = _bp(off.s.reshape(-1, 1), P, 1)
    for r, c in ((rv, cv + sc), (rv + sc, cv)):
        okr = np.where(valid_row, (r % s0 == 0) & (r >= 0) & (r // s0 < rows), True)
        okc = np.where(valid_col, (c >= 0) & (c < cols), True)
        if okr.all() and okc.all():
            return r // s0, c
    return None


class _Fast:
    """The check on kept-apart values: per program, vectors along rows, columns and k only."""

    def __init__(self, run, a, b):
        self.run, self.ok = run, False
        P = run.P
        if a.off.d is not None or b.off.d is not None:
            return
        sa = _separate_mask(a.mask, a.off.shape, P)
        sb = _separate_mask(b.mask, b.off.shape, P)
        if sa is None or sb is None:
            return
        self.ai, self.bi = run.binding[a.arg], run.binding[b.arg]
        vSa, vRa, vTa = sa            # A [R, T]: axis 0 the output rows, axis 1 the contraction
        vSb, vTb, vCb = sb            # B [T, C]: axis 0 the contraction, axis 1 the output columns
        ca = _coords_sep(a.off, self.ai, 0, vRa, vTa, P)
        cb = _coords_sep(b.off, self.bi, 1, vCb, vTb, P)
        if ca is None or cb is None:
            return
        self.m, ka = ca               # m [P, R], ka [P, T]
        self.n, kb = cb               # n [P, C], kb [P, T]
        vS = (vSa & vSb).reshape(-1, 1)
        live = vS & vRa.any(axis=1, keepdims=True) & vCb.any(axis=1, keepdims=True)
        if np.any(live & (vTa != vTb)):
            # a k read by one operand and masked out of the other: 0 * x is not 0 for every x the IR cannot see
            raise _Fail("unproven", "the two operands are masked differently along k")
        vT = vTa & vTb & vS
        if np.any(vT & (ka != kb)):
            raise _Fail("violation", "the two operands are read at different k for one contraction index")
        self.vT, self.k = vT, ka
        self.vR = vRa & vS
        self.vC = vCb & vS
        nt = vT.sum(axis=1)
        self.elements = int((self.vR.sum(axis=1) * nt + self.vC.sum(axis=1) * nt).sum())
        self.ok = True

    def scale(self, leaf, role):
        """True when this scale holds for every consumed element (or raises the violation); False when its address
        or mask does not split along the output axis (the caller goes dense)."""
        info = self.run.binding[leaf.arg]
        vinfo = self.ai if role == "activation_scale" else self.bi
        if info.pair != vinfo.serial:
            raise _Fail("violation", f"the scale read from {leaf.arg} is issue {info.serial}, paired with issue "
                                     f"{info.pair}; the operand it multiplies is issue {vinfo.serial}")
        if leaf.off is None:
            raise _Fail("unproven", f"the scale address is chosen by data ({leaf.taint})")
        off = leaf.off
        axis = 0 if role == "activation_scale" else 1
        if off.d is not None or not off.axes() <= {axis} or len(off.shape) != 2 or len(info.stride) != 2:
            return False
        P = self.run.P
        s0, s1 = int(info.stride[0]), int(info.stride[1])
        n_o = off.shape[axis]
        S = _bp(off.along(axis), P, n_o)                            # the scale read, by output row or column
        gk = int(vinfo.block[1])
        if role == "activation_scale":
            wantO, valid_o, what = self.m * s0, self.vR, "rows"
        else:
            wantO, valid_o, what = (self.n // int(vinfo.block[0])) * s0, self.vC, "columns"
        wantT = (self.k // gk) * s1                                 # by k
        read = np.ones(S.shape, dtype=bool)                         # the lanes this load reads
        if leaf.mask is not None:
            sm = _separate_mask(leaf.mask, off.shape, P)
            if sm is None:
                return False
            mS, m0, m1 = sm
            need = (m0 if axis == 0 else m1) & mS.reshape(-1, 1)
            read = need
            if np.any(valid_o & ~need):
                raise _Fail("violation", f"a scale of {leaf.arg} needed by a consumed element is masked out")
        _scale_in_bounds(leaf, info, S, read)
        c = _constant_over(wantT, self.vT)
        if c is None:
            big = np.iinfo(np.int64).max
            lo = np.where(self.vT, wantT, big).min(axis=1)
            hi = np.where(self.vT, wantT, -big).max(axis=1)
            p = int(np.nonzero(self.vT.any(axis=1) & (lo != hi))[0][0])
            groups = sorted(set(int(x) // gk for x in self.k[p][self.vT[p]]))
            raise _Fail("violation", f"one {leaf.arg} value is multiplied into an output {what[:-1]} for k of "
                                     f"{len(groups)} scale groups: the K tile spans groups of {gk} and reads one "
                                     f"scale", {"program_chunk_index": p, "groups_in_tile": groups[:8]})
        want = wantO + c.reshape(-1, 1)
        vo = valid_o & self.vT.any(axis=1).reshape(-1, 1)
        wrong = vo & (S != want)
        if wrong.any():
            p, i = (int(x[0]) for x in np.nonzero(wrong))
            g, w = int(S[p, i]), int(want[p, i])
            ex = {"program_chunk_index": p, "output_index": i,
                  "scale_read": [g // s0 if s0 else g, (g % s0) // s1 if s1 and s0 else g],
                  "scale_declared": [w // s0 if s0 else w, (w % s0) // s1 if s1 and s0 else w]}
            raise _Fail("violation", f"{int(wrong.sum())} output {what} are multiplied by a scale of {leaf.arg} that "
                                     f"is not theirs (block {1 if axis == 0 else int(vinfo.block[0])}x{gk})", ex)
        return True


def _extent(info) -> Optional[int]:
    """How many elements a tensor's layout spans with no gap (a permuted contiguous layout); None otherwise."""
    shape, stride = [int(x) for x in info.shape], [int(x) for x in info.stride]
    if not shape or len(shape) != len(stride) or any(n <= 0 for n in shape):
        return None
    need = 1
    for s, n in sorted((s, n) for s, n in zip(stride, shape) if n > 1):
        if s != need:
            return None
        need *= n
    return need


def _scale_in_bounds(leaf, info, addr, read):
    """Every lane a scale load reads (its own mask aside) must address an element of the scale tensor - also in a K
    tile whose operands are all masked out: what is read there multiplies a zero product, and an out-of-bounds value
    can be infinite or NaN (0 x inf is NaN). Before 2026-10-03 (L5.4d) such lanes were not checked."""
    ext = _extent(info)
    if ext is None:
        raise Unmodelled(f"{leaf.arg}: a scale layout with gaps (strides {tuple(info.stride)}); reads outside it are "
                         f"not decided")
    addr = np.asarray(addr)
    out = np.broadcast_to(read, addr.shape) & ((addr < 0) | (addr >= ext))
    if out.any():
        idx = tuple(int(x[0]) for x in np.nonzero(out))
        raise _Fail("violation", f"{int(out.sum())} lanes read a scale of {leaf.arg} outside its "
                                 f"{tuple(int(x) for x in info.shape)} elements (an offset of "
                                 f"{int(addr[idx])} of {ext})", {"program_chunk_index": idx[0]})


def _dense_coords(run, leaf, role):
    """(row, col) of every element a load reads, element by element, with its validity mask."""
    info = run.binding[leaf.arg]
    if len(info.stride) != 2 or int(info.stride[1]) != 1:
        raise Unmodelled(f"{leaf.arg}: a layout whose innermost stride is not 1")
    s0 = int(info.stride[0])
    off = leaf.off.full()
    row, col = off // s0, off % s0
    valid = np.ones(off.shape, dtype=bool) if leaf.mask is None else np.broadcast_to(_as_bool_full(leaf.mask),
                                                                                    off.shape)
    if (valid & ((col >= info.shape[1]) | (row >= info.shape[0]) | (off < 0))).any():
        raise _Fail("violation", f"the dot reads {leaf.arg} outside its {info.shape} elements")
    return row, col, valid, info


def _dense_rows_cols(run, a, b):
    """(rows (P, R), columns (P, C), elements, k (P, T), k valid (P, T)), element by element; -1 where a row,
    column or k is not read."""
    am, ak, av, _ = _dense_coords(run, a, "activation")      # [P, R, T]
    bn, bk, bv, _ = _dense_coords(run, b, "weight")          # [P, T, C]
    rv, tva = av.any(axis=2), av.any(axis=1)
    cv, tvb = bv.any(axis=1), bv.any(axis=2)
    if not np.array_equal(av, rv[:, :, None] & tva[:, None, :]) or \
            not np.array_equal(bv, tvb[:, :, None] & cv[:, None, :]):
        raise Unmodelled("an operand mask that is not a row (or column) condition times a k condition")
    live = (rv.any(axis=1) & cv.any(axis=1))[:, None]
    if np.any(live & (tva != tvb)):
        raise _Fail("unproven", "the two operands are masked differently along k")
    ka = np.where(av, ak, -1).max(axis=1)
    kb = np.where(bv, bk, -1).max(axis=2)
    if not np.array_equal(np.where(av, ak, ka[:, None, :]), np.broadcast_to(ka[:, None, :], ak.shape)) or \
            not np.array_equal(np.where(bv, bk, kb[:, :, None]), np.broadcast_to(kb[:, :, None], bk.shape)):
        raise _Fail("violation", "an operand is read at different k along one contraction index")
    both = (ka >= 0) & (kb >= 0)
    if not np.array_equal(np.where(both, ka, 0), np.where(both, kb, 0)):
        raise _Fail("violation", "the two operands are read at different k for one contraction index")
    rows = np.where(av, am, -1).max(axis=2)
    cols = np.where(bv, bn, -1).max(axis=1)
    if not np.array_equal(np.where(av, am, rows[:, :, None]), np.broadcast_to(rows[:, :, None], am.shape)) or \
            not np.array_equal(np.where(bv, bn, cols[:, None, :]), np.broadcast_to(cols[:, None, :], bn.shape)):
        raise _Fail("violation", "an operand row changes along the contraction")
    return rows, cols, int(av.sum() + bv.sum()), ka, both & live


def _dense_scale(run, leaf, role, a, b):
    """The scale check element by element (when an address or a mask does not split along the axes)."""
    info = run.binding[leaf.arg]
    if role == "activation_scale":
        vrow, vk, valid, vinfo = _dense_coords(run, a, "activation")
        axis, gr = "row", 1
    else:
        vrow, vk, valid, vinfo = _dense_coords(run, b, "weight")
        axis, gr = "col", int(vinfo.block[0])
    gk = int(vinfo.block[1])
    if info.pair != vinfo.serial:
        raise _Fail("violation", f"the scale read from {leaf.arg} is issue {info.serial}, paired with issue "
                                 f"{info.pair}; the operand it multiplies is issue {vinfo.serial}")
    if leaf.off is None:
        raise _Fail("unproven", f"the scale address is chosen by data ({leaf.taint})")
    s0, s1 = int(info.stride[0]), int(info.stride[1])
    off = leaf.off.full()
    if axis == "row":
        if not np.array_equal(off, np.broadcast_to(off[:, :, :1], off.shape)):
            raise Unmodelled("an activation scale that varies along the output columns")
        got = off[:, :, 0][:, :, None]
    else:
        if not np.array_equal(off, np.broadcast_to(off[:, :1, :], off.shape)):
            raise Unmodelled("a weight scale that varies along the output rows")
        got = off[:, 0, :][:, None, :]
    want = (vrow // gr) * s0 + (vk // gk) * s1
    read = np.ones(off.shape, dtype=bool)
    if leaf.mask is not None:
        read = np.broadcast_to(_as_bool_full(leaf.mask), off.shape)
        sm = _as_bool_full(leaf.mask)
        sm = sm[:, :, 0][:, :, None] if axis == "row" else sm[:, 0, :][:, None, :]
        if (valid & ~np.broadcast_to(sm, valid.shape)).any():
            raise _Fail("violation", f"a scale of {leaf.arg} needed by a consumed element is masked out")
    _scale_in_bounds(leaf, info, off, read)
    wrong = valid & (np.broadcast_to(got, want.shape) != want)
    if wrong.any():
        idx = tuple(int(x[0]) for x in np.nonzero(wrong))
        g, w = int(np.broadcast_to(got, want.shape)[idx]), int(want[idx])
        ex = {"program_chunk_index": idx[0], "operand_row": int(vrow[idx]), "k": int(vk[idx]),
              "scale_read": [g // s0 if s0 else g, (g % s0) // s1 if s1 and s0 else g],
              "scale_declared": [w // s0 if s0 else w, (w % s0) // s1 if s1 and s0 else w]}
        raise _Fail("violation", f"{int(wrong.sum())} consumed elements are multiplied by a scale of {leaf.arg} that "
                                 f"is not theirs (block {gr}x{gk})", ex)


def _grid_pids(grid, start, stop):
    """Program ids (x, y, z) of programs start..stop in launch order (x fastest)."""
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    lin = np.arange(start, stop, dtype=np.int64)
    return [lin % gx, (lin // gx) % gy, lin // (gx * gy)]


_POINTER_KEEPS = ("tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.addptr", "tt.bitcast", "tt.trans",
                  "tt.advance", "tt.make_tensor_ptr")
_WRITES = ("tt.store", "tt.atomic_rmw", "tt.atomic_cas", "tt.descriptor_store")


def written_args(ttir: str):
    """The names of a kernel's pointer arguments it may write through (stores and atomics), followed back from each
    write's pointer to the arguments it was made from, through pointer arithmetic, broadcasts, selects, branches and
    loop-carried values; None when a written pointer comes from something else (an integer cast to a pointer, a
    pointer loaded from memory): then every pointer argument may be written. No launch values are needed (writeguard,
    L5.4e: a Triton launch is refused only for the arguments it writes)."""
    return _args_through(ttir, _WRITES)


_READS = ("tt.load", "tt.atomic_rmw", "tt.atomic_cas", "tt.descriptor_load")


def read_args(ttir: str):
    """The names of a kernel's pointer arguments it may read through (loads and atomics), as written_args finds the
    written ones; None when a read pointer comes from something else."""
    return _args_through(ttir, _READS)


def _args_through(ttir: str, kinds):
    f = parse(ttir)
    bases = {name: {name[1:]} for name, typ in f.args if "!tt.ptr" in typ}
    written = set()
    unknown = []

    def add(name, srcs):
        if not srcs:
            return False
        cur = bases.setdefault(name, set())
        if srcs <= cur:
            return False
        cur |= srcs
        return True

    def visit(ops):
        changed = False
        for op in ops:
            n = op.name
            if n in _POINTER_KEEPS and op.operands and op.results:
                changed |= add(op.results[0], bases.get(op.operands[0], set()))
            elif n == "arith.select" and len(op.operands) == 3 and op.results:
                changed |= add(op.results[0], bases.get(op.operands[1], set()) | bases.get(op.operands[2], set()))
            elif n == "scf.for":
                names = [k for k, _v in op.extra.get("iter", [])]
                for k, v in op.extra.get("iter", []):
                    changed |= add(k, bases.get(v, set()))
                changed |= visit(op.body)
                ys = next((o for o in op.body if o.name == "scf.yield"), None)
                base = op.results[0].split("#")[0] if op.results else None
                for i, v in enumerate(ys.operands if ys is not None else []):
                    if i < len(names):
                        changed |= add(names[i], bases.get(v, set()))
                    if base:
                        changed |= add(f"{base}#{i}", bases.get(v, set()) | bases.get(names[i] if i < len(names)
                                                                                     else "", set()))
                if len(op.results) == 1 and names:
                    changed |= add(op.results[0], bases.get(names[0], set()))
            elif n == "scf.if":
                for branch in (op.body, op.extra.get("else", [])):
                    changed |= visit(branch)
                    ys = next((o for o in branch if o.name == "scf.yield"), None)
                    for name, v in zip(op.results, ys.operands if ys is not None else []):
                        changed |= add(name, bases.get(v, set()))
            elif n in kinds and op.operands:
                b = bases.get(op.operands[0])
                if b:
                    written.update(b)
                else:
                    unknown.append(op.operands[0])
            else:
                changed |= visit(op.body)
        return changed

    for _ in range(64):                 # to a fixed point: loops carry pointers around
        unknown.clear()
        if not visit(f.body):
            break
    return None if unknown else sorted(written)


def check_launch(ttir: str, binding: Dict[str, Tensor], ints: Dict[str, int], grid,
                 chunk_elements: int = CHUNK_ELEMENTS) -> Verdict:
    """The verdict on one launch: the kernel's IR, what each pointer argument holds (binding, by parameter name),
    the integer arguments by name (constexpr ones are folded into the IR already), the grid. The values are kept
    apart along their axes when they can be (fast); when one cannot, the launch is evaluated again element by
    element (dense, in smaller chunks of programs)."""
    t0 = time.perf_counter()
    try:
        fn = parse(ttir)
    except Unmodelled as e:
        return Verdict("unproven", str(e), seconds=time.perf_counter() - t0)
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    total = int(gx) * int(gy) * int(gz)
    biggest = longest = 1
    for op in _walk(fn.body):
        sh = _shape(op.rtype) or (1,)
        biggest = max(biggest, int(np.prod(sh)))
        longest = max(longest, max(sh))
    outs = [t for t in binding.values() if t.role == "output"]
    if len(outs) != 1 or len(outs[0].shape) != 2:
        return Verdict("unproven", f"the launch has {len(outs)} tensors bound as its output (one is needed)",
                       seconds=time.perf_counter() - t0)
    M, N = (int(x) for x in outs[0].shape)
    dense = False
    for attempt in (("fast", "dense") if FAST else ("dense",)):
        per = max(1, chunk_elements // (longest if attempt == "fast" else biggest))
        terms = elements = 0
        stores = []
        try:
            for start in range(0, total, per):
                run = _Run(fn, binding, ints, _grid_pids((gx, gy, gz), start, min(total, start + per)),
                           dense=attempt == "dense")
                run.run()
                terms += run.terms
                elements += run.elements
                stores += run.stores
                dense = dense or run.went_dense
            break
        except _Dense:
            continue
        except _Fail as e:
            return Verdict(e.verdict, e.why, terms, elements, total, time.perf_counter() - t0, e.example,
                           dense or attempt == "dense")
        except Unmodelled as e:
            return Verdict("unproven", str(e), terms, elements, total, time.perf_counter() - t0, None,
                           dense or attempt == "dense")
    tiled = _tiling(stores, M, N)
    if tiled is not None:
        return Verdict(tiled[0], tiled[1], terms, elements, total, time.perf_counter() - t0, tiled[2], dense)
    return Verdict("proven", f"every element of the {M}x{N} output is stored once, as the whole product: the sum "
                             f"over all K of its activation and weight values times their producers' scales "
                             f"({terms} scaled dot terms over {total} programs)",
                   terms, elements, total, time.perf_counter() - t0, None, dense)


def _walk(ops):
    for op in ops:
        yield op
        yield from _walk(op.body)
