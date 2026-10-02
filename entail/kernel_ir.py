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


_LOC = re.compile(r'\s*loc\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)')
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
    body, _ = _block(lines, i + 1)
    return Func(name, args, body)


def _block(lines, i):
    ops = []
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("}"):
            return ops, i + 1
        op = _op(line)
        i += 1
        if line.endswith("{"):
            op.body, i = _block(lines, i)
        ops.append(op)
    return ops, i


def _op(line: str) -> Op:
    results = []
    if " = " in line and line.startswith("%"):
        lhs, rhs = line.split(" = ", 1)
        results = _results(lhs)
    else:
        rhs = line
    rhs = rhs.rstrip("{").strip()
    name = rhs.split()[0]
    rest = rhs[len(name):].strip()
    op = Op(results, name, [], rest)
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
    if "->" in op.rtype:
        op.rtype = op.rtype.split("->")[-1].strip()
    if " to " in op.rtype:                  # a conversion prints "source type to result type"
        op.rtype = op.rtype.split(" to ")[-1].strip()
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

class E:
    """An integer (or 0/1 boolean: b) tensor over the programs: s (P|1,), v {axis: (P|1, n)}, or dense d."""
    __slots__ = ("shape", "s", "v", "d", "b")

    def __init__(self, shape, s=None, v=None, d=None, b=False):
        self.shape, self.s, self.v, self.d, self.b = tuple(shape), s, dict(v or {}), d, b
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


def _tdiv(a, b):
    if np.any(b == 0):
        raise Unmodelled("a division by zero in the index arithmetic")
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) ^ (b < 0), -q, q)


_GENERAL = {"arith.muli": np.multiply, "arith.minsi": np.minimum, "arith.maxsi": np.maximum,
            "arith.minui": np.minimum, "arith.maxui": np.maximum,
            "arith.andi": np.bitwise_and, "arith.ori": np.bitwise_or, "arith.xori": np.bitwise_xor,
            "arith.divsi": _tdiv, "arith.divui": _tdiv,
            "arith.remsi": lambda a, b: a - b * _tdiv(a, b), "arith.remui": lambda a, b: a - b * _tdiv(a, b),
            "arith.shli": np.left_shift, "arith.shrsi": np.right_shift, "arith.shrui": np.right_shift}
_PASS = ("arith.extsi", "arith.extui", "arith.trunci", "arith.index_cast", "arith.truncf", "arith.extf",
         "tt.bitcast", "arith.sitofp", "arith.fptosi")


class _Run:
    def __init__(self, fn: Func, binding: Dict[str, Tensor], ints: Dict[str, int], pids, dense: bool):
        self.fn, self.binding, self.ints, self.pids, self.allow_dense = fn, binding, ints, pids, dense
        self.P = int(pids[0].shape[0])
        self.terms = 0
        self.elements = 0
        self.went_dense = False
        self.stores = []           # per store to the output: (row lo, rows, column lo, columns) of the programs storing

    # -- kept-apart arithmetic --

    def _dense(self, shape, arr, b=False):
        if not self.allow_dense:
            raise _Dense()
        self.went_dense = True
        return E(shape, d=np.asarray(arr), b=b)

    def _general(self, f, args, shape, b=False):
        """f over operands (E) of one shape: on the scalars, on the one axis they vary along, or dense."""
        if any(a.d is not None for a in args) or len(set().union(*(a.axes() for a in args))) > 1:
            return self._dense(shape, np.asarray(f(*[a.full() for a in args])).astype(np.int64), b)
        axes = set().union(*(a.axes() for a in args))
        if not axes:
            return E(shape, s=np.asarray(f(*[a.s for a in args])).astype(np.int64), b=b)
        pos = axes.pop()
        n = shape[pos]
        vals = [a.along(pos, n) for a in args]
        return E(shape, v={pos: np.asarray(f(*vals)).astype(np.int64)}, b=b)

    def _add(self, a, b, sign=1):
        if a.d is not None or b.d is not None:
            return self._dense(a.shape, a.full() + sign * b.full())
        v = dict(a.v)
        for pos, arr in b.v.items():
            v[pos] = v[pos] + sign * arr if pos in v else sign * arr
        return E(a.shape, s=a.s + sign * b.s, v=v)

    def _mul(self, a, b):
        if a.d is None and b.d is None and (a.scalar_only() or b.scalar_only()):
            sc, other = (a, b) if a.scalar_only() else (b, a)
            x = sc.s
            return E(other.shape, s=other.s * x, v={p: arr * x.reshape(-1, 1) for p, arr in other.v.items()})
        return self._general(np.multiply, [a, b], a.shape)

    def _int(self, x):
        """An E for an integer operand (a mask becomes its dense product)."""
        if isinstance(x, Mk):
            return self._dense(x.shape, x.full().astype(np.int64), True)
        return x

    def run(self):
        env = {}
        for name, typ in self.fn.args:
            key = name[1:]
            if typ.startswith("!tt.ptr"):
                env[name] = Ptr(key, E(()))
            elif key in self.ints:
                env[name] = E((), s=np.full((1,), int(self.ints[key]), dtype=np.int64))
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
            m = re.match(r'(?:dense<)?([-0-9.eE+a-z]+)>?', op.text)
            et = _elem(op.rtype)
            if et.startswith("f") or et.startswith("bf"):
                try:
                    val = float(m.group(1)) if m else None
                except ValueError:             # a hex bit pattern: not read here
                    val = None
                r = F("opaque", stash={"const": True, "value": val})
            else:
                v = m.group(1)
                v = 1 if v == "true" else 0 if v == "false" else int(float(v))
                r = E(shape, s=np.full((1,), v, dtype=np.int64), b=et == "i1")
        elif n == "tt.get_program_id":
            axis = {"x": 0, "y": 1, "z": 2}[op.text.split()[0]]
            r = E((), s=self.pids[axis])
        elif n == "tt.make_range":
            s = int(re.search(r'start = (-?\d+)', op.text).group(1))
            e = int(re.search(r'end = (-?\d+)', op.text).group(1))
            r = E((e - s,), v={0: np.arange(s, e, dtype=np.int64)[None, :]})
        elif n in ("arith.addi", "arith.subi"):
            a, b = args
            if isinstance(a, T) or isinstance(b, T):
                r = T(a.why if isinstance(a, T) else b.why)
            else:
                r = self._add(self._int(a), self._int(b), 1 if n == "arith.addi" else -1)
        elif n == "arith.andi" and all(isinstance(x, (E, Mk)) and (isinstance(x, Mk) or x.b) for x in args):
            fs = []
            for x in args:
                fs += x.f if isinstance(x, Mk) else [x]
            r = Mk(shape or args[0].shape, fs) if len(fs) > 1 else fs[0]
        elif n in _GENERAL:
            a, b = args
            if isinstance(a, T) or isinstance(b, T):
                r = T(a.why if isinstance(a, T) else b.why)
            elif n == "arith.muli":
                r = self._mul(self._int(a), self._int(b))
            else:
                a, b = self._int(a), self._int(b)
                r = self._general(_GENERAL[n], [a, b], a.shape, b=a.b and b.b)
        elif n == "arith.cmpi":
            pred = op.text.split(",")[0].strip()
            a, b = args
            if isinstance(a, T) or isinstance(b, T):
                r = T(a.why if isinstance(a, T) else b.why)
            else:
                a, b = self._int(a), self._int(b)
                r = self._general(_CMP[pred], [a, b], a.shape, b=True)
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
                r = self._general(lambda x, y, z: np.where(x.astype(bool), y, z), [c, a, b], a.shape)
        elif n in ("tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape"):
            r = self._shape_op(n, op, args[0], shape)
        elif n in _PASS:
            r = args[0]
            if isinstance(r, F) and n not in ("arith.truncf", "arith.extf"):
                # a float made into an integer or reinterpreted: a value that depends on data from here on
                r = T(f"{n} of a float value") if not _elem(op.rtype).startswith(("f", "bf")) else \
                    F("opaque", stash={"why": f"{n} of a float value"})
        elif n == "tt.addptr":
            p, o = args
            if isinstance(o, T) or p.taint is not None:
                r = Ptr(p.arg, p.off, taint=(o.why if isinstance(o, T) else p.taint))
            else:
                o = self._int(o)
                off = p.off
                if off.shape != o.shape:
                    off = self._shape_op("tt.splat", op, off, o.shape) if not off.shape else off
                r = Ptr(p.arg, self._add(off, o))
        elif n == "tt.load":
            p = args[0]
            mask = args[1] if len(args) > 1 else None
            if isinstance(mask, T):
                raise Unmodelled("a load masked by data read at run time")
            et = _elem(op.rtype)
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
        if isinstance(r, E) and not r.b and n.startswith("arith.") and _elem(op.rtype) == "i32":
            lo, hi = _int_range(r)
            if lo < -2 ** 31 or hi > 2 ** 31 - 1:
                raise Unmodelled(f"an i32 index computation ({n}) leaves the i32 range ({lo}..{hi}); the IR wraps")
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
            return E(shape, s=x.s, b=x.b)
        if n == "tt.expand_dims":
            axis = int(re.search(r'axis = (\d+)', op.text).group(1))
            new = x.shape[:axis] + (1,) + x.shape[axis:]
            if x.d is not None:
                return E(new, d=np.expand_dims(x.d, axis + 1), b=x.b)
            return E(new, s=x.s, v={(p + 1 if p >= axis else p): a for p, a in x.v.items()}, b=x.b)
        if n == "tt.broadcast":
            if x.d is not None:
                return E(shape, d=np.broadcast_to(x.d, (x.d.shape[0],) + tuple(shape)), b=x.b)
            s, v = x.s, {}
            for p, a in x.v.items():
                if x.shape[p] == 1 and shape[p] > 1:
                    s = s + a[:, 0]
                else:
                    v[p] = a
            return E(shape, s=s, v=v, b=x.b)
        return self._dense(shape, x.full().reshape((-1,) + tuple(shape)), x.b)

    def _for(self, op, env):
        lb, ub, st = (self._get(env, op.extra[k]) for k in ("lb", "ub", "step"))
        for v in (lb, ub, st):
            if not isinstance(v, E) or not v.scalar_only() or np.unique(v.s).size != 1:
                raise Unmodelled("a loop whose bounds differ between programs or depend on data")
        lo, hi, step = int(lb.s.flat[0]), int(ub.s.flat[0]), int(st.s.flat[0])
        if step <= 0:
            raise Unmodelled("a loop with a non-positive step")
        carried = [self._get(env, v) for _k, v in op.extra["iter"]]
        names = [k for k, _v in op.extra["iter"]]
        for it in range(lo, hi, step):
            inner = dict(env)
            inner[op.extra["iv"]] = E((), s=np.full((1,), it, dtype=np.int64))
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


def _int_range(x):
    """(least, greatest) value of an integer value over all programs and elements."""
    if x.d is not None:
        return int(x.d.min()), int(x.d.max())
    lo, hi = x.s.astype(np.int64), x.s.astype(np.int64)
    for arr in x.v.values():
        lo = lo + arr.min(axis=1)
        hi = hi + arr.max(axis=1)
    return int(lo.min()), int(hi.max())


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
        if leaf.mask is not None:
            sm = _separate_mask(leaf.mask, off.shape, P)
            if sm is None:
                return False
            mS, m0, m1 = sm
            need = (m0 if axis == 0 else m1) & mS.reshape(-1, 1)
            if np.any(valid_o & ~need):
                raise _Fail("violation", f"a scale of {leaf.arg} needed by a consumed element is masked out")
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
    if leaf.mask is not None:
        sm = _as_bool_full(leaf.mask)
        sm = sm[:, :, 0][:, :, None] if axis == "row" else sm[:, 0, :][:, None, :]
        if (valid & ~np.broadcast_to(sm, valid.shape)).any():
            raise _Fail("violation", f"a scale of {leaf.arg} needed by a consumed element is masked out")
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
