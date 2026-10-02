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
it. For every `tt.dot` whose result is multiplied by loaded scales and accumulated, and for every element of both
operands the dot consumes (masked-out elements contribute nothing and are skipped):

  scaled_dot_reads   the activation scale multiplied into output row i is, for every k the row consumes, the one
                     issued for (m, k // group_k) with that activation; the weight scale multiplied into output
                     column j is, for every k, the one issued for (n // block_n, k // block_k) with that weight
  contraction        both operands are read at the same k for each contraction index
  store              the accumulated value of (row of A, column of B) is stored at C[m, n]

The verdict, per launch:
  proven       every term of every iteration of every program holds the meaning above, whatever the data
  violation    an element is multiplied by a scale that is not its own (the first such is described)
  possible     the scale read is chosen by a value the kernel loads at run time (data the IR cannot see), and one
               of the choices is not the element's own scale
  unproven     the IR does something this module does not model (an op, a data-dependent address, a loop whose
               bounds differ between programs, a layout it cannot invert): nothing is claimed
A launch that is not proven gets no claim from here; the caller decides (the L5.4a whole-output check, or refuse).
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
                r = F("opaque", stash={"const": True})
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
                r = F("leaf", arg=p.arg, off=(p.off if p.taint is None else None), mask=mask, taint=p.taint)
            else:
                r = T(f"a value loaded from {p.arg}")
        elif n == "tt.dot":
            a, b = args[0], args[1]
            if not (isinstance(a, F) and isinstance(b, F) and a.kind == "leaf" and b.kind == "leaf"):
                raise Unmodelled("a dot whose operands are not loads")
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
            stash = None
            for x in (a, b):
                if isinstance(x, F) and self._has_dot(x):
                    stash = self._check_term(x)
            prev = [x.stash for x in (a, b) if isinstance(x, F) and x.kind == "acc" and x.stash]
            r = F("acc", stash=stash or (prev[0] if prev else None))
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
                         mask=None if x.mask is None else self._shape_op(n, op, x.mask, shape), taint=x.taint)
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
        scales = []
        for f in parts:
            if f.kind == "dot" or (f.kind == "opaque" and (f.stash or {}).get("const")):
                continue
            scales.append(f)
        fast = _Fast(self, a, b) if FAST and not self.went_dense else None
        if fast is not None and not fast.ok:
            fast = None
            if not self.allow_dense:
                raise _Dense()
        seen = {"activation_scale": 0, "weight_scale": 0}
        for f in scales:
            for alt, data_choice in self._alternatives(f):
                role = self._scale_role(alt)
                seen[role] = seen.get(role, 0) + 1
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
        self.terms += 1
        if fast is not None:
            self.elements += fast.elements
            return {"rows": fast.m, "rows_valid": fast.vR, "cols": fast.n, "cols_valid": fast.vC}
        rows, cols, count = _dense_rows_cols(self, a, b)
        self.elements += count
        return {"rows": rows, "rows_valid": rows >= 0, "cols": cols, "cols_valid": cols >= 0}

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
        ptr, val = args[0], args[1]
        mask = args[2] if len(args) > 2 else None
        if not isinstance(val, F) or val.kind != "acc" or not val.stash:
            return
        info = self.binding.get(ptr.arg)
        if info is None or info.role != "output":
            raise _Fail("violation", f"the accumulated product is stored to {ptr.arg}, which is not the output")
        if ptr.taint is not None:
            raise _Fail("unproven", f"the output address is chosen by data ({ptr.taint})")
        if isinstance(mask, T):
            raise Unmodelled("a store masked by data")
        st = val.stash
        s0, s1 = int(info.stride[0]), int(info.stride[1])
        P = self.P
        off = ptr.off
        sep = _separate_mask(mask, off.shape, P) if FAST and off.d is None and off.axes() <= {0, 1} \
            and len(off.shape) == 2 else None
        if sep is not None:
            vS, v0, v1 = sep
            R, C = off.shape
            rows, cols = _bp(st["rows"], P, R), _bp(st["cols"], P, C)
            vR = v0 & _bp(st["rows_valid"], P, R)
            vC = v1 & _bp(st["cols_valid"], P, C)
            u = _bp(off.v.get(0, np.zeros((1, R), dtype=np.int64)), P, R) + _bp(off.s.reshape(-1, 1), P, 1) \
                - rows * s0
            w = _bp(off.v.get(1, np.zeros((1, C), dtype=np.int64)), P, C) - cols * s1
            live = vS & vR.any(axis=1) & vC.any(axis=1)
            if not live.any():
                return
            uc, wc = _constant_over(u, vR), _constant_over(w, vC)
            if uc is None or wc is None or np.any((uc + wc)[live] != 0):
                raise _Fail("violation", "an output value is stored where another row or column belongs")
            return
        if not self.allow_dense:
            raise _Dense()
        self.went_dense = True
        rows, cols = st["rows"], st["cols"]
        want = rows[:, :, None] * s0 + cols[:, None, :] * s1
        offd = np.broadcast_to(off.full(), np.broadcast_shapes(off.full().shape, want.shape))
        want = np.broadcast_to(want, offd.shape)
        m = np.ones(offd.shape, dtype=bool) if mask is None else np.broadcast_to(_as_bool_full(mask), offd.shape)
        m = m & np.broadcast_to(st["rows_valid"][:, :, None] & st["cols_valid"][:, None, :], offd.shape)
        if (m & (offd != want)).any():
            raise _Fail("violation", "an output value is stored where another row or column belongs")


def _bp(x, P, n):
    """x broadcast to (P, n)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.broadcast_to(x, (P, n))


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
    am, ak, av, _ = _dense_coords(run, a, "activation")
    bn, bk, bv, _ = _dense_coords(run, b, "weight")
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
    return rows, cols, int(av.sum() + bv.sum())


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
    dense = False
    for attempt in (("fast", "dense") if FAST else ("dense",)):
        per = max(1, chunk_elements // (longest if attempt == "fast" else biggest))
        terms = elements = 0
        try:
            for start in range(0, total, per):
                run = _Run(fn, binding, ints, _grid_pids((gx, gy, gz), start, min(total, start + per)),
                           dense=attempt == "dense")
                run.run()
                terms += run.terms
                elements += run.elements
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
    if terms == 0:
        return Verdict("unproven", "no scaled dot term was found in the kernel", 0, 0, total,
                       time.perf_counter() - t0, None, dense)
    return Verdict("proven", f"{terms} scaled dot terms over {total} programs hold the producers' scale mapping",
                   terms, elements, total, time.perf_counter() - t0, None, dense)


def _walk(ops):
    for op in ops:
        yield op
        yield from _walk(op.body)
