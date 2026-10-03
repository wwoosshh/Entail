"""kernel_types: one rule for any Triton kernel, read from its own IR (ROADMAP M19 L6).

A tensor handed to a kernel carries a meaning (Meaning): what each of its axes enumerates (a name shared across
tensors - "token", "hidden", "feature" - or none), how many of the base coordinates one element stands for (a scale
tensor's grouped axis), its physical layout, and, for a quantized value, which issued scale is its own. The rule held
to the kernel, for one launch, whatever the data:

    values combined in one arithmetic operation are paired only where their meanings agree, and each element of
    the output is produced once from all of what its meaning covers.

That is all. Which scale multiplies which value, that both operands of a contraction are read at the same k, that a
sum covers every k once, that a value is stored at its own output element, that nothing else is written: every one
of these is this rule applied to one operation. There is no contract per kernel; the kernel's own IR says which
elements it combines, and the meanings say whether they belong together.

How it is read: kernel_ir's evaluator follows every integer and address of the launch for every program and loop
iteration without data (its integer meaning, loops, branches and chunking are reused here, unchanged). This module
gives the float values a type instead of a kind: for every lane, its coordinates on every named axis, which lanes
hold real elements, which axes it has already been summed over (and which coordinates that sum covered), which
quantized values it derives from and whether their scales have been applied, and what else was done to it (a
constant factor, a non-linear function, an addend that is not a term). Loads give coordinates (the address, taken
apart by the tensor's strides), elementwise operations pair them, dot and reduce turn one axis into a sum, stores
pair the value with the output's own coordinates and record what was covered.

A coordinate is kept as small as it is: one that varies only along the rows of a tile is a (programs, rows, 1)
array, one that is the same for every lane a (programs, 1, 1) array; numpy broadcasts them when they meet. So a
64 x 128 tile costs its 64 + 128 coordinates, not 8,192 (the launches of an engine run have thousands of programs
and tens of loop iterations each).

Verdicts (per launch; the caller decides what to do with one that is not proven):
  proven      the rule holds for every operation and every element, whatever the data; the output meaning, where
              it was not declared, is inferred (Verdict.inferred)
  violation   an operation pairs elements whose meanings disagree, a sum misses or repeats a coordinate, an output
              element is never stored, something else is written, or a declared output gets more than its meaning
  possible    the pairing depends on a value the kernel reads at run time, and one of the choices disagrees
  unproven    something this module does not model, or a meaning it was not given (nothing is claimed)
"""
import itertools
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from . import kernel_ir as KI
from .kernel_ir import E, Mk, Ptr, T, Unmodelled, _as_bool_full, _Dense, _Fail, _ranges, _tiling

CHUNK_ELEMENTS = 1 << 22
MAX_ALTS = 8
VERDICTS = ("proven", "violation", "possible", "unproven")
_ELEMENTWISE_BINARY = {"arith.mulf": "mul", "arith.divf": "div", "arith.addf": "add", "arith.subf": "sub",
                       "arith.maxnumf": "max", "arith.minnumf": "min", "arith.maximumf": "max",
                       "arith.minimumf": "min"}
_UNARY_LINEAR = ("arith.negf",)


# --- meanings -------------------------------------------------------------------------------------------------------

@dataclass
class Axis:
    name: Optional[str]        # shared across tensors; None: this axis means nothing the rule can pair
    size: int
    group: int = 1             # one coordinate here stands for `group` base coordinates (a grouped scale)


@dataclass
class Meaning:
    """What a pointer argument holds, as its producer issued it (or as the launch alone can tell: shape and strides,
    with unnamed axes)."""
    axes: Tuple[Axis, ...] = ()
    shape: tuple = ()
    stride: tuple = ()
    kind: str = "value"        # value (a tensor read), scale (of the value `pair`), output, sums, other
    serial: int = 0
    pair: int = 0              # value: the serial of its scale (0: none); scale: the serial of its value
    reduced: Dict[str, Tuple[int, int]] = field(default_factory=dict)   # output: axes it must cover completely
    strict: bool = False       # output: nothing beyond the paired values and the declared sums may reach it
    basis: Optional[str] = None    # an integer tensor: what its numbers mean (a position, a kv block, ...)
    label: Optional[str] = None    # a name a loaded stride refers to ("stride:<label>:<dim>")

    def names(self):
        return [a.name for a in self.axes]


def dense(shape, stride, names=None, **kw) -> Meaning:
    """A meaning for a tensor of this shape and strides, its axes named (or unnamed)."""
    names = list(names or [None] * len(shape))
    return Meaning(tuple(Axis(n, int(s)) for n, s in zip(names, shape)), tuple(int(x) for x in shape),
                   tuple(int(x) for x in stride), **kw)


def block_fp8_matmul(M, N, K, group_k=128, block_n=128, a_serial=1, b_serial=3, as_serial=2, bs_serial=4,
                     out_stride=None, as_stride=None, bs_stride=None):
    """The meanings of vLLM's block FP8 matmul as L5.4 issued them: A[token, hidden] with As[token, hidden/group],
    B[feature, hidden] with Bs[feature/block, hidden/block], C[token, feature] = the complete sum over hidden."""
    NB = -(-K // group_k)
    return {
        "A": Meaning((Axis("token", M), Axis("hidden", K)), (M, K), (K, 1), "value", a_serial, as_serial),
        "As": Meaning((Axis("token", M), Axis("hidden", K, group_k)), (M, NB), as_stride or (NB, 1), "scale",
                      as_serial, a_serial),
        "B": Meaning((Axis("feature", N), Axis("hidden", K)), (N, K), (K, 1), "value", b_serial, bs_serial),
        "Bs": Meaning((Axis("feature", N, block_n), Axis("hidden", K, group_k)), (N // block_n, NB),
                      bs_stride or (NB, 1), "scale", bs_serial, b_serial),
        "C": Meaning((Axis("token", M), Axis("feature", N)), (M, N), out_stride or (N, 1), "output",
                     reduced={"hidden": (0, K)}, strict=True),
    }


@dataclass
class Verdict:
    verdict: str
    why: str = ""
    checks: int = 0            # pairings decided
    programs: int = 0
    seconds: float = 0.0
    example: Optional[dict] = None
    inferred: Optional[dict] = None    # the meaning of an output whose axes were not named

    def to_json(self):
        return {"verdict": self.verdict, "why": self.why, "checks": self.checks, "programs": self.programs,
                "seconds": round(self.seconds, 4), "example": self.example, "inferred": self.inferred}


# --- typed float values ---------------------------------------------------------------------------------------------

class V:
    """A float tensor value in the kernel, typed by where its elements come from. Every array has the rank
    1 + len(shape) (the program axis first) and is as small as its variation: a size of 1 where it is constant."""
    __slots__ = ("shape", "coords", "valid", "sums", "serials", "applied", "extras", "fn", "const", "leaf",
                 "masked_zero", "data_addr", "alts", "scale_of", "scale_serial", "data_valid")

    def __init__(self, shape, coords=None, valid=None, sums=None, serials=frozenset(), applied=None, extras=(),
                 fn=False, const=None, leaf=None, masked_zero=True, data_addr=None, alts=None, scale_of=0,
                 scale_serial=0, data_valid=False):
        self.data_valid = data_valid          # which lanes hold elements is decided by data the kernel read
        self.shape = tuple(shape)
        self.coords = dict(coords or {})      # name -> (array, group)
        self.valid = valid                    # bool array or None (every lane holds a real element)
        self.sums = {k: list(v) for k, v in (sums or {}).items()}   # name -> [(lo (P,), count (P,)), ...]
        self.serials = frozenset(serials)     # quantized values this derives from (their issue serials)
        self.applied = dict(applied or {})    # scale serial -> times applied
        self.extras = tuple(extras)           # what else reached it (constant factors, non-term addends)
        self.fn = fn                          # a non-linear function was applied
        self.const = const                    # a constant's value (None: not a constant)
        self.leaf = leaf                      # the argument it was loaded from (a direct load)
        self.masked_zero = masked_zero        # masked-out lanes hold an exact zero
        self.data_addr = data_addr            # loaded at an address chosen by data (why)
        self.alts = alts                      # alternatives chosen by data: a list of V (then the rest is unused)
        self.scale_of = scale_of              # a scale leaf: the serial of the value it scales
        self.scale_serial = scale_serial

    def is_zero(self):
        return self.const == 0.0

    def copy(self, **kw):
        out = V(self.shape, self.coords, self.valid, self.sums, self.serials, self.applied, self.extras, self.fn,
                self.const, self.leaf, self.masked_zero, self.data_addr, self.alts, self.scale_of,
                self.scale_serial, self.data_valid)
        for k, v in kw.items():
            setattr(out, k, v)
        return out


class S(T):
    """A symbolic integer: data the kernel read, or made from what it read, as a linear form over opaque leaves
    (terms: leaf key -> (basis, coefficient: an int, or "stride:<label>:<dim>" for a stride it loaded)) plus a
    launch-known part (const: E or None), with the basis of what the whole means (None: unknown)."""
    __slots__ = ("basis", "terms", "const", "shape")

    def __init__(self, why, basis=None, terms=(), const=None, shape=()):
        super().__init__(why)
        self.basis, self.terms, self.const, self.shape = basis, dict(terms), const, tuple(shape)

    def leaf(self):
        """(key, basis) when this is one leaf with coefficient 1, else None."""
        if len(self.terms) == 1:
            (k, (b, c, kn)), = self.terms.items()
            if c == 1 and kn is None:
                return k, b
        return None


class Sym:
    """A coordinate chosen by data: a symbolic leaf plus a launch-known offset."""
    __slots__ = ("key", "basis", "off")

    def __init__(self, key, basis, off):
        self.key, self.basis, self.off = key, basis, off

    def same(self, other):
        if not isinstance(other, Sym) or self.key != other.key:
            return False
        a, b = np.asarray(self.off), np.asarray(other.off)
        try:
            shape = np.broadcast_shapes(a.shape, b.shape)
            return bool(np.array_equal(np.broadcast_to(a, shape), np.broadcast_to(b, shape)))
        except ValueError:
            return False


RELATIONS = {}            # (basis a, op, basis b) -> the basis of a op b; declared by whoever attaches the bases


def relate(a, op, b, result):
    """Declare what two bases make: relate("kv_block", "*", "block_size", "kv_slot")."""
    RELATIONS[(a, op, b)] = result


def _basis_of(op, a, b):
    """The basis of a op b from the relations, or the one side's basis when the other means nothing."""
    r = RELATIONS.get((a, op, b))
    if r is not None:
        return r
    if op in ("+", "-") and b is None:
        return a
    if op == "+" and a is None:
        return b
    return None


def _opaque(why, key, basis, shape):
    return S(why, basis, {key: (basis, 1, None)}, None, shape)


def _coordinate_basis(basis):
    """A basis that names a coordinate (a stride, or a pointer, is not one)."""
    return None if basis is None or str(basis).startswith(("stride:", "ptr:")) else basis


def _full(a, shape, P):
    """An array broadcast to (P, *shape) (a view, no copy)."""
    return np.broadcast_to(np.asarray(a), (P,) + tuple(shape))


def _ones(shape):
    return np.ones((1,) + tuple(shape), dtype=bool)


def _alts(v):
    return v.alts if v.alts is not None else [v]


def _combine_alts(outs):
    if len(outs) == 1:
        return outs[0]
    if len(outs) > MAX_ALTS:
        raise Unmodelled(f"more than {MAX_ALTS} alternatives chosen by data")
    return V((), alts=outs)


def _alt_map(f, *vs):
    """f over every combination of the operands' alternatives; a violation in one of them is "possible" (the data
    the kernel reads decides which it takes)."""
    outs = []
    for combo in itertools.product(*[_alts(v) for v in vs]):
        try:
            outs.append(f(*combo))
        except _Fail as e:
            if e.verdict == "violation":
                raise _Fail("possible", f"a value chosen at run time: {e.why}", e.example)
            raise
    return _combine_alts(outs)


def _first(bad):
    """The index of the first True in a boolean array (program first, then the lanes)."""
    return tuple(int(t[0]) for t in np.nonzero(bad))


def _same_coord(a, b):
    """Two coordinates (arrays or Sym) that are the same, lane for lane."""
    if isinstance(a, Sym) or isinstance(b, Sym):
        return isinstance(a, Sym) and a.same(b)
    a, b = np.asarray(a), np.asarray(b)
    try:
        shape = np.broadcast_shapes(a.shape, b.shape)
    except ValueError:
        return False
    return bool(np.array_equal(np.broadcast_to(a, shape), np.broadcast_to(b, shape)))


# --- evaluation -----------------------------------------------------------------------------------------------------

class _Typed(KI._Run):
    """kernel_ir's evaluator with typed float values."""

    def __init__(self, fn, meanings, ints, pids, dense, chunk_index=0, grid=(1, 1, 1)):
        super().__init__(fn, {}, ints, pids, dense)
        self.meanings = meanings
        self.checks = 0
        self.chunk_index = chunk_index
        self.grid = tuple(int(g) for g in (list(grid) + [1, 1, 1])[:3])
        self.inferred = {}         # output argument -> {dim: axis name, ...}
        self.stores = {}           # output argument -> [(row lo, rows, column lo, columns) per storing program]
        self.active = np.ones(self.P, dtype=bool)   # the programs the current branch is taken by
        self.iters = ()            # the loop iterations in progress (a leaf's key tells them apart)
        self.data_depth = 0        # inside a branch or loop decided by data: stores cover what data decides
        self.pointers = {}         # argument -> [target meaning names] (a table of pointers it holds)

    def _key(self, op):
        return (op.results[0] if op.results else op.name, self.iters)

    def _leaf(self, op, basis, shape=()):
        return _opaque(f"a value loaded from {op.name}", self._key(op), basis, shape)

    # -- symbolic integers --

    def _sym_int(self, n, op, args, shape):  # noqa: C901
        """An integer operation with a symbolic operand: the linear form and the basis it makes."""
        shape = tuple(shape or ())
        why = "a value made from values read at run time"
        if n in KI._INT_CASTS:
            x = args[0]
            return S(x.why, x.basis, x.terms, x.const, shape) if isinstance(x, S) else T(x.why)
        if n == "arith.cmpi":
            return T("a comparison with a value read at run time")
        a, b = args
        sa, sb = isinstance(a, S), isinstance(b, S)
        if (isinstance(a, T) and not sa) or (isinstance(b, T) and not sb):
            return T((a if isinstance(a, T) else b).why)
        if n in ("arith.addi", "arith.subi"):
            sign = 1 if n == "arith.addi" else -1
            sym = "+" if sign == 1 else "-"
            if sa and sb:
                ba, bb = a.basis, b.basis
                rel = RELATIONS.get((ba, sym, bb))
                if rel is not None:
                    return _opaque(why, self._key(op), rel, shape)
                # two meanings added without a rule mean nothing as a value, but as an address each term still
                # addresses its own axis: the linear form is kept
                terms = dict(a.terms)
                for k, (bs, c, kn) in b.terms.items():
                    c0 = terms.get(k, (None, 0, None))[1]
                    if isinstance(c, str) or isinstance(c0, str) or kn is not None:
                        if sign == -1 or k in terms:
                            return _opaque(why, self._key(op), None, shape)
                        terms[k] = (bs, c, kn)
                    elif k in terms:
                        terms[k] = (terms[k][0], c0 + sign * c, None)
                    else:
                        terms[k] = (bs, c if sign == 1 else -c, None)
                terms = {k: v for k, v in terms.items() if v[1] != 0}
                return S(a.why, _basis_of(sym, ba, bb), terms, self._const_add(a.const, b.const, sign, shape), shape)
            x, e = (a, b) if sa else (b, a)
            e = self._int(e)
            if not isinstance(e, E):
                raise Unmodelled(f"{n} of a symbolic value and a {type(e).__name__}")
            if sa:
                return S(x.why, x.basis, x.terms, self._const_add(x.const, e, sign, shape), shape)
            terms = {}
            for k, (bs, c, kn) in x.terms.items():
                if isinstance(c, str) or kn is not None:
                    return _opaque(why, self._key(op), None, shape)
                terms[k] = (bs, -c, None)
            return S(x.why, None, terms, self._const_add(e, x.const, -1, shape), shape)
        if n == "arith.muli":
            if sa and sb:
                la, lb = a.leaf(), b.leaf()
                if la and lb and a.const is None and b.const is None:
                    if lb[1] is not None and str(lb[1]).startswith("stride:"):
                        return S(a.why, a.basis, {la[0]: (la[1], lb[1], None)}, None, shape)
                    if la[1] is not None and str(la[1]).startswith("stride:"):
                        return S(b.why, b.basis, {lb[0]: (lb[1], la[1], None)}, None, shape)
                return _opaque("a product of values read at run time", self._key(op),
                               RELATIONS.get((a.basis, "*", b.basis)), shape)
            x, e = (a, b) if sa else (b, a)
            e = self._int(e)
            if not isinstance(e, E):
                return _opaque("a product of a value read at run time", self._key(op), None, shape)
            lx = x.leaf()
            if lx and x.const is None and lx[1] is not None and str(lx[1]).startswith("stride:"):
                # a launch-known number times a stride the kernel loaded: a known coordinate on that stride's axis
                return S(x.why, None, {("known",) + self._key(op): (None, lx[1], e)}, None, shape)
            if not e.scalar_only() or np.unique(e.s).size != 1:
                return _opaque("a product of a value read at run time", self._key(op), None, shape)
            m = int(e.s.flat[0])
            if m == 1:
                return S(x.why, x.basis, x.terms, x.const, shape)
            terms = {}
            for k, (bs, c, kn) in x.terms.items():
                if isinstance(c, str) or kn is not None:
                    return _opaque("a product of a value read at run time", self._key(op), None, shape)
                terms[k] = (bs, c * m, None)
            const = None if x.const is None else self._mul(x.const, E((), s=np.full((1,), m, dtype=np.int64),
                                                                        w=x.const.w))
            return S(x.why, RELATIONS.get((x.basis, "*", None)), terms, const, shape)
        if n in ("arith.divsi", "arith.divui", "arith.remsi", "arith.remui", "arith.floordivsi",
                 "arith.ceildivsi", "arith.ceildivui"):
            sym = "//" if "div" in n else "%"
            ba = a.basis if sa else None
            bb = b.basis if sb else None
            return _opaque("a quotient of values read at run time", self._key(op), RELATIONS.get((ba, sym, bb)),
                           shape)
        return _opaque(why, self._key(op), None, shape)

    def _const_add(self, x, y, sign, shape):
        """x + sign * y for launch-known parts that may be None."""
        if x is None and y is None:
            return None
        if x is None:
            return y if sign == 1 else self._mul(y, E((), s=np.full((1,), -1, dtype=np.int64), w=y.w))
        if y is None:
            return x
        if x.shape != y.shape:
            if not x.shape:
                x = E(y.shape, s=x.s, w=x.w)
            elif not y.shape:
                y = E(x.shape, s=y.s, w=y.w)
            else:
                raise Unmodelled("a symbolic sum of launch-known parts of different shapes")
        return self._add(x, y, sign)

    def _ops(self, ops, env):
        """kernel_ir's, with cf.cond_br: the two blocks follow, each ending in a return."""
        i = 0
        while i < len(ops):
            op = ops[i]
            if op.name == "cf.cond_br":
                labels = ["^" + x for x in re.findall(r'\^(bb\d+)', op.text)]
                blocks = {}
                cur = None
                for o in ops[i + 1:]:
                    if o.name.startswith("^bb"):
                        cur = o.name.rstrip(":")
                        blocks[cur] = []
                    elif cur is not None:
                        blocks[cur].append(o)
                if len(labels) != 2 or any(lb not in blocks for lb in labels):
                    raise Unmodelled("a conditional branch whose blocks are not found")
                for lb in labels:
                    if any(o.name in ("cf.br", "cf.cond_br") for o in blocks[lb]):
                        raise Unmodelled("a conditional branch whose block branches again")
                cond = self._get(env, op.operands[0])
                self._branch(cond, blocks[labels[0]], blocks[labels[1]], env)
                return
            if op.name.startswith("^bb"):
                raise Unmodelled("a block label without a branch to it")
            self._op(op, env)
            i += 1

    def _branch(self, cond, then_ops, else_ops, env):
        """Both blocks of a two-way branch: by program when the launch decides the condition, under data when the
        kernel reads it."""
        if isinstance(cond, T):
            self.data_depth += 1
            try:
                for body in (then_ops, else_ops):
                    inner = dict(env)
                    self._ops(body, inner)
            finally:
                self.data_depth -= 1
            return
        c = self._int(cond)
        if not isinstance(c, E) or not c.scalar_only():
            raise Unmodelled("a branch whose condition differs between lanes")
        cv = np.broadcast_to(c.s.astype(bool), (self.P,))
        saved = self.active
        try:
            for take, body in ((cv, then_ops), (~cv, else_ops)):
                if not (saved & take).any():
                    continue
                self.active = saved & take
                inner = dict(env)
                self._ops(body, inner)
        finally:
            self.active = saved

    def _for(self, op, env):
        """kernel_ir's loop, or one symbolic iteration when the bounds are read at run time (or differ between
        programs): the induction variable is a leaf with the bounds' basis, and what the body stores covers what
        the data decides."""
        lb, ub, st = (self._get(env, op.extra[k]) for k in ("lb", "ub", "step"))
        plain = all(isinstance(v, E) and v.scalar_only() and np.unique(v.s).size == 1 for v in (lb, ub, st))
        if plain:
            saved = self.iters
            try:
                self.iters = saved + ((op.extra["iv"], "unrolled"),)
                return super()._for(op, env)
            finally:
                self.iters = saved
        if isinstance(st, T) or not isinstance(st, E):
            raise Unmodelled("a loop whose step is read at run time")
        basis = None
        for v in (lb, ub):
            if isinstance(v, S) and _coordinate_basis(v.basis) is not None:
                basis = v.basis
                break
        iv = _opaque("a loop index whose bounds are read at run time", (op.extra["iv"], self.iters), basis, ())
        carried = [self._get(env, v) for _k, v in op.extra["iter"]]
        names = [kk for kk, _v in op.extra["iter"]]
        inner = dict(env)
        inner[op.extra["iv"]] = iv
        for kk, v in zip(names, carried):
            inner[kk] = v
        saved = self.iters
        self.data_depth += 1
        try:
            self.iters = saved + ((op.extra["iv"], "symbolic"),)
            self._ops(op.body, inner)
        finally:
            self.iters = saved
            self.data_depth -= 1
        base = op.results[0].split("#")[0] if op.results else None
        for i, _v in enumerate(carried):
            if base:
                env[f"{base}#{i}"] = T("a value carried by a loop whose bounds are read at run time")
        if len(op.results) == 1 and carried:
            env[op.results[0]] = T("a value carried by a loop whose bounds are read at run time")

    def _active_lanes(self, ndim):
        return self.active.reshape((-1,) + (1,) * ndim)

    # -- a branch taken by some programs and not others --

    def _if(self, op, env, cond):
        """scf.if whose condition is decided by the launch but differs between programs (one program pads, the
        others compute): both branches are followed, each with the programs that take it active; what they yield is
        merged program by program. A condition read from data, or one shared by every program: kernel_ir's."""
        if isinstance(cond, T):
            branches = [op.body, op.extra.get("else", [])]
            self.data_depth += 1
            yields = []
            try:
                for body in branches:
                    inner = dict(env)
                    self._ops(body, inner)
                    yields.append(inner.get("__yield__", []))
            finally:
                self.data_depth -= 1
            for name, a, b in zip(op.results, yields[0], yields[1]):
                env[name] = self._merge_data(a, b)
            return
        c = self._int(cond)
        if not isinstance(c, E) or not c.scalar_only():
            raise Unmodelled("a branch whose condition differs between lanes")
        cv = np.broadcast_to(c.s.astype(bool), (self.P,))
        if cv.all() or not cv.any():
            return super()._if(op, env, cond)
        branches = [op.body, op.extra.get("else", [])]
        saved = self.active
        yields = []
        try:
            for take, body in ((cv, branches[0]), (~cv, branches[1])):
                self.active = saved & take
                inner = dict(env)
                self._ops(body, inner)
                yields.append(inner.get("__yield__", []))
        finally:
            self.active = saved
        for name, a, b in zip(op.results, yields[0], yields[1]):
            env[name] = self._merge(cv, a, b)

    def _merge_data(self, a, b):
        """What a branch decided by data yields: a symbolic value when both sides are integers (their common basis,
        if any), a typed value only when both sides mean the same."""
        if isinstance(a, V) and isinstance(b, V):
            if a.const is not None and b.const is not None:
                return V(a.shape, const=a.const if a.const == b.const else None, fn=a.const != b.const)
            if a.shape == b.shape and set(a.coords) == set(b.coords) and set(a.sums) == set(b.sums) and \
                    a.serials == b.serials and a.applied == b.applied and \
                    all(a.coords[k][1] == b.coords[k][1] and _same_coord(a.coords[k][0], b.coords[k][0])
                        for k in a.coords):
                return a.copy(extras=tuple(dict.fromkeys(a.extras + b.extras)), fn=a.fn or b.fn)
            return self._select_v(None, a, b)
        if isinstance(a, V) or isinstance(b, V):
            raise Unmodelled("a float chosen by data against a non-float")
        if isinstance(a, Ptr) or isinstance(b, Ptr):
            raise Unmodelled("a pointer chosen by data read at run time")
        ba = a.basis if isinstance(a, S) else None
        bb = b.basis if isinstance(b, S) else None
        if ba == bb:
            basis = ba
        elif bb is None and isinstance(b, E):
            basis = ba
        elif ba is None and isinstance(a, E):
            basis = bb
        else:
            basis = None
        shape = a.shape if isinstance(a, (S, E)) else b.shape if isinstance(b, (S, E)) else ()
        return _opaque("a value chosen by data read at run time", ("select", self.iters, id(a), id(b)), basis, shape)

    def _select_v(self, cond, a, b):
        """a or b, typed values: by data (cond None) they are alternatives, unless one side is a constant (then the
        other value, noted as sometimes replaced by a constant); by a launch-known condition (cond: an i1 E over the
        lanes) they are taken lane by lane."""
        if a.alts is not None or b.alts is not None:
            alts = _alts(a) + _alts(b)
            typed = [v for v in alts if v.const is None]
            consts = [v for v in alts if v.const is not None]
            if typed and consts:
                note = "some elements replaced by the constants " + ", ".join(str(v.const) for v in consts)
                return _combine_alts([v.copy(extras=v.extras + (note,), fn=True) for v in typed])
            return _combine_alts(alts)
        if a.const is not None and b.const is not None:
            return V(np.broadcast_shapes(a.shape, b.shape), const=a.const if a.const == b.const else None,
                     fn=a.const != b.const)
        if a.const is not None or b.const is not None:
            c, o = (a, b) if a.const is not None else (b, a)
            return o.copy(shape=np.broadcast_shapes(a.shape, b.shape),
                          extras=o.extras + (f"some elements replaced by the constant {c.const}",), fn=True)
        if cond is None:
            return _combine_alts([a, b])
        shape = np.broadcast_shapes(a.shape, b.shape)
        sel = np.asarray(cond.full()).astype(bool)
        if sel.ndim != 1 + len(shape):
            sel = sel.reshape((-1,) + tuple(shape))
        if set(a.coords) != set(b.coords) or set(a.sums) != set(b.sums) or a.serials != b.serials or \
                a.applied != b.applied:
            raise Unmodelled("a select, lane by lane, between values that differ in meaning")
        coords = {}
        for k in a.coords:
            (xa, ga), (xb, gb) = a.coords[k], b.coords[k]
            if ga != gb or isinstance(xa, Sym) or isinstance(xb, Sym):
                raise Unmodelled("a select, lane by lane, between coordinates of different kinds")
            coords[k] = (np.where(sel, xa, xb), ga)
        va = _ones(shape) if a.valid is None else a.valid
        vb = _ones(shape) if b.valid is None else b.valid
        sums = {}
        for k in a.sums:
            if len(a.sums[k]) != len(b.sums[k]) or any(not (np.array_equal(la, lb) and np.array_equal(ca_, cb))
                                                     for (la, ca_), (lb, cb) in zip(a.sums[k], b.sums[k])):
                raise Unmodelled("a select between values summed over different ranges")
            sums[k] = a.sums[k]
        return V(shape, coords, np.where(sel, va, vb), sums, a.serials, a.applied,
                 tuple(dict.fromkeys(a.extras + b.extras)), a.fn or b.fn,
                 masked_zero=a.masked_zero and b.masked_zero, data_addr=a.data_addr or b.data_addr,
                 data_valid=a.data_valid or b.data_valid)

    def _merge(self, cv, a, b):
        """a where the program's condition holds, else b."""
        if isinstance(a, T) or isinstance(b, T):
            return T(a.why if isinstance(a, T) else b.why)
        if isinstance(a, V) and isinstance(b, V):
            if a.alts is not None or b.alts is not None:
                raise Unmodelled("a branch that yields values chosen by data")
            if a.const is not None and b.const is not None:
                return V(a.shape, const=a.const if a.const == b.const else None, fn=a.const != b.const)
            if a.const is not None or b.const is not None or a.shape != b.shape or set(a.coords) != set(b.coords) \
                    or set(a.sums) != set(b.sums) or a.serials != b.serials or a.applied != b.applied:
                raise Unmodelled("a branch whose two values differ in meaning")
            ndim = len(a.shape)
            sel = cv.reshape((-1,) + (1,) * ndim)
            coords = {}
            for k in a.coords:
                (xa, ga), (xb, gb) = a.coords[k], b.coords[k]
                if ga != gb:
                    raise Unmodelled("a branch whose two values group an axis differently")
                coords[k] = (np.where(sel, xa, xb), ga)
            va = _ones(a.shape) if a.valid is None else a.valid
            vb = _ones(b.shape) if b.valid is None else b.valid
            sums = {}
            for k in a.sums:
                if len(a.sums[k]) != len(b.sums[k]):
                    raise Unmodelled("a branch whose two values sum over different numbers of ranges")
                sums[k] = [(np.where(cv, np.broadcast_to(la, cv.shape), np.broadcast_to(lb, cv.shape)),
                            np.where(cv, np.broadcast_to(ca, cv.shape), np.broadcast_to(cb, cv.shape)))
                           for (la, ca), (lb, cb) in zip(a.sums[k], b.sums[k])]
            return V(a.shape, coords, np.where(sel, va, vb), sums, a.serials, a.applied,
                     tuple(dict.fromkeys(a.extras + b.extras)), a.fn or b.fn,
                     masked_zero=a.masked_zero and b.masked_zero, data_addr=a.data_addr or b.data_addr)
        if isinstance(a, Ptr) and isinstance(b, Ptr):
            if a.arg != b.arg or a.taint is not None or b.taint is not None:
                raise Unmodelled("a branch that yields pointers into different tensors")
            return Ptr(a.arg, self._merge(cv, a.off, b.off))
        a, b = self._int(a), self._int(b)
        if not (isinstance(a, E) and isinstance(b, E)):
            raise Unmodelled("a branch whose two values are of different kinds")
        if a.shape != b.shape:
            raise Unmodelled("a branch whose two values differ in shape")
        c = E((), s=cv.astype(np.int64), b=True, w=1)
        if a.shape:
            c = E(a.shape, s=cv.astype(np.int64), b=True, w=1)
        return self._general(lambda x, y, z: np.where(x.astype(bool), y, z), [c, a, b], a.shape, b=a.b and b.b,
                             w=a.w)

    def run(self):
        """kernel_ir's binding of the arguments, with a float scalar argument a constant value."""
        env = {}
        for name, typ in self.fn.args:
            key = name[1:]
            if not typ.startswith("!tt.ptr") and key in self.ints and isinstance(self.ints[key], float):
                if not (typ.startswith("f") or typ.startswith("bf")):
                    raise Unmodelled(f"argument {key} is a float in the launch but {typ} in the IR")
                env[name] = V((), const=float(self.ints[key]))
            elif typ.startswith("!tt.ptr"):
                env[name] = Ptr(key, E(()))
            elif key in self.ints:
                w = KI._width(typ)
                v = int(self.ints[key])
                if w is None:
                    raise Unmodelled(f"argument {key} has the type {typ}, not an integer type this module models")
                if not KI._fits(v, v, w) or abs(v) > KI._LIMIT:
                    raise Unmodelled(f"argument {key} = {v} is not a value of its type {typ}")
                env[name] = E((), s=np.full((1,), v, dtype=np.int64), b=(w == 1), w=w)
            else:
                raise Unmodelled(f"argument {key} has no value in the launch")
        self._ops(self.fn.body, env)

    # -- addresses as coordinates --

    def _dims(self, arg, m):
        dims = [(int(s), int(n), i) for i, (s, n) in enumerate(zip(m.stride, m.shape)) if int(n) > 1]
        dims.sort(key=lambda t: -t[0])
        for (sa, na, _ia), (sb, nb, _ib) in zip(dims, dims[1:]):
            if sa < nb * sb:
                raise Unmodelled(f"{arg}: a layout whose dimensions overlap (strides {tuple(m.stride)})")
        for s, _n, _i in dims:
            if s <= 0:
                raise Unmodelled(f"{arg}: a stride of {s}")
        return dims

    def _coords_of(self, arg, off, valid, what, bounds=True):
        """{dim: coordinate array} of a load or store through `off` in the tensor `arg`, each array of rank
        1 + len(off.shape) and as small as it varies, for the lanes `valid` (a bool array or None); every valid lane
        must address an element of the tensor."""
        m = self.meanings.get(arg)
        if m is None:
            raise Unmodelled(f"{what} through {arg}, whose shape and strides the launch did not give")
        P, ndim = self.P, len(off.shape)
        dims = self._dims(arg, m)
        lead = (1,) * ndim
        coords = None
        syms = {}
        known = {}
        if isinstance(off, S):
            # the symbolic terms, each a multiple of one stride (an int, or the stride the kernel loaded)
            for k, (basis, c, kn) in off.terms.items():
                dim = None
                if isinstance(c, str):
                    parts = c.split(":")
                    if len(parts) == 3 and parts[1] == (m.label or arg) and parts[2].isdigit():
                        dim = int(parts[2])
                else:
                    for st, n, i in dims:
                        if abs(c) == st:                 # a term subtracted (end - count) addresses the same axis
                            dim = i
                            break
                    if dim is None and not dims and m.shape:
                        dim = 0
                if dim is None:
                    raise _Fail("unproven", f"{what} addresses {arg} by a value read at run time that this module "
                                            f"cannot take apart by the strides ({off.why})")
                if kn is not None:                   # a launch-known coordinate on this axis
                    known.setdefault(dim, []).append(kn)
                    continue
                keys, b0 = syms.get(dim, ((), None))
                basis = _coordinate_basis(basis)
                sign = -1 if (not isinstance(c, str) and c < 0) else 1
                syms[dim] = (keys + ((k, sign),), _basis_of("+" if sign == 1 else "-", b0, basis) if keys else basis)
            off = off.const if off.const is not None else E(off.shape, s=np.zeros((1,), dtype=np.int64), w=None)
            if off.shape != tuple(lead) and len(off.shape) != ndim:
                raise Unmodelled(f"{what}: a symbolic address whose known part has another shape")
        if off.d is None:
            # kept apart: the scalar part in mixed radix, then each lane axis's part a multiple of one stride
            coords = {}
            rem = np.array(off.s, dtype=np.int64).reshape(-1)
            for s, n, i in dims:
                c = np.floor_divide(rem, s)
                rem = rem - c * s
                coords[i] = c.reshape((-1,) + lead)
            if np.any(rem != 0):
                coords = None
            else:
                taken = set()
                for j, vec in off.v.items():
                    vec = np.asarray(vec)
                    lane = [1] * ndim
                    lane[j] = vec.shape[1]
                    done = False
                    for s, n, i in dims:
                        if i in taken or np.any(vec % s):
                            continue
                        c = vec // s
                        if np.any(c < 0) or np.any(c >= n):
                            continue                  # a carry into the next dimension: taken apart below
                        coords[i] = coords[i] + c.reshape((-1,) + tuple(lane))
                        taken.add(i)
                        done = True
                        break
                    if not done:
                        coords = None
                        break
        if coords is None:
            o = _full(off.full(), off.shape, P)
            rem = o.copy()
            coords = {}
            for s, n, i in dims:
                c = np.floor_divide(rem, s)
                rem = rem - c * s
                coords[i] = c
            if not dims and m.shape:
                coords[0] = rem
                rem = np.zeros_like(rem)
                dims = [(1, 1, 0)]
            if valid is None:
                v = True
            else:
                v = _full(valid, off.shape, P)
            if bounds and np.any(v & (rem != 0)):
                raise _Fail("violation", f"{what} addresses {arg} between its elements (strides {tuple(m.stride)})",
                            {"program_chunk_index": self.chunk_index})
        for i, (s, n) in enumerate(zip(m.stride, m.shape)):
            if i not in coords:
                coords[i] = np.zeros((1,) + lead, dtype=np.int64)
        for i, kns in known.items():
            for kn in kns:
                arr = np.asarray(kn.full())
                arr = arr.reshape((-1,) + lead) if not kn.shape else arr.reshape((-1,) + tuple(kn.shape))
                coords[i] = coords[i] + arr
        for s, n, i in dims:
            if i in syms or not bounds:
                continue                     # a coordinate chosen by data: its bounds are the data's
            c = coords[i]
            out = (c < 0) | (c >= n)
            bad = out if valid is None else (valid & out)
            if np.any(bad):
                idx = _first(np.broadcast_to(bad, (P,) + tuple(off.shape)))
                cc = np.broadcast_to(c, (P,) + tuple(off.shape))
                raise _Fail("violation", f"{what} addresses {arg} outside its {tuple(m.shape)} elements (coordinate "
                                         f"{int(cc[idx])} on axis {i} of size {n})",
                            {"program_chunk_index": self.chunk_index, "lane": list(idx[1:])})
        for i, (k, basis) in syms.items():
            ax = m.axes[i] if i < len(m.axes) else None
            if ax is not None and ax.name is not None and basis is not None and ax.name != basis:
                raise _Fail("violation", f"{what} addresses {arg} along its axis '{ax.name}' with a number that "
                                         f"means a {basis}", {"program_chunk_index": self.chunk_index, "axis": i,
                                                                "basis": basis})
            coords[i] = Sym(k, basis, coords[i])
        return coords

    def _load(self, op, p, mask, bounds=True):
        """A float load as a typed value (masked_zero is set by the caller)."""
        m = self.meanings.get(p.arg)
        shape = KI._shape(op.rtype) or ()
        if m is None:
            return V(shape, leaf=p.arg, extras=(f"a value of {p.arg}, whose meaning the launch did not give",))
        if p.taint is not None or p.off is None:
            return V(shape, leaf=p.arg, data_addr=p.taint or "a data-dependent address")
        valid = None if mask is None else _as_bool_full(mask)
        if valid is not None and valid.ndim != 1 + len(shape):
            valid = valid.reshape((-1,) + tuple(shape))
        if not self.active.all():          # lanes of programs not in this branch hold nothing
            valid = self._active_lanes(len(shape)) & (_ones(shape) if valid is None else valid)
        cs = self._coords_of(p.arg, p.off, valid, "a load", bounds=bounds)
        coords = {}
        for i, ax in enumerate(m.axes):
            key = ax.name if ax.name is not None else (p.arg, i)
            coords[key] = (cs[i], ax.group)
        serials = frozenset([m.serial]) if m.kind == "value" and m.pair else frozenset()
        applied = {m.pair: 0} if m.kind == "value" and m.pair else {}
        return V(shape, coords, valid, {}, serials, applied, (), False, None, p.arg, True, None, None,
                 m.pair if m.kind == "scale" else 0, m.serial if m.kind == "scale" else 0)

    # -- pairing --

    def _eq(self, a, ga, b, gb, where):
        """Lanes (a bool array, broadcast) where coordinates a (group ga) and b (group gb) disagree, among `where`."""
        if ga == gb:
            return where & (a != b)
        g = max(ga, gb)
        if g % ga or g % gb:
            raise Unmodelled(f"groups of {ga} and {gb} on one axis")
        return where & ((a * ga) // g != (b * gb) // g)

    def _pair(self, x, y, how):
        """x and y combined elementwise: their coordinates must agree on every axis they share, where both hold
        real elements; returns the combined coordinates and validity. On a value whose lanes are the data's, a
        disagreement is undecided (the lanes may be the masked ones), not a violation."""
        try:
            return self._pair_checked(x, y, how)
        except _Fail as e:
            if e.verdict == "violation" and (x.data_valid or y.data_valid):
                raise _Fail("unproven", f"whether {how} pairs its elements depends on data read at run time "
                                        f"(lanes masked by data): {e.why}", e.example)
            raise

    def _pair_checked(self, x, y, how):
        P = self.P
        shape = np.broadcast_shapes(x.shape, y.shape)
        full = (P,) + tuple(shape)
        vx = _ones(shape) if x.valid is None else x.valid
        vy = _ones(shape) if y.valid is None else y.valid
        both = vx & vy
        coords = {}
        for key in set(x.coords) | set(y.coords):
            cx, cy = x.coords.get(key), y.coords.get(key)
            if isinstance(key, str) and ((cx is not None and key in y.sums) or (cy is not None and key in x.sums)):
                # a coordinate on an axis the other side has summed over (a scale applied to a sum over k): every
                # summed coordinate must map to it; the axis stays summed, it is not a coordinate of the result
                side, other_sums = (x, y.sums) if cx is not None else (y, x.sums)
                if key in side.sums:
                    raise Unmodelled(f"a value with both a coordinate and a sum on the axis '{key}'")
                c, g = side.coords[key]
                lead = (-1,) + (1,) * len(shape)
                for lo, cnt in other_sums[key]:
                    has = cnt > 0
                    first, last = lo // g, (lo + np.maximum(cnt, 1) - 1) // g
                    if np.any(has & (first != last)):
                        p = int(np.nonzero(has & (first != last))[0][0])
                        groups = list(range(int(first[p]), int(last[p]) + 1))
                        raise _Fail("violation", f"{how} applies one coordinate on the axis '{key}' (groups of {g}) "
                                                 f"to a sum over {int(cnt[p])} coordinates spanning {len(groups)} "
                                                 f"groups", {"program_chunk_index": p, "groups_in_tile": groups[:8]})
                    want = first.reshape(lead)
                    bad = both & has.reshape(lead) & (c != want)
                    self.checks += 1
                    if np.any(bad):
                        idx = _first(np.broadcast_to(bad, full))
                        raise _Fail("violation", f"{how} applies a coordinate on the axis '{key}' (group "
                                                 f"{int(np.broadcast_to(c, full)[idx])} of {g}) to a sum over group "
                                                 f"{int(first[idx[0]])}", {"program_chunk_index": idx[0],
                                                                          "lane": list(idx[1:]), "axis": key})
            elif cx is not None and cy is not None and isinstance(key, str):
                if isinstance(cx[0], Sym) or isinstance(cy[0], Sym):
                    if not _same_coord(cx[0], cy[0]):
                        raise _Fail("unproven", f"whether {how} pairs the same '{key}' depends on data read at "
                                                f"run time")
                    coords[key] = cx
                    continue
                bad = self._eq(cx[0], cx[1], cy[0], cy[1], both)
                self.checks += 1
                if np.any(bad):
                    idx = _first(np.broadcast_to(bad, full))
                    ax, ay = np.broadcast_to(cx[0], full), np.broadcast_to(cy[0], full)
                    raise _Fail("violation", f"{how} pairs elements that disagree on the axis '{key}': "
                                             f"{int(ax[idx]) * cx[1]} with {int(ay[idx]) * cy[1]} (base coordinates)",
                                {"program_chunk_index": self.chunk_index, "lane": list(idx[1:]), "axis": key,
                                 "x": int(ax[idx]), "x_group": cx[1], "y": int(ay[idx]), "y_group": cy[1]})
                coords[key] = cx if cx[1] <= cy[1] else cy
            else:
                coords[key] = cx if cx is not None else cy
        return shape, coords, vx, vy

    def _binary(self, kind, x, y):
        if x.alts is not None or y.alts is not None:
            return _alt_map(lambda a, b: self._binary(kind, a, b), x, y)
        shape = np.broadcast_shapes(x.shape, y.shape)
        if x.const is not None and y.const is not None:
            vals = {"mul": x.const * y.const, "add": x.const + y.const, "sub": x.const - y.const,
                    "div": (x.const / y.const if y.const else None), "max": max(x.const, y.const),
                    "min": min(x.const, y.const)}
            return V(shape, const=vals.get(kind))
        if kind in ("mul", "div"):
            for c, o in ((x, y), (y, x)):
                if c.const is not None:
                    if kind == "div" and c is x:
                        return o.copy(shape=shape, fn=True, extras=o.extras + (f"a constant {c.const} divided by "
                                                                                f"it",))
                    if c.const == 1.0:
                        return o.copy(shape=shape)
                    if c.const == 0.0 and kind == "mul":
                        return V(shape, const=0.0)
                    return o.copy(shape=shape, extras=o.extras + (f"the constant factor {c.const}",))
            _shape, coords, vx, vy = self._pair(x, y, "a multiplication" if kind == "mul" else "a division")
            applied = dict(x.applied)
            for k, v in y.applied.items():
                applied[k] = applied.get(k, 0) + v
            for s, o in ((x, y), (y, x)):
                if s.scale_serial:
                    if s.scale_of not in o.serials:
                        raise _Fail("violation", f"a value is multiplied by the scale {s.leaf} (issue "
                                                 f"{s.scale_serial}, the scale of issue {s.scale_of}) that is not "
                                                 f"its own (it derives from "
                                                 f"{sorted(o.serials) if o.serials else 'no quantized value'})")
                    applied[s.scale_serial] = applied.get(s.scale_serial, 0) + 1
            sums = dict(x.sums)
            for k, v in y.sums.items():
                if k in sums:
                    raise Unmodelled(f"a product of two sums over the axis '{k}'")
                sums[k] = v
            return V(shape, coords, vx & vy, sums, x.serials | y.serials, applied, x.extras + y.extras,
                     x.fn or y.fn, masked_zero=x.masked_zero and y.masked_zero,
                     data_addr=x.data_addr or y.data_addr, data_valid=x.data_valid or y.data_valid)
        if kind in ("add", "sub"):
            if y.const == 0.0:
                return x.copy(shape=shape)
            if x.const == 0.0 and kind == "add":
                return y.copy(shape=shape)
            if x.const == 0.0 and kind == "sub":
                return y.copy(shape=shape, extras=y.extras + ("negated",))
            if x.const is not None or y.const is not None:
                o = y if x.const is not None else x
                c = x if x.const is not None else y
                return o.copy(shape=shape, extras=o.extras + (f"the constant addend {c.const}",))
            _shape, coords, vx, vy = self._pair(x, y, "an addition" if kind == "add" else "a subtraction")
            sums = {}
            extras = x.extras + y.extras
            for k in set(x.sums) | set(y.sums):
                if k in x.sums and k in y.sums:
                    sums[k] = x.sums[k] + y.sums[k]
                else:
                    sums[k] = x.sums.get(k) or y.sums.get(k)
                    extras = extras + (f"an addend that is not a term of the sum over '{k}'",)
            if kind == "sub":
                extras = extras + ("a subtracted addend",)
            applied = dict(x.applied)
            for k, v in y.applied.items():
                if applied.get(k, v) != v:
                    raise Unmodelled(f"a sum of values with the scale of issue {k} applied a different number "
                                     f"of times")
                applied[k] = v
            return V(shape, coords, vx | vy, sums, x.serials | y.serials, applied, extras, x.fn or y.fn,
                     masked_zero=x.masked_zero and y.masked_zero, data_addr=x.data_addr or y.data_addr,
                     data_valid=x.data_valid or y.data_valid)
        _shape, coords, vx, vy = self._pair(x, y, f"a {kind}")      # max, min: paired, no longer a plain term
        return V(shape, coords, vx & vy, {}, x.serials | y.serials, {}, x.extras + y.extras +
                 (f"a {kind}",), True, data_addr=x.data_addr or y.data_addr,
                 data_valid=x.data_valid or y.data_valid)

    def _dot(self, a, b, c):
        if a.alts is not None or b.alts is not None:
            return _alt_map(lambda x, y: self._dot(x, y, c), a, b)
        for leaf in (a, b):
            if leaf.leaf is None:
                raise Unmodelled("a dot whose operands are not loads")
            if leaf.data_valid:
                raise _Fail("unproven", f"the lanes of {leaf.leaf} in the dot are masked by data read at run time")
            if not leaf.masked_zero:
                raise _Fail("unproven", f"masked-out lanes of {leaf.leaf} in the dot are not loaded as zero")
            if leaf.data_addr:
                raise _Fail("unproven", f"the dot reads {leaf.leaf} at an address chosen by data ({leaf.data_addr})")
            if leaf.extras:
                raise _Fail("unproven", f"the dot reads {leaf.leaf}: {leaf.extras[0]}")
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise Unmodelled("a dot of operands that are not 2-D")
        P = self.P
        R, Tn = a.shape
        Tb, C = b.shape
        if Tn != Tb:
            raise Unmodelled("a dot whose contraction lengths differ")
        va = _full(_ones(a.shape) if a.valid is None else a.valid, a.shape, P)
        vb = _full(_ones(b.shape) if b.valid is None else b.valid, b.shape, P)
        rv, tva = va.any(axis=2), va.any(axis=1)
        tvb, cv = vb.any(axis=2), vb.any(axis=1)
        if not np.array_equal(va, rv[:, :, None] & tva[:, None, :]) or \
                not np.array_equal(vb, tvb[:, :, None] & cv[:, None, :]):
            raise Unmodelled("an operand mask that is not a row (or column) condition times a k condition")
        live = (rv.any(axis=1) & cv.any(axis=1))[:, None]
        if np.any(live & (tva != tvb)):
            raise _Fail("unproven", "the two operands are masked differently along k")
        vT = tva & tvb & live
        # each coordinate of an operand is carried (it does not vary along k) or contracted (it varies only along k)
        carried, contracted = {"a": {}, "b": {}}, {"a": {}, "b": {}}
        for side, v, name in ((a, va, "a"), (b, vb, "b")):
            kax, oax = (2, 1) if name == "a" else (1, 2)       # the lane axis of k, the lane axis of rows / columns
            for key, (arr, g) in side.coords.items():
                if arr.shape[kax] == 1:
                    carried[name][key] = (np.take(arr, 0, axis=kax), g)
                elif arr.shape[oax] == 1:
                    contracted[name][key] = (np.take(arr, 0, axis=oax), g)
                else:                                          # varies along both lanes: decided by the data
                    per_o = np.where(v, arr, -1).max(axis=kax)
                    per_k = np.where(v, arr, -1).max(axis=oax)
                    if np.array_equal(np.where(v, arr, np.expand_dims(per_o, kax)),
                                      np.broadcast_to(np.expand_dims(per_o, kax), v.shape)):
                        carried[name][key] = (per_o, g)
                    elif np.array_equal(np.where(v, arr, np.expand_dims(per_k, oax)),
                                        np.broadcast_to(np.expand_dims(per_k, oax), v.shape)):
                        contracted[name][key] = (per_k, g)
                    else:
                        raise Unmodelled(f"an operand coordinate on '{key}' that varies along both lane axes")
        ca, cb = contracted["a"], contracted["b"]
        shared = [k for k in ca if k in cb and isinstance(k, str)]
        if not shared:
            raise _Fail("unproven", "the dot contracts axes that share no name (" +
                        ", ".join(str(k) for k in list(ca) + list(cb)) + ")")
        sums = {}
        for key in shared:
            (ka, ga), (kb, gb) = ca[key], cb[key]
            if ga != 1 or gb != 1:
                raise Unmodelled(f"a dot over the grouped axis '{key}'")
            ka, kb = np.broadcast_to(ka, (P, Tn)), np.broadcast_to(kb, (P, Tn))
            bad = vT & (ka != kb)
            self.checks += 1
            if np.any(bad):
                raise _Fail("violation", f"the two operands are read at different '{key}' for one contraction "
                                         f"index", {"program_chunk_index": self.chunk_index})
            lo, cnt = _ranges(ka, vT, f"a contraction tile reads one '{key}' twice",
                              f"a contraction tile whose '{key}' are not one range")
            sums[key] = [(lo, cnt)]
        for key in list(ca) + list(cb):
            if key not in shared:
                raise _Fail("unproven", f"the dot contracts the axis {key} of one operand against another name")
        coords = {}
        for key, (arr, g) in carried["a"].items():
            coords[key] = (arr[:, :, None], g)
        for key, (arr, g) in carried["b"].items():
            if key in coords:
                raise Unmodelled(f"both operands carry the axis '{key}' into the dot's output")
            coords[key] = (arr[:, None, :], g)
        applied = dict(a.applied)
        for k, v in b.applied.items():
            applied[k] = applied.get(k, 0) + v
        out = V((R, C), coords, rv[:, :, None] & cv[:, None, :], sums, a.serials | b.serials, applied, (), False)
        if c is not None and not (c.const == 0.0):
            return self._binary("add", c, out)
        return out

    def _reduce(self, op, x):
        if x.alts is not None:
            return _alt_map(lambda a: self._reduce(op, a), x)
        axis = int(re.search(r'axis = (\d+)', op.text).group(1))
        names = {o.name for o in KI._walk(op.body)}
        kind = "sum" if "arith.addf" in names else ("max" if {"arith.maxnumf", "arith.maximumf"} & names else
                                                   "min" if {"arith.minnumf", "arith.minimumf"} & names else None)
        if kind is None:
            raise Unmodelled(f"a reduction whose combiner is {sorted(names)}")
        P = self.P
        shape = tuple(n for i, n in enumerate(x.shape) if i != axis)
        v = _full(_ones(x.shape) if x.valid is None else x.valid, x.shape, P)
        lane = axis + 1
        coords, sums = {}, {k: list(r) for k, r in x.sums.items()}
        for key, (arr, g) in x.coords.items():
            if arr.shape[lane] == 1:
                coords[key] = (np.take(arr, 0, axis=lane), g)       # constant along the reduced lanes: carried
            elif kind == "sum" and not isinstance(key, str):
                continue                                           # an axis without a name: nothing to cover
            elif kind == "sum":
                if g != 1:
                    raise Unmodelled(f"a sum over the grouped axis '{key}'")
                if any(s != 1 for i, s in enumerate(arr.shape) if i not in (0, lane)):
                    raise Unmodelled(f"a sum over '{key}' whose coordinates differ between the other lanes")
                vm = np.moveaxis(v, lane, -1).reshape(P, -1, x.shape[axis])
                if not np.array_equal(vm, np.broadcast_to(vm[:, :1, :], vm.shape)):
                    raise Unmodelled(f"a sum over '{key}' whose lanes are masked differently")
                moved = np.broadcast_to(np.take(np.moveaxis(arr, lane, -1), 0, axis=1) if arr.ndim > 2 else arr,
                                        (P, x.shape[axis]))
                lo, cnt = _ranges(moved, vm[:, 0, :], f"a sum reads one '{key}' twice",
                                  f"a sum whose '{key}' are not one range")
                sums[key] = sums.get(key, []) + [(lo, cnt)]
            elif kind in ("max", "min"):
                continue                                           # the coordinate along the reduced lanes is gone
            else:
                raise Unmodelled(f"a {kind} over the axis {key}")
        return V(shape, coords, v.any(axis=lane), sums, x.serials, x.applied,
                 x.extras + ((f"a {kind}",) if kind != "sum" else ()), x.fn or kind != "sum",
                 data_addr=x.data_addr)

    def _shape_v(self, n, op, x, shape):
        if x.alts is not None:
            return _alt_map(lambda a: self._shape_v(n, op, a, shape), x)
        if x.const is not None:
            return V(shape, const=x.const)
        ndim = len(shape)

        def f(arr):
            if n == "tt.splat":
                return np.asarray(arr).reshape((-1,) + (1,) * ndim)
            if n == "tt.expand_dims":
                axis = int(re.search(r'axis = (\d+)', op.text).group(1))
                return np.expand_dims(arr, axis + 1)
            if n == "tt.broadcast":
                return arr                                        # numpy broadcasts when the arrays meet
            if n == "tt.reshape":
                if isinstance(arr, Sym):
                    raise Unmodelled("a reshape of a coordinate chosen by data")
                arr = np.asarray(arr)
                return np.broadcast_to(arr, (arr.shape[0],) + tuple(x.shape)).reshape((arr.shape[0],) +
                                                                                          tuple(shape))
            if n == "tt.trans":
                order = [int(t) for t in re.search(r'order = array<i32: ([0-9, ]+)>', op.text).group(1).split(",")]
                return np.transpose(arr, (0,) + tuple(o + 1 for o in order))
            raise Unmodelled(f"{n} of a typed value")
        return x.copy(shape=tuple(shape), coords={k: (f(a), g) for k, (a, g) in x.coords.items()},
                      valid=None if x.valid is None else f(x.valid))

    def _store(self, args):
        ptr, val = args[0], args[1]
        mask = args[2] if len(args) > 2 else None
        if not isinstance(ptr, Ptr):
            raise Unmodelled("a store through a value that is not a pointer")
        m = self.meanings.get(ptr.arg)
        if m is None:
            raise _Fail("unproven", f"the kernel writes to {ptr.arg}, whose meaning the launch did not give")
        if m.kind != "output":
            raise _Fail("violation", f"the kernel writes to {ptr.arg}, which is {m.kind} it reads, not its output")
        if ptr.taint is not None:
            raise _Fail("unproven", f"the output address is chosen by data ({ptr.taint})")
        if not isinstance(val, V):
            if isinstance(val, (T, E, Mk)):
                return self._store_int(ptr, val, mask, m)
            raise Unmodelled(f"a store of a {type(val).__name__}")
        if isinstance(mask, T):
            raise Unmodelled("a store of a float value masked by data")
        stored = None if mask is None else _as_bool_full(mask)
        for alt in _alts(val):
            try:
                self._store_one(ptr, alt, stored, m)
            except _Fail as e:
                if val.alts is not None and e.verdict == "violation":
                    raise _Fail("possible", f"a value chosen at run time: {e.why}", e.example)
                raise

    def _store_int(self, ptr, val, mask, m):
        """An integer stored: its basis must be the output's, and it lands where the address says."""
        P = self.P
        shape = tuple(ptr.off.shape)
        inferred = self.inferred.setdefault(ptr.arg, {})
        if isinstance(mask, T):
            sv = None
            inferred["coverage"] = "data-dependent (a store masked by data)"
        else:
            sv = np.ones((P,) + shape, dtype=bool) if mask is None else _full(_as_bool_full(mask), shape, P)
            if not self.active.all():
                sv = sv & self._active_lanes(len(shape))
        cs = self._coords_of(ptr.arg, ptr.off, sv, "a store", bounds=sv is not None)
        basis = _coordinate_basis(val.basis) if isinstance(val, S) else None
        if m.basis is not None:
            if basis is not None and basis != m.basis:
                raise _Fail("violation", f"a number that means a {basis} is stored into {ptr.arg}, which holds "
                                         f"{m.basis}s", {"program_chunk_index": self.chunk_index})
            if basis is None:
                inferred["values"] = (f"a value whose basis is not known is stored ({val.why})" if isinstance(val, T)
                                      else "launch-known values")
        self.checks += 1
        if sv is None or self.data_depth or any(isinstance(c, Sym) for c in cs.values()):
            inferred["coverage"] = inferred.get("coverage") or "data-dependent"
            return
        self._footprint(ptr.arg, cs, sv, shape)

    def _footprint(self, arg, cs, sv, shape):
        P = self.P
        full = (P,) + shape
        live = sv.reshape(P, -1).any(axis=1)
        if len(shape) == 2:
            sr, sc = sv.any(axis=2), sv.any(axis=1)
            if not np.array_equal(sv, sr[:, :, None] & sc[:, None, :]):
                raise Unmodelled("a store mask that is not a set of rows times a set of columns")
            r = cs[0] if 0 in cs else np.zeros((1, 1, 1), dtype=np.int64)
            c = cs[1] if 1 in cs else np.zeros((1, 1, 1), dtype=np.int64)
            rows = np.where(sr, np.broadcast_to(r[:, :, 0], (P, shape[0])), -1) if r.shape[2] == 1 else \
                np.where(sv, r, -1).max(axis=2)
            cols = np.where(sc, np.broadcast_to(c[:, 0, :], (P, shape[1])), -1) if c.shape[1] == 1 else \
                np.where(sv, c, -1).max(axis=1)
            rlo, rcnt = _ranges(rows, sr, "one output row is stored twice by one program",
                                "the rows a program stores are not one range", twice="unproven")
            clo, ccnt = _ranges(cols, sc, "one output column is stored twice by one program",
                                "the columns a program stores are not one range", twice="unproven")
            self.stores.setdefault(arg, []).append((rlo[live], rcnt[live], clo[live], ccnt[live]))
        elif len(shape) == 1:
            x = np.broadcast_to(cs[0], full)
            lo, cnt = _ranges(np.where(sv, x, -1), sv, "one output element is stored twice by one program",
                              "the elements a program stores are not one range", twice="unproven")
            self.stores.setdefault(arg, []).append((lo[live], cnt[live], np.zeros(int(live.sum()), dtype=np.int64),
                                                    np.ones(int(live.sum()), dtype=np.int64)))
        elif len(shape) == 0:
            x = np.broadcast_to(cs[0], (P,))
            self.stores.setdefault(arg, []).append((x[live], np.ones(int(live.sum()), dtype=np.int64),
                                                    np.zeros(int(live.sum()), dtype=np.int64),
                                                    np.ones(int(live.sum()), dtype=np.int64)))
        else:
            raise Unmodelled(f"a store of {len(shape)} lane dimensions")

    def _store_one(self, ptr, val, stored, m):
        P = self.P
        shape = tuple(ptr.off.shape)
        full = (P,) + shape
        sv = np.ones(full, dtype=bool) if stored is None else _full(stored, shape, P)
        if not self.active.all():
            sv = sv & self._active_lanes(len(shape))
        if val.data_addr:
            raise _Fail("unproven", f"the value stored was read at an address chosen by data ({val.data_addr})")
        cs = self._coords_of(ptr.arg, ptr.off, sv, "a store")
        symbolic = any(isinstance(c, Sym) for c in cs.values()) or val.data_valid
        if val.data_valid:
            try:
                return self._store_typed(ptr, val, sv, cs, m, symbolic, inferred_note="data-dependent (lanes "
                                                                                      "masked by data)")
            except _Fail as e:
                if e.verdict == "violation":
                    raise _Fail("unproven", f"whether the value is stored at its own place depends on data read at "
                                            f"run time (lanes masked by data): {e.why}", e.example)
                raise
        return self._store_typed(ptr, val, sv, cs, m, symbolic)

    def _store_typed(self, ptr, val, sv, cs, m, symbolic, inferred_note=None):
        P = self.P
        shape = tuple(ptr.off.shape)
        full = (P,) + shape
        vv = _ones(shape) if val.valid is None else val.valid
        if np.any(sv & ~vv):
            raise _Fail("violation", "the kernel stores output lanes whose value has no element (masked-out operand "
                                     "lanes)", {"program_chunk_index": self.chunk_index})
        inferred = self.inferred.setdefault(ptr.arg, {})
        for i, ax in enumerate(m.axes):
            oc = cs[i]
            if ax.name is not None:
                c = val.coords.get(ax.name)
                if c is None:
                    raise _Fail("unproven", f"the value stored has no coordinate on the output's axis '{ax.name}' "
                                            f"(it derives from {sorted(str(k) for k in val.coords) or 'nothing'})")
                if isinstance(oc, Sym) or isinstance(c[0], Sym):
                    if not _same_coord(oc, c[0]):
                        raise _Fail("unproven", f"whether the value is stored at its own '{ax.name}' depends on "
                                                f"data read at run time")
                    continue
                bad = self._eq(oc, 1, c[0], c[1], sv)
                self.checks += 1
                if np.any(bad):
                    idx = _first(np.broadcast_to(bad, full))
                    raise _Fail("violation", f"an output value is stored where another '{ax.name}' belongs "
                                             f"(stored at {int(np.broadcast_to(oc, full)[idx])}, the value's is "
                                             f"{int(np.broadcast_to(c[0], full)[idx]) * c[1]})",
                                {"program_chunk_index": self.chunk_index, "lane": list(idx[1:]), "axis": ax.name})
            else:
                found = None
                for key, (arr, g) in val.coords.items():
                    if isinstance(oc, Sym) or isinstance(arr, Sym):
                        if g == 1 and _same_coord(oc, arr):
                            found = key
                            break
                        continue
                    if g == 1 and not np.any(self._eq(oc, 1, arr, 1, sv)):
                        found = key
                        break
                if found is None and np.any(sv):
                    raise _Fail("unproven", f"no coordinate of the value stored matches the output's axis {i}")
                if found is not None:
                    was = inferred.get(i)
                    if was is not None and was != found:
                        raise _Fail("unproven", f"the output's axis {i} matches {was} in one store and {found} in "
                                                f"another")
                    inferred[i] = found
        for name in m.reduced:
            if name not in val.sums:
                raise _Fail("violation", f"the output is the sum over '{name}', but the value stored is not summed "
                                         f"over it")
        live = sv.reshape(P, -1).any(axis=1)
        if symbolic or self.data_depth:
            live = np.zeros(P, dtype=bool)        # nothing is claimed complete under data
        for name, rs in val.sums.items():
            if name in m.reduced:
                lo, hi = m.reduced[name]
                bad = _covers(rs, lo, hi, live)
                if bad is not None:
                    p, why = bad
                    raise _Fail("violation", f"the sum over '{name}' stored {why}", {"program_chunk_index": p})
            elif m.strict:
                raise _Fail("violation", f"the value stored is summed over '{name}', which the output does not reduce")
            else:
                inferred.setdefault("sums", {})[name] = "partial or complete, not declared"
        if m.strict:
            if val.extras:
                raise _Fail("violation", f"the output gets more than its meaning: {val.extras[0]}")
            if val.fn:
                raise _Fail("unproven", "the value stored is a non-linear function of the paired values")
            for s in sorted(val.serials):       # a quantized value means value x its scale: applied exactly once
                value = next((v for v in self.meanings.values() if v.kind == "value" and v.serial == s), None)
                n = val.applied.get(value.pair, 0) if value is not None else 0
                if n != 1:
                    raise _Fail("violation", f"the value of issue {s} reaches the output with its scale applied "
                                             f"{n} times (once is its meaning)")
        else:
            inferred["pending_scales"] = sorted(k for k, v in val.applied.items() if v == 0)
            if val.extras:
                inferred["extras"] = list(val.extras)
        if symbolic or self.data_depth:
            if m.strict or m.reduced:
                raise _Fail("unproven", "whether the declared output is covered depends on data read at run time")
            inferred["coverage"] = inferred.get("coverage") or inferred_note or "data-dependent"
            return
        self._footprint(ptr.arg, cs, sv, shape)

    # -- the ops --

    def _op(self, op, env):  # noqa: C901
        n = op.name
        args = [self._get(env, a) for a in op.operands]
        shape = KI._shape(op.rtype)
        r = None
        if n == "arith.constant":
            et = KI._elem(op.rtype)
            if et.startswith("f") or et.startswith("bf"):
                mm = re.match(r'(?:dense<)?([-0-9.eE+a-z]+)>?', op.text)
                try:
                    val = float(mm.group(1)) if mm else None
                except ValueError:
                    val = None
                r = V(shape or (), const=val)
            else:
                return super()._op(op, env)
        elif n == "tt.load":
            p = args[0]
            mask = args[1] if len(args) > 1 else None
            et = KI._elem(op.rtype)
            if et.startswith("f") or et.startswith("bf"):
                if not isinstance(p, Ptr):
                    raise Unmodelled("a load from a non-pointer")
                other = args[2] if len(args) > 2 else None
                if isinstance(mask, T):
                    r = self._load(op, p, None, bounds=False)
                    r.data_valid = True
                else:
                    r = self._load(op, p, mask)
                r.masked_zero = mask is None or (isinstance(other, V) and other.is_zero())
            else:
                if not isinstance(p, Ptr):
                    raise Unmodelled("a load from a non-pointer")
                m = self.meanings.get(p.arg)
                if m is not None and p.taint is None and p.off is not None and \
                        (m.basis is not None or m.kind == "pointers" or any(a.name for a in m.axes)):
                    # an integer with a meaning: its address is held to the tensor's axes like a float's
                    valid = None if mask is None or isinstance(mask, T) else _as_bool_full(mask)
                    if valid is not None and valid.ndim != 1 + len(shape or ()):
                        valid = valid.reshape((-1,) + tuple(shape or ()))
                    if not self.active.all():
                        valid = self._active_lanes(len(shape or ())) & (_ones(shape or ()) if valid is None
                                                                        else valid)
                    cs = self._coords_of(p.arg, p.off, valid, "a load", bounds=not isinstance(mask, T))
                    if m.kind == "pointers":
                        c0 = cs.get(0)
                        if isinstance(c0, Sym) or np.unique(np.asarray(c0)).size != 1:
                            raise Unmodelled(f"a pointer read from {p.arg} at a position this chunk does not share")
                        which = int(np.asarray(c0).flat[0])
                        targets = self.pointers.get(p.arg) or []
                        if not 0 <= which < len(targets):
                            raise _Fail("violation", f"a pointer read from {p.arg} at {which}, which holds "
                                                     f"{len(targets)} pointers")
                        r = _opaque(f"a pointer read from {p.arg}", self._key(op), f"ptr:{targets[which]}",
                                    shape or ())
                    else:
                        r = self._leaf(op, m.basis, shape or ())
                else:
                    return super()._op(op, env)
        elif (n in KI._INT_BINARY or n in KI._INT_CASTS or n == "arith.cmpi") and any(isinstance(a, S) for a in args):
            r = self._sym_int(n, op, args, shape)
        elif n == "tt.addptr" and (isinstance(args[1], S) or (isinstance(args[0], Ptr) and
                                                             isinstance(args[0].off, S))):
            p, o = args
            if p.taint is not None:
                r = p
            elif not isinstance(o, S):
                o = self._int(o)
                if isinstance(o, T) or not isinstance(o, E):
                    r = Ptr(p.arg, p.off, taint=getattr(o, "why", "an offset that is not an integer"))
                else:
                    base = p.off
                    r = Ptr(p.arg, S(base.why, base.basis, base.terms,
                                     self._const_add(base.const, o, 1, shape or base.shape), shape or base.shape))
            else:
                base = p.off
                if isinstance(base, S):
                    r = Ptr(p.arg, self._sym_int("arith.addi", op, [base, o], shape or o.shape))
                else:
                    known = base is not None and (base.v or base.d is not None or np.any(base.s != 0))
                    const = self._const_add(o.const, base, 1, shape or o.shape) if known else o.const
                    r = Ptr(p.arg, S(o.why, o.basis, o.terms, const, shape or o.shape))
        elif n == "tt.int_to_ptr" and isinstance(args[0], S) and str(args[0].basis).startswith("ptr:"):
            r = Ptr(args[0].basis[4:], E(()))
        elif n == "tt.dot":
            a, b = args[0], args[1]
            c = args[2] if len(args) > 2 else None
            if not (isinstance(a, V) and isinstance(b, V)):
                raise Unmodelled("a dot whose operands are not float values")
            if c is not None and not isinstance(c, V):
                raise Unmodelled("a dot that accumulates into a non-float")
            r = self._dot(a, b, c)
        elif n in _ELEMENTWISE_BINARY:
            x, y = args
            if not (isinstance(x, V) and isinstance(y, V)):
                raise Unmodelled(f"{n} of a non-float")
            r = self._binary(_ELEMENTWISE_BINARY[n], x, y)
        elif n in _UNARY_LINEAR:
            x = args[0]
            r = x.copy(extras=x.extras + ("negated",)) if x.alts is None else \
                _combine_alts([a.copy(extras=a.extras + ("negated",)) for a in x.alts])
        elif n.startswith("math.") or n in ("tt.extern_elementwise",):
            xs = [a for a in args if isinstance(a, V)]
            if not xs:
                return super()._op(op, env)
            x = xs[0]
            r = x.copy(shape=tuple(shape or x.shape), fn=True, sums={}, extras=x.extras + (n,)) if x.alts is None \
                else _combine_alts([a.copy(fn=True, sums={}, extras=a.extras + (n,)) for a in x.alts])
        elif n in KI._FLOAT_CASTS:
            x = args[0]
            if isinstance(x, V):
                if n in ("arith.truncf", "arith.extf"):
                    r = x
                elif n in ("arith.fptosi", "arith.fptoui"):
                    r = T(f"{n} of a float value")
                else:
                    raise Unmodelled(f"{n} of a float")
            elif isinstance(x, (E, Mk)) and n in ("arith.sitofp", "arith.uitofp"):
                r = V(shape or (), fn=True, extras=(f"{n}: a float made from an integer",))
            elif isinstance(x, T) and n in ("arith.sitofp", "arith.uitofp"):
                r = V(shape or (), fn=True, extras=(f"{n}: a float made from data read at run time",),
                      data_valid=True)
            else:
                return super()._op(op, env)
        elif n == "arith.select" and isinstance(args[0], T) and not any(isinstance(a, V) for a in args) and \
                any(isinstance(a, S) for a in args[1:]):
            r = self._merge_data(args[1], args[2])
        elif n == "arith.select" and any(isinstance(a, V) for a in args):
            c, a, b = args
            if not (isinstance(a, V) and isinstance(b, V)):
                raise Unmodelled("a select between a float and something else")
            if isinstance(c, T):
                r = self._select_v(None, a, b)
            else:
                cc = self._int(c)
                if not isinstance(cc, E) or cc.w != 1:
                    raise Unmodelled("a select whose condition is not i1")
                if cc.scalar_only() and np.unique(cc.s).size == 1:
                    r = a if int(cc.s.flat[0]) else b
                else:
                    r = self._select_v(cc, a, b)
        elif n in ("tt.splat", "tt.broadcast", "tt.expand_dims", "tt.reshape", "tt.trans") and \
                isinstance(args[0], V):
            r = self._shape_v(n, op, args[0], shape)
        elif n == "tt.bitcast" and isinstance(args[0], V):
            r = T(f"{n} of a float value") if KI._width(op.rtype) else args[0].copy(fn=True, extras=args[0].extras +
                                                                                     (f"{n} of a float",))
        elif n == "tt.reduce":
            x = args[0]
            if isinstance(x, V):
                r = self._reduce(op, x)
            else:
                return super()._op(op, env)
        elif n == "tt.atomic_rmw":
            ptr = args[0] if args else None
            if not isinstance(ptr, Ptr):
                raise Unmodelled("an atomic through a value that is not a pointer")
            m = self.meanings.get(ptr.arg)
            if m is not None and m.kind == "output":
                if m.strict or m.reduced:
                    raise _Fail("violation", f"the kernel writes to its output {ptr.arg} with an atomic (each element "
                                             f"is to be stored once)")
                inf = self.inferred.setdefault(ptr.arg, {})
                inf["coverage"] = "accumulated by atomics (what the data decides)"
                r = T("the old value of an atomic")
                for name in op.results[:1]:
                    env[name] = r
                return
            if m is None or m.kind != "sums" or "add" not in op.text.split(",")[0]:
                raise _Fail("violation", f"the kernel writes to {ptr.arg} with an atomic other than an add into a "
                                         f"sums buffer")
            r = T("the old value of an atomic add")
        elif n == "tt.store":
            self._store(args)
            return
        elif n == "tt.get_num_programs":
            axis = {"x": 0, "y": 1, "z": 2}[op.text.split()[0]] if op.text.split() and op.text.split()[0] in "xyz" \
                else int(re.search(r'axis = (\d+)', op.text).group(1))
            r = E((), s=np.full((1,), self.grid[axis], dtype=np.int64), w=32)
        elif n == "arith.cmpf":
            r = T("a comparison of float values")
        else:
            return super()._op(op, env)
        for name in op.results[:1]:
            env[name] = r


def _covers(ranges, lo, hi, live):
    """None when the per-program ranges [(start (P,), count (P,))] cover [lo, hi) exactly once for every live
    program; else (program, why)."""
    if not ranges:
        return (int(np.nonzero(live)[0][0]) if live.any() else 0), "nothing"
    starts = np.stack([np.broadcast_to(r[0], live.shape) for r in ranges], axis=1)
    counts = np.stack([np.broadcast_to(r[1], live.shape) for r in ranges], axis=1)
    for p in np.nonzero(live)[0]:
        s, c = starts[p], counts[p]
        s, c = s[c > 0], c[c > 0]
        order = np.argsort(s)
        s, c = s[order], c[order]
        if s.size == 0:
            return int(p), "nothing"
        if s[0] != lo:
            return int(p), f"from {int(s[0])}, not from {lo}"
        ends = s + c
        if np.any(s[1:] < ends[:-1]):
            i = int(np.nonzero(s[1:] < ends[:-1])[0][0])
            return int(p), f"{int(s[i + 1])}..{int(min(ends[i], ends[i + 1]) - 1)} twice"
        if np.any(s[1:] > ends[:-1]):
            i = int(np.nonzero(s[1:] > ends[:-1])[0][0])
            return int(p), f"{int(ends[i])}..{int(s[i + 1]) - 1} never"
        if ends[-1] != hi:
            return int(p), (f"{int(ends[-1])}..{hi - 1} never" if ends[-1] < hi else f"up to {int(ends[-1]) - 1}, "
                                                                                    f"past {hi - 1}")
    return None


def check_launch(ttir: str, meanings: Dict[str, Meaning], ints: Dict[str, int], grid,
                 chunk_elements: int = CHUNK_ELEMENTS, pointers: Optional[Dict[str, list]] = None) -> Verdict:
    """The verdict on one launch: the kernel's IR, the meaning of each pointer argument (by parameter name), the
    integer (and float) arguments by name, the grid. Every argument of kind "output" is an output: a declared one
    (strict, or with reduced axes) must be covered completely, one whose axes were not named gets its meaning
    inferred, with the part of it the launch covers."""
    t0 = time.perf_counter()
    try:
        fn = KI.parse(ttir)
    except Unmodelled as e:
        return Verdict("unproven", str(e), seconds=time.perf_counter() - t0)
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    total = int(gx) * int(gy) * int(gz)
    longest = 1
    for op in KI._walk(fn.body):
        sh = KI._shape(op.rtype) or (1,)
        longest = max(longest, int(np.prod(sh)) if len(sh) <= 1 else max(sh))
    outs = [k for k, m in meanings.items() if m.kind == "output"]
    if not outs:
        return Verdict("unproven", "the launch has no tensor bound as its output", seconds=time.perf_counter() - t0)
    per = max(1, chunk_elements // max(longest, 1))
    checks = 0
    stores = {}
    inferred = {}
    try:
        for ci, start in enumerate(range(0, total, per)):
            run = _Typed(fn, meanings, ints, KI._grid_pids((gx, gy, gz), start, min(total, start + per)), True, ci,
                         (gx, gy, gz))
            run.pointers = pointers or {}
            run.run()
            checks += run.checks
            for k, v in run.stores.items():
                stores.setdefault(k, []).extend(v)
            for k, v in run.inferred.items():
                for kk, vv in v.items():
                    was = inferred.setdefault(k, {}).get(kk)
                    if was is not None and was != vv:
                        return Verdict("unproven", f"the output {k}'s {kk} is inferred as {was} in one chunk and "
                                                   f"{vv} in another", checks, total, time.perf_counter() - t0)
                    inferred[k][kk] = vv
    except _Fail as e:
        return Verdict(e.verdict, e.why, checks, total, time.perf_counter() - t0, e.example)
    except Unmodelled as e:
        return Verdict("unproven", str(e), checks, total, time.perf_counter() - t0)
    except _Dense:
        return Verdict("unproven", "a value could not be evaluated", checks, total, time.perf_counter() - t0)
    notes = []
    for name in outs:
        out = meanings[name]
        declared = out.strict or bool(out.reduced)          # a declared output must be covered completely
        if not declared and "coverage" in (inferred.get(name) or {}):
            continue                                         # what the launch covers is the data's; nothing claimed
        if len(out.shape) == 2:
            M, N = (int(x) for x in out.shape)
        elif len(out.shape) == 1:
            M, N = int(out.shape[0]), 1
        elif len(out.shape) == 0:
            M, N = 1, 1
        else:
            return Verdict("unproven", f"an output of {len(out.shape)} dimensions (coverage is decided for 1-D and "
                                       f"2-D)", checks, total, time.perf_counter() - t0)
        tiled = _tiling(stores.get(name, []), M, N)
        if tiled is not None:
            partial = tiled[0] == "violation" and ("never stored" in tiled[1] or "never stores" in tiled[1])
            if declared or not partial:
                why = tiled[1] if len(outs) == 1 else f"{name}: {tiled[1]}"
                return Verdict(tiled[0], why, checks, total, time.perf_counter() - t0, tiled[2])
            inferred.setdefault(name, {})["coverage"] = "partial: " + tiled[1]
        inf = inferred.get(name)
        if inf:
            inferred[name] = {("axis_%d" % k if isinstance(k, int) else k):
                              (v if isinstance(v, (str, list, dict)) else str(v)) for k, v in inf.items()}
            notes.append(f"{name}: {inferred[name]}")
    return Verdict("proven", f"every stored element of the output{'s' if len(outs) > 1 else ''} "
                             f"{', '.join(f'{n} {tuple(meanings[n].shape)}' for n in outs)} is stored once from values "
                             f"paired on their meanings ({checks} pairings over {total} programs)" +
                   (f"; inferred: {'; '.join(notes)}" if notes else ""),
                   checks, total, time.perf_counter() - t0, None, {k: v for k, v in inferred.items() if v} or None)
