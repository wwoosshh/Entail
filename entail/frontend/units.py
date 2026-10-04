"""frontend.units: integers that say what they number or count, for host-side code (ROADMAP M21.1).

A scheduler works in Python integers: a slot, a block, a row of a batch, the tokens a sequence is known to have, the
tokens computed, the slots reserved. All of them are ints, and nothing stops one from standing where another is
meant. Here an integer carries its meaning (facts.Index, facts.Count or facts.Positions) from where it is made;
arithmetic keeps the meaning or refuses; a function states what it takes (@takes) and what it returns (@returns); and
a tensor made from such integers (tensor()) carries the meaning into a traced program, whose bind compares it with
the program's declared input. A plain int where a meaning is required is refused: the meaning is declared once,
where the value is made.

One rule per meaning, none per bug:
  count     a count plus or minus a count of the same length keeps that length; counts of two lengths (known and
            computed) are not added or subtracted - a sequence moves between lengths only where the code says so
            (num(..., Count(...)) at the place that makes the new length). A plain integer added to a count keeps
            the count. Counts compare with counts of any length (how many are left) and with positions.
  index     an index plus or minus a plain integer stays in its numbering (the next slot); two indices are not added;
            a block times its block size is the block's first slot, a slot floor-divided by the block size is its
            block (any other factor is refused); indices of two units or two pools are not compared or subtracted.
  position  a position plus or minus a plain integer is a position; two positions are not added; positions compare
            with positions and counts.
Anything else (a product of counts, an index times an arbitrary number) gives a plain int: it no longer says what it
is, so it is refused wherever a meaning is required.
"""
import functools
import inspect
import weakref

from ..core import RoleError
from ..facts import Count, Index, Positions

MEANINGS = (Index, Count, Positions)


def _what(x):
    return getattr(x, "fact", None) if isinstance(x, Num) else None


def _say(f):
    return str(f) if f is not None else "a plain integer"


class Num(int):
    """An int with a meaning (an Index, a Count or a Positions fact). It is an int everywhere an int is read (list
    indices, range, tensor construction); arithmetic and comparison follow the rules in the module docstring."""

    def __new__(cls, value, fact):
        if not isinstance(fact, MEANINGS):
            raise RoleError(f"num: a meaning is an Index, a Count or a Positions fact, got {type(fact).__name__}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise RoleError(f"num: expected an integer, got {value!r}")
        out = super().__new__(cls, int(value))
        out.fact = fact
        return out

    def __repr__(self):
        return f"{int(self)} ({self.fact})"

    __str__ = __repr__

    def __hash__(self):
        return int.__hash__(self)

    # --- + and - -------------------------------------------------------------------------------------------
    def _plus(self, other, sign, op):
        a, b = self.fact, _what(other)
        if b is None:
            if not isinstance(other, int) or isinstance(other, bool):
                return NotImplemented
            return Num(int(self) + sign * int(other), a)
        if isinstance(a, Count) and isinstance(b, Count):
            if a.of != b.of:
                raise RoleError(f"{op}: {a} and {b} are two lengths of a sequence; say where one becomes the other "
                                f"instead of adding them")
            return Num(int(self) + sign * int(other), a)
        if sign < 0 and type(a) is type(b) and not isinstance(a, Count):
            if a != b:
                raise RoleError(f"{op}: subtracts {b} from {a}")
            return int(self) - int(other)            # a distance, in units of the numbering
        raise RoleError(f"{op}: {'adds' if sign > 0 else 'subtracts'} {b} {'to' if sign > 0 else 'from'} {a}")

    def __add__(self, other):
        return self._plus(other, 1, "add")

    def __radd__(self, other):
        return self._plus(other, 1, "add")

    def __sub__(self, other):
        return self._plus(other, -1, "subtract")

    def __rsub__(self, other):
        if _what(other) is None and isinstance(other, int):
            raise RoleError(f"subtract: {int(other)} minus {self.fact}: a plain integer minus a meaning has none")
        return NotImplemented

    # --- * // % --------------------------------------------------------------------------------------------
    def __mul__(self, other):
        a = self.fact
        if _what(other) is not None:
            raise RoleError(f"multiply: {a} times {other.fact}")
        if isinstance(a, Index) and a.unit == "block":
            if int(other) != a.block:
                raise RoleError(f"multiply: a block of {a.block} slots times {int(other)}: only its block size "
                                f"makes it its first slot")
            return Num(int(self) * int(other), Index("slot", a.pool, a.block))
        return int(self) * int(other)

    __rmul__ = __mul__

    def __floordiv__(self, other):
        a = self.fact
        if _what(other) is not None:
            raise RoleError(f"divide: {a} by {other.fact}")
        if isinstance(a, Index):
            if a.unit != "slot" or a.block is None or int(other) != a.block:
                raise RoleError(f"divide: {a} by {int(other)}: only a slot of a pool in blocks, by its block size, "
                                f"gives its block")
            return Num(int(self) // int(other), Index("block", a.pool, a.block))
        return int(self) // int(other)

    def __mod__(self, other):
        if _what(other) is not None:
            raise RoleError(f"remainder: {self.fact} by {other.fact}")
        return int(self) % int(other)

    # --- comparisons ---------------------------------------------------------------------------------------
    def _cmp_ok(self, other, op):
        a, b = self.fact, _what(other)
        if b is None:
            return
        if isinstance(a, Index) or isinstance(b, Index):
            if a != b:
                raise RoleError(f"compare ({op}): {a} with {b}")
            return
        # counts and positions are all numbers of tokens of one sequence: they compare

    def __eq__(self, other):
        self._cmp_ok(other, "==")
        return int.__eq__(self, other)

    def __ne__(self, other):
        self._cmp_ok(other, "!=")
        return int.__ne__(self, other)

    def __lt__(self, other):
        self._cmp_ok(other, "<")
        return int.__lt__(self, other)

    def __le__(self, other):
        self._cmp_ok(other, "<=")
        return int.__le__(self, other)

    def __gt__(self, other):
        self._cmp_ok(other, ">")
        return int.__gt__(self, other)

    def __ge__(self, other):
        self._cmp_ok(other, ">=")
        return int.__ge__(self, other)


def num(value, fact):
    """value as an integer with a meaning: where the meaning is made (a block the allocator handed out, the length of a
    request's tokens, the count of tokens a step wrote)."""
    return Num(value, fact)


def nums(values, fact):
    return [Num(v, fact) for v in values]


def check(value, fact, where):
    """value must be a Num (or a list or tuple of them) meaning exactly `fact`."""
    items = value if isinstance(value, (list, tuple)) else [value]
    for v in items:
        f = _what(v)
        if f != fact:
            raise RoleError(f"{where}: takes {fact}, got {_say(f)}" + (f" ({int(v)})" if isinstance(v, int) else ""))
    return value


def takes(**required):
    """@takes(upto=Count("computed"), block=Index("block", "gpu", 16)): the named arguments must mean that."""
    def wrap(fn):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def run(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            for name, fact in required.items():
                if name in bound.arguments:
                    check(bound.arguments[name], fact, f"{fn.__qualname__}({name}=)")
            return fn(*args, **kwargs)
        run.takes = dict(required)
        return run
    return wrap


def returns(fact):
    """@returns(Count("known")): the result (an int, or a list of ints) means that - declared where it is made."""
    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            out = fn(*args, **kwargs)
            if isinstance(out, (list, tuple)):
                return type(out)(Num(int(v), fact) for v in out)
            return Num(int(out), fact)
        run.returns = fact
        return run
    return wrap


# --- tensors made from meanings ------------------------------------------------------------------------------

_MADE = {}   # id(tensor) -> (a weak reference to it, the meaning it was made with); gone when the tensor is


def _remember(t, fact):
    key = id(t)
    _MADE[key] = (weakref.ref(t, lambda _, k=key: _MADE.pop(k, None)), fact)


def tensor(values, fact=None, *, device=None, dtype=None):
    """A tensor of integers that keeps their meaning: every value must mean `fact` (or, when fact is None, all must
    mean one thing, which becomes the tensor's). A traced program's bind compares it with the input's declared type."""
    import torch

    flat = _flatten(values)
    if fact is None:
        found = {_what(v) for v in flat}
        if len(found) != 1 or None in found:
            raise RoleError(f"tensor: the values mean {sorted(_say(f) for f in found)}; one meaning is needed")
        fact = found.pop()
    for v in flat:
        if _what(v) != fact:
            raise RoleError(f"tensor: a value means {_say(_what(v))}, the tensor {fact}")
    t = torch.tensor(_plain(values), dtype=dtype or torch.int64, device=device)
    _remember(t, fact)
    return t


def meaning(t):
    """The meaning a tensor was made with (tensor()), or None."""
    entry = _MADE.get(id(t))
    if entry is None or entry[0]() is not t:
        return None
    return entry[1]


def _flatten(values):
    if isinstance(values, (list, tuple)):
        out = []
        for v in values:
            out.extend(_flatten(v))
        return out
    return [values]


def _plain(values):
    if isinstance(values, (list, tuple)):
        return [_plain(v) for v in values]
    return int(values)
