"""frontend.graph: types, symbolic values, tracing and the program that runs afterwards (ROADMAP M8.1).

A program is a Python function whose inputs are keyword arguments. It is traced once from the types of its inputs
(`trace`): every frontend operation it calls checks the roles and facts of what it is given and records one node;
nothing runs on data. The traced program (`Program`) then runs the nodes' lowerings on plain tensors - the checks are
behind it. Only what the data alone can say is looked at when it runs: that each input has the dtype and the fixed
sizes its type declares (once, when the program is bound to its tensors, `Program.bind`).

A value written in place (a KV cache) has versions: the write returns the next version, and the one before it may not
be read after the write (TIME, rolebench 09/10/14). Nothing here stops at run time: a program that traced is a
program whose roles agree.
"""
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

from ..core import RoleError


@dataclass(frozen=True)
class T:
    """A value's type: its named dims (and their sizes where fixed), dtype, what it is, and the facts it carries.

    kind   what the value is: "query", "key", "value", "hidden", "logits", "token_ids", "positions", "last_key",
           "weight", ... A key cannot be passed where a value is taken.
    yields for a weight: the kind of what a linear map with it produces ("query" for a query projection) - the role
           is fixed where the data is made, not where it is used (rolec's cache handing out Key and Value)
    facts  vocabulary facts (entail.facts): Layout (storage format), Quantized, Reduction, Positions, Rotary,
           ModelProps ... one per class
    """
    dims: Tuple[str, ...]
    dtype: str = "bfloat16"
    kind: str = "data"
    sizes: Tuple[Optional[int], ...] = ()
    facts: Tuple[object, ...] = ()
    yields: str = ""

    def __post_init__(self):
        if len(set(self.dims)) != len(self.dims):
            raise RoleError(f"type: dims {self.dims} name one dim twice")
        if self.sizes and len(self.sizes) != len(self.dims):
            raise RoleError(f"type: {len(self.sizes)} sizes for dims {self.dims}")
        seen = set()
        for f in self.facts:
            name = type(f).__name__
            if name in seen:
                raise RoleError(f"type: two {name} facts")
            seen.add(name)

    def fact(self, cls):
        name = cls if isinstance(cls, str) else cls.__name__
        return next((f for f in self.facts if type(f).__name__ == name), None)

    def size(self, dim):
        if not self.sizes or dim not in self.dims:
            return None
        return self.sizes[self.dims.index(dim)]

    def but(self, **changes):
        """This type with some parts changed; facts=(...) replaces the facts of those classes only."""
        facts = changes.pop("facts", None)
        out = replace(self, **changes)
        if facts is not None:
            names = {type(f).__name__ for f in facts}
            out = replace(out, facts=tuple(f for f in out.facts if type(f).__name__ not in names) + tuple(facts))
        return out

    def without(self, cls):
        name = cls if isinstance(cls, str) else cls.__name__
        return replace(self, facts=tuple(f for f in self.facts if type(f).__name__ != name))

    def __str__(self):
        dims = ", ".join(f"{d}={s}" if s is not None else d
                         for d, s in zip(self.dims, self.sizes or (None,) * len(self.dims)))
        facts = "".join(f" {f}" for f in self.facts)
        return f"{self.kind}[{dims}] {self.dtype}{facts}"


class Sym:
    """A value while a program is traced: its type, where it came from, and - once it was written in place - who
    wrote it (after which it may not be read)."""
    __slots__ = ("type", "id", "origin", "written_by")

    def __init__(self, t: T, sid: int, origin: str):
        self.type, self.id, self.origin, self.written_by = t, sid, origin, None

    def __repr__(self):
        return f"<{self.origin}: {self.type}>"


@dataclass
class Node:
    op: str
    inputs: Dict[str, object]            # role -> Sym, or a constant fixed at trace time
    outputs: Tuple[Sym, ...]
    run: Callable                        # run(**tensors by role) -> tuple of tensors, in the order of outputs
    note: str = ""                       # what the trace decided here (a lowering, a repair)


@dataclass
class Graph:
    inputs: Dict[str, object] = field(default_factory=dict)    # the program's inputs: Sym, or nested dict/list
    nodes: List[Node] = field(default_factory=list)
    outputs: object = None
    notes: List[str] = field(default_factory=list)            # repairs made while tracing (resolved), one line each
    options: Dict[str, object] = field(default_factory=dict)
    count: int = 0

    def sym(self, t: T, origin: str) -> Sym:
        self.count += 1
        return Sym(t, self.count, origin)


_TRACING: List[Graph] = []


def current() -> Graph:
    if not _TRACING:
        raise RoleError("frontend operations run while a program is traced (frontend.trace), and only then")
    return _TRACING[-1]


def option(name, default=None):
    return current().options.get(name, default)


def fail(op: str, text: str):
    raise RoleError(f"{op}: {text}")


def take(op: str, role: str, value, kind=None, dims=None) -> Sym:
    """The argument `role` of `op` as a traced value of the kind (or kinds) it takes, with the dims it needs."""
    current()   # an operation called outside a trace says so first
    if not isinstance(value, Sym):
        fail(op, f"{role} takes a traced value, got {type(value).__name__}")
    if value.written_by is not None:
        fail(op, f"{role} reads {value.origin} after {value.written_by} wrote it; read the value that write "
                 f"returned")
    kinds = (kind,) if isinstance(kind, str) else kind
    if kinds and value.type.kind not in kinds:
        fail(op, f"{role} takes {' or '.join(kinds)}, got {value.type.kind} ({value.origin})")
    if dims is not None:
        missing = [d for d in dims if d not in value.type.dims]
        if missing:
            fail(op, f"{role} needs dims {', '.join(missing)}; it has {', '.join(value.type.dims)} ({value.origin})")
    return value


def node(op: str, inputs: Dict[str, object], out_types, run: Callable, note: str = "", written=()) -> Tuple[Sym, ...]:
    """Record one operation: its inputs, the types of what it makes, and how it runs. `written` names the input roles
    it writes in place: those values may not be read again (their next version is among the outputs)."""
    g = current()
    outs = tuple(g.sym(t, f"{op}#{len(g.nodes)}") for t in out_types)
    for role in written:
        inputs[role].written_by = f"{op}#{len(g.nodes)}"
    g.nodes.append(Node(op, dict(inputs), outs, run, note))
    return outs


def said(text: str):
    """A repair made while tracing - the resolution the library would make at run time, made once, here."""
    current().notes.append(text)


# --- tracing and running -----------------------------------------------------------------------------------------

def _symbolize(g: Graph, value, path):
    if isinstance(value, T):
        return g.sym(value, path)
    if isinstance(value, dict):
        return {k: _symbolize(g, v, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_symbolize(g, v, f"{path}[{i}]") for i, v in enumerate(value))
    return value   # a constant: a Python number, a fact, a config value


def trace(fn: Callable, options: Optional[dict] = None, **inputs) -> "Program":
    """Trace fn(**inputs) - inputs given as types (T), nested in dicts and lists as the program takes them, or as
    constants - and return the program. Every check runs here; a role or fact that does not agree raises RoleError
    before anything runs (as a type checker fails a build). `options` choose lowerings, e.g. {"attention":
    "triton"}."""
    g = Graph(options=dict(options or {}))
    g.inputs = {k: _symbolize(g, v, k) for k, v in inputs.items()}
    _TRACING.append(g)
    try:
        g.outputs = fn(**g.inputs)
    finally:
        _TRACING.pop()
    return Program(g)


def _flat(structure, values, out, path=""):
    """Pairs (Sym, value) of a program's inputs, walking its structure and the values given for it alike."""
    if isinstance(structure, Sym):
        out.append((structure, values, path))
    elif isinstance(structure, dict):
        if not isinstance(values, dict):
            raise RoleError(f"program input {path}: expected a dict, got {type(values).__name__}")
        for k, s in structure.items():
            if k not in values:
                raise RoleError(f"program input {path}.{k} is missing")
            _flat(s, values[k], out, f"{path}.{k}")
    elif isinstance(structure, (list, tuple)):
        if not isinstance(values, (list, tuple)) or len(values) != len(structure):
            raise RoleError(f"program input {path}: expected {len(structure)} items")
        for i, (s, v) in enumerate(zip(structure, values)):
            _flat(s, v, out, f"{path}[{i}]")
    return out


def _check_tensor(sym: Sym, value, path):
    """What only the data can say, looked at once: the dtype and the fixed sizes the type declares."""
    t = sym.type
    shape = tuple(getattr(value, "shape", ()))
    dtype = str(getattr(value, "dtype", "")).replace("torch.", "")
    if len(shape) != len(t.dims):
        raise RoleError(f"program input {path}: {len(shape)} dims, the type declares {t.dims}")
    for d, want, have in zip(t.dims, t.sizes or (None,) * len(t.dims), shape):
        if want is not None and want != have:
            raise RoleError(f"program input {path}: dim {d} is {have}, the type declares {want}")
    if t.kind != "weight" and dtype and dtype != t.dtype:
        raise RoleError(f"program input {path}: dtype {dtype}, the type declares {t.dtype}")
    for f in t.facts:   # what integers number or count (v13) is said where the integers are made (units.tensor); a
        # pool's Index says how its slots are numbered, which is not about the floating values it holds
        if type(f).__name__ in ("Index", "Count") and t.dtype.startswith(("int", "uint")):
            from .units import meaning
            made = meaning(value)
            if made != f:
                raise RoleError(f"program input {path}: the type declares {f}; the tensor was made "
                                f"{'meaning ' + str(made) if made is not None else 'without a declared meaning'} "
                                f"(make it with frontend.units.tensor)")


class Program:
    """A traced program: its nodes run in order on plain tensors. bind() checks a set of tensors against the input
    types once and returns a function of nothing that runs the program on them (what a CUDA graph captures);
    calling the program checks and runs in one go."""

    def __init__(self, graph: Graph):
        self.graph = graph

    @property
    def notes(self):
        return list(self.graph.notes)

    def lines(self):
        """The program as a person reads it: one line per operation, with the types it made and what it decided."""
        out = []
        for n in self.graph.nodes:
            ins = ", ".join(f"{r}={v.origin if isinstance(v, Sym) else v!r}" for r, v in n.inputs.items())
            outs = ", ".join(str(s.type) for s in n.outputs)
            out.append(f"{n.outputs[0].origin if n.outputs else n.op}: {n.op}({ins}) -> {outs}"
                       + (f"   [{n.note}]" if n.note else ""))
        return out

    def content_inputs(self, op: str):
        """The program inputs (by name) that reach what the operation `op` stores (its `src`): what a key for the stored
        item has to cover (paged.check_identity). Left out: weights (the program's own, not per request) and inputs
        that say where or how much (an Index or a Count: slots, tables, rows, lengths, and the pools themselves, whose
        contents were stored under their own keys)."""
        g = self.graph
        owner = {}
        for name, structure in g.inputs.items():
            for s in _syms(structure):
                owner[s.id] = (name, s)
        producer = {o.id: n for n in g.nodes for o in n.outputs}
        found, seen = set(), set()
        stack = [n.inputs["src"] for n in g.nodes if n.op == op and isinstance(n.inputs.get("src"), Sym)]
        if not stack:
            raise RoleError(f"content_inputs: the program has no {op} that stores a src")
        while stack:
            s = stack.pop()
            if s.id in seen:
                continue
            seen.add(s.id)
            if s.id in owner:
                name, sym = owner[s.id]
                addressing = any(type(f).__name__ in ("Index", "Count") for f in sym.type.facts)
                if sym.type.kind != "weight" and not addressing:
                    found.add(name)
                continue
            n = producer.get(s.id)
            if n is not None:
                stack.extend(v for v in n.inputs.values() if isinstance(v, Sym))
        return found

    def _env(self, values, names):
        env = {}
        for k in names:
            s = self.graph.inputs[k]
            if not _has_sym(s):
                continue   # a constant, fixed when the program was traced
            if k not in values:
                raise RoleError(f"program input {k} is missing")
            pairs = _flat(s, values[k], [], k)
            for sym, value, path in pairs:
                if isinstance(sym, Sym):
                    _check_tensor(sym, value, path)
                    env[sym.id] = value
        return env

    def _steps(self):
        """The nodes as (run, [(role, symbol id, is a symbol)], [output ids]), made once: running is then lookups."""
        if getattr(self, "_plan", None) is None:
            self._plan = [(n.run, [(r, v.id, True) if isinstance(v, Sym) else (r, v, False) for r, v in n.inputs.items()],
                           [s.id for s in n.outputs]) for n in self.graph.nodes]
        return self._plan

    def _execute(self, env):
        for run, ins, outs in self._steps():
            got = run(**{r: (env[v] if sym else v) for r, v, sym in ins})
            for i, t in zip(outs, got):
                env[i] = t
        return _gather(self.graph.outputs, env)

    def bind(self, **values) -> Callable:
        env = self._env(values, list(self.graph.inputs))

        def run():
            return self._execute(dict(env))

        return run

    def prepare(self, **fixed) -> Callable:
        """Binds the inputs that stay (weights, pools) once and returns run(**rest): the inputs that change from step
        to step are checked at every call (what only their data can say), the fixed ones were checked here."""
        unknown = set(fixed) - set(self.graph.inputs)
        if unknown:
            raise RoleError(f"prepare: the program has no inputs {sorted(unknown)}")
        env_fixed = self._env(fixed, list(fixed))
        rest = [k for k in self.graph.inputs if k not in fixed]

        def run(**values):
            env = dict(env_fixed)
            env.update(self._env(values, rest))
            return self._execute(env)

        return run

    def __call__(self, **values):
        return self.bind(**values)()


def _syms(structure):
    if isinstance(structure, Sym):
        return [structure]
    if isinstance(structure, dict):
        return [s for v in structure.values() for s in _syms(v)]
    if isinstance(structure, (list, tuple)):
        return [s for v in structure for s in _syms(v)]
    return []


def _has_sym(structure):
    if isinstance(structure, Sym):
        return True
    if isinstance(structure, dict):
        return any(_has_sym(v) for v in structure.values())
    if isinstance(structure, (list, tuple)):
        return any(_has_sym(v) for v in structure)
    return False


def _gather(structure, env):
    if isinstance(structure, Sym):
        return env[structure.id]
    if isinstance(structure, dict):
        return {k: _gather(v, env) for k, v in structure.items()}
    if isinstance(structure, (list, tuple)):
        return type(structure)(_gather(v, env) for v in structure)
    return structure
