"""The one rule over a compiled graph (ROADMAP M19 L6, step 3).

vLLM's compile backend receives the model's whole computation as an FX graph, with the real tensors (the weights
among them) as its inputs. This module walks that graph once, at compile time, and applies the same rule as
kernel_types applies inside a Triton kernel: values computed together pair only where their meanings agree, and a
value with a scale is used with its own scale, once. A kernel written in C++ cannot be read, so what it is handed is
held to the meaning it declares for its arguments (its signature: which tensor is the packed weight, which is its
scale, how the packed shape encodes the sizes, which numeric formats it reads). Nothing here is about one model or
one bug; the vocabulary is the operations the graph contains.

A tensor's meaning in the graph (GV) is its fact where a producer attached one (kernel_check.fact_of), carried
through views, elementwise operations and the kernels whose outputs the rule can name; an axis the rule cannot
name is None and pairs with anything. The verdict is per kernel site (a custom operation in the graph): proven when
the meanings it needed were there and agreed, unproven when they were not given, violation when they disagreed.
"""
import collections
import operator
import time


class Violation(Exception):
    pass


class Unknown(Exception):
    """A meaning this module cannot follow (nothing is claimed about it)."""


class GV:
    """A tensor value in the graph, typed by where its elements come from."""
    __slots__ = ("names", "shape", "dtype", "kind", "serial", "pair", "groups", "packed", "applied", "basis", "src",
                 "life")

    def __init__(self, names, shape, dtype, kind="value", serial=0, pair=0, groups=None, packed=None, applied=None,
                 basis=None, src=None, life="value"):
        self.names = tuple(names)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.kind = kind
        self.serial = serial
        self.pair = pair
        self.groups = tuple(groups) if groups is not None else (1,) * len(self.names)
        self.packed = packed
        self.applied = dict(applied or {})
        self.basis = basis
        self.src = src
        self.life = life

    def copy(self, **kw):
        out = GV(self.names, self.shape, self.dtype, self.kind, self.serial, self.pair, self.groups, self.packed,
                 self.applied, self.basis, self.src, self.life)
        for k, v in kw.items():
            setattr(out, k, v)
        out.names, out.shape = tuple(out.names), tuple(out.shape)
        if out.groups is None or len(out.groups) != len(out.names):
            out.groups = (1,) * len(out.names)
        return out

    def named(self):
        return any(n is not None for n in self.names)


def from_fact(f, shape, dtype, src=None):
    shape = tuple(shape)
    if f is None or len(f.get("names", ())) != len(shape):
        return GV([None] * len(shape), shape, dtype, src=src)
    return GV(f["names"], shape, dtype, f.get("kind", "value"), f.get("serial", 0), f.get("pair", 0),
              f.get("groups"), f.get("packed"), basis=f.get("basis"), src=src, life=f.get("life", "value"))


def _dims(t):
    """A tensor's sizes: ints where known, the symbol's text where symbolic."""
    out = []
    for s in t.shape:
        # a symbolic size is kept as its text: asking for its value (int()) would pin the symbol to one size for
        # the whole compilation, and the engine's dynamic shapes would then fail
        out.append(s if isinstance(s, int) else str(s))
    return tuple(out)


def _same_size(a, b):
    return a == b


def _norm(d, n):
    return d + n if d < 0 else d


def _op_name(node):
    """The name an operation is known by here: '.method' for a tensor method, the op's own name for a custom
    operator, module.name for a function."""
    t = node.target
    if node.op == "call_method":
        return "." + str(t)
    if node.op != "call_function":
        return node.op
    tag = getattr(t, "__entail_op__", None)
    if tag:
        return tag
    s = str(t)
    if not s.startswith("<"):
        return s                                   # torch.ops.x.y (an OpOverloadPacket / OpOverload prints its name)
    name = getattr(t, "__name__", None) or getattr(t, "__qualname__", None) or s
    mod = getattr(t, "__module__", None)
    if name == "_get_data_attr":
        return name
    if t is operator.getitem or (mod == "_operator" or mod == "operator"):
        return "operator." + name
    if mod in (None, "torch", "torch._C", "torch._VF", "torch.functional", "builtins"):
        return "torch." + name
    return f"{mod}.{name}"


_COMPARED = [0]           # comparisons of two named axes made so far (a site is proven only when it made one)


def _pair_names(a, b, how, where):
    if a is not None and b is not None:
        _COMPARED[0] += 1
        if a != b:
            raise Violation(f"{how} pairs elements that disagree on {where}: '{a}' with '{b}'")
    return a if a is not None else b


def _pair(a, b, how):
    """a and b combined elementwise (broadcast from the right): their named axes must agree; a scale meets its own
    value, once."""
    if a is None:
        return b
    if b is None:
        return a
    na, nb = len(a.names), len(b.names)
    n = max(na, nb)
    names, shape = [], []
    for i in range(n):
        ia, ib = i - (n - na), i - (n - nb)
        xa = a.names[ia] if ia >= 0 else None
        xb = b.names[ib] if ib >= 0 else None
        sa = a.shape[ia] if ia >= 0 else 1
        sb = b.shape[ib] if ib >= 0 else 1
        if isinstance(sa, int) and isinstance(sb, int) and sa == 1 and sb != 1:
            xa = None                              # a broadcast axis holds one element: it pairs with everything
        if isinstance(sa, int) and isinstance(sb, int) and sb == 1 and sa != 1:
            xb = None
        names.append(_pair_names(xa, xb, how, f"axis {i}"))
        shape.append(sa if not (isinstance(sa, int) and sa == 1) else sb)
    applied = dict(a.applied)
    for k, v in b.applied.items():
        applied[k] = applied.get(k, 0) + v
    kind, serial, pair = "value", 0, 0
    scale, value = (a, b) if a.kind == "scale" else ((b, a) if b.kind == "scale" else (None, None))
    if scale is not None:
        if scale.pair and value.serial and scale.pair != value.serial:
            raise Violation(f"{how} applies the scale of issue {scale.pair} to a value of issue {value.serial}")
        if scale.serial:
            applied[scale.serial] = applied.get(scale.serial, 0) + 1
            if applied[scale.serial] > 1:
                raise Violation(f"{how} applies the scale of issue {scale.serial} a second time")
        serial = value.serial
    elif a.kind == b.kind:
        kind = a.kind
        serial = a.serial if a.serial == b.serial else 0
        pair = a.pair if a.pair == b.pair else 0
    return GV(names, shape, a.dtype, kind, serial, pair, None, None, applied)


def _reshape(x, shape):
    """Axes kept by a reshape: those unchanged from the left and from the right; the rest are not named."""
    shape = tuple(shape)
    names = [None] * len(shape)
    i = 0
    while i < min(len(shape), len(x.names)) and _same_size(shape[i], x.shape[i]):
        names[i] = x.names[i]
        i += 1
    j = 0
    while j < min(len(shape), len(x.names)) - i and _same_size(shape[-1 - j], x.shape[-1 - j]):
        names[-1 - j] = x.names[-1 - j]
        j += 1
    return x.copy(names=names, shape=shape, groups=None)


def _slice(x, idx):
    if not isinstance(idx, tuple):
        idx = (idx,)
    if any(isinstance(k, GV) for k in idx):
        raise Unknown("an index chosen by data")
    n_int = sum(1 for k in idx if k is not None and k is not Ellipsis)
    names, shape = [], []
    pos = 0
    for k in idx:
        if k is Ellipsis:
            fill = len(x.names) - n_int
            while fill > 0:
                names.append(x.names[pos])
                shape.append(x.shape[pos])
                pos += 1
                fill -= 1
        elif k is None:
            names.append(None)
            shape.append(1)
        elif isinstance(k, slice):
            names.append(x.names[pos])
            shape.append(x.shape[pos] if k == slice(None) else None)
            pos += 1
        elif isinstance(k, int):
            pos += 1
        else:
            raise Unknown(f"an index of type {type(k).__name__}")
    names += list(x.names[pos:])
    shape += list(x.shape[pos:])
    return x.copy(names=names, shape=[s if s is not None else "?" for s in shape], groups=None)


_PASS = {".contiguous", ".clone", ".to", ".float", ".half", ".bfloat16", ".type", ".type_as", ".detach", ".cuda",
         ".cpu", ".long", ".int", ".bool", ".double", ".neg", ".sigmoid", ".tanh", ".exp", ".expand", ".expand_as",
         ".masked_fill", ".fill_",
         "torch.clone", "torch.neg", "torch.tanh", "torch.exp", "torch.sigmoid", "torch.broadcast_to",
         "operator.neg", "torch.nn.functional.silu", "torch.nn.functional.gelu", "torch.nn.functional.relu",
         "torch.nn.functional.softmax", "torch.nn.functional.dropout", "torch.pow", ".pow", "torch.rsqrt",
         "torch.sqrt", "torch.nn.functional.pad"}
_BINARY = {"operator.mul": "a multiplication", "operator.add": "an addition", "operator.sub": "a subtraction",
           "operator.truediv": "a division", "torch.mul": "a multiplication", "torch.add": "an addition",
           "torch.sub": "a subtraction", "torch.div": "a division", ".mul": "a multiplication", ".add": "an addition",
           ".sub": "a subtraction", ".div": "a division", ".mul_": "a multiplication", ".add_": "an addition",
           "torch.maximum": "a maximum", "torch.minimum": "a minimum", "torch.where": "a selection"}
_FRESH = {"torch.empty", "torch.zeros", "torch.ones", "torch.full", "torch.empty_like", "torch.zeros_like",
          "torch.ones_like", "torch.full_like", "torch.arange", "torch.tensor"}
_DTYPES_HALF = ("torch.bfloat16", "torch.float16")
# the weight types Marlin is told by its b_q_type argument (vLLM's ScalarType ids, as vLLM 0.30 makes them; read from
# vllm.scalar_type when it is there): (name, bits, floating)
_MARLIN_TYPES = {1125899907892224: ("uint4b8", 4, False), 1125899923621888: ("uint8b128", 8, False),
                 1125899906843648: ("uint4", 4, False), 1125899906909952: ("int8", 8, False),
                 1125899906844672: ("uint8", 8, False), 2814749767172868: ("float8_e4m3fn", 8, True),
                 562949953487106: ("float4_e2m1f", 4, True)}


def _marlin_type(type_id):
    """(name, bits, floating) of the weight type a Marlin call is told, or None."""
    if not isinstance(type_id, int):
        return None
    try:
        from vllm.scalar_type import ScalarType      # the engine's own reading of its ids
        t = ScalarType.from_id(type_id)
        return (str(t), int(t.size_bits), bool(t.is_floating_point()))
    except Exception:  # noqa: BLE001 - no vLLM here, or an id it does not know: the table above
        return _MARLIN_TYPES.get(type_id)


class _Walk:
    def __init__(self, fact_of):
        self.fact_of = fact_of
        self.env = {}
        self.sites = []            # {"target", "node", "verdict", "why"}
        self.unknown = collections.Counter()
        self.checks = 0

    # -- values --

    def value(self, node):
        import torch

        if node in self.env:
            return self.env[node]
        ev = node.meta.get("example_value") if hasattr(node, "meta") else None
        if isinstance(ev, torch.Tensor):
            return GV([None] * ev.dim(), _dims(ev), str(ev.dtype), src=node.name)
        return None

    def arg(self, a):
        import torch.fx

        if isinstance(a, torch.fx.Node):
            return self.value(a)
        if isinstance(a, (list, tuple)):
            return type(a)(self.arg(x) for x in a)
        if isinstance(a, slice):
            return slice(self.arg(a.start), self.arg(a.stop), self.arg(a.step))
        return a

    def site(self, node, target, verdict, why=""):
        self.sites.append({"target": target, "node": node.name, "verdict": verdict, "why": why})

    # -- the rule per operation --

    def op(self, node):  # noqa: C901
        import torch

        name = _op_name(node)
        args = [self.arg(a) for a in node.args]
        kwargs = {k: self.arg(v) for k, v in node.kwargs.items()}
        ev = node.meta.get("example_value") if hasattr(node, "meta") else None
        x = args[0] if args else None
        self.writes_constant(node, name, args, kwargs)
        if name == "_get_data_attr":
            return x
        if name in _PASS:
            return x if isinstance(x, GV) else None
        if name in (".view", ".reshape", "torch.reshape", ".flatten", ".unflatten", "torch.flatten", ".view_as"):
            if isinstance(x, GV) and isinstance(ev, torch.Tensor):
                return _reshape(x, _dims(ev))
            return None
        if name in (".transpose", "torch.transpose", ".swapaxes"):
            if not isinstance(x, GV):
                return None
            d0, d1 = _norm(int(args[1]), len(x.names)), _norm(int(args[2]), len(x.names))
            names, shape = list(x.names), list(x.shape)
            names[d0], names[d1] = names[d1], names[d0]
            shape[d0], shape[d1] = shape[d1], shape[d0]
            return x.copy(names=names, shape=shape, groups=None)
        if name in (".permute", "torch.permute"):
            if not isinstance(x, GV):
                return None
            dims = args[1] if isinstance(args[1], (list, tuple)) else args[1:]
            dims = [_norm(int(d), len(x.names)) for d in dims]
            return x.copy(names=[x.names[d] for d in dims], shape=[x.shape[d] for d in dims], groups=None)
        if name in (".t", "torch.t"):
            return x.copy(names=x.names[::-1], shape=x.shape[::-1], groups=None) if isinstance(x, GV) else None
        if name in (".unsqueeze", "torch.unsqueeze"):
            if not isinstance(x, GV):
                return None
            d = _norm(int(args[1]), len(x.names) + 1)
            return x.copy(names=x.names[:d] + (None,) + x.names[d:], shape=x.shape[:d] + (1,) + x.shape[d:],
                          groups=None)
        if name in (".squeeze", "torch.squeeze"):
            if not isinstance(x, GV):
                return None
            if len(args) > 1:
                d = _norm(int(args[1]), len(x.names))
                keep = [i for i in range(len(x.names)) if i != d]
            else:
                keep = [i for i in range(len(x.names)) if x.shape[i] != 1]
            return x.copy(names=[x.names[i] for i in keep], shape=[x.shape[i] for i in keep], groups=None)
        if name in (".index_select", "torch.index_select"):
            # elements along one axis chosen by the data: that axis means nothing the rule can name, the rest stay
            if not isinstance(x, GV) or not isinstance(ev, torch.Tensor):
                return None
            d = _norm(int(args[1]), len(x.names))
            names = list(x.names)
            names[d] = None
            return x.copy(names=names, shape=_dims(ev), groups=None)
        if name in (".chunk", "torch.chunk", ".split", "torch.split", ".unbind", "torch.unbind",
                    "torch.tensor_split"):
            if not isinstance(x, GV) or not isinstance(ev, (tuple, list)):
                return None
            out = []
            for piece in ev:
                shape = _dims(piece) if isinstance(piece, torch.Tensor) else x.shape
                names = list(x.names) if len(shape) == len(x.names) else [None] * len(shape)
                out.append(x.copy(names=names, shape=shape, groups=None))
            return tuple(out)
        if name == "operator.getitem":
            obj, idx = args[0], args[1]
            if isinstance(obj, (tuple, list)):
                return obj[idx] if isinstance(idx, int) and -len(obj) <= idx < len(obj) else None
            if isinstance(obj, GV):
                return _slice(obj, idx)
            return None
        if name in ("torch.cat", "torch.concat", "torch.stack"):
            parts = [p for p in (args[0] if args else kwargs.get("tensors", ())) if isinstance(p, GV)]
            if not parts or not isinstance(ev, torch.Tensor):
                return None
            dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
            dim = _norm(int(dim), len(parts[0].names))
            names = list(parts[0].names)
            for p in parts[1:]:
                if len(p.names) != len(names):
                    raise Unknown("a concatenation of values of different ranks")
                for i in range(len(names)):
                    names[i] = _pair_names(names[i], p.names[i], f"{name}", f"axis {i}")
                    self.checks += 1
            if name == "torch.stack":
                names.insert(dim, None)
            return GV(names, _dims(ev), str(ev.dtype))
        if name in _BINARY:
            vs = [a for a in args[:2] if isinstance(a, GV)]
            if name == "torch.where":
                vs = [a for a in args[1:3] if isinstance(a, GV)]
            if not vs:
                return None
            if len(vs) == 1:
                return vs[0].copy(groups=None)
            self.checks += 1
            r = _pair(vs[0], vs[1], _BINARY[name])
            if isinstance(ev, torch.Tensor):
                r.shape, r.dtype = _dims(ev), str(ev.dtype)
            return r
        if name in _FRESH:
            return GV([None] * ev.dim(), _dims(ev), str(ev.dtype)) if isinstance(ev, torch.Tensor) else None
        if name in ("torch.nn.functional.embedding", "torch.embedding"):
            ids, table = (args[0], args[1]) if name.endswith("functional.embedding") else (args[1], args[0])
            if not (isinstance(ids, GV) and isinstance(table, GV)):
                return None
            self.checks += 1
            if ids.basis is not None and table.names[0] is not None and ids.basis != table.names[0]:
                raise Violation(f"token ids that are {ids.basis}s index a table whose rows are {table.names[0]}s")
            if ids.basis is not None and table.names[0] is not None:
                self.site(node, name, "proven")
            else:
                self.site(node, name, "unproven", "the ids' basis and the table's rows were not both given a "
                                                  "meaning: nothing to compare")
            return GV(tuple(ids.names) + tuple(table.names[1:]), _dims(ev) if isinstance(ev, torch.Tensor) else
                      ids.shape + table.shape[1:], str(ev.dtype) if isinstance(ev, torch.Tensor) else table.dtype)
        if name == "_C.marlin_gemm":
            return self.marlin_gemm(node, name, args, kwargs, ev)
        if name.startswith("vllm_ir.rms_norm"):
            return self.rms_norm(node, name, args, ev, residual=None)
        if name.startswith("vllm_ir.fused_add_rms_norm"):
            return self.rms_norm(node, name, [args[0], args[2], args[3]], ev, residual=args[1])
        if name.startswith("vllm.unified_attention_with_output"):
            q, k, v = args[0], args[1], args[2]
            out_node = node.args[3]
            how = "attention"
            if all(isinstance(t, GV) for t in (q, k, v)):
                self.checks += 1
                before = _COMPARED[0]
                _pair_names(q.names[0], k.names[0], how, "the token axis")
                _pair_names(q.names[0], v.names[0], how, "the token axis")
                _pair_names(q.names[-1], k.names[-1], how, "the head dimension")
                compared = _COMPARED[0] > before
                self.site(node, "vllm.unified_attention_with_output", "proven" if compared else "unproven",
                          "" if compared else "no two named axes to compare")
                self.env[out_node] = q.copy(groups=None)
            return None
        if name.startswith("vllm.unified_kv_cache_update"):
            k, v = args[0], args[1]
            if isinstance(k, GV) and isinstance(v, GV):
                self.checks += 1
                _pair(k, v, "the KV cache update")
            return None
        if isinstance(ev, torch.Tensor):
            self.unknown[name] += 1
            return GV([None] * ev.dim(), _dims(ev), str(ev.dtype))
        return None

    def writes_constant(self, node, name, args, kwargs):
        """An operation that writes in place into a constant (a weight): a violation, whatever it computes."""
        targets = []
        if node.op == "call_method" and name.endswith("_") and not name.endswith("__"):
            targets.append(args[0] if args else None)
        schema = getattr(node.target, "_schema", None)
        if schema is not None:
            for i, a in enumerate(schema.arguments):
                ai = getattr(a, "alias_info", None)
                if ai is not None and getattr(ai, "is_write", False):
                    targets.append(args[i] if i < len(args) else kwargs.get(a.name))
        for t in targets:
            if isinstance(t, GV) and t.life == "const":
                raise Violation(f"{name} writes in place into a constant (a weight, {t.src}) after it was loaded")

    def rms_norm(self, node, name, args, ev, residual):
        x, w = args[0], args[1]
        if not isinstance(x, GV):
            return None
        how = "the RMS normalization"
        before = _COMPARED[0]
        if isinstance(w, GV):
            self.checks += 1
            _pair_names(x.names[-1], w.names[0], how, "the normalized axis")
            if isinstance(x.shape[-1], int) and isinstance(w.shape[0], int) and x.shape[-1] != w.shape[0]:
                raise Violation(f"{how} scales {x.shape[-1]} features with {w.shape[0]} weights")
        if isinstance(residual, GV):
            self.checks += 1
            _pair(x, residual, "the residual addition")
        compared = _COMPARED[0] > before
        self.site(node, name.split(".")[0] + "." + name.split(".")[1], "proven" if compared else "unproven",
                  "" if compared else "no two named axes to compare")
        out = x.copy(kind="value", serial=0, pair=0, groups=None)
        if isinstance(ev, (tuple, list)):
            return (out, out.copy())
        return out

    def marlin_gemm(self, node, name, args, kwargs, ev):
        import torch

        def at(i, key):
            return kwargs[key] if key in kwargs else (args[i] if i < len(args) else None)

        a, b, bias, s = at(0, "a"), at(2, "b_q_weight"), at(3, "b_bias"), at(4, "b_scales")
        a_scales, wtype = at(5, "a_scales"), _marlin_type(at(9, "b_q_type_id"))
        size_m, size_n, size_k = at(10, "size_m"), at(11, "size_n"), at(12, "size_k")
        how = "the Marlin matmul"
        missing = [nm for nm, t in (("a", a), ("b_q_weight", b), ("b_scales", s)) if not isinstance(t, GV)]
        if missing:
            self.site(node, name, "unproven", f"arguments that are not tensors: {missing}")
            return None
        if wtype is None:
            self.site(node, name, "unproven", f"the weight type it is told ({at(9, 'b_q_type_id')}) is not one "
                                              f"this check knows")
            return None
        # what the kernel reads, by the weight type it is told: the activation in a half format (or quantized, with
        # its own scales), the weight packed by Marlin (16 rows of hidden per int32 row, 32 / bits values per int32
        # word along the features), its scales in the activation's format - for 4-bit floating weights in the
        # formats of their group scales (E4M3 for NVFP4, E8M0 for MXFP4)
        self.checks += 1
        wname, bits, floating = wtype
        if a.dtype not in _DTYPES_HALF and not (isinstance(a_scales, GV) and
                                                a.dtype in ("torch.float8_e4m3fn", "torch.int8")):
            raise Violation(f"{how} reads the activation as a half-precision value but it is {a.dtype}")
        if b.dtype != "torch.int32":
            raise Violation(f"{how} reads the packed weight as int32 words but it is {b.dtype}")
        if floating and bits == 4:
            if s.dtype not in ("torch.float8_e4m3fn", "torch.float8_e8m0fnu"):
                raise Violation(f"{how} reads {wname} weights with group scales in E4M3 (NVFP4) or E8M0 (MXFP4) "
                                f"but they are {s.dtype}")
        elif a.dtype in _DTYPES_HALF and s.dtype != a.dtype:
            raise Violation(f"{how} reads the scales in the activation's format ({a.dtype}) but they are {s.dtype}")
        elif a.dtype not in _DTYPES_HALF and s.dtype not in _DTYPES_HALF:
            raise Violation(f"{how} reads the scales in a half format (the format of its output) but they are "
                            f"{s.dtype}")
        if isinstance(bias, GV) and a.dtype in _DTYPES_HALF and bias.dtype != a.dtype:
            raise Violation(f"{how} reads the bias in the activation's format ({a.dtype}) but it is {bias.dtype}")
        pk = b.shape[0] * 16 if isinstance(b.shape[0], int) else None
        pn = b.shape[1] * (32 // bits) // 16 if len(b.shape) > 1 and isinstance(b.shape[1], int) else None
        if isinstance(size_k, int) and pk is not None and size_k != pk:
            raise Violation(f"{how} is told size_k = {size_k} but the packed weight holds {pk} rows of hidden")
        if isinstance(size_n, int) and pn is not None and size_n != pn:
            raise Violation(f"{how} is told size_n = {size_n} but the packed weight holds {pn} features")
        if isinstance(size_k, int) and isinstance(a.shape[-1], int) and a.shape[-1] != size_k:
            raise Violation(f"{how} contracts {size_k} of hidden but the activation's last axis has "
                            f"{a.shape[-1]} elements")
        if isinstance(size_m, int) and isinstance(a.shape[0], int) and a.shape[0] != size_m:
            raise Violation(f"{how} is told size_m = {size_m} but the activation has {a.shape[0]} rows")
        if a.names[-1] is not None and a.names[-1] in ("token", "batch", "token_slot", "position"):
            raise Violation(f"{how} contracts the activation's '{a.names[-1]}' axis as if it were hidden features")
        for nm in a.names[:-1]:
            if nm in ("hidden", "feature"):
                raise Violation(f"{how} treats the activation's '{nm}' axis as rows of tokens")
        # the weight and its scale: issued together by one producer
        why = []
        if b.packed is None or b.packed.get("form") != "marlin_fp8":
            why.append("the packed weight's meaning was not given")
        if s.kind != "scale" or not s.pair:
            why.append("the scales' meaning was not given")
        if not why:
            if s.pair != b.serial or b.pair != s.serial:
                raise Violation(f"{how} is handed the weight of issue {b.serial} with the scale of issue "
                                f"{s.serial} (which belongs to issue {s.pair})")
            if isinstance(size_k, int) and b.packed.get("size_k") not in (None, size_k) and \
                    b.packed.get("size_k") > size_k:
                raise Violation(f"{how} is told size_k = {size_k} but the layer's weight was packed with "
                                f"{b.packed['size_k']} of hidden")
            if isinstance(size_n, int) and b.packed.get("size_n") not in (None, size_n) and \
                    b.packed.get("size_n") > size_n:
                raise Violation(f"{how} is told size_n = {size_n} but the layer's weight was packed with "
                                f"{b.packed['size_n']} features")
            g = s.groups[0] if s.groups else 1
            if isinstance(s.shape[0], int) and isinstance(size_k, int) and s.shape[0] not in (1,) and \
                    s.shape[0] * g != size_k and b.packed.get("group") not in (None, size_k):
                raise Violation(f"{how} is handed {s.shape[0]} groups of scales of {g} for {size_k} of hidden")
            if isinstance(s.shape[-1], int) and isinstance(size_n, int) and s.shape[-1] != size_n:
                raise Violation(f"{how} is handed scales for {s.shape[-1]} features but computes {size_n}")
            self.site(node, name, "proven")
        else:
            self.site(node, name, "unproven", "; ".join(why))
        names = tuple(a.names[:-1]) + ("feature",)
        shape = _dims(ev) if isinstance(ev, torch.Tensor) else a.shape[:-1] + (size_n,)
        applied = dict(a.applied)
        if s.serial:
            applied[s.serial] = applied.get(s.serial, 0) + 1
        return GV(names, shape, str(ev.dtype) if isinstance(ev, torch.Tensor) else a.dtype, "value", 0, 0, None,
                  None, applied)


def check_graph(gm, example_inputs, fact_of):
    """The rule over a graph, once: per kernel site a verdict, every disagreement listed."""
    import torch

    t0 = time.perf_counter()
    w = _Walk(fact_of)
    nodes = list(gm.graph.nodes)
    placeholders = [n for n in nodes if n.op == "placeholder"]
    inputs_with_facts = 0
    violations = []
    for i, n in enumerate(placeholders):
        t = example_inputs[i] if i < len(example_inputs) else None
        if isinstance(t, torch.Tensor):
            f = fact_of(t)
            if f is not None:
                inputs_with_facts += 1
                if f.get("unwritten") and n.users:
                    # an input holding elements nothing wrote (M19 L7): whatever reads it reads no value there
                    violations.append({"node": n.name, "op": "input", "inputs": [],
                                       "why": f"the graph reads {n.name}, but {f['unwritten']} of its elements were "
                                              f"never written while the model was loaded"})
            w.env[n] = from_fact(f, _dims(t), str(t.dtype), src=n.name)
        else:
            w.env[n] = None
    for n in nodes:
        if n.op in ("placeholder", "output", "get_attr"):
            continue
        try:
            w.env[n] = w.op(n)
        except Violation as e:
            violations.append({"node": n.name, "op": _op_name(n), "why": str(e),
                               "inputs": [a.name for a in n.args if isinstance(a, torch.fx.Node)][:6]})
            w.site(n, _op_name(n), "violation", str(e))
            w.env[n] = None
        except Unknown as e:
            w.unknown[f"{_op_name(n)} ({e})"] += 1
            w.env[n] = None
        except Exception as e:  # noqa: BLE001 - the rule failed on this node: nothing is claimed about it
            w.unknown[f"{_op_name(n)} (the check raised {type(e).__name__}: {str(e)[:80]})"] += 1
            w.env[n] = None
    sites = collections.defaultdict(collections.Counter)
    for s in w.sites:
        sites[s["target"]][s["verdict"]] += 1
    verdict = "violation" if violations else ("checked" if (w.sites or w.checks) else "unproven")
    return {"verdict": verdict, "nodes": len(nodes), "inputs": len(placeholders), "inputs_with_facts": inputs_with_facts,
            "checks": w.checks, "sites": {k: dict(v) for k, v in sites.items()},
            "unproven_sites": [s for s in w.sites if s["verdict"] == "unproven"][:12],
            "violations": violations, "unknown_ops": dict(w.unknown.most_common(20)),
            "seconds": round(time.perf_counter() - t0, 3)}
