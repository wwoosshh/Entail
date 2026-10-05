"""declarations: what an engine already declares about the values it makes, read where it makes them and turned into
meanings (ROADMAP M19 L6 step 1, M22.4; ENTAIL=types). kernel_check holds the facts, kernel_types the rule.

The engine's own declarations are the source: its weight parameter classes and their attributes (output_dim,
input_dim, ...), the scales a layer keeps beside a weight, what its functions take and return. Each engine has one
table (data/<engine>_declarations.json) and one adapter that says where to read; the reading is this module and is
the same for every engine. Nothing here is written for one model, one quantization method or one bug.

  Declarations(table)            the reader of one engine's table
    .declare()                   the table's relations and merges, given to the rule
    .around(orig)                a weight-processing function wrapped: every parameter's declaration is read before
                                 it runs; after it, the tensors the layers now hold get their meanings - matched by
                                 name, and by shape to what was declared (the same shape, or a transpose)
    .wrap_function(holder, name, spec)
                                 a function wrapped: its tensor arguments and what it returns get meanings
    .rotations(model)            every rotary layer, while it runs, says what the query and key it is handed mean:
                                 tokens, heads of head_size features, and in each head the rotation pairs its own
                                 is_neox_style declares (kernel_check.scope_push)
A tensor that already has a meaning (attached by a more specific producer) keeps it.

Table (keys): layers (a layer kind - a class name in a module's MRO - -> the names of a weight's output and input
axes; a MoE layer's [expert, out, in] layout), scales (a value parameter -> its scale parameters), functions
(module:function or module:Class.method -> args: name -> [axes, basis] or "moe_weight"; pairs: [value, scale]
arguments; returns: per returned tensor [axes, basis], {"like": arg}, {"scale_of": index}; issue: a returned value
and its scale made together - {"value": i, "scale": j, "names": [...], "scale_names": [...], "scale_groups": [n or
an argument's name, ...]}), relations and merges (for the rule), mutable (parameters the engine updates in place
after loading; every other parameter is a constant), rotations (the rotary layer kinds - class names in a module's
MRO - and the attributes that say its head size, rotary dimension and pairing, and where its forward takes the query
and key: [[name, position], ...]).
"""
import functools
import inspect

from . import kernel_check, kernel_types


class Declarations:
    def __init__(self, table, cuda_only=True):
        self.table = table or {}
        self.cuda_only = cuda_only          # meanings go on the GPU's tensors (tests read CPU ones)
        self.stats = {}

    def choice(self, kind):
        return self.table.get(kind)

    def _count(self, k):
        self.stats[k] = self.stats.get(k, 0) + 1

    def declare(self):
        for a, op, b, result in self.choice("relations") or []:
            kernel_types.relate(a, op, b, result)
        for outer, inner, result in self.choice("merges") or []:
            kernel_types.merge(outer, inner, result)

    # --- a value and its scale --------------------------------------------------------------------------------------

    def _ok(self, t):
        return t is not None and hasattr(t, "data_ptr") and (t.is_cuda or not self.cuda_only)

    @staticmethod
    def _has(t):
        try:
            return kernel_check.fact_of(t) is not None
        except Exception:  # noqa: BLE001
            return True

    def attach(self, t, names, basis=None, **kw):
        try:
            if self._ok(t) and t.dim() == len(names) and not self._has(t):
                kernel_check.attach(t, names, kind="index" if basis else kw.pop("kind", "value"), basis=basis, **kw)
                self._count("attached")
                return True
        except Exception:  # noqa: BLE001 - never the engine's problem
            self._count("attach_failed")
        return False

    def issue(self, value, value_names, scale, scale_names=None, scale_groups=None):
        """A value and its scale, as one producer made them: their meanings and their pairing. The scale's axes, when
        not given, follow the value's by broadcast."""
        if value is None or scale is None or not hasattr(scale, "shape"):
            return False
        if not (self._ok(value) and self._ok(scale)):
            return False
        if self._has(value) or self._has(scale):
            return False
        if value.dim() != len(value_names):
            return False
        if scale_names is None or scale_groups is None:
            axes = scale_axes(tuple(value.shape), list(value_names), tuple(scale.shape))
            if axes is None:
                self._count("scale_not_aligned")
                return False
            scale_names, scale_groups = axes
        if scale.dim() != len(scale_names) or len(scale_groups) != len(scale_names):
            self._count("scale_not_aligned")
            return False
        try:
            kernel_check.issue_pair(value, scale, list(value_names), list(scale_names), list(scale_groups))
            self._count("issued")
            return True
        except Exception:  # noqa: BLE001
            self._count("attach_failed")
            return False

    # --- the weights, after loading ---------------------------------------------------------------------------------

    def _kind(self, module):
        layers = self.choice("layers") or {}
        for cls in type(module).__mro__:
            if cls.__name__ in layers:
                return cls.__name__, layers[cls.__name__]
        return None, None

    @staticmethod
    def _declaration(p):
        d = {"shape": tuple(int(n) for n in p.shape), "dtype": str(p.dtype), "cls": type(p).__name__}
        for attr in ("output_dim", "input_dim", "packed_dim", "packed_factor", "pack_factor", "quant_method",
                     "is_transposed"):
            try:
                v = getattr(p, attr, None)
            except Exception:  # noqa: BLE001 - a property that raises
                v = None
            if v is not None:
                d[attr] = v.value if hasattr(v, "value") else v
        return d

    @staticmethod
    def _names(kind_row, d, shape):
        """The axis names a declaration gives a tensor of `shape` (None: it does not say)."""
        if kind_row is None:
            return None
        if "moe" in kind_row:
            layout = kind_row["moe_transposed"] if d.get("is_transposed") else kind_row["moe"]
            return list(layout) if len(shape) == 3 else None
        out, inn = d.get("output_dim"), d.get("input_dim")
        if out is None and inn is None:
            return None
        names = [None] * len(shape)
        if out is not None and 0 <= int(out) < len(shape):
            names[int(out)] = kind_row["out"]
        if inn is not None and 0 <= int(inn) < len(shape):
            names[int(inn)] = kind_row["in"]
        return names

    def snapshot(self, model):
        """Every parameter's declaration, by module and name, before the weights are processed."""
        snap = {}
        for mname, module in model.named_modules():
            kname, row = self._kind(module)
            params = dict(module.named_parameters(recurse=False))
            if not params:
                continue
            decl = {}
            for pname, p in params.items():
                d = self._declaration(p)
                d["names"] = self._names(row, d, d["shape"])
                decl[pname] = d
            snap[mname] = (kname, row, decl)
        return snap

    def apply(self, model, snap):
        scales_of = self.choice("scales") or {}
        modules = dict(model.named_modules())
        for mname, (_kname, _row, decl) in snap.items():
            module = modules.get(mname)
            if module is None:
                continue
            now = {}
            for pname in decl:
                t = getattr(module, pname, None)
                if t is not None and hasattr(t, "shape"):
                    now[pname] = t
            done = set()
            for vname, snames in scales_of.items():
                if vname not in now or vname not in decl or decl[vname]["names"] is None:
                    continue
                names = matched((decl[vname]["names"], decl[vname]["shape"]), tuple(now[vname].shape))
                for sname in snames:
                    if sname in now and names is not None:
                        if self.issue(now[vname], names, now[sname]):
                            done.update((vname, sname))
                        break
            for pname, t in now.items():
                if pname in done or decl[pname]["names"] is None:
                    continue
                names = matched((decl[pname]["names"], decl[pname]["shape"]), tuple(t.shape))
                if names is not None and self.attach(t, names):
                    self._count("weight_named")
        self.constants(model)
        self.rotations(model)

    def constants(self, model):
        """After loading, every parameter is a constant (nothing writes it again), except the ones the engine updates
        in place (the table's "mutable")."""
        mutable = set(self.choice("mutable") or [])
        for _mname, module in model.named_modules():
            for pname, p in module.named_parameters(recurse=False):
                if pname in mutable or not self._ok(p):
                    continue
                try:
                    kernel_check.set_life(p, "const")
                    self._count("constant")
                except Exception:  # noqa: BLE001
                    self._count("attach_failed")

    def rotations(self, model):
        """Every rotary layer of the model (a class the table's "rotations" names, in the module's MRO): while it runs,
        the query and key it is handed mean tokens, heads of head_size features, and in each head the rotation pairs
        its own is_neox_style declares (split: j with j + rotary_dim/2; interleaved: 2i with 2i+1). A kernel launched
        inside the layer sees that meaning on the tensors of their shape and dtype (the layer's copies and outputs)."""
        spec = self.choice("rotations")
        if not spec:
            return 0
        classes = set(spec.get("classes") or [])
        n = 0
        for _mname, module in model.named_modules():
            if getattr(module, "_entail_rotation", False) or \
                    not any(c.__name__ in classes for c in type(module).__mro__):
                continue
            try:
                module.register_forward_pre_hook(functools.partial(_rotation_enter, spec), with_kwargs=True)
                module.register_forward_hook(_rotation_leave, with_kwargs=True, always_call=True)
                module._entail_rotation = True
                n += 1
            except Exception:  # noqa: BLE001 - never the engine's problem
                self._count("rotation_hook_failed")
        if n:
            self.stats["rotation_layers"] = self.stats.get("rotation_layers", 0) + n
        return n

    def around(self, orig, model_arg=0):
        """`orig` (a function that processes a model's weights after loading, the model its argument `model_arg`)
        with the declarations read before it and the meanings attached after it."""
        @functools.wraps(orig)
        def run(*a, **k):
            model = a[model_arg] if len(a) > model_arg else k.get("model")
            snap = None
            if model is not None:
                try:
                    snap = self.snapshot(model)
                except Exception:  # noqa: BLE001
                    self._count("snapshot_failed")
            out = orig(*a, **k)
            if snap is not None:
                try:
                    self.apply(model, snap)
                except Exception:  # noqa: BLE001
                    self._count("apply_failed")
            return out
        return run

    # --- the functions: their arguments and what they return -------------------------------------------------------

    def wrap_function(self, orig, spec):
        """`orig` with its tensor arguments and returns given the meanings `spec` declares; None when its parameters
        cannot be read."""
        try:
            params = list(inspect.signature(orig).parameters)
        except (TypeError, ValueError):
            return None
        moe = (self.choice("layers") or {}).get("FusedMoE", {}).get("moe")

        @functools.wraps(orig)
        def run(*a, **k):
            bound = {}
            try:
                bound = dict(zip(params, a))
                bound.update(k)
                for left, right in spec.get("pairs", []):
                    value, scale = bound.get(left), bound.get(right)
                    if value is not None and moe and len(getattr(value, "shape", ())) == 3:
                        self.issue(value, moe, scale)
                for arg, meaning in spec.get("args", {}).items():
                    t = bound.get(arg)
                    if t is None or not hasattr(t, "dim"):
                        continue
                    if meaning == "moe_weight":
                        if moe and t.dim() == 3:
                            self.attach(t, list(moe))
                        continue
                    names, basis = meaning
                    self.attach(t, list(names), basis)
            except Exception:  # noqa: BLE001
                self._count("attach_failed")
            out = orig(*a, **k)
            try:
                self._returns(spec, bound, out)
            except Exception:  # noqa: BLE001
                self._count("attach_failed")
            return out
        return run

    def _returns(self, spec, bound, out):
        outs = out if isinstance(out, tuple) else (out,)
        iss = spec.get("issue")
        if iss:
            vi, si = iss.get("value", 0), iss.get("scale", 1)
            if vi < len(outs) and si < len(outs):
                groups = [bound.get(g) if isinstance(g, str) else g for g in iss.get("scale_groups", [])]
                if all(isinstance(g, int) and g > 0 for g in groups):
                    self.issue(outs[vi], iss["names"], outs[si], iss.get("scale_names", iss["names"]), groups)
                else:
                    self._count("issue_groups_unknown")
        for i, meaning in enumerate(spec.get("returns") or []):
            if i >= len(outs) or outs[i] is None or not hasattr(outs[i], "dim"):
                continue
            t = outs[i]
            if isinstance(meaning, dict) and "like" in meaning:
                src = bound.get(meaning["like"])
                f = kernel_check.fact_of(src) if src is not None and hasattr(src, "dim") else None
                scale_i = next((j for j, mm in enumerate(spec["returns"]) if isinstance(mm, dict) and
                                mm.get("scale_of") == i), None)
                scale = outs[scale_i] if scale_i is not None and scale_i < len(outs) else None
                if f is not None and t is not src and t.dim() == len(f["names"]):
                    if scale is not None and hasattr(scale, "dim"):
                        self.issue(t, f["names"], scale)
                    else:
                        self.attach(t, list(f["names"]))
            elif isinstance(meaning, list):
                names, basis = meaning
                self.attach(t, list(names), basis)


def rotation_view(shape, stride, head_size, rotary_dim, neox):
    """(names, view shape, view strides) of a query or key of `shape` - (tokens, features) or (tokens, heads,
    head_size) - whose heads have head_size features, the first rotary_dim of them rotated in pairs: split-wise (neox:
    feature j with j + rotary_dim/2, the halves an unnamed axis, the pair's index "freq") or interleaved (2i with
    2i+1). The features past rotary_dim keep the view's coordinates and are rotated by nothing. None when the shape
    does not split so."""
    P, rd = int(head_size), int(rotary_dim)
    if P <= 0 or rd <= 0 or rd > P or rd % 2 or P % 2:
        return None
    if len(shape) == 2:
        T, F = (int(x) for x in shape)
        if F % P:
            return None
        H, st, sh, sf = F // P, int(stride[0]), P * int(stride[1]), int(stride[1])
    elif len(shape) == 3:
        T, H, P3 = (int(x) for x in shape)
        if P3 != P:
            return None
        st, sh, sf = (int(x) for x in stride)
    else:
        return None
    if neox:
        half = rd // 2
        if P % half:
            return None
        return ["token", "head", None, "freq"], (T, H, P // half, half), (st, sh, half * sf, sf)
    return ["token", "head", "freq", None], (T, H, P // 2, 2), (st, sh, 2 * sf, sf)


def _rotation_fact(values, head_size, rotary_dim, neox, shape, stride, dtype):
    """The meaning, inside a rotary layer, of a tensor handed to a kernel: a query or key (a declared value's dtype,
    its tokens and its features)."""
    if len(shape) not in (2, 3):
        return None
    tokens, features = int(shape[0]), 1
    for n in shape[1:]:
        features *= int(n)
    for vshape, vdtype in values:
        vf = 1
        for n in vshape[1:]:
            vf *= int(n)
        if vdtype == dtype and len(vshape) >= 2 and int(vshape[0]) == tokens and vf == features:
            view = rotation_view(shape, stride, head_size, rotary_dim, neox)
            return None if view is None else kernel_check.scoped_fact(*view)
    return None


def _rotation_enter(spec, module, args, kwargs):
    """A rotary layer starts running: what its query and key mean, from its own attributes."""
    if kernel_check._compiling():
        return None
    fact_for = None
    try:
        P = int(getattr(module, spec.get("head_size", "head_size")))
        rd = int(getattr(module, spec.get("rotary_dim", "rotary_dim")))
        neox = bool(getattr(module, spec.get("neox", "is_neox_style")))
        values = []
        for name, i in spec.get("values") or []:
            t = kwargs.get(name) if name in kwargs else (args[i] if i < len(args) else None)
            if t is not None and hasattr(t, "shape") and hasattr(t, "dtype"):
                values.append((tuple(int(x) for x in t.shape), t.dtype))
        if values:
            fact_for = functools.partial(_rotation_fact, values, P, rd, neox)
    except Exception:  # noqa: BLE001 - an attribute it does not have: it says nothing
        fact_for = None
    kernel_check.scope_push(fact_for)
    return None


def _rotation_leave(module, args, kwargs, out):
    if not kernel_check._compiling():
        kernel_check.scope_pop()
    return None


def scale_axes(value_shape, value_names, scale_shape):
    """A scale's axes by broadcast against its value's, aligned from the first axis: the same size is the same axis,
    a size that divides is that axis in groups, size 1 is no coordinate. None when they do not align."""
    if len(scale_shape) > len(value_shape):
        if all(int(n) == 1 for n in scale_shape[len(value_shape):]):
            scale_shape = scale_shape[:len(value_shape)]
        else:
            return None
    names, groups = [], []
    for i, n in enumerate(scale_shape):
        n, v = int(n), int(value_shape[i])
        if n == v:
            names.append(value_names[i])
            groups.append(1)
        elif n == 1:
            names.append(None)
            groups.append(1)
        elif n > 0 and v % n == 0:
            names.append(value_names[i])
            groups.append(v // n)
        elif n > 0 and (n - 1) * -(-v // n) < v:
            names.append(value_names[i])          # groups of ceil(v / n), the last one short
            groups.append(-(-v // n))
        else:
            return None
    return names + [None] * (len(scale_shape) - len(names)), groups + [1] * (len(scale_shape) - len(groups))


def matched(before, after_shape):
    """The names of a processed tensor from its declaration: the same shape keeps them, a transpose swaps them."""
    names, shape = before
    if tuple(after_shape) == tuple(shape):
        return list(names)
    if len(shape) == 2 and tuple(after_shape) == (shape[1], shape[0]) and shape[0] != shape[1]:
        return [names[1], names[0]]
    if len(shape) == 3 and tuple(after_shape) == (shape[0], shape[2], shape[1]) and shape[1] != shape[2]:
        return [names[0], names[2], names[1]]
    return None
