"""Adapter v2: where vLLM 0.30's GPU worker makes the numbers its bookkeeping kernels pass around, and what they
mean (ROADMAP M19 L6; the third place meanings are attached, for ENTAIL=types; entail/kernel_check.py holds the
facts, entail/kernel_types.py the rule).

  hooks        InputBuffers.__init__ (v1/worker/gpu/input_batch.py): input_ids, positions, query_start_loc, seq_lens
               BlockTables.__init__ and init_block_table_layout_tensors (v1/worker/gpu/block_table.py): the block
                   tables, the slot mappings, the tables of pointers to them, their strides, the block sizes
               InputBatch.__init__: idx_mapping, expanded_idx_mapping, cu_num_logits, logits_indices
  read_choice  the tensors themselves and the layout the worker chose (block sizes, strides)
  handles      none: meanings are attached, nothing is changed or decided here
What a number means (its basis) and what each axis enumerates:
  position        a token's place in its sequence           request_state   a slot in the worker's request table
  token           a place in the batch's token array        batch_request   a place in this step's request list
  kv_block        a block of the KV cache (per group)       block_slot      a place in a request's block list
  block_offset    a token's place inside a KV block         kv_slot         a slot of the KV cache (block x size + offset)
  block_size      a KV block's size in tokens               stride:block_table:0   the block table's row stride
The relations declared (how the worker's kernels derive one from another): position // block_size = block_slot,
position % block_size = block_offset, kv_block * block_size = kv_slot, kv_slot + block_offset = kv_slot.
"""
import functools
import sys

from .. import kernel_check, kernel_types

engine = "vllm"
versions = "vLLM 0.30.0"
_WRAPPED = {}
_STATS = {}


def hooks():
    from .base import Hook

    return [Hook("vllm.v1.worker.gpu.input_batch.InputBuffers.__init__", "load"),
            Hook("vllm.v1.worker.gpu.input_batch.InputBatch.__init__", "request"),
            Hook("vllm.v1.worker.gpu.block_table.BlockTables.__init__", "load"),
            Hook("vllm.v1.worker.gpu.block_table.BlockTables.init_block_table_layout_tensors", "load"),
            Hook("vllm.v1.worker.gpu.states.RequestState.__init__", "load")] + \
        [Hook(f"vllm.v1.worker.gpu.input_batch.{name}", "request") for name in _functions()]


def read_choice(kind, obj):
    """The tensors a worker object holds and what each means: (attribute, axes, basis)."""
    if kind == "input_buffers":
        return [("input_ids", ["token"], "token_id"), ("positions", ["token"], "position"),
                ("query_start_loc", ["batch_request_bound"], "token"), ("seq_lens", ["batch_request"], "position"),
                ("is_padding", ["token"], None)]
    if kind == "input_batch":
        return [("idx_mapping", ["batch_request"], "request_state"),
                ("expanded_idx_mapping", ["logit"], "request_state"),
                ("cu_num_logits", ["batch_request_bound"], "logit"), ("logits_indices", ["logit"], "token"),
                ("positions", ["token"], "position"), ("input_ids", ["token"], "token_id"),
                ("query_start_loc", ["batch_request_bound"], "token"), ("seq_lens", ["batch_request"], "position")]
    return []


def handles():
    return {}


def _count(k):
    _STATS[k] = _STATS.get(k, 0) + 1


def _wrap(cls, name, make):
    orig = cls.__dict__.get(name)
    if orig is None or getattr(orig, "__entail_types__", False):
        return 0
    run = make(orig)
    run.__entail_types__ = True
    setattr(cls, name, run)
    _WRAPPED[(cls, name)] = orig
    return 1


def _attach(t, names, basis=None, **kw):
    try:
        if t is not None and hasattr(t, "data_ptr") and t.is_cuda:
            kernel_check.attach(t, names, kind="index" if basis else "value", basis=basis, **kw)
            _count("attached")
    except Exception:  # noqa: BLE001 - never the engine's problem
        _count("attach_failed")


def declare_relations():
    kernel_types.relate("position", "//", "block_size", "block_slot")
    kernel_types.relate("position", "%", "block_size", "block_offset")
    kernel_types.relate("kv_block", "*", "block_size", "kv_slot")
    kernel_types.relate("kv_slot", "+", "block_offset", "kv_slot")
    kernel_types.relate("position", "+", None, "position")
    kernel_types.relate("token", "+", None, "token")


def install_input_batch():
    mod = sys.modules.get("vllm.v1.worker.gpu.input_batch")
    if mod is None:
        return 0
    n = 0
    buffers = getattr(mod, "InputBuffers", None)
    if buffers is not None:
        def make(orig):
            @functools.wraps(orig)
            def run(self, *a, **k):
                orig(self, *a, **k)
                for attr, names, basis in read_choice("input_buffers", self):
                    _attach(getattr(self, attr, None), names, basis)
            return run
        n += _wrap(buffers, "__init__", make)
    batch = getattr(mod, "InputBatch", None)
    if batch is not None:
        def make2(orig):
            @functools.wraps(orig)
            def run(self, *a, **k):
                orig(self, *a, **k)
                for attr, names, basis in read_choice("input_batch", self):
                    _attach(getattr(self, attr, None), names, basis)
            return run
        n += _wrap(batch, "__init__", make2)
    return n


def _functions():
    """The worker's functions that launch the bookkeeping kernels, and what each tensor argument means (the data
    file data/vllm_index_meanings.json): these buffers are made fresh each step, or before the InputBatch exists, so
    their meanings are attached at the call."""
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                        "vllm_index_meanings.json")
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    return {k: {arg: (list(v[0]), v[1]) for arg, v in spec.items()} for k, spec in table.items()
            if k != "_" and k != "classes"}


def _classes():
    """The worker's objects that hold tensors (the "classes" section of the same data file): for each class, its
    attributes and what each means. A staged-write buffer (StagedWriteTensor) is its .gpu tensor; its write
    buffers then mean places along that tensor's axes."""
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data",
                        "vllm_index_meanings.json")
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    return {k: {attr: (list(v[0]), v[1]) for attr, v in spec.items()} for k, spec in table.get("classes", {}).items()}


def _attach_holder(obj, names, basis, **kw):
    """A tensor, or an object holding one (.gpu): the meaning goes on the GPU tensor; a staged-write buffer's write
    buffers are tagged with it, so what they hand to the write kernel means places along its axes."""
    t = getattr(obj, "gpu", obj)
    if t is None or not hasattr(t, "dim"):
        return
    if t.dim() == len(names):
        _attach(t, names, basis, **kw)
    if hasattr(obj, "write_indices"):
        try:
            obj.__entail_meaning__ = (list(names), basis)
        except Exception:  # noqa: BLE001
            pass
    for attr, role in (("write_indices", 0), ("write_starts", 1), ("write_cu_lens", None), ("write_contents", "v")):
        pool = getattr(obj, attr, None)
        if pool is not None and hasattr(pool, "copy_to_uva"):
            pool.__entail_meaning__ = (["write"], names[role] if isinstance(role, int) and role < len(names)
                                       else (basis if role == "v" else None))


def _wrap_function(mod, name, table):
    import inspect

    orig = getattr(mod, name, None)
    if orig is None or getattr(orig, "__entail_types__", False):
        return 0
    try:
        params = list(inspect.signature(orig).parameters)
    except (TypeError, ValueError):
        return 0

    @functools.wraps(orig)
    def run(*a, **k):
        try:
            bound = dict(zip(params, a))
            bound.update(k)
            for arg, (names, basis) in table.items():
                t = bound.get(arg)
                if t is not None and hasattr(t, "dim") and t.dim() == len(names):
                    _attach(t, names, basis)
        except Exception:  # noqa: BLE001
            _count("attach_failed")
        return orig(*a, **k)

    run.__entail_types__ = True
    setattr(mod, name, run)
    _WRAPPED[(mod, name)] = orig
    return 1


def install_functions():
    n = 0
    for key, table in _functions().items():
        module, _, name = key.rpartition(":")
        holder = sys.modules.get(module or "vllm.v1.worker.gpu.input_batch")
        if holder is not None and "." in name:          # module:Class.method
            cname, _, name = name.partition(".")
            holder = getattr(holder, cname, None)
        if holder is not None:
            n += _wrap_function(holder, name, table)
    return n


def install_classes():
    """The sampler's and the worker's state objects: their tensors get their meanings when the object is made."""
    n = 0
    for key, table in _classes().items():
        module, _, cname = key.rpartition(":")
        mod = sys.modules.get(module)
        cls = getattr(mod, cname, None) if mod is not None else None
        if cls is None:
            continue

        def make(orig, table=table):
            @functools.wraps(orig)
            def run(self, *a, **k):
                orig(self, *a, **k)
                for attr, (names, basis) in table.items():
                    try:
                        _attach_holder(getattr(self, attr, None), names, basis)
                    except Exception:  # noqa: BLE001
                        _count("attach_failed")
            return run
        n += _wrap(cls, "__init__", make)
    return n


def install_write_buffers():
    """A staged write's index buffers are made at the write (copied into a pooled UVA tensor): the tensor the
    write kernel gets carries the meaning its owner was tagged with."""
    mod = sys.modules.get("vllm.v1.worker.gpu.buffer_utils")
    cls = getattr(mod, "UvaBufferPool", None) if mod is not None else None
    if cls is None:
        return 0

    def make(orig):
        @functools.wraps(orig)
        def run(self, *a, **k):
            t = orig(self, *a, **k)
            m = getattr(self, "__entail_meaning__", None)
            if m is not None:
                try:
                    _attach(t, list(m[0]), m[1])
                except Exception:  # noqa: BLE001
                    _count("attach_failed")
            return t
        return run
    n = _wrap(cls, "copy_to_uva", make)

    # the staged values themselves, when they are copied fresh for the write (not through a pool): they are the
    # buffer's values, in write order
    swt = getattr(mod, "StagedWriteTensor", None)
    if swt is not None and hasattr(mod, "async_tensor_h2d"):
        def make_apply(orig):
            @functools.wraps(orig)
            def run(self, *a, **k):
                m = getattr(self, "__entail_meaning__", None)
                if m is None:
                    return orig(self, *a, **k)
                saved = mod.async_tensor_h2d

                def h2d(*aa, **kk):
                    t = saved(*aa, **kk)
                    _attach(t, ["content"], m[1])
                    return t
                mod.async_tensor_h2d = h2d
                try:
                    return orig(self, *a, **k)
                finally:
                    mod.async_tensor_h2d = saved
            return run
        n += _wrap(swt, "apply_write", make_apply)
    return n


def install_request_state():
    mod = sys.modules.get("vllm.v1.worker.gpu.states")
    cls = getattr(mod, "RequestState", None) if mod is not None else None
    if cls is None:
        return 0

    def make(orig):
        @functools.wraps(orig)
        def run(self, *a, **k):
            orig(self, *a, **k)
            for attr, names, basis in (("all_token_ids", ["request_state", "position"], "token_id"),
                                       ("prefill_len", ["request_state"], "position"),
                                       ("total_len", ["request_state"], "position"),
                                       ("num_computed_tokens", ["request_state"], "position")):
                try:
                    _attach_holder(getattr(self, attr, None), names, basis)
                except Exception:  # noqa: BLE001
                    _count("attach_failed")
            t = getattr(self, "last_sampled_tokens", None)
            if t is not None and hasattr(t, "dim"):
                _attach(t, ["request_state"] + [None] * (t.dim() - 1), "token_id")
        return run
    return _wrap(cls, "__init__", make)


def install_block_tables():
    mod = sys.modules.get("vllm.v1.worker.gpu.block_table")
    cls = getattr(mod, "BlockTables", None) if mod is not None else None
    if cls is None:
        return 0

    def make(orig):
        @functools.wraps(orig)
        def run(self, *a, **k):
            orig(self, *a, **k)
            try:
                for i, bt in enumerate(getattr(self, "block_tables", []) or []):
                    _attach_holder(bt, ["request_state", "block_slot"], "kv_block", label="block_table")
                for i, bt in enumerate(getattr(self, "input_block_tables", []) or []):
                    _attach(bt, ["batch_request", "block_slot"], "kv_block", label="block_table")
                _attach(getattr(self, "slot_mappings", None), ["kv_group", "token"], "kv_slot")
                nb = getattr(self, "num_blocks", None)
                _attach(getattr(nb, "gpu", None), ["kv_group", "request_state"], "block_slot")
                _attach(getattr(self, "block_table_strides", None), ["kv_group"], "stride:block_table:0")
                _attach(getattr(self, "block_sizes_tensor", None), ["kv_group"], "block_size")
                _attach(getattr(self, "kernel_block_sizes_tensor", None), ["kv_group"], "block_size")
                _attach(getattr(self, "slot_mapping_enabled", None), ["kv_group"])
                tables = [bt.gpu for bt in getattr(self, "block_tables", []) or []]
                if tables and getattr(self, "block_table_ptrs", None) is not None:
                    kernel_check.attach(self.block_table_ptrs, ["kv_group"], kind="pointers", pointers=tables)
                inputs = list(getattr(self, "input_block_tables", []) or [])
                if inputs and getattr(self, "input_block_table_ptrs", None) is not None:
                    kernel_check.attach(self.input_block_table_ptrs, ["kv_group"], kind="pointers", pointers=inputs)
                _count("block_tables_attached")
            except Exception:  # noqa: BLE001
                _count("attach_failed")
        return run
    n = _wrap(cls, "__init__", make)
    # the layout tensors are remade after the block tables are resized
    n += _wrap(cls, "init_block_table_layout_tensors", lambda orig: _relayout(orig, make))
    return n


def _relayout(orig, make):
    @functools.wraps(orig)
    def run(self, *a, **k):
        orig(self, *a, **k)
        try:
            tables = [bt.gpu for bt in getattr(self, "block_tables", []) or []]
            if tables and getattr(self, "block_table_ptrs", None) is not None:
                kernel_check.attach(self.block_table_ptrs, ["kv_group"], kind="pointers", pointers=tables)
            inputs = list(getattr(self, "input_block_tables", []) or [])
            if inputs and getattr(self, "input_block_table_ptrs", None) is not None:
                kernel_check.attach(self.input_block_table_ptrs, ["kv_group"], kind="pointers", pointers=inputs)
            _attach(getattr(self, "block_table_strides", None), ["kv_group"], "stride:block_table:0")
            _attach(getattr(self, "block_sizes_tensor", None), ["kv_group"], "block_size")
            _attach(getattr(self, "kernel_block_sizes_tensor", None), ["kv_group"], "block_size")
            _attach(getattr(self, "slot_mapping_enabled", None), ["kv_group"])
        except Exception:  # noqa: BLE001
            _count("attach_failed")
    return run


def install():
    declare_relations()
    return install_input_batch() + install_functions() + install_block_tables() + install_request_state()


def stats():
    return dict(_STATS)
