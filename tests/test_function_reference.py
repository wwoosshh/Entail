"""Tests for engine functions held against entail's own definitions (ROADMAP M19 L3; entail/definitions.py,
adapters/function_reference.py): the wrapper on stand-in modules (no vLLM or SGLang), and the definitions against
plain formulas. CPU torch. Run: python tests/test_function_reference.py"""
import inspect
import io
import os
import sys
import types
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from entail import core, definitions, kernel_reference_contract as krc, load  # noqa: E402
from entail.adapters import function_reference as fr  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402


def gate_definition(a, b, scale=1.0, _dtype=None):
    dt = _dtype or a.dtype
    return (F.silu(a.to(dt)) * b.to(dt) * scale).to(a.dtype)


DEF = definitions.Definition("fake_engine.ops:gate", "fake", ("a", "b"), gate_definition, None, "silu(a) * b * scale")


def stand_in(kind):
    """A module holding `gate` as a kernel computes it."""
    mod = types.ModuleType("fake_engine.ops")

    def gate(a, b, scale=1.0):
        if kind == "faithful":            # another op order, same values
            return b * (a * torch.sigmoid(a)) * scale
        if kind == "swapped":             # the operands the other way round
            return F.silu(b) * a * scale
        if kind == "packed_rows":         # reads its inputs as if their rows were packed (stride = row length)
            ra = torch.as_strided(a, a.shape, (a.shape[1], 1), a.storage_offset())
            rb = torch.as_strided(b, b.shape, (b.shape[1], 1), b.storage_offset())
            return F.silu(ra) * rb * scale
        if kind == "in_place":            # right, but writes into its input
            a.copy_(F.silu(a))
            return a * b * scale
        if kind == "uncovered":
            return F.silu(a) * b * scale
        raise AssertionError(kind)

    mod.gate = gate
    return mod


def decided(fn):
    n = len(load.LEDGER.decisions)
    was = core.mode()
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            out = fn()
    finally:
        core.set_mode(was)
    return out, load.LEDGER.decisions[n:]


def setup(kind, d=DEF):
    fr.reset()
    torch.manual_seed(0)
    mod = stand_in(kind)
    orig = mod.gate
    assert fr.wrap(mod, "gate", d) and not fr.wrap(mod, "gate", d)
    return mod, orig


def test_a_faithful_function_passes_once_and_the_engine_gets_its_output():
    mod, orig = setup("faithful")
    a, b = torch.randn(300, 16), torch.randn(300, 16)
    out, ds = decided(lambda: mod.gate(a, b, scale=2.0))
    assert torch.equal(out, orig(a, b, scale=2.0)), "the engine gets the function's own output"
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS and ds[0].name == "KernelReference", ds
    assert "64 of 300 rows" in ds[0].note and "entail's definition" in ds[0].note, ds[0].note
    out, ds = decided(lambda: mod.gate(a, b))
    assert not ds, "decided once per function and process"
    assert fr.stats()["decided"] == 1 and fr.stats()["instrumented"] == 1


def test_a_mismatch_is_sent_to_the_definition_from_the_call_compared():
    mod, orig = setup("swapped")
    a, b = torch.randn(100, 16), torch.randn(100, 16)
    out, ds = decided(lambda: mod.gate(a, b, 3.0))
    assert len(ds) == 1 and ds[0].verdict is Verdict.RESOLVED and ds[0].rule == RULES["resolved"], ds
    assert "sent to entail's definition" in (ds[0].resolution or ""), ds[0].resolution
    assert torch.equal(out, gate_definition(a, b, 3.0, _dtype=torch.float32)), "the compared call gets the definition"
    a2, b2 = torch.randn(7, 16).to(torch.bfloat16), torch.randn(7, 16).to(torch.bfloat16)
    out2, ds2 = decided(lambda: mod.gate(a2, b2, scale=0.5))
    assert not ds2 and out2.dtype == torch.bfloat16
    assert torch.equal(out2, gate_definition(a2, b2, 0.5, _dtype=torch.float32)), "later calls too, in float32"
    s = fr.stats()
    assert s["resolved"] == 1 and s["sent_to_definition"] == 1 and s["definition_calls"] == 2, s


def test_after_a_capture_or_under_refuse_a_mismatch_stays_broken():
    mod, orig = setup("swapped")
    a, b = torch.randn(100, 16), torch.randn(100, 16)
    was = fr._capturing
    fr._capturing = lambda: True
    try:
        out, ds = decided(lambda: mod.gate(a, b))
    finally:
        fr._capturing = was
    assert not ds and torch.equal(out, orig(a, b)), "inside a capture nothing is compared"
    out, ds = decided(lambda: mod.gate(a, b))
    assert ds[0].verdict is Verdict.BROKEN and "not repaired: a CUDA graph captured in this process holds the " \
        "kernel" in ds[0].note, ds[0].note
    assert torch.equal(out, orig(a, b)), "unrepaired: the engine gets the function's output"
    mod, orig = setup("swapped")
    pol = core.policy()
    core.set_policy("refuse")
    try:
        out, ds = decided(lambda: mod.gate(a, b))
    finally:
        core.set_policy(pol)
    assert ds[0].verdict is Verdict.BROKEN and torch.equal(out, orig(a, b)), ds


def test_a_repaired_function_inside_a_later_capture():
    """Captured after the repair: a capturable definition goes into the graph; one that synchronises with the host
    cannot, so the graph gets the kernel and that is said once, as the mismatch it is."""
    a, b = torch.randn(100, 16), torch.randn(100, 16)
    for capturable in (True, False):
        mod, orig = setup("swapped", DEF._replace(capturable=capturable))
        out, ds = decided(lambda: mod.gate(a, b))
        assert ds[0].verdict is Verdict.RESOLVED, ds
        was = fr._capturing
        fr._capturing = lambda: True
        try:
            out, ds = decided(lambda: mod.gate(a, b))
            out2, ds2 = decided(lambda: mod.gate(a, b))
        finally:
            fr._capturing = was
        if capturable:
            assert not ds and torch.equal(out, gate_definition(a, b, _dtype=torch.float32))
        else:
            assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "cannot be captured" in ds[0].note, ds
            assert torch.equal(out, orig(a, b)) and not ds2, "the graph gets the kernel; said once"
    assert not definitions.DEFINITIONS[0].capturable, "the fused MoE's definition synchronises with the host"


def test_a_function_that_misreads_strides_is_caught_on_a_strided_call():
    """The slice keeps the call's strides: a kernel that reads rows as packed misreads the slice as it misreads the
    real input (a clone would hand it packed rows, and it would pass)."""
    fused = torch.randn(200, 32)
    a, b = fused[:, :16], fused[:, 16:]                   # views of one projection: row stride 32
    assert not a.is_contiguous()
    mod, orig = setup("packed_rows")
    out, ds = decided(lambda: mod.gate(a, b))
    assert len(ds) == 1 and ds[0].verdict is Verdict.RESOLVED, ds
    assert torch.equal(out, gate_definition(a, b, _dtype=torch.float32))
    mod, orig = setup("packed_rows")
    out, ds = decided(lambda: mod.gate(a.contiguous(), b.contiguous()))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, "packed inputs: the same kernel is right"
    cols = torch.randn(200, 32)[:, ::2]                    # an inner stride of 2
    k = krc.kept(cols[:64])
    assert k.stride() == cols.stride() and torch.equal(k, cols[:64]) and k.data_ptr() != cols.data_ptr()
    e = torch.randn(1, 16).expand(64, 16)
    assert krc.kept(e).is_contiguous() and torch.equal(krc.kept(e), e), "an expanded tensor is copied contiguous"
    s = krc.sliced((a,), 200, 64)
    assert s[0].stride() == a.stride(), "the custom-op slices keep strides too"


def test_an_in_place_function_is_compared_on_its_own_copies():
    mod, orig = setup("in_place")
    a, b = torch.randn(100, 16), torch.randn(100, 16)
    keep = a.clone()
    out, ds = decided(lambda: mod.gate(a, b))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS, ds
    assert torch.equal(a, F.silu(keep)), "the real call still ran on the real input, once"


def test_a_call_the_definition_does_not_cover_is_unknown_once():
    def narrow(a, b, scale=1.0, _dtype=None):
        raise NotImplementedError("another scheme")

    d = definitions.Definition("fake_engine.ops:gate", "fake", ("a", "b"), narrow, None, "none")
    mod, orig = setup("uncovered", d)
    a, b = torch.randn(100, 16), torch.randn(100, 16)
    out, ds = decided(lambda: mod.gate(a, b))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "another scheme" in ds[0].note, ds
    assert torch.equal(out, orig(a, b))
    out, ds = decided(lambda: mod.gate(a, b))
    assert not ds, "said once"


def vllm_dummy():
    """A stand-in for the kernel reference adapter's mark of vLLM's dummy runs (profile run, warm-ups)."""
    vk = types.ModuleType("entail.adapters.vllm_kernel_reference")
    vk._STATE = {"dummy": 1}
    return vk


def test_warm_ups_are_decided_on_a_probe_and_empty_or_vacuous_real_calls_are_not():
    mod, orig = setup("swapped")
    out, ds = decided(lambda: mod.gate(torch.randn(0, 16), torch.randn(0, 16)))
    assert not ds and out.shape == (0, 16), "a call with no token decides nothing"
    was = sys.modules.get("entail.adapters.vllm_kernel_reference")
    sys.modules["entail.adapters.vllm_kernel_reference"] = vllm_dummy()
    try:
        out, ds = decided(lambda: mod.gate(torch.randn(100, 16), torch.randn(100, 16)))
    finally:
        if was is None:
            del sys.modules["entail.adapters.vllm_kernel_reference"]
        else:
            sys.modules["entail.adapters.vllm_kernel_reference"] = was
    assert len(ds) == 1 and ds[0].verdict is Verdict.RESOLVED and "warm-up call of 100 rows" in ds[0].note, ds
    mod, orig = setup("faithful")
    was_tries = fr.TRIES
    fr.TRIES = 3
    try:
        for _ in range(10):         # an unmarked dummy batch of one size class: decided once, on a probe
            out, ds = decided(lambda: mod.gate(torch.ones(100, 16), torch.ones(100, 16)))
        assert fr.stats()["warm_decided"] == 1, fr.stats()
        for _ in range(2):          # rows that differ, and an output of zeros: decides nothing, counted
            out, ds = decided(lambda: mod.gate(torch.randn(100, 16), torch.zeros(100, 16)))
            assert not ds
        out, ds = decided(lambda: mod.gate(torch.randn(100, 16), torch.zeros(100, 16)))
        assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "first 3 real calls" in ds[0].note, ds
    finally:
        fr.TRIES = was_tries
    sig = inspect.signature(stand_in("faithful").gate)
    assert fr._prepare(sig, DEF, (torch.randn(10, 4), torch.randn(9, 4)), {}) is None, "rows that disagree: not cut"


def test_an_unmarked_warm_up_is_decided_before_the_capture_and_the_graph_holds_the_definition():
    """SGLang's order, which the second L3 stage measured broken: warm-ups of one row over and over, the capture,
    then real calls. The warm-up is decided on a probe made from it (strides kept), the repair precedes the
    capture, and the capture records the definition."""
    fused = torch.ones(100, 32)
    a, b = fused[:, ::2], fused[:, 1::2]        # an inner stride of 2, one row over and over
    mod, orig = setup("packed_rows", DEF._replace(capturable=True))
    out, ds = decided(lambda: mod.gate(a, b))
    assert len(ds) == 1 and ds[0].verdict is Verdict.RESOLVED and "warm-up call of 100 rows" in ds[0].note, ds
    was = fr._capturing
    fr._capturing = lambda: True
    try:
        out, ds = decided(lambda: mod.gate(a, b))
    finally:
        fr._capturing = was
    assert not ds and torch.equal(out, gate_definition(a, b, _dtype=torch.float32)), "the capture gets the definition"


def test_warm_ups_catch_a_launch_that_is_wrong_only_at_large_sizes():
    """vllm#52576's shape: the tuning table gives another tile at larger sizes. The real call's 64-row slice takes
    the small launch; warm-ups are compared at the engine's own size, once per size class."""
    mod = types.ModuleType("fake_engine.ops")

    def gate(a, b, scale=1.0):
        out = F.silu(a) * b * scale
        return out if a.shape[0] <= 64 else out + 0.5 * out.flip(-1)

    mod.gate = gate
    fr.reset()
    assert fr.wrap(mod, "gate", DEF)
    out, ds = decided(lambda: mod.gate(torch.ones(32, 16), torch.ones(32, 16)))
    assert ds[0].verdict is Verdict.PASS and "warm-up call of 32 rows" in ds[0].note, ds
    was = fr._capturing
    fr._capturing = lambda: True
    try:
        decided(lambda: mod.gate(torch.ones(32, 16), torch.ones(32, 16)))   # class 32 captured, verified there
    finally:
        fr._capturing = was
    out, ds = decided(lambda: mod.gate(torch.ones(256, 16), torch.ones(256, 16)))
    assert ds[0].verdict is Verdict.RESOLVED and "probe of 256 rows" in ds[0].note, ds
    assert "graphs captured before hold the kernel at the sizes where it matched" in ds[0].resolution, ds[0].resolution


def test_once_the_real_call_is_decided_rows_are_not_looked_at_again():
    mod, orig = setup("faithful")
    out, ds = decided(lambda: mod.gate(torch.randn(100, 16), torch.randn(100, 16)))
    assert ds[0].verdict is Verdict.PASS
    seen = []
    was = fr._uniform_call
    fr._uniform_call = lambda *a: seen.append(1) or False
    try:
        out, ds = decided(lambda: mod.gate(torch.ones(100, 16), torch.ones(100, 16)))
    finally:
        fr._uniform_call = was
    assert not ds and not seen, "after the decision the hot path does not examine rows (no device sync)"


def test_the_moe_probe_routes_each_token_to_distinct_experts():
    x, w1, w2, tw, ids, I = _moe_inputs()
    cut = {"hidden_states": x, "w1": w1, "w2": w2, "topk_weights": tw, "topk_ids": ids}
    got = definitions.probe_experts(cut, krc.generator("cpu"))
    new_ids, new_w = got["topk_ids"], got["topk_weights"]
    assert new_ids.shape == ids.shape and new_ids.dtype == ids.dtype and new_w.dtype == tw.dtype
    assert int(new_ids.min()) >= 0 and int(new_ids.max()) < w1.shape[0]
    assert all(len(set(r.tolist())) == r.numel() for r in new_ids), "no expert twice for one token"
    assert torch.allclose(new_w.sum(-1), torch.ones(new_w.shape[0], dtype=new_w.dtype))
    assert len({tuple(r.tolist()) for r in new_ids}) > 1, "tokens are routed differently"
    s = definitions.probe_block_scales({"As": torch.zeros(8, 4)}, krc.generator("cpu"))["As"]
    assert bool((s >= 0.5).all()) and bool((s < 1.5).all())


def test_under_a_stop_policy_the_decision_raises_once_and_the_wrapper_steps_aside():
    mod, orig = setup("swapped")
    pol = core.policy()
    core.set_policy("refuse")
    os.environ["ENTAIL_ON_BROKEN"] = "stop"
    try:
        try:
            decided(lambda: mod.gate(torch.randn(100, 16), torch.randn(100, 16)))
        except core.RoleError:
            pass
        else:
            raise AssertionError("a stop policy must stop at the mismatch")
        out, ds = decided(lambda: mod.gate(torch.randn(100, 16), torch.randn(100, 16)))
        assert not ds, "the second call compares nothing"
    finally:
        del os.environ["ENTAIL_ON_BROKEN"]
        core.set_policy(pol)


def test_install_wraps_the_registered_functions_of_loaded_modules():
    fr.reset()
    made = []
    for d in definitions.DEFINITIONS:
        modname, attr = d.target.split(":")
        if modname in sys.modules:
            continue
        m = types.ModuleType(modname)
        holder, fname = m, attr
        if "." in attr:                                  # a class's method (M19 L3.3d)
            cls, fname = attr.split(".")
            holder = type(cls, (), {fname: lambda self, *a, **k: None})
            setattr(m, cls, holder)
        else:
            setattr(m, fname, lambda *a, **k: None)
        sys.modules[modname] = m
        made.append((modname, holder, fname, holder.__dict__[fname] if isinstance(holder, type) else getattr(m, fname)))
    try:
        assert fr.install() == len(made) and fr.install() == 0, "each function wrapped once"
        for modname, holder, fname, f in made:
            got = holder.__dict__[fname] if isinstance(holder, type) else getattr(holder, fname)
            assert got is not f and got.__entail_definition__ is not None
        fr.uninstall()
        for modname, holder, fname, f in made:
            got = holder.__dict__[fname] if isinstance(holder, type) else getattr(holder, fname)
            assert got is f, "uninstall puts the function back"
    finally:
        for modname, _h, _f, _g in made:
            del sys.modules[modname]
        fr.reset()
    hooks = fr.hooks()
    assert len(hooks) == len(definitions.DEFINITIONS)


def _pos_inputs():
    """Three requests in state rows 5, 0, 2 with 3, 1 and 2 query tokens; computed tokens by state row."""
    idx = torch.tensor([5, 0, 2], dtype=torch.int32)
    qsl = torch.tensor([0, 3, 4, 6, 6, 6], dtype=torch.int32)
    nct = torch.tensor([10, 0, 7, 0, 0, 20], dtype=torch.int32)
    return idx, qsl, nct


def test_the_positions_definition_against_hand_values():
    idx, qsl, nct = _pos_inputs()
    pos, seq = torch.full((8,), -9, dtype=torch.int64), torch.full((6,), -9, dtype=torch.int32)
    definitions.prepare_pos_seq_lens(idx, qsl, nct, pos, seq)
    assert pos.tolist() == [20, 21, 22, 10, 7, 8, -9, -9], pos
    assert seq.tolist() == [23, 11, 9, 0, 0, 0], seq


def _pos_module(kind):
    mod = types.ModuleType("fake_engine.input_batch")

    def prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens):
        definitions.prepare_pos_seq_lens(idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens)
        if kind == "off_by_one":                         # positions from the query offset + 1: a lost base
            n = int(query_start_loc[idx_mapping.shape[0]])
            pos[:n] += 1

    mod.prepare_pos_seq_lens = prepare_pos_seq_lens
    return mod


POS = definitions.Definition("fake_engine.input_batch:prepare_pos_seq_lens", "fake", ("idx_mapping",),
                             definitions.prepare_pos_seq_lens, None, "positions", writes=("pos", "seq_lens"),
                             whole=True, exact=True)


def test_written_integer_outputs_are_held_exactly_and_repaired():
    for kind, verdict in (("faithful", Verdict.PASS), ("off_by_one", Verdict.RESOLVED)):
        fr.reset()
        mod = _pos_module(kind)
        assert fr.wrap(mod, "prepare_pos_seq_lens", POS)
        idx, qsl, nct = _pos_inputs()
        pos, seq = torch.full((8,), -9, dtype=torch.int64), torch.zeros(6, dtype=torch.int32)
        out, ds = decided(lambda: mod.prepare_pos_seq_lens(idx, qsl, nct, pos, seq))
        assert len(ds) == 1 and ds[0].verdict is verdict, (kind, ds)
        assert pos.tolist() == [20, 21, 22, 10, 7, 8, -9, -9], "the engine's buffer holds the definition's values"
        if kind == "off_by_one":
            assert "compared element by element" in ds[0].note and "3 differ" not in ds[0].note, ds[0].note
            assert "6 differ" in ds[0].note, ds[0].note


class _Staged:
    def __init__(self, t):
        self.gpu = t


class BlockTables:
    """vLLM 0.30's BlockTables in the shape the definition reads: block tables per group (state rows x blocks),
    kernel block sizes, which groups map slots, a persistent slot-mapping buffer."""

    def __init__(self, kind="faithful", enabled=(True, False)):
        self.kind = kind
        self.num_kv_cache_groups = 2
        self.kernel_block_sizes = [4, 4]
        self._slot_mapping_enabled = list(enabled)
        self.cp_size = 1
        tables = torch.arange(6 * 8, dtype=torch.int32).reshape(6, 8) * 3 + 1
        self.block_tables = [_Staged(tables), _Staged(tables + 100)]
        self.slot_mappings = torch.full((2, 10), 777, dtype=torch.int64)

    def compute_slot_mappings(self, idx_mapping, query_start_loc, positions, num_tokens_padded, out=None):
        got = definitions.compute_slot_mappings(self, idx_mapping, query_start_loc, positions, num_tokens_padded, out)
        if self.kind == "next_block":                    # reads the block after the token's: an index-base slip
            slots = self.slot_mappings if out is None else out
            slots[0, :3] += 4
        return got


SLOTS = definitions.Definition("fake_engine.block_table:BlockTables.compute_slot_mappings", "fake", ("idx_mapping",),
                               definitions.compute_slot_mappings, None, "slots", whole=True,
                               fresh=definitions.fresh_slot_mappings, exact=True)


def test_the_slot_mapping_definition_against_hand_values():
    bt = BlockTables()
    idx, qsl, _ = _pos_inputs()
    positions = torch.tensor([20, 21, 22, 10, 7, 8, 0, 0], dtype=torch.int64)
    got = definitions.compute_slot_mappings(bt, idx, qsl, positions, 8)
    t = bt.block_tables[0].gpu
    rows = {5: [20, 21, 22], 0: [10], 2: [7, 8]}
    want = []
    for s, ps in rows.items():
        for p in ps:
            want.append(int(t[s, p // 4]) * 4 + p % 4)
    assert got[0, :6].tolist() == want, (got[0], want)
    assert got[0, 6:].tolist() == [-1, -1] and bt.slot_mappings[0, 8:].tolist() == [-1, -1], "pad after the batch"
    assert got[1].tolist() == [-1] * 8, "a group that maps no slots gets the pad id"


def test_a_method_is_wrapped_on_its_class_and_its_engine_buffer_is_left_to_the_real_call():
    for kind, verdict in (("faithful", Verdict.PASS), ("next_block", Verdict.RESOLVED)):
        fr.reset()
        cls = type("BlockTables", (BlockTables,), {})
        cls.compute_slot_mappings = BlockTables.compute_slot_mappings
        assert fr.wrap(cls, "compute_slot_mappings", SLOTS)
        bt = cls(kind)
        idx, qsl, _ = _pos_inputs()
        positions = torch.tensor([20, 21, 22, 10, 7, 8, 0, 0], dtype=torch.int64)
        seen = []
        real = fr._prepare

        def spy(*a, **k):
            seen.append(bt.slot_mappings.clone())
            return real(*a, **k)

        fr._prepare = spy
        try:
            out, ds = decided(lambda: bt.compute_slot_mappings(idx, qsl, positions, 8))
        finally:
            fr._prepare = real
        assert len(ds) == 1 and ds[0].verdict is verdict, (kind, ds)
        assert seen and bool((seen[0] == 777).all()), "before the real call the persistent buffer is untouched"
        want = definitions.compute_slot_mappings(BlockTables(), idx, qsl, positions, 8)
        assert torch.equal(out, want), "the real call's slots are the definition's (repaired) or the kernel's (right)"
        uninstalled = cls.__dict__["compute_slot_mappings"]
        assert uninstalled.__entail_definition__ is SLOTS
    fr.reset()


def test_every_definition_takes_its_functions_arguments():
    for d in definitions.DEFINITIONS:
        params = inspect.signature(d.fn).parameters
        assert "_dtype" in params and all(r in params for r in d.rows), d.target
        assert d.noise in (None, "bfloat16", "float16"), d.target


def _moe_inputs(T=16, E=4, H=32, I=16, K=2):
    torch.manual_seed(0)
    x = torch.randn(T, H, dtype=torch.float64)
    w1, w2 = torch.randn(E, 2 * I, H, dtype=torch.float64) * 0.2, torch.randn(E, H, I, dtype=torch.float64) * 0.2
    tw = torch.rand(T, K, dtype=torch.float64) + 0.5
    ids = torch.stack([torch.randperm(E)[:K] for _ in range(T)]).to(torch.int32)
    return x, w1, w2, tw, ids, I


def _moe_loop(x, w1, w2, tw, ids, I, q1=None, q2=None):
    """Per token and slot, as the model's own reference loop computes it; q1/q2 quantize the input and the
    intermediate (given all intermediates, for a dynamic per-tensor scale)."""
    T, K = ids.shape
    hs = {}
    for t in range(T):
        for j in range(K):
            e = int(ids[t, j])
            xi = q1(x)[t] if q1 else x[t]
            h = w1[e] @ xi
            hs[t, j] = F.silu(h[:I]) * h[I:]
    if q2:
        qa = q2(torch.stack(list(hs.values())))
        hs = dict(zip(hs.keys(), qa))
    out = torch.zeros_like(x)
    for (t, j), h in hs.items():
        out[t] += tw[t, j] * (w2[int(ids[t, j])] @ h)
    return out


def test_the_moe_definition_against_the_reference_loop():
    x, w1, w2, tw, ids, I = _moe_inputs()
    out = definitions.fused_experts(x, w1, w2, tw, ids)
    assert torch.allclose(out, _moe_loop(x, w1, w2, tw, ids, I), atol=1e-10)

    def q8(v, s):
        return torch.clamp(torch.round(v / s), -128, 127) * s

    ch1 = torch.linspace(0.02, 0.08, w1.shape[0] * w1.shape[1], dtype=torch.float64).view(w1.shape[0], -1, 1)
    ch2 = torch.linspace(0.02, 0.08, w2.shape[0] * w2.shape[1], dtype=torch.float64).view(w2.shape[0], -1, 1)
    iw1, iw2 = torch.round(w1 / ch1).clamp(-128, 127), torch.round(w2 / ch2).clamp(-128, 127)

    def cfg(**kw):
        base = dict(use_int8_w8a8=True, quant_dtype=torch.int8, weight_quant_dtype=torch.int8,
                    per_act_token_quant=False, block_shape=None, w1_scale=ch1, w2_scale=ch2, a1_scale=None,
                    a2_scale=None, w1_bias=None, w2_bias=None, w1_zp=None, w2_zp=None)
        base.update(kw)
        return SimpleNamespace(**base)

    dq1, dq2 = iw1 * ch1, iw2 * ch2
    s = torch.tensor(0.05, dtype=torch.float64)
    got = definitions.fused_experts(x, iw1, iw2, tw, ids, quant_config=cfg(a1_scale=s, a2_scale=s))
    want = _moe_loop(x, dq1, dq2, tw, ids, I, lambda v: q8(v, s), lambda v: q8(v, s))
    assert torch.allclose(got, want, atol=1e-9), "static per-tensor activations, per-channel weight scales"
    got = definitions.fused_experts(x, iw1, iw2, tw, ids, quant_config=cfg(per_act_token_quant=True))
    row = lambda v: q8(v, v.abs().amax(-1, keepdim=True) / 127)  # noqa: E731
    assert torch.allclose(got, _moe_loop(x, dq1, dq2, tw, ids, I, row, row), atol=1e-9), "per token"
    got = definitions.fused_experts(x, iw1, iw2, tw, ids, quant_config=cfg())
    whole = lambda v: q8(v, v.abs().amax() / 127)  # noqa: E731
    assert torch.allclose(got, _moe_loop(x, dq1, dq2, tw, ids, I, whole, whole), atol=1e-9), "dynamic per tensor"
    one = SimpleNamespace(**{**vars(cfg(a1_scale=s, a2_scale=s)), "w1_scale": ch1[:, :1, :] * 0 + 0.05,
                             "w2_scale": ch2[:, :1, :] * 0 + 0.05})
    got = definitions.fused_experts(x, iw1, iw2, tw, ids, quant_config=one)
    want = _moe_loop(x, iw1 * 0.05, iw2 * 0.05, tw, ids, I, lambda v: q8(v, s), lambda v: q8(v, s))
    assert torch.allclose(got, want, atol=1e-9), "one weight scale per expert"
    for kw in (dict(expert_map=torch.arange(4)), dict(activation="gelu"),
               dict(quant_config=SimpleNamespace(use_int8_w8a8=False, quant_dtype="fp8", weight_quant_dtype="fp8")),
               dict(quant_config=cfg(w1_bias=torch.zeros(4, 32))),
               dict(quant_config=cfg(block_shape=[128, 128])),
               dict(quant_config=cfg(), apply_router_weight_on_input=True)):
        try:
            definitions.fused_experts(x, iw1, iw2, tw, ids, **kw)
        except NotImplementedError:
            continue
        raise AssertionError(f"not covered, must say so: {kw}")


def test_the_block_fp8_and_gdn_definitions_against_their_formulas():
    torch.manual_seed(0)
    M, N, K, gn, gk = 8, 256, 512, 128, 128
    A = (torch.randn(M, K) * 0.5).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K) * 0.5).to(torch.float8_e4m3fn)
    As, Bs = torch.rand(M, K // gk) + 0.1, torch.rand(N // gn, K // gk) + 0.1
    a = A.double() * As.double().repeat_interleave(gk, 1)
    b = B.double() * Bs.double().repeat_interleave(gn, 0).repeat_interleave(gk, 1)
    got = definitions.w8a8_triton_block_scaled_mm(A, B, As, Bs, [gn, gk], torch.float64, _dtype=torch.float64)
    assert torch.allclose(got, a @ b.T, atol=1e-9)
    assert definitions.w8a8_triton_block_scaled_mm(A, B, As, Bs, [gn, gk]).dtype == torch.float16, "the default"
    A_log, dt_bias = torch.randn(16), torch.randn(16)
    ga, gb = torch.randn(64, 32)[:, ::2], torch.randn(64, 16)
    g, beta = definitions.fused_gdn_gating(A_log, ga, gb, dt_bias)
    assert g.shape == (1, 64, 16) and g.dtype == torch.float32 and beta.dtype == torch.float32
    assert torch.allclose(g[0], -torch.exp(A_log) * F.softplus(ga + dt_bias), atol=1e-6)
    assert torch.allclose(beta[0], torch.sigmoid(gb), atol=1e-7)
    g2, _ = definitions.fused_gdn_gating(A_log, ga, gb, dt_bias, beta=2.0, threshold=5.0)
    assert torch.allclose(g2[0], -torch.exp(A_log) * F.softplus(ga + dt_bias, beta=2.0, threshold=5.0), atol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
