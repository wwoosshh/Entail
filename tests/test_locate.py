"""Tests for locating where meaning broke (ROADMAP M7.1, LIBRARY_DESIGN.md 12): the rules on the ledger, the operation
that made a fact untrue on the way, layers compared with a reference, and the same answer from the record files.
Run: python tests/test_locate.py"""
import contextlib
import io
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import boundaries, cli, core, diagnose, load, record, tally  # noqa: E402
from entail.contracts import RULES, Contract, Decision, Verdict  # noqa: E402
from entail.core import RoleError, boundary  # noqa: E402
from entail.facts import Certainty, Fact, Layout, Rotary, Source  # noqa: E402
from entail.propagate import RolePropagation  # noqa: E402


def fresh():
    load.LEDGER.decisions.clear()
    load.LEDGER.layers.clear()
    boundaries.PASSES.clear()
    boundaries.REPEATS.clear()
    tally.reset()


def row(boundary, verdict, name="Rotary", rule=None, consumer="engine.rotary", declared="config.json#rope_theta",
        lost_by=None):
    return {"boundary": boundary, "consumer": consumer, "name": name, "verdict": verdict,
            "rule": rule or {"pass": RULES["match"], "broken": RULES["no_resolution"],
                             "unknown": RULES["cannot_check"]}.get(verdict, RULES["match"]),
            "declared": {"source": {"kind": "file", "where": declared}} if declared else None, "lost_by": lost_by}


def test_the_first_boundary_that_broke_is_the_problem_area():
    found = record.locate([row("load:a", "pass"), row("load:b", "broken"), row("container:c", "refused")],
                          output_wrong=True)
    assert (found.broken_at, found.broken, found.all_intact) == ("load:b", ("load:b", "container:c"), False), found
    assert found.suspects == ("boundary load:b", "boundary container:c"), found.suspects
    assert found.intact == ("load:a",) and "inside a layer" not in found.suspects, "a break explains a wrong output"
    assert found.lines()[0] == "[entail] where: meaning broke at load:b (and at container:c)", found.lines()


def test_every_boundary_held_and_the_output_is_wrong_points_inside_a_layer():
    rows = [row("load:a", "pass"), row("load:b", "resolved")]
    found = record.locate(rows, passes={"container:kv": 40}, output_wrong=True)
    assert (found.broken_at, found.all_intact, found.unchecked) == (None, True, ()), found
    assert found.suspects == ("inside a layer",) and "not in the plumbing" in found.why[0], found.why
    assert found.intact == ("load:a", "load:b", "container:kv"), found.intact
    quiet = record.locate(rows)   # the output was not said to be wrong: nothing is suspect
    assert quiet.all_intact and quiet.suspects == () and quiet.lines() == [
        "[entail] where: every checked boundary kept its meaning (2 boundaries)"], quiet.lines()


def test_a_boundary_that_was_not_checked_keeps_it_and_its_neighbours_suspect():
    found = record.locate([row("load:a", "pass"), row("load:attention", "unknown", name="ModelProps",
                                                     consumer="engine.attention.custom")],
                          skipped=["container:kv"], output_wrong=True)
    assert found.all_intact and found.broken_at is None, "what was checked held"
    assert found.suspects[:2] == (
        "boundary load:attention and beside it file: config.json#rope_theta, engine.attention.custom",
        "boundary container:kv and the layers beside it"), found.suspects
    assert found.unchecked[0].startswith("load:attention (ModelProps: " + RULES["cannot_check"]), found.unchecked
    assert "inside a captured graph" in found.unchecked[1], found.unchecked
    assert found.suspects[-1] == "inside a layer", "the layers themselves stay suspect too"


def test_nothing_checked_says_so():
    found = record.locate([], output_wrong=True)
    assert (found.all_intact, found.lines()[0]) == (False, "[entail] where: no boundary was checked, so the ledger "
                                                           "cannot say where"), found


def test_the_operation_that_made_a_fact_untrue_is_named():
    """Diagnosis mode: a transpose between two boundaries; the next boundary names it (Decision.lost_by)."""
    fresh()
    core.set_mode("debug")
    try:
        @boundary("pack", returns=Layout("dense"))
        def pack(*, w):
            return w.clone()

        @boundary("kernel", w=Layout("dense"))
        def kernel(*, w):
            return w

        with RolePropagation():
            w = pack(w=torch.zeros(4, 8))
            moved = w.transpose(0, 1)
            try:
                kernel(w=moved)
            except RoleError:
                pass
        found = diagnose.locate(output_wrong=True)
    finally:
        core.set_mode("off")
    assert found.lost_by == ("Layout at boundary:kernel, made untrue by transpose",), found.lost_by
    assert found.unchecked[0].startswith("boundary:kernel (Layout: " + RULES["invalidated"]), found.unchecked
    assert found.suspects[0] == "operation transpose on the way to boundary:kernel", found.suspects
    assert found.lines()[0] == ("[entail] where: meaning was lost on the way: Layout at boundary:kernel, made "
                                "untrue by transpose"), found.lines()
    note = load.LEDGER.decisions[-1].note
    assert "made untrue by transpose (it was declared by boundary: pack.returns)" in note, note


def planted_softmax(x):
    return torch.softmax(x * 1.25, dim=-1)   # the planted fault: a wrong scale inside the layer


def exact_softmax(x):
    return torch.softmax(x.double(), dim=-1).to(x.dtype)


def test_a_layer_compared_with_a_reference_narrows_the_fault():
    """Every boundary held; the watched layer whose output its reference does not reproduce is where the fault is,
    and a layer that agrees is cleared."""
    fresh()

    class Layers:
        norm = staticmethod(lambda x: x / x.norm(dim=-1, keepdim=True))
        attend = staticmethod(planted_softmax)

    core.set_mode("debug")
    try:
        load.enforce([Decision(Contract("load:m", "engine.attention", ("Rotary",)), "Rotary", Verdict.PASS,
                               RULES["match"])], quiet_pass=True)
        with diagnose.watch(Layers, "norm", lambda x: x / torch.linalg.vector_norm(x, dim=-1, keepdim=True),
                            label="norm"), \
                diagnose.watch(Layers, "attend", exact_softmax, label="attention", calls=2), \
                contextlib.redirect_stdout(io.StringIO()) as said:
            x = torch.randn(3, 8)
            for _ in range(3):
                Layers.attend(Layers.norm(x) * 4)
        found = diagnose.locate(output_wrong=True)
    finally:
        core.set_mode("off")
    assert Layers.attend is planted_softmax, "the watched function is put back"
    assert [(c["layer"], c["agrees"]) for c in load.LEDGER.layers] == [("norm", True), ("attention", False),
                                                                      ("attention", False)], load.LEDGER.layers
    assert found.suspects == ("inside attention",) and found.all_intact, found
    assert "attention differs from exact_softmax on the same inputs" in found.why[0], found.why
    assert "compared: attention differs" in said.getvalue(), "a layer that differs is said"


def test_every_layer_is_compared_once():
    """A method watched on its class runs for every layer: each is compared on its first call (calls per module),
    and the comparison names the layer that differs by its place."""
    fresh()

    class Norm(torch.nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return x * self.scale

    layers = [Norm(1.0), Norm(1.0), Norm(1.5), Norm(1.0)]   # the third one computes wrong
    core.set_mode("debug")
    try:
        with diagnose.watch(Norm, "forward", lambda self, x: x * 1.0, label="norm"):
            x = torch.ones(2, 4)
            for _ in range(3):
                for layer in layers:
                    layer(x)
        found = diagnose.locate(output_wrong=True)
    finally:
        core.set_mode("off")
    assert [(c["instance"], c["agrees"]) for c in load.LEDGER.layers] == [(0, True), (1, True), (2, False),
                                                                          (3, True)], load.LEDGER.layers
    assert found.suspects == ("inside norm",) and "norm (instance 2) differs" in found.why[0], found.why


def test_watching_does_nothing_outside_debug_mode():
    fresh()

    class Layer:
        run = staticmethod(planted_softmax)

    core.set_mode("load")
    try:
        with diagnose.watch(Layer, "run", exact_softmax):
            Layer.run(torch.randn(2, 4))
    finally:
        core.set_mode("off")
    assert load.LEDGER.layers == [], "compared only at the diagnosis site"


def test_the_record_files_give_the_same_answer():
    """Engines check in child processes: `entail locate` reads what every process wrote."""
    fresh()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "record.jsonl")
        old = os.environ.get("ENTAIL_RECORD")
        os.environ["ENTAIL_RECORD"] = path
        try:
            declared = Fact("Rotary", Rotary("llama3", 500000.0), Source("file", "config.json#rope_theta"),
                            Certainty.DECLARED)
            chosen = Fact("Rotary", Rotary("llama3", None), Source("engine", "rotary_embedding"), Certainty.VERIFIED)
            with contextlib.redirect_stdout(io.StringIO()):
                load.enforce([Decision(Contract("load:rope", "engine.rotary", ("Rotary",)), "Rotary", Verdict.PASS,
                                       RULES["match"], declared=declared, chosen=declared),
                              Decision(Contract("load:rope_user", "engine.rotary", ("Rotary",)), "Rotary",
                                       Verdict.BROKEN, RULES["policy_refuses"], declared=declared, chosen=chosen)])
                tally.write_summary({"container:kv": {"checks": 8, "passed": {"kv_needed": 8}}})
                with open(path, "a", encoding="utf-8") as f:
                    f.write("not json\n")
                    f.write(json.dumps({"pid": 1, "layer": "attention", "reference": "eager", "agrees": True,
                                        "max_rel": 0.001, "tol": 0.03}) + "\n")
            here = diagnose.locate(output_wrong=True)
            rows, passes, layers, skipped = record.read_records([path])
            there = record.locate(rows, passes, layers, True, skipped)
            assert (there.broken_at, there.suspects) == (here.broken_at, here.suspects) == (
                "load:rope_user", ("boundary load:rope_user",)), (here, there)
            assert there.intact == ("load:rope", "container:kv") and len(there.layers) == 1, there
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cli.main(["locate", path, "--wrong", "--json"])
            shown = json.loads(out.getvalue())
            assert (code, shown["broken_at"], shown["records"]) == (1, "load:rope_user", [path]), shown
        finally:
            if old is None:
                os.environ.pop("ENTAIL_RECORD", None)
            else:
                os.environ["ENTAIL_RECORD"] = old


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
