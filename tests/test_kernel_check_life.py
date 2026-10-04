"""Tests for the lifetime part of a meaning (ROADMAP M19 L7): a constant (a loaded weight) is written by nothing after
it is made - a launch that writes it, or a launch that reads it after an in-place change, is a violation; a value
that is written and read is not; two live values in one memory are reported when the second is made; an in-place
operation on a constant in a compiled graph is a violation. torch (CPU) only.
Run: python tests/test_kernel_check_life.py
"""
import os
import sys
import tempfile

import torch
import torch.fx as fx

os.environ.setdefault("ENTAIL_LOG_DIR", tempfile.mkdtemp(prefix="entail_life_test_"))   # what the check records

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import graph_types as GT  # noqa: E402
from entail import kernel_check as KC  # noqa: E402


def main():
    w = torch.zeros(8, 4)
    KC.attach(w, ["feature", "hidden"])
    KC.set_life(w, "const")
    f = KC.fact_of(w)
    assert f["life"] == "const", f
    assert KC._life_check("k", {"w": w}, {"w": f}, set()) == [], "an unchanged constant read"
    bad = KC._life_check("k", {"w": w}, {"w": f}, {"w"})
    assert bad and "writes w, a constant" in bad[0], bad
    print("ok a launch that writes a loaded weight: violation:", bad[0])

    w.add_(1.0)                                         # an in-place change: torch's version counter moves
    bad = KC._life_check("k", {"w": w}, {"w": KC.fact_of(w)}, set())
    assert bad and "written after it was loaded" in bad[0] and "in-place torch" in bad[0], bad
    print("ok a weight read after an in-place change: violation:", bad[0])

    v = torch.zeros(4)
    KC.attach(v, ["token"])
    KC._note_writes({"v": v}, {"v"})
    assert KC._life_check("k", {"v": v}, {"v": KC.fact_of(v)}, {"v"}) == []
    v.mul_(2.0)
    assert KC._life_check("k", {"v": v}, {"v": KC.fact_of(v)}, set()) == []
    print("ok a value written and changed in place, then read: no violation (only constants are held still)")

    w2 = torch.zeros(8, 4)
    KC.attach(w2, ["feature", "hidden"])
    KC.set_life(w2, "const")
    KC._note_writes({"w2": w2}, {"w2"})                 # a launch this process saw wrote it
    bad = KC._life_check("k", {"w2": w2}, {"w2": KC.fact_of(w2)}, set())
    assert bad and "a kernel this process saw" in bad[0], bad
    print("ok a weight a kernel wrote, then read: violation:", bad[0])

    buf = bytearray(4096)
    a = torch.frombuffer(buf, dtype=torch.float32)
    before = KC.stats().get("life_alias", 0)
    KC.attach(a, ["x"])
    b = torch.frombuffer(buf, dtype=torch.float32, offset=256)   # another storage over the same bytes
    KC.attach(b, ["y"])
    assert KC.stats().get("life_alias", 0) == before + 1, KC.stats()
    c = torch.zeros(16)
    KC.attach(c, ["z"])
    assert KC.stats().get("life_alias", 0) == before + 1, KC.stats()
    print("ok two live storages over the same bytes: reported when the second gets its meaning; a separate one: no")

    # another value written over a live one: a launch writes through b, whose bytes lie inside a's memory
    assert KC._life_check("k", {"a": a}, {"a": KC.fact_of(a)}, set()) == [], "nothing written yet"
    before_write = KC._note_writes({"b": b}, {"b"})
    assert before_write and "other live value" in before_write[0], before_write
    print("ok a launch about to write through b reaches a's memory: reported before it runs:", before_write[0])
    bad = KC._life_check("k", {"a": a}, {"a": KC.fact_of(a)}, set())
    assert bad and "another value" in bad[0], bad
    assert KC._life_check("k", {"b": b}, {"b": KC.fact_of(b)}, set()) == [], "the writer's own value is its new one"
    assert KC._life_check("k", {"c": c}, {"c": KC.fact_of(c)}, set()) == [], "a separate memory is untouched"
    print("ok a launch writes through one storage over another's bytes: reading the other is a violation:", bad[0])
    KC._note_writes({"c": c}, {"c"})
    assert KC._life_check("k", {"c": c}, {"c": KC.fact_of(c)}, set()) == [], "a value its own launch rewrote"
    print("ok a value written through its own storage, then read: no violation")

    # the compiled graph: an in-place op on a constant placeholder
    g = fx.Graph()
    x, wt = g.placeholder("x"), g.placeholder("w")
    xt, wtt = torch.zeros(4, 8), torch.zeros(8)
    x.meta["example_value"], wt.meta["example_value"] = xt, wtt
    n = g.call_method("add_", (wt, 1.0))
    n.meta["example_value"] = wtt
    m = g.call_function(torch.mul, (x, wt))
    m.meta["example_value"] = xt
    g.output(m)
    facts = {xt.data_ptr(): {"names": ["token", "hidden"], "kind": "value", "serial": 0, "pair": 0, "groups": [1, 1]},
             wtt.data_ptr(): {"names": ["hidden"], "kind": "value", "serial": 0, "pair": 0, "groups": [1],
                              "life": "const"}}
    v = GT.check_graph(fx.GraphModule(torch.nn.Module(), g), [xt, wtt], lambda t: facts.get(t.data_ptr()))
    assert v["verdict"] == "violation" and "constant" in v["violations"][0]["why"], v
    print("ok an in-place op on a weight in a compiled graph: violation:", v["violations"][0]["why"])
    facts[wtt.data_ptr()]["life"] = "value"
    v = GT.check_graph(fx.GraphModule(torch.nn.Module(), g), [xt, wtt], lambda t: facts.get(t.data_ptr()))
    assert not v["violations"], v
    print("ok the same op on a value: no lifetime violation")


if __name__ == "__main__":
    main()
    print("all ok")
