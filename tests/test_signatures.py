"""Tests for the signatures of library-owned boundaries (ROADMAP M4.2) and for reading vocabulary v1 under v2: the
table is data, checked against the vocabulary when it is read, and a file written for v1 may not use a field v1 did
not have. Pure Python. Run: python tests/test_signatures.py"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import caps, signatures  # noqa: E402
from entail.facts import Certainty, Fact, Layout, Source  # noqa: E402

ROW = {"producer": "e.quant_method.M", "writes": "step", "reads": "apply", "value": "weight",
       "facts": {"Layout": {"kind": "dense", "orientation": "out_in"}}, "moves": "identity",
       "evidence": "measured", "version": "1", "ref": "a result file"}


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected ValueError containing {text!r}")


def test_the_packaged_table_reads_and_signs_what_the_audit_measured():
    t = signatures.default_table()
    unq = t.lookup("vllm.quant_method.UnquantizedLinearMethod")
    fp8 = t.lookup("vllm.quant_method.Fp8PerTensorOnlineLinearMethod")
    assert unq.declared("Layout").value == Layout("dense", orientation="out_in")
    assert fp8.declared("Layout").value == Layout("strided", dtype="float8_e4m3fn", orientation="in_out",
                                                  scale_granularity="per_tensor")
    assert all(r.evidence == "measured" and r.moves == "identity" for r in t.rows)
    assert t.why_not("vllm.quant_method.Fp8KVCacheMethod") and t.lookup("vllm.quant_method.Nope") is None


def test_declared_and_taken_name_their_step():
    s = signatures.from_rows([ROW]).lookup("e.quant_method.M")
    d, c = s.declared("Layout"), s.taken("Layout")
    assert str(d.source) == "boundary: e.quant_method.M.step.writes.weight" and d.certainty is Certainty.DECLARED
    assert str(c.source) == "boundary: e.quant_method.M.apply.takes.weight" and c.value == d.value
    assert s.declared("Quantized") is None


def test_a_bad_row_is_refused_with_the_reason():
    bad = [({**ROW, "producer": "M"}, "producer must be named engine.role.name"),
           ({**ROW, "facts": {"Layout": {"kind": "dense", "axis": 1}}}, "Layout takes the fields"),
           ({**ROW, "facts": {"Layout": {"kind": "dense", "orientation": "sideways"}}},
            "Layout.orientation: unknown value 'sideways'"),
           ({**ROW, "facts": {"Colour": {}}}, "unknown fact name 'Colour'"),
           ({**ROW, "facts": {}}, "facts must name at least one vocabulary fact"),
           ({**ROW, "moves": "rotate"}, "moves 'rotate' is not one of ['identity']"),
           ({**ROW, "evidence": "hunch"}, "evidence 'hunch' is not one of"),
           ({**ROW, "ref": " "}, "every row needs a ref"),
           ({**ROW, "colour": 1}, "unknown fields ['colour']")]
    for row, text in bad:
        raises(lambda: signatures.from_rows([row]), text)
    raises(lambda: signatures.from_rows([ROW, ROW]), "e.quant_method.M appears twice")
    raises(lambda: signatures.from_rows([ROW], {"e.quant_method.M": "x"}), "both signed and listed as holding no")
    raises(lambda: signatures.from_rows([], {"e.quant_method.N": ""}), "say why its layers hold no value")


def test_a_file_written_for_v1_may_not_use_what_v2_added():
    raises(lambda: signatures.from_rows([ROW], vocab_version=1), "Layout.orientation is not in vocabulary v1")
    v1 = {**ROW, "facts": {"Layout": {"kind": "dense"}}}
    assert signatures.from_rows([v1], vocab_version=1).lookup("e.quant_method.M")
    d = tempfile.mkdtemp()
    for name, data, text in (
            ("v9.json", {"schema": 1, "vocab_version": 9, "rows": []}, "written for vocabulary v9"),
            ("s2.json", {"schema": 2, "vocab_version": 2, "rows": []}, "schema 2, this library reads 1")):
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        raises(lambda: signatures.load_table(path), text)


def test_a_v1_fact_is_read_under_v2_but_may_not_state_a_v2_field():
    src = Source("manifest", "m.json#Layout")
    old = Fact("Layout", Layout("q8_0", packing="split"), src, Certainty.DECLARED, vocab_version=1)
    assert old.value.orientation is None and old.vocab_version == 1
    raises(lambda: Fact("Layout", Layout("dense", orientation="in_out"), src, Certainty.DECLARED, vocab_version=1),
           "fact Layout was written with vocabulary v1, which has no Layout.orientation (added in v2)")


def test_the_capability_table_written_for_v1_still_reads_and_may_not_use_v2_fields():
    assert caps.load_table().rows   # data/caps.json is still marked vocabulary v1
    d = tempfile.mkdtemp()
    path = os.path.join(d, "caps.json")
    row = {"consumer": "e.linear.k", "fact": "Layout.orientation", "honours": True, "evidence": "code", "ref": "x"}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "vocab_version": 1, "rows": [row]}, f)
    raises(lambda: caps.load_table(path), "Layout.orientation is not in vocabulary v1 (added in v2)")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"schema": 1, "vocab_version": 2, "rows": [row]}, f)
    assert caps.load_table(path).rows[0].fact == "Layout.orientation"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
