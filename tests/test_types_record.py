"""Tests for what ENTAIL=types leaves for the platform (ROADMAP M22.2): every decided launch configuration and
compiled graph counted at the boundary kernel:types, a broken launch and a broken read as decisions (kernel:types,
kernel:life), and at exit one line on what was checked and what was not read. The platform's Kernels node shows them.
No GPU: the checker's own record functions are called as a launch would call them.
Run: python tests/test_types_record.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
_DIR = tempfile.mkdtemp(prefix="entail_types_record_")
os.environ["ENTAIL_LOG_DIR"] = _DIR
os.environ["ENTAIL_RECORD"] = os.path.join(_DIR, "record.jsonl")
os.environ["ENTAIL_QUIET"] = "all"

from entail import kernel_check as KC  # noqa: E402
from entail import kernel_types  # noqa: E402
from entail import tally  # noqa: E402
from entail.platform import graph as PG  # noqa: E402


def lines():
    with open(os.environ["ENTAIL_RECORD"], encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    for verdict in ("proven", "proven", "unproven"):
        KC._count(f"verdict_{verdict}")
        KC._tallied(KC.BOUNDARY_TYPES, verdict)
    KC._count("graph_checked")
    KC._tallied(KC.BOUNDARY_TYPES, "checked")
    KC._count("verdict_violation")
    KC._tallied(KC.BOUNDARY_TYPES, "violation")
    KC._broken("vllm.kern.fused_kernel", kernel_types.Verdict("violation", "a load addresses b_scale_ptr along its "
                                                                           "axis 'feature' with a number that means "
                                                                           "an expert"))
    KC._broken("block", kernel_types.Verdict("violation", "block runs with its parameter bias, but 8 of its 8 "
                                                          "elements were never written"), where="block")
    tally.write_summary(tally.stats())
    KC._end_line()

    rows = lines()
    decisions = [r for r in rows if r.get("verdict") and r.get("boundary")]
    assert {(r["boundary"], r["verdict"]) for r in decisions} == {("kernel:types", "broken"),
                                                                   ("kernel:life", "broken")}, decisions
    print("ok a broken launch and a broken read are decisions in the record, at kernel:types and kernel:life")

    g = PG.graph(rows)
    kern = [n for n in g["nodes"] if n["id"] == "kernel"]
    assert kern and kern[0]["state"] == "broken", g["nodes"]
    d = PG.node_detail(rows, "kernel")
    c = d["counts"]["kernel:types"]
    assert c["checks"] == 5 and c["passed"] == 3 and c["skipped"] == 1, c
    said = " ".join(x["text"] for x in d["said"])
    assert "checked 4 kernel launch configurations (2 proven, 1 broken, 1 not decided) and 1 compiled graphs" in said \
        and "not read: the inside of C++ kernels" in said and "Python computation" in said, said
    print("ok the Kernels node: broken, 5 checks (3 held, 1 not decided), and the end line:", said[:150])


if __name__ == "__main__":
    main()
