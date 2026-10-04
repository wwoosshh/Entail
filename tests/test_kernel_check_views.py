"""Tests for the meaning of a view (ROADMAP M19 L6 step 1): a slice, a selected row or column, a transpose, a merge
of two axes the vocabulary names, each read back from the tensor it views - with the coordinate it starts at - and a
view that walks the tensor in a way no axis does, which means nothing. torch (CPU) only.
Run: python tests/test_kernel_check_views.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import kernel_check as KC  # noqa: E402
from entail import kernel_types as KT  # noqa: E402


def names(t):
    f = KC.fact_of(t)
    return None if f is None else (list(f["names"]), list(f.get("origins") or [0] * t.dim()))


def main():
    table = torch.zeros(6, 4, dtype=torch.int32)
    KC.attach(table, ["request_state", "block_slot"], kind="index", basis="kv_block")
    assert names(table) == (["request_state", "block_slot"], [0, 0])
    col = table[:, 0]
    assert names(col) == (["request_state"], [0]), names(col)
    f = KC.fact_of(col)
    assert f["basis"] == "kv_block" and f["kind"] == "index", f
    print("ok a column of a block table: [request_state] from 0, its numbers kv blocks")
    rows = table[2:5]
    assert names(rows) == (["request_state", "block_slot"], [2, 0]), names(rows)
    print("ok rows 2..4: the same axes, from request_state 2")
    t = table.t()
    assert names(t) == (["block_slot", "request_state"], [0, 0]), names(t)
    print("ok a transpose: the axes swap")
    part = table[1:3, 1:3]
    assert names(part) == (["request_state", "block_slot"], [1, 1]), names(part)
    print("ok a block of it: both axes, from (1, 1)")
    one = table[3]
    assert names(one) == (["block_slot"], [0]), names(one)
    print("ok one row: [block_slot]")

    ids = torch.zeros(5, 2, dtype=torch.int32)
    KC.attach(ids, ["token", "k"], kind="index", basis="expert")
    flat = ids.view(-1)
    assert KC.fact_of(flat) is None, KC.fact_of(flat)
    KT.merge("token", "k", "token_slot")
    assert names(flat) == (["token_slot"], [0]), names(flat)
    print("ok a router table read flat: nothing until the vocabulary names the merge, then [token_slot]")

    odd = torch.as_strided(table, (3,), (3,))          # walks the table by 3: no axis of it
    assert KC.fact_of(odd) is None, KC.fact_of(odd)
    print("ok a view no axis walks: no meaning")

    m = KC._meaning_from(rows.shape, rows.stride(), KC.fact_of(rows))
    assert [a.origin for a in m.axes] == [2, 0] and [a.name for a in m.axes] == ["request_state", "block_slot"], m
    print("ok the meaning handed to the rule carries the view's origin")


if __name__ == "__main__":
    main()
    print("all ok")
