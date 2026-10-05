"""Tests for kernel_types on numbers that mean something (ROADMAP M19 L6): two bookkeeping kernels of vLLM 0.30's GPU
worker (tests/data/kernel_ir/vllm_*.ttir, compiled by Triton 3.7.1), read with the meanings entail's vLLM adapter
attaches (entail/adapters/vllm_index_meanings.py): a position, a place in the batch's tokens, a request slot, a KV
block, a KV slot. Their loops and masks are decided by data the kernels read, so what they cover is the data's;
what the rule decides is that every number is used along the axis it means and stored where numbers of its meaning
belong. A table of pointers (the block tables of the KV cache groups) is followed to the tensors it points to.
Mutations of the meanings (a request slot used as a batch index, a KV block stored as a KV slot) are violations.
numpy only. Run: python tests/test_kernel_types_index.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import kernel_types as KT  # noqa: E402
from entail.kernel_types import Axis, Meaning, check_launch, relate  # noqa: E402

DATA = os.path.join(HERE, "data", "kernel_ir")


def ttir(name):
    with open(os.path.join(DATA, name + ".ttir"), encoding="utf-8") as f:
        return f.read()


def idx(size, axis, basis, kind="index", label=None):
    return Meaning((Axis(axis, size),), (size,), (1,), kind, basis=basis, label=label)


def relations():
    relate("position", "//", "block_size", "block_slot")
    relate("position", "%", "block_size", "block_offset")
    relate("kv_block", "*", "block_size", "kv_slot")
    relate("kv_slot", "+", "block_offset", "kv_slot")


def pos_seq_lens(seq_basis="position", computed_basis="position"):
    ms = {"pos_ptr": idx(1024, "token", "position", "output"),
          "seq_lens_ptr": idx(256, "batch_request", seq_basis, "output"),
          "idx_mapping_ptr": idx(256, "batch_request", "request_state"),
          "query_start_loc_ptr": idx(257, "batch_request_bound", "token"),
          "num_computed_tokens_ptr": idx(256, "request_state", computed_basis)}
    return check_launch(ttir("vllm_prepare_pos_seq_lens"), ms, {"max_num_reqs": 256}, (7,))


def slot_mappings(table_axis0="request_state", slots_basis="kv_slot"):
    table = Meaning((Axis(table_axis0, 256), Axis("block_slot", 128)), (256, 128), (128, 1), "index",
                    basis="kv_block", label="block_table")
    ms = {"idx_mapping": idx(256, "batch_request", "request_state"),
          "query_start_loc": idx(257, "batch_request_bound", "token"),
          "pos": idx(1024, "token", "position"),
          "block_table_ptrs": idx(1, "kv_group", None, "pointers"),
          "block_table_strides": idx(1, "kv_group", "stride:block_table:0"),
          "block_sizes": idx(1, "kv_group", "block_size"),
          "kernel_block_sizes": idx(1, "kv_group", "block_size"),
          "slot_mapping_enabled": idx(1, "kv_group", None),
          "slot_mappings_ptr": Meaning((Axis("kv_group", 1), Axis("token", 1024)), (1, 1024), (1024, 1), "output",
                                       basis=slots_basis),
          "@block_table_ptrs[0]": table}
    ints = {"max_num_tokens": 1024, "slot_mappings_stride": 1024, "cp_rank": 0}
    return check_launch(ttir("vllm_compute_slot_mappings"), ms, ints, (1, 7), pointers={"block_table_ptrs":
                                                                                           ["@block_table_ptrs[0]"]})


def main():
    relations()
    v = pos_seq_lens()
    assert v.verdict == "proven", v
    inf = v.inferred or {}
    assert "data-dependent" in str(inf.get("pos_ptr", {}).get("coverage")), v
    assert "coverage" not in inf.get("seq_lens_ptr", {}), v     # every request's length is stored by the launch
    print(f"ok prepare_pos_seq_lens: {v.verdict}; positions covered as the data decides, sequence lengths "
          f"completely ({v.checks} pairings)")

    v = pos_seq_lens(seq_basis="token")
    assert v.verdict == "violation" and "position" in v.why and "token" in v.why, v
    print("ok a position stored where places in the batch's tokens belong:", v.why[:110])

    v = slot_mappings()
    assert v.verdict == "proven", v
    inf = v.inferred or {}
    assert "data-dependent" in str(inf.get("slot_mappings_ptr", {}).get("coverage")), v
    print(f"ok compute_slot_mappings: {v.verdict}; the block table is reached through its pointer, a KV slot "
          f"(block x size + offset) is stored where KV slots belong ({v.checks} pairings)")

    v = slot_mappings(table_axis0="batch_request")
    assert v.verdict == "violation" and "request_state" in v.why and "batch_request" in v.why, v
    print("ok a request slot used as a batch index into the block table:", v.why[:120])

    v = slot_mappings(slots_basis="kv_block")
    assert v.verdict == "violation" and "kv_slot" in v.why and "kv_block" in v.why, v
    print("ok a KV slot stored where KV block numbers belong:", v.why[:110])

    # axes of size 1 (ROADMAP M22.2): vLLM 0.30's drafter with one request reads next_prefill_tokens [lookahead 1,
    # request_state 1], strides (1, 1), at lookahead * stride + req_state_idx. Both axes have the stride 1, so the
    # stride cannot say which axis the request slot is on; the meaning names one. The recorded launch was a false
    # alarm in 2.3.0 (the slot put on 'lookahead').
    v = drafter_prepare_prefill()
    assert v.verdict == "proven", v
    print(f"ok the drafter's prepare-prefill with one request: proven ({v.checks} pairings); a request slot read "
          f"along the size-1 axis its meaning names")
    v = drafter_prepare_prefill(rename={"next_prefill_tokens_ptr": ["lookahead", "batch_request"]})
    assert v.verdict == "violation" and "request_state" in v.why, v
    print("ok the same tensor whose axes name no request slot: violation:", v.why[:110])


def drafter_prepare_prefill(rename=None):
    import json

    from entail import kernel_check, kernel_ir

    with open(os.path.join(DATA, "vllm_speculator_prepare_prefill_inputs.json"), encoding="utf-8") as f:
        rec = json.load(f)
    text = ttir("vllm_speculator_prepare_prefill_inputs")
    written = kernel_ir.written_args(text)
    meanings = {}
    for k, t in rec["tensors"].items():
        fact = dict(t["fact"]) if t["fact"] is not None else None
        if fact is not None and rename and k in rename:
            fact["names"] = rename[k]
        m = kernel_check._meaning_from(t["shape"], t["stride"], fact)
        if written is not None and k in written:
            m.kind = "output"
        meanings[k] = m
    return check_launch(text, meanings, rec["scalars"], tuple(rec["grid"]))


if __name__ == "__main__":
    main()
