"""Tests for the platform's node model (ROADMAP product track P1; LIBRARY_DESIGN.md 13.3, 13.4): the node table in
data/nodes.json, record lines of version 2 (v, t, run), and a launch's lines as the nodes of its workflow.
Pure Python. Run: python tests/test_platform_graph.py"""
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import record  # noqa: E402
from entail.platform import graph  # noqa: E402

# Boundary names seen in the research workspace's record files (2,094 files, 171 names, 2026-09-28), one or more per
# family, with the node each belongs to.
KNOWN = {
    "load:transformers.config": "config", "load:vllm.config.rope_parameters": "rotary",
    "load:transformers.config.rope_scaling[full_attention]": "rotary", "load:sglang.model_config": "config",
    "load:transformers.generation_config": "config", "load:transformers.tokenizer": "tokenizer",
    "load:transformers.tokenizer.ids": "tokenizer", "load:vllm.input_processor": "tokenizer",
    "load:vllm.loader": "weights", "load:vllm.weights": "weights", "load:sglang.linear": "weights",
    "load:vllm.quant_method.process_weights_after_loading": "weights", "load:vllm.attention": "rotary",
    "load:transformers.paged_attention": "rotary", "load:vllm.rotary_pairing": "rotary",
    "load:vllm.adapter_config": "adapter", "load:sglang.adapter_config": "adapter", "start:vllm.paths": "self_check",
    "start:sglang.paths": "self_check", "request:vllm.chat_template": "request",
    "request:vllm.template_settings": "request", "request:vllm.request_fields": "request",
    "request:vllm.reasoning_history": "request", "request:vllm.parser_settings": "request",
    "request:vllm.multimodal": "request", "request:vllm.scoring.token_type_ids": "request",
    "request:transformers.chat_template": "request", "request:vllm.parser": "response",
    "request:sglang.logprobs": "response", "load:vllm.tool_parser": "response",
    "container:vllm.allocate_slots": "cache", "container:vllm.request.block_hashes": "cache",
    "container:transformers.cache_update": "cache", "container:sglang.prepare_for_decode": "cache",
    "request:transformers.generate.beam_reorder": "cache", "kernel:vllm.custom_op": "kernel", "kernel:triton": "kernel",
    "kernel:definition": "kernel", "kernel:vllm.fused_moe_kernel": "kernel",
    "load:sglang.fp8_block_kernel_config": "kernel", "load:sglang.fused_moe_kernel_config": "kernel",
    "load:diffusers.single_file": "checkpoint", "load:diffusers.pipeline": "checkpoint",
    "load:comfyui.checkpoint": "checkpoint", "load:comfyui.prediction": "prediction",
    "load:diffusers.latent_scale": "vae", "load:comfyui.latent_scale": "vae", "load:diffusers.lora": "image_lora",
    "load:comfyui.lora": "image_lora", "boundary:attention.q": "user",
}


def test_the_node_table_is_whole():
    m = graph.model()
    ids = [n["id"] for n in m["nodes"]]
    assert len(ids) == len(set(ids)) and ids[-1] == "other", ids
    for n in m["nodes"]:
        assert n["flow"] in m["flows"] and isinstance(n["step"], int) and n["ko"] and n["en"], n
        for p in n["patterns"]:
            re.compile(p)
    assert graph.node_of("anything:at all") == "other"


def test_every_known_boundary_goes_to_its_node():
    wrong = {b: (graph.node_of(b), want) for b, want in KNOWN.items() if graph.node_of(b) != want}
    assert not wrong, wrong
    assert graph.engine_of("load:vllm.attention") == "vllm" and graph.engine_of("kernel:triton") is None


def test_record_lines_carry_version_time_and_launch():
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "r.jsonl")
    old = {k: os.environ.get(k) for k in ("ENTAIL_RECORD", "ENTAIL_RUN_ID")}
    try:
        os.environ["ENTAIL_RECORD"] = path
        os.environ.pop("ENTAIL_RUN_ID", None)
        record.write_json({"pid": 7, "boundary": "load:vllm.attention", "verdict": "pass"})
        first = os.environ["ENTAIL_RUN_ID"]           # named on the first line, then inherited
        record.write_json({"pid": 8, "timing": "load:vllm.attention", "ms": 1.5})
        record.close_files()
        lines = [json.loads(x) for x in open(path, encoding="utf-8")]
        assert [x["v"] for x in lines] == [2, 2] and all(isinstance(x["t"], float) for x in lines)
        assert [x["run"] for x in lines] == [first, first] and lines[0]["pid"] == 7 and lines[1]["ms"] == 1.5
    finally:
        record.close_files()
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _launch():
    run = "4242-1790000000"
    return [
        {"v": 2, "t": 10.0, "run": run, "pid": 1, "boundary": "load:vllm.attention", "verdict": "pass",
         "name": "ModelProps", "rule": "match"},
        {"v": 2, "t": 11.0, "run": run, "pid": 1, "boundary": "load:transformers.tokenizer", "verdict": "broken",
         "name": "Tokenization", "rule": "tokenizer_ids", "note": "probe 3 differs"},
        {"v": 2, "t": 12.0, "run": run, "pid": 2, "boundary": "container:vllm.allocate_slots", "verdict": "unknown",
         "name": "KvExtent", "rule": "not decided"},
        {"v": 2, "t": 13.0, "run": run, "pid": 2, "boundaries": {
            "container:vllm.allocate_slots": {"checks": 5, "passed": {"kv_needed": 5}, "skipped": 0},
            "kernel:vllm.custom_op": {"checks": 0, "skipped": 3, "passed": {}}}},
        {"v": 2, "t": 13.5, "run": run, "pid": 2, "timing": "kernel:vllm.custom_op", "ms": 2.25},
        {"v": 2, "t": 14.0, "run": run, "pid": 2, "timing": "kernel:vllm.custom_op", "ms": 0.75},
    ]


def test_a_launch_becomes_its_workflow():
    g = graph.graph(_launch())
    by = {n["id"]: n for n in g["nodes"]}
    assert [f["id"] for f in g["flows"]] == ["llm"], g["flows"]
    order = g["flows"][0]["nodes"]
    assert order[:4] == ["config", "tokenizer", "weights", "rotary"] and "other" not in order, order
    assert g["flows"][0]["edges"][0] == ["config", "tokenizer"]
    assert by["rotary"]["state"] == "pass" and by["tokenizer"]["state"] == "broken", by
    assert by["cache"]["state"] == "unknown" and by["cache"]["checks"] == 5 and by["cache"]["passed"] == 5
    assert by["kernel"]["state"] == "unchecked" and by["kernel"]["skipped"] == 3
    assert by["kernel"]["timed_calls"] == 2 and by["kernel"]["ms"] == 3.0
    assert by["weights"]["state"] == "none" and by["weights"]["boundaries"] == []
    assert g["state"] == "broken" and g["locate"]["broken_at"] == "load:transformers.tokenizer", g["locate"]
    assert g["start"] == 10.0 and g["end"] == 14.0 and g["pids"] == [1, 2] and g["engines"] == ["transformers", "vllm"]
    d = graph.node_detail(_launch(), "kernel")
    assert d["counts"]["kernel:vllm.custom_op"]["skipped"] == 3 and d["timing"]["kernel:vllm.custom_op"]["calls"] == 2
    t = graph.node_detail(_launch(), "tokenizer")
    assert len(t["decisions"]) == 1 and t["decisions"][0]["note"] == "probe 3 differs" and t["ko"] == "토크나이저"


def test_launches_group_by_run_and_old_files_by_file():
    old = [("record-2026-09-26.jsonl", {"pid": 5, "boundary": "load:vllm.attention", "verdict": "pass"})]
    new = [("record-2026-09-28.jsonl", x) for x in _launch()]
    groups = graph.launches(old + new)
    assert set(groups) == {"file:record-2026-09-26.jsonl", "4242-1790000000"}, set(groups)
    s = graph.summaries(groups)
    assert s[0]["run"] == "4242-1790000000" and s[0]["state"] == "broken" and s[-1]["start"] is None, s


def test_an_image_launch_draws_the_image_flow():
    lines = [{"v": 2, "t": 1.0, "run": "r", "pid": 1, "boundary": "load:comfyui.prediction", "verdict": "resolved",
              "name": "Prediction"}]
    g = graph.graph(lines)
    assert [f["id"] for f in g["flows"]] == ["image"] and g["flows"][0]["nodes"][1] == "prediction"
    assert {n["id"]: n["state"] for n in g["nodes"]}["prediction"] == "resolved"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
