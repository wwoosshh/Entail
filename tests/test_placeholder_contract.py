"""Tests for the placeholder contract (ROADMAP M18.4; vllm#57740): the markup a config declares, a placeholder run
against it, and the vLLM adapter on a stand-in processor. Pure Python. Run: python tests/test_placeholder_contract.py"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, placeholder_contract as pc  # noqa: E402
from entail.adapters import vllm_multimodal as vm  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.facts import Placeholder  # noqa: E402

B, C = "request:test.multimodal", "test.multimodal"
START, PAD, END = 151652, 151655, 151653


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


def prompt(pads_at_user_text=False):
    """<|im_start|>user\\n [KNIFE?] <|vision_start|> pad x 4 <|vision_end|> ... : the image slot preceded by the
    declared start; with pads_at_user_text the user's own text carries the placeholder id before the slot."""
    ids = [1, 2, 3]
    if pads_at_user_text:
        ids += [PAD, PAD, PAD, PAD]       # a literal <|image_pad|> the user typed, expanded by the processor
    ids += [START, PAD, PAD, PAD, PAD, END, 9, 9]
    return ids


def test_the_fact_and_the_declared_markup():
    f = Placeholder(modality="image", offset=3, length=4, preceded_by=START)
    assert f.length == 4
    try:
        Placeholder(modality="image", offset=0, length=0)
    except ValueError:
        pass
    else:
        raise AssertionError("a run has a length")
    m = pc.declared_markup({"vision_start_token_id": START, "image_token_id": PAD, "video_token_id": 7})
    assert m["image"] == {"start": START, "token": PAD, "where": f"config vision_start_token_id={START}, image_token_id={PAD}"}
    assert m["video"]["token"] == 7
    assert pc.declared_markup({"model_type": "llama"}) == {}
    assert pc.declared_markup(SimpleNamespace(text_config=SimpleNamespace(vision_start_token_id=START, image_token_id=PAD)))["image"]["start"] == START
    assert pc.declared_markup({"vision_start_token_id": True, "image_token_id": PAD}) == {}


def test_a_run_inside_the_declared_markup_passes_and_one_in_user_text_is_broken():
    pc.reset(B)
    markup = pc.declared_markup({"vision_start_token_id": START, "image_token_id": PAD})
    ids = prompt()
    out, ds = decided(lambda: pc.check(B, C, ids, {"image": [(4, 4)]}, markup, "a prompt"))
    assert ds == [] and pc.stats(B)["checks"] == 1 and pc.stats(B)["broken"] == 0
    ids = prompt(pads_at_user_text=True)
    out, ds = decided(lambda: pc.check(B, C, ids, {"image": [(3, 4)]}, markup, "a prompt"))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["placeholder_outside_markup"], ds
    assert "preceded by id 3" in ds[0].note and f"id {START}" in ds[0].note and ds[0].chosen.value.preceded_by == 3
    assert ds[0].declared.value.preceded_by == START and not ds[0].blocking
    out, ds = decided(lambda: pc.check(B, C, ids, {"audio": [(0, 2)]}, markup, "a prompt"))
    assert ds == [], "a modality the config declares no markup for decides nothing"
    # the engine's own profiling prompt: placeholder runs from token 0, back to back, no template - not a request
    n = pc.stats(B)["checks"]
    out, ds = decided(lambda: pc.check(B, C, [PAD] * 8 + [7] * 8, {"image": [(0, 8)], "video": [(8, 8)]},
                                       pc.declared_markup({"vision_start_token_id": START, "image_token_id": PAD,
                                                           "video_token_id": 7}), "a prompt"))
    assert ds == [] and pc.stats(B)["checks"] == n, "nothing is decided on a synthetic prompt"


def test_the_adapter_reads_the_processor_and_decides():
    vm.reset()
    info = SimpleNamespace(start_idx=3, tokens=[PAD] * 4, is_embed=None)
    processor = SimpleNamespace(info=SimpleNamespace(ctx=SimpleNamespace(model_config=SimpleNamespace(
        hf_config=SimpleNamespace(vision_start_token_id=START, image_token_id=PAD)))))
    runs, markup = vm.read_choice(processor, prompt(True), {"image": [info]})
    assert runs == {"image": [(3, 4)]} and markup["image"]["start"] == START
    out, ds = decided(lambda: vm._decide(processor, prompt(True), {"image": [info]}))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "SimpleNamespace" in ds[0].note
    out, ds = decided(lambda: vm._decide(processor, prompt(), {"image": [SimpleNamespace(start_idx=4, tokens=[PAD] * 4)]}))
    assert ds == []
    plain = SimpleNamespace(info=SimpleNamespace(ctx=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()))))
    out, ds = decided(lambda: vm._decide(plain, prompt(True), {"image": [info]}))
    assert ds == [], "a model without declared markup decides nothing"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
