"""Tests for fact vocabulary v1 and the fact envelope (ROADMAP M1.1). Every closed set rejects what is outside it, with
the exact reason. Run: python tests/test_facts_v1.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail.facts import (FACT_KINDS, VOCAB_VERSION, VOCABULARY, Assumed, Certainty, Epoch, Fact, LatentScale,  # noqa
                          Layout, ModelProps, Origin, Positions, Prediction, Quantized, Reduction, Rotary, Source,
                          Template, Valid, vocabulary_class)


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        assert text in str(e), (text, str(e))
        return str(e)
    raise AssertionError(f"expected ValueError containing {text!r}")


def test_closed_sets_reject_with_the_reason():
    cases = [
        (lambda: Layout("q9_9"), "unknown layout kind 'q9_9'; closed set is"),
        (lambda: Layout("q8_0", packing="zigzag"), "Layout.packing: unknown value 'zigzag'; closed set is"),
        (lambda: Layout("fp8_block", scale_format="fp64"), "Layout.scale_format: unknown value 'fp64'"),
        (lambda: Layout("dense", dtype="float7"), "Layout.dtype: unknown value 'float7'"),
        (lambda: Layout("dense", block=(0,)), "Layout.block: expected a tuple of positive ints, got (0,)"),
        (lambda: Layout("dense", orientation="sideways"), "Layout.orientation: unknown value 'sideways'"),
        (lambda: Layout("dense", scale_granularity="per_row"), "Layout.scale_granularity: unknown value 'per_row'"),
        (lambda: Template(tool_call_format="xml"), "Template.tool_call_format: unknown value 'xml'"),
        (lambda: Quantized("float7"), "Quantized.dtype: unknown value 'float7'"),
        (lambda: Quantized("float8_e4m3fn", scale=0), "Quantized.scale: expected > 0, got 0"),
        (lambda: Positions("relative"), "Positions.frame: unknown value 'relative'"),
        (lambda: Positions("chunk_relative", -1), "Positions.offset: expected >= 0, got -1"),
        (lambda: Rotary("ntk"), "Rotary.rope_type: unknown value 'ntk'"),
        (lambda: Rotary(theta=-1.0), "Rotary.theta: expected > 0, got -1.0"),
        (lambda: Valid(length=-1), "Valid.length: expected >= 0, got -1"),
        (lambda: Valid(window=0), "Valid.window: expected > 0, got 0"),
        (lambda: ModelProps(softcap=True), "ModelProps.softcap: expected a number, got True"),
        (lambda: ModelProps(sliding_window=4096.0), "ModelProps.sliding_window: expected an int, got 4096.0"),
        (lambda: ModelProps(tie_word_embeddings="yes"), "ModelProps.tie_word_embeddings: expected a bool, got 'yes'"),
        (lambda: Prediction("v", "yes"), "Prediction.zsnr: expected a bool, got 'yes'"),
        (lambda: Prediction("velocity"), "unknown prediction kind 'velocity'"),
        (lambda: LatentScale(0), "LatentScale.scale: expected > 0, got 0"),
        (lambda: LatentScale(0.13, shift="x"), "LatentScale.shift: expected a number, got 'x'"),
        (lambda: Template(chat_template_sha256="abc"), "Template.chat_template_sha256: expected 64 lowercase hex"),
        (lambda: Template(reasoning_history="summarise"), "Template.reasoning_history: unknown value 'summarise'"),
        (lambda: Reduction("Q"), "Reduction.state: unknown value 'Q'"),
        (lambda: Reduction("S"), "a sharded value ('S') must say which dim it is sharded on"),
        (lambda: Epoch(-1), "Epoch.version: expected >= 0, got -1"),
        (lambda: Assumed(("batch",)), "Assumed.conditions: expected a tuple of (name, value) pairs"),
        (lambda: Assumed((("b", 1), ("a", 2))), "pairs must be sorted by name"),
        (lambda: Origin("temperature", "guess"), "Origin.came_from: unknown value 'guess'"),
        (lambda: Origin("", "user"), "Origin.setting: expected a setting name"),
    ]
    for fn, text in cases:
        raises(fn, text)


def test_values_the_existing_code_uses_are_accepted():
    """Every value the older modules, the tests and rolebench construct (found by grep, M1.1) stays valid."""
    Layout("dense"), Layout("strided"), Layout("q8_0", packing="interleaved"), Layout("q8_0", packing="split")
    Layout("fp8_block", scale_format="fp32"), Layout("fp8_block", scale_format="ue8m0")
    Positions("absolute"), Positions("chunk_relative", 256), Positions("chunk_relative", offset=16)
    Quantized("float8_e4m3fn", scale=0.5), Quantized("float32", scale=None), Quantized("bfloat16", scale=None)
    Reduction("P"), Reduction("R"), Valid(length=3), Valid(length=0)
    ModelProps(softcap=50.0), ModelProps(softcap=50), ModelProps(sliding_window=4096), ModelProps()
    Prediction("eps", False), Prediction("v", True), Prediction("v")
    Rotary("llama3", theta=500000.0, factor=32.0, original_max_position=8192), LatentScale(0.13025, shift=-0.1)
    Template("a" * 64, "keep"), Epoch(0), Assumed((("batch", 4), ("seq", 128))), Origin("temperature", "user")


def test_every_name_has_a_class_and_all_ten_kinds_are_covered():
    assert set(VOCABULARY.values()) == FACT_KINDS, set(VOCABULARY.values()) ^ FACT_KINDS
    for name in VOCABULARY:
        assert vocabulary_class(name).__name__ == name
    raises(lambda: vocabulary_class("Colour"), f"unknown fact name 'Colour'; vocabulary v{VOCAB_VERSION} has")


def test_the_envelope_checks_itself():
    src = Source("file", "m.safetensors#__metadata__.modelspec.prediction_type")
    f = Fact("Prediction", Prediction("v"), src, Certainty.DECLARED)
    assert f.kind == "PROPERTY" and f.vocab_version == VOCAB_VERSION and str(src).startswith("file: ")
    Fact("Prediction", None, src, Certainty.UNKNOWN)
    raises(lambda: Fact("Colour", Prediction("v"), src, Certainty.DECLARED), "unknown fact name 'Colour'")
    raises(lambda: Fact("Prediction", Prediction("v"), src, Certainty.DECLARED, vocab_version=9),
           "fact Prediction was written with vocabulary v9; this library reads v1, v2, v3")
    assert Fact("Prediction", Prediction("v"), src, Certainty.DECLARED, vocab_version=1).vocab_version == 1
    raises(lambda: Fact("Prediction", Prediction("v"), src, "declared"), "Fact.certainty: expected a Certainty")
    raises(lambda: Fact("Prediction", None, src, Certainty.DECLARED),
           "a fact has no value exactly when its certainty is unknown (value None, certainty declared)")
    raises(lambda: Fact("Prediction", Prediction("v"), src, Certainty.UNKNOWN), "certainty unknown")
    raises(lambda: Fact("Prediction", Layout("dense"), src, Certainty.DECLARED),
           "fact Prediction: holds a Layout, expected Prediction")
    raises(lambda: Source("guess", "x"), "Source.kind: unknown value 'guess'")
    raises(lambda: Source("file", ""), "Source.where: expected an address")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
