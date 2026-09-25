"""Tests for the vLLM scoring adapter (ROADMAP M15.1; vllm#58138): what it reads from the padded token type ids and
how its handle repairs them, on fakes - no vLLM. The core rule (TokenType) is tested in test_request_contract.py.
Run: python tests/test_vllm_scoring.py"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, request_contract  # noqa: E402
from entail.adapters import vllm_scoring  # noqa: E402
from entail.contracts import Verdict  # noqa: E402

TOK = SimpleNamespace(pad_token_type_id=0, truncation_side="right")


def params(truncate=-1, max_input=64):
    return SimpleNamespace(truncate_prompt_tokens=truncate, max_input_tokens=max_input)


def vllm_pads(given, pad_length):
    """What vLLM 0.30 does: pad with the last real id."""
    return given + [given[-1]] * (pad_length - len(given))


def test_nothing_padded_means_nothing_to_decide():
    given = [0, 0, 1, 1]
    assert vllm_scoring.read_choice(TOK, params(), given, list(given)) == (None, None, 4)


def test_the_padding_the_engine_gave_and_the_declared_type_are_read():
    given = [0, 0, 0, 1, 1]
    out = vllm_pads(given, 8)
    assert vllm_scoring.read_choice(TOK, params(), given, out) == (0, 1, 5)


def test_truncation_before_padding_is_followed():
    given = [0] * 10
    out = [0] * 4 + [1] * 4   # truncated to 4 (the tok params say so), then padded to 8 with 1
    assert vllm_scoring.read_choice(TOK, params(truncate=4, max_input=8), given, out) == (0, 1, 4)


def test_a_tokenizer_that_does_not_say_gives_no_declared_type():
    given = [0, 1]
    out = vllm_pads(given, 4)
    assert vllm_scoring.read_choice(SimpleNamespace(), params(), given, out) == (None, 1, 2)


def test_the_handle_writes_the_declared_type_over_the_padding_in_place():
    out = [0, 0, 1, 1, 1, 1]
    vllm_scoring.handles(out, 4)["set_pad_type"](0)
    assert out == [0, 0, 1, 1, 0, 0]


def test_the_boundary_reports_the_padding_and_leaves_it_since_vllm_cannot_carry_the_declared_type():
    """vLLM 0.30 keeps token types as the index of the first 1 (caps.json: honours false, measured): a pad type
    after the document cannot be carried, so the decision is broken under the default policy and the ids stay."""
    given = [0, 0, 0, 1, 1]
    out = vllm_pads(given, 8)
    n = len(load.LEDGER.decisions)
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            vllm_scoring._decide(TOK, params(), given, out)
    finally:
        core.set_mode("off")
    d = load.LEDGER.decisions[n:]
    assert len(d) == 1 and d[0].verdict is Verdict.BROKEN and d[0].resolution is None, d
    assert "compressed form" in d[0].note and out == [0, 0, 0, 1, 1, 1, 1, 1]
    request_contract.reset(vllm_scoring.BOUNDARY)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
