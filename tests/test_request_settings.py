"""Tests for the request-setting name rule in the core (ROADMAP M17.1b; data/request_settings.json): a setting the
request gives under a name the template honoured and the reasoning parser reads under another name, the repair
that hands the value to the parser, the per-version parser table (the retrospective on vllm#43728 at 0.22.0), and
the vLLM serve adapter's parser wrapper on a fake parser class. Pure Python, no engine.
Run: python tests/test_request_settings.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, request_contract as rc  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402

B, C = "request:test.parser_settings", "test.reasoning_parser.kimi_k2"


def quiet(fn):
    import io
    from contextlib import redirect_stdout

    n = len(load.LEDGER.decisions)
    was = core.mode()
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            out = fn()
    finally:
        core.set_mode(was)
    return out, load.LEDGER.decisions[n:]


def test_the_table_knows_the_parsers_per_version():
    assert rc.parser_reads("vllm", "kimi_k2", "0.22.0") == ["thinking"]
    assert rc.parser_reads("vllm", "kimi_k2", "0.30.0") == ["thinking", "enable_thinking"]
    assert rc.parser_reads("vllm", "kimi_k2", None) == ["thinking", "enable_thinking"]      # the engine's row
    assert rc.parser_reads("vllm", "qwen3", "0.22.0") == ["enable_thinking"]
    assert rc.parser_reads("vllm", "hermes", "0.30.0") is None
    assert rc.parser_reads("sglang", "qwen3", None) is None


def test_a_name_the_template_honoured_and_the_parser_does_not_read_is_broken_or_resolved():
    given = {"enable_thinking": False}
    out, rec = quiet(lambda: rc.setting_names(B, C, given, {"enable_thinking"}, ["thinking"], "the request",
                                              "kimi_k2 (0.22.0)"))
    assert len(out) == 1 and out[0].verdict is Verdict.BROKEN and out[0].rule == RULES["setting_name_not_read"], out
    assert "runs on its default" in out[0].note
    handed = {}
    out, rec = quiet(lambda: rc.setting_names(B, C, given, {"enable_thinking"}, ["thinking"], "the request",
                                              "kimi_k2 (0.22.0)",
                                              handles={"apply_setting_name": lambda t: handed.update([t]) or True}))
    assert out[0].verdict is Verdict.RESOLVED and out[0].target == ("thinking", False), out
    assert handed == {"thinking": False}


def test_the_retrospective_on_vllm_43728_by_the_0_22_0_row():
    reads = rc.parser_reads("vllm", "kimi_k2", "0.22.0")
    ck = {"enable_thinking": False}
    out, _ = quiet(lambda: rc.setting_names(B, C, ck, {"enable_thinking"}, reads, "the request", "kimi_k2 (0.22.0)",
                                            handles={"apply_setting_name": lambda t: ck.__setitem__(*t) or True}))
    assert out[0].verdict is Verdict.RESOLVED and ck == {"enable_thinking": False, "thinking": False}
    reads30 = rc.parser_reads("vllm", "kimi_k2", "0.30.0")
    out, _ = quiet(lambda: rc.setting_names(B, C, {"enable_thinking": False}, {"enable_thinking"}, reads30,
                                            "the request", "kimi_k2 (0.30.0)"))
    assert [d.verdict for d in out] == [Verdict.PASS], out


def test_nothing_is_decided_when_the_template_did_not_read_it_or_the_parser_reads_no_name_of_the_group():
    # Qwen3's template reads enable_thinking; a request giving `thinking` is unread by the template (the template
    # rule says so) and the parser must not be pushed away from what the template did
    out, _ = quiet(lambda: rc.setting_names(B, C, {"thinking": False}, {"enable_thinking"}, ["enable_thinking"],
                                            "the request", "qwen3"))
    assert out == []
    out, _ = quiet(lambda: rc.setting_names(B, C, {"enable_thinking": False}, {"enable_thinking"}, [], "the request",
                                            "a tool parser"))
    assert out == []
    out, _ = quiet(lambda: rc.setting_names(B, C, {}, {"enable_thinking"}, ["thinking"], "the request", "kimi_k2"))
    assert out == []
    out, _ = quiet(lambda: rc.setting_names(B, C, {"enable_thinking": None}, {"enable_thinking"}, ["thinking"],
                                            "the request", "kimi_k2"))
    assert out == []


def test_the_serve_adapter_wraps_the_parser_class_and_hands_the_name_over():
    from entail.adapters import vllm_serve

    seen = {}

    class Fake:
        def __init__(self, tokenizer, tools=None, *args, **kwargs):
            seen.update(kwargs.get("chat_template_kwargs") or {})

    class Tok:
        chat_template = "{% if enable_thinking is defined and not enable_thinking %}off{% endif %}"

    cls_ = vllm_serve._wrap_parser_cls(Fake, "kimi_k2", reads=["thinking"])
    assert cls_.__name__ == "Fake" and issubclass(cls_, Fake)
    _, rec = quiet(lambda: cls_(Tok(), None, chat_template_kwargs={"enable_thinking": False}, model_config=None))
    assert seen == {"enable_thinking": False, "thinking": False}, seen
    assert any(d.verdict is Verdict.RESOLVED and d.contract.boundary == vllm_serve.SETTING_NAMES for d in rec), rec
    assert vllm_serve._wrap_parser_cls(Fake, "hermes", reads=None) is Fake         # a parser the table does not know
    assert vllm_serve._wrap_parser_cls(None, "kimi_k2") is None
    assert vllm_serve._wrap_parser_cls(("not", "a", "class"), "kimi_k2") == ("not", "a", "class")
    assert "enable_thinking" in vllm_serve._template_reads(Tok.chat_template)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
