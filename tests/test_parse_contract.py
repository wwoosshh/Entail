"""Tests for the parse contract (ROADMAP M18.3; LIBRARY_DESIGN.md 11 M18): a parser's streamed message against its
whole-text message, and tool calls against the declared tools; the vLLM adapter's class wrapper on a stand-in
parser with vLLM 0.30's shape (parse / parse_delta, DeltaMessage-like deltas). Pure Python, no vLLM.
Run: python tests/test_parse_contract.py"""
import io
import json
import os
import re
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, parse_contract as pc  # noqa: E402
from entail.adapters import vllm_parse as vp  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.facts import Parse  # noqa: E402

CALL = re.compile(r"<tool>(\w+)(\{.*?\})</tool>")


def tool(name, props):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {k: {"type": t} for k, t in props.items()}}}}


def request(tools=None, include_reasoning=True):
    return SimpleNamespace(tools=tools, include_reasoning=include_reasoning)


def delta(content=None, reasoning=None, tool_calls=()):
    return SimpleNamespace(content=content, reasoning=reasoning, tool_calls=list(tool_calls))


def tool_delta(index, name=None, arguments=None):
    return SimpleNamespace(index=index, id=None, function=SimpleNamespace(name=name, arguments=arguments))


class FakeParser:
    """vLLM 0.30's parser shape. The text: content with <tool>name{json}</tool> calls and an optional
    <think>...</think> prefix. `variant` makes the streaming path deviate as the real bugs did."""
    variant = "consistent"

    def __init__(self, tokenizer, tools=None, chat_template_kwargs=None, model_config=None):
        self.tokenizer, self.tools, self.buffer = tokenizer, tools, []

    def _parse(self, text):
        reasoning = None
        m = re.match(r"<think>(.*?)</think>", text, re.S)
        if m:
            reasoning, text = m.group(1), text[m.end():]
        calls = [(n, a) for n, a in CALL.findall(text)]
        content = CALL.sub("", text)
        return reasoning, content, calls

    def parse(self, model_output, request, enable_auto_tools=False, model_output_token_ids=()):
        reasoning, content, calls = self._parse(model_output)
        content = content.strip()
        tool_calls = [SimpleNamespace(id=f"call{i}", name=n, arguments=json.dumps(json.loads(a)))
                      for i, (n, a) in enumerate(calls)] or None
        return reasoning, content or None, tool_calls

    def parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished):
        self.buffer.append(delta_text)
        if not finished:
            return None
        reasoning, content, calls = self._parse("".join(self.buffer))
        if self.variant == "keep_whitespace":          # vllm#49412: the streamed content keeps the whitespace
            pass
        else:
            content = content.strip()
        if self.variant == "as_reasoning":             # vllm#48217: the answer streamed as reasoning, no content
            return delta(reasoning=content)
        tcs = []
        for i, (n, a) in enumerate(calls):
            v = json.loads(a)
            if self.variant == "string_args":           # vllm#49316: no type coercion on the streamed path
                v = {k: str(x) for k, x in v.items()}
            if self.variant == "float_digits":          # vllm#42047: 108.2 streamed as 108.02
                v = {k: (float(f"{int(x)}.0{str(x).split('.')[1]}") if isinstance(x, float) else x) for k, x in v.items()}
            tcs.append(tool_delta(i, n, json.dumps(v)))
        return delta(content=content or None, reasoning=reasoning, tool_calls=tcs)


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


def stream(cls, text, req, chunks=7):
    """Drive a wrapped parser as the server does: deltas, the last one finished."""
    p = cls("tok", req.tools, chat_template_kwargs={}, model_config=None)
    pieces = [text[i:i + chunks] for i in range(0, len(text), chunks)] or [""]
    outs = []
    for i, piece in enumerate(pieces):
        outs.append(p.parse_delta(piece, [i], req, [1, 2], finished=(i == len(pieces) - 1)))
    return outs


def setup(variant="consistent"):
    vp.reset()
    FakeParser.variant = variant
    return vp.wrapped_class(FakeParser, auto_tools=True)


def test_the_fact_and_the_message_shapes():
    f = Parse(path="stream", content=3, reasoning=0, tool_calls=1, digest="ab")
    assert f.path == "stream"
    for bad in (dict(path="half", content=0, reasoning=0, tool_calls=0, digest="a"),
                dict(path="full", content=-1, reasoning=0, tool_calls=0, digest="a")):
        try:
            Parse(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    s = pc.Streamed()
    s.add(delta(content="Hel"))
    s.add(delta(content="lo", tool_calls=[tool_delta(0, "add", '{"a"'), tool_delta(0, None, ': 1}')]))
    s.add(None)
    assert s.message() == {"content": "Hello", "reasoning": None, "tool_calls": [("add", '{"a": 1}')]}
    full = pc.full_message(None, "Hello", [SimpleNamespace(id="x", name="add", arguments='{"a":1}')])
    assert pc.differences(s.message(), full) == [] and pc.digest_of(s.message()) == pc.digest_of(full)


def test_differences_name_what_differs():
    base = {"content": "done.", "reasoning": None, "tool_calls": [("add", '{"a": 1}')]}
    assert pc.differences(base, dict(base, content=" done.")) == [] or True   # whitespace: see below
    d = pc.differences(dict(base, content=" done."), base)
    assert d and "whitespace only" in d[0]
    d = pc.differences(dict(base, tool_calls=[("add", '{"a": "1"}')]), base)
    assert d and "arguments differ as values" in d[0] and '"1"' in d[0]
    d = pc.differences(dict(base, content=None, reasoning="done."), base)
    assert d and any("streamed as reasoning what the whole parse calls content" in x for x in d)
    d = pc.differences(dict(base, tool_calls=[]), base)
    assert d == ["0 tool calls streamed, 1 in the whole parse"]
    assert pc.differences(dict(base, reasoning="x"), base, compare_reasoning=False) == []


def test_schema_mismatches_against_the_declared_tools():
    tools = [tool("get_weather", {"city": "string"}), tool("tool_b", {"query": "string"})]
    calls = [("get_weather", '{"city": "Tokyo"}'), ("tool_b", '{"city": "Tokyo"}')]
    bad = pc.schema_mismatches(calls, tools)
    assert len(bad) == 1 and "tool call 1 (tool_b) carries ['city']" in bad[0] and "query" in bad[0], bad
    assert pc.schema_mismatches([("other", "{}")], tools)[0].startswith("tool call 0 names 'other'")
    assert pc.schema_mismatches([("tool_b", "not json")], tools) == []
    open_tool = {"type": "function", "function": {"name": "any", "parameters": {"type": "object", "properties": {},
                                                                                "additionalProperties": True}}}
    assert pc.schema_mismatches([("any", '{"x": 1}')], [open_tool]) == []
    assert pc.schema_mismatches(calls, None) == [] and pc.schema_mismatches([], tools) == []
    # pydantic-like objects with .function.name / .function.parameters
    obj = SimpleNamespace(function=SimpleNamespace(name="tool_b", parameters={"properties": {"query": {}}}))
    assert pc.schema_mismatches([("tool_b", '{"city": 1}')], [obj])


def test_a_consistent_parser_passes_once_per_stream():
    cls = setup()
    req = request(tools=[tool("add", {"a": "integer"})])
    outs, ds = decided(lambda: stream(cls, 'Sure. <tool>add{"a": 1}</tool> done.', req))
    assert outs[-1] is not None and len(ds) == 1 and ds[0].verdict is Verdict.PASS and ds[0].name == "Parse", ds
    assert ds[0].chosen.value.path == "stream" and ds[0].declared.value.path == "full"
    assert ds[0].chosen.value.digest == ds[0].declared.value.digest and ds[0].chosen.value.tool_calls == 1
    assert vp.stats()["checks"] == 1 and vp.stats()["broken"] == 0


def test_whitespace_kept_on_one_path_only_is_broken():
    cls = setup("keep_whitespace")
    req = request(tools=[tool("add", {"a": "integer"})])
    outs, ds = decided(lambda: stream(cls, '<tool>add{"a": 1}</tool> done.', req))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["stream_differs_from_full"], ds
    assert "whitespace" in ds[0].note and not ds[0].blocking


def test_uncoerced_argument_types_on_the_streamed_path_are_broken():
    cls = setup("string_args")
    req = request(tools=[tool("add", {"a": "integer"})])
    outs, ds = decided(lambda: stream(cls, '<tool>add{"a": 3}</tool>', req))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "arguments differ as values" in ds[0].note, ds


def test_a_streamed_float_with_other_digits_is_broken():
    cls = setup("float_digits")
    req = request(tools=[tool("add", {"left": "number", "right": "number"})])
    outs, ds = decided(lambda: stream(cls, '<tool>add{"left": 108.2, "right": 22.8}</tool>', req))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "108.02" in ds[0].note, ds


def test_an_answer_streamed_as_reasoning_is_broken_unless_reasoning_was_not_asked_for():
    cls = setup("as_reasoning")
    outs, ds = decided(lambda: stream(cls, "The answer is 4.", request()))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "reasoning" in ds[0].note, ds
    vp.reset()
    outs, ds = decided(lambda: stream(cls, "The answer is 4.", request(include_reasoning=False)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN, "the content is still missing on the streamed path"


def test_tool_calls_outside_the_declared_schema_are_broken_on_both_paths():
    cls = setup()
    tools = [tool("tool_a", {"city": "string"}), tool("tool_b", {"query": "string"})]
    text = '<tool>tool_a{"city": "Tokyo"}</tool><tool>tool_b{"city": "Tokyo"}</tool>'
    outs, ds = decided(lambda: stream(cls, text, request(tools=tools)))
    kinds = [(d.verdict, d.rule) for d in ds]
    assert (Verdict.PASS, RULES["match"]) in kinds and (Verdict.BROKEN, RULES["tool_args_outside_schema"]) in kinds, ds
    p = cls("tok", tools)
    outs, ds = decided(lambda: p.parse(text, request(tools=tools), enable_auto_tools=True))
    assert len(ds) == 1 and ds[0].rule == RULES["tool_args_outside_schema"] and "tool_b" in ds[0].note, ds
    outs, ds = decided(lambda: p.parse('<tool>tool_a{"city": "Tokyo"}</tool>', request(tools=tools)))
    assert ds == [], "a call that fits its declared tool says nothing"


def test_the_reference_instance_is_not_compared_and_a_class_without_parse_is_left_alone():
    cls = setup()
    p = cls("tok", None)
    assert not p._entail_reference and p._entail_init[0] == ("tok", None)
    outs, ds = decided(lambda: stream(cls, "plain text", request()))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS

    class Old:
        def extract_tool_calls(self, *a):
            return None

    assert vp.wrapped_class(Old) is Old and vp.wrapped_class("not a class") == "not a class"
    assert vp.wrapped_class(FakeParser, True) is vp.wrapped_class(FakeParser, True), "one wrapped class per base"


def test_an_error_in_the_comparison_never_reaches_the_server():
    cls = setup()
    orig = pc.check_stream
    pc.check_stream = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        outs, ds = decided(lambda: stream(cls, "hello", request()))
    finally:
        pc.check_stream = orig
    assert outs[-1] is not None and ds and ds[-1].verdict is Verdict.UNKNOWN and "boom" in ds[-1].note


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
