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


def tool(name, props, closed=False, required=()):
    params = {"type": "object", "properties": {k: {"type": t} for k, t in props.items()}}
    if closed:
        params["additionalProperties"] = False
    if required:
        params["required"] = list(required)
    return {"type": "function", "function": {"name": name, "parameters": params}}


def request(tools=None, include_reasoning=True, max_tokens=None, tool_choice=None):
    return SimpleNamespace(tools=tools, include_reasoning=include_reasoning, max_tokens=max_tokens,
                           max_completion_tokens=None, tool_choice=tool_choice)


def delta(content=None, reasoning=None, tool_calls=()):
    return SimpleNamespace(content=content, reasoning=reasoning, tool_calls=list(tool_calls))


def tool_delta(index, name=None, arguments=None):
    return SimpleNamespace(index=index, id=None, function=SimpleNamespace(name=name, arguments=arguments))


class FakeParser:
    """vLLM 0.30's parser shape. The text: content with <tool>name{json}</tool> calls and an optional
    <think>...</think> prefix. `variant` makes the streaming path deviate as the real bugs did."""
    variant = "consistent"
    built = 0

    def __init__(self, tokenizer, tools=None, chat_template_kwargs=None, model_config=None):
        FakeParser.built += 1
        if tokenizer == "boom" and FakeParser.built % 2 == 0:     # every second build is the reference's
            raise RuntimeError("no second parser")
        self.tokenizer, self.tools, self.buffer = tokenizer, tools, []
        self._reasoning_parser = object()
        self._stream_state = SimpleNamespace(reasoning_ended=True)

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
        if self.variant == "promoted":                 # vLLM's fallback: the open reasoning sent again as content
            self._stream_state.reasoning_ended = False
            return delta(content=content, reasoning=content)
        if self.variant == "open_reasoning":           # the block never closed and nothing was promoted
            self._stream_state.reasoning_ended = False
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


def stream(cls, text, req, chunks=7, tokenizer="tok"):
    """Drive a wrapped parser as the server does: deltas, the last one finished."""
    p = cls(tokenizer, req.tools, chat_template_kwargs={}, model_config=None)
    pieces = [text[i:i + chunks] for i in range(0, len(text), chunks)] or [""]
    outs = []
    for i, piece in enumerate(pieces):
        outs.append(p.parse_delta(piece, [i], req, [1, 2], finished=(i == len(pieces) - 1)))
    return outs


def setup(variant="consistent"):
    vp.reset()
    FakeParser.variant = variant
    FakeParser.built = 0
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
    assert pc.differences(s.message(), full) == ([], []) and pc.digest_of(s.message()) == pc.digest_of(full)


def test_differences_name_what_differs():
    base = {"content": "done.", "reasoning": None, "tool_calls": [("add", '{"a": 1}')]}
    d, notes = pc.differences(dict(base, content=" done."), base)
    assert d == [] and notes and "whitespace only" in notes[0], "surrounding whitespace is a note, not a difference"
    d, notes = pc.differences(dict(base, tool_calls=[("add", '{"a": "1"}')]), base)
    assert d and "arguments differ as values" in d[0] and '"1"' in d[0]
    d, notes = pc.differences(dict(base, content=None, reasoning="done."), base)
    assert d and any("streamed as reasoning what the whole parse calls content" in x for x in d)
    d, notes = pc.differences(dict(base, reasoning="done."), base)
    assert d == [] and notes and "again as content" in notes[0], "reasoning promoted to content by the fallback: a note"
    d, notes = pc.differences(dict(base, tool_calls=[]), base)
    assert d == ["0 tool calls streamed, 1 in the whole parse"]
    assert pc.differences(dict(base, reasoning="x"), base, compare_reasoning=False) == ([], [])
    d, notes = pc.differences(dict(base, content="all done."), base)
    assert d and "content differs from character 0" in d[0]


def test_schema_mismatches_against_the_declared_tools():
    tools = [tool("get_weather", {"city": "string"}), tool("tool_b", {"query": "string"}, closed=True)]
    calls = [("get_weather", '{"city": "Tokyo"}'), ("tool_b", '{"city": "Tokyo"}')]
    bad, notes = pc.schema_mismatches(calls, tools)
    assert len(bad) == 1 and "tool call 1 (tool_b) carries ['city']" in bad[0] and "forbids" in bad[0], bad
    assert notes == []
    bad, notes = pc.schema_mismatches(calls, tools, raw_text='<tool>tool_b{"city": "Tokyo"}</tool>')
    assert "the model's text carries it" in bad[0]
    bad, notes = pc.schema_mismatches(calls, tools, raw_text='<tool>tool_b{"town": "Tokyo"}</tool>')
    assert "the parser added it" in bad[0]
    bad, notes = pc.schema_mismatches([("get_weather", '{"city": "Tokyo", "units": "C"}')], tools)
    assert bad == [] and len(notes) == 1 and "does not forbid additional properties" in notes[0], notes
    bad, notes = pc.schema_mismatches([("tool_b", '{}')], [tool("tool_b", {"query": "string"}, required=["query"])])
    assert len(bad) == 1 and "lacks its required ['query']" in bad[0], bad
    assert pc.schema_mismatches([("other", "{}")], tools)[0][0].startswith("tool call 0 names 'other'")
    assert pc.schema_mismatches([("tool_b", "not json")], tools) == ([], [])
    open_tool = {"type": "function", "function": {"name": "any", "parameters": {"type": "object", "properties": {},
                                                                                "additionalProperties": True}}}
    assert pc.schema_mismatches([("any", '{"x": 1}')], [open_tool]) == ([], [])
    assert pc.schema_mismatches(calls, None) == ([], []) and pc.schema_mismatches([], tools) == ([], [])
    # pydantic-like objects with .function.name / .function.parameters
    obj = SimpleNamespace(function=SimpleNamespace(name="tool_b", parameters={"properties": {"query": {}},
                                                                             "additionalProperties": False}))
    assert pc.schema_mismatches([("tool_b", '{"city": 1}')], [obj])[0]


def test_unfinished_outputs_are_named():
    streamed = {"tool_calls": [("add", '{"a": 1')]}
    assert pc.unfinished(True, False, False, streamed).startswith("the stream reached")
    assert "reasoning block" in pc.unfinished(False, True, False, streamed)
    assert "never became valid JSON" in pc.unfinished(False, False, True, streamed)
    assert pc.unfinished(False, False, True, {"tool_calls": [("add", '{"a": 1}')]}) is None
    assert pc.unfinished(False, False, False, streamed) is None


def test_a_consistent_parser_passes_once_per_stream_and_is_counted_not_recorded():
    cls = setup()
    req = request(tools=[tool("add", {"a": "integer"})])
    outs, ds = decided(lambda: stream(cls, 'Sure. <tool>add{"a": 1}</tool> done.', req))
    assert outs[-1] is not None and ds == [], "a silent pass is the request path's ordinary case: counted only"
    assert vp.stats()["checks"] == 2 and vp.stats()["broken"] == 0, vp.stats()   # the stream and its schema


def test_whitespace_kept_on_one_path_only_is_noted_not_broken():
    """M18.5: on a live server every tool-call stream of vLLM's hermes parser streamed two newlines as content and
    parsed nothing whole; a newline loses no meaning, so surrounding whitespace is a note on a recorded pass."""
    cls = setup("keep_whitespace")
    req = request(tools=[tool("add", {"a": "integer"})])
    outs, ds = decided(lambda: stream(cls, '<tool>add{"a": 1}</tool> done.', req))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS and "whitespace only" in ds[0].note, ds
    assert ds[0].chosen.value.path == "stream" and ds[0].declared.value.path == "full"


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


def test_reasoning_the_fallback_promoted_to_content_is_a_note_and_an_open_block_is_unknown():
    cls = setup("promoted")
    outs, ds = decided(lambda: stream(cls, "The answer is 4.", request()))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS and "again as content" in ds[0].note, ds
    cls = setup("open_reasoning")
    outs, ds = decided(lambda: stream(cls, "The answer is 4.", request()))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "ended inside its reasoning block" in ds[0].note, ds
    assert "streamed as reasoning what the whole parse calls content" in ds[0].note, "the difference is still named"


def test_an_output_cut_by_the_token_limit_is_unknown_not_broken():
    cls = setup("string_args")
    req = request(tools=[tool("add", {"a": "integer"})], max_tokens=3)
    outs, ds = decided(lambda: stream(cls, '<tool>add{"a": 3}</tool>', req))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "token limit" in ds[0].note and "differ as values" in ds[0].note, ds
    cls = setup("string_args")
    outs, ds = decided(lambda: stream(cls, '<tool>add{"a": 3}</tool>', request(tools=[tool("add", {"a": "integer"})], max_tokens=300)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN, "the limit was not reached: the difference stands"


def test_tool_calls_outside_the_declared_schema_are_broken_on_both_paths():
    cls = setup()
    tools = [tool("tool_a", {"city": "string"}), tool("tool_b", {"query": "string"}, closed=True)]
    text = '<tool>tool_a{"city": "Tokyo"}</tool><tool>tool_b{"city": "Tokyo"}</tool>'
    outs, ds = decided(lambda: stream(cls, text, request(tools=tools)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["tool_args_outside_schema"], ds
    assert "the model's text carries it" in ds[0].note, ds[0].note
    p = cls("tok", tools)
    outs, ds = decided(lambda: p.parse(text, request(tools=tools), enable_auto_tools=True))
    assert len(ds) == 1 and ds[0].rule == RULES["tool_args_outside_schema"] and "tool_b" in ds[0].note, ds
    outs, ds = decided(lambda: p.parse('<tool>tool_a{"city": "Tokyo"}</tool>', request(tools=tools)))
    assert ds == [], "a call that fits its declared tool says nothing"
    open_tools = [tool("tool_a", {"city": "string"})]
    outs, ds = decided(lambda: p.parse('<tool>tool_a{"city": "Tokyo", "units": "C"}</tool>', request(tools=open_tools)))
    assert len(ds) == 1 and ds[0].verdict is Verdict.PASS and "does not forbid additional properties" in ds[0].note, ds


def test_the_reference_instance_is_not_compared_and_a_class_without_parse_is_left_alone():
    cls = setup()
    p = cls("tok", None)
    assert not p._entail_reference and p._entail_init[0] == ("tok", None)
    outs, ds = decided(lambda: stream(cls, "plain text", request()))
    assert ds == [] and vp.stats()["checks"] == 1

    class Old:
        def extract_tool_calls(self, *a):
            return None

    assert vp.wrapped_class(Old) is Old and vp.wrapped_class("not a class") == "not a class"
    assert vp.wrapped_class(FakeParser, True) is vp.wrapped_class(FakeParser, True), "one wrapped class per base"


def test_a_reference_that_cannot_be_built_is_unknown_once_per_class():
    cls = setup()
    outs, ds = decided(lambda: stream(cls, "hello", request(), tokenizer="boom"))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "could not be built" in ds[0].note, ds
    outs, ds = decided(lambda: stream(cls, "hello again", request(), tokenizer="boom"))
    assert ds == [], "said once per class"


def test_an_error_in_the_comparison_never_reaches_the_server():
    cls = setup()
    orig = pc.check_stream
    pc.check_stream = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        outs, ds = decided(lambda: stream(cls, "hello", request()))
    finally:
        pc.check_stream = orig
    assert outs[-1] is not None and ds and ds[-1].verdict is Verdict.UNKNOWN and "boom" in ds[-1].note


def test_logprobs_that_cover_the_parsed_out_reasoning_are_broken():
    """M18.4, sglang#25055: with separate_reasoning the logprobs covered the whole raw output (the <think> span and
    its markers) while message.content held the parsed answer."""
    from entail.adapters import sglang_serve

    B2 = sglang_serve.LOGPROBS
    pc.reset(B2)
    content, reasoning = "The answer is 4.", "\nOkay, 2 + 2.\n"
    tokens = ["<think>", "\n", "Okay", ",", " 2", " +", " 2", ".", "\n", "</think>", "\n\n", "The", " answer", " is", " 4", "."]
    out, ds = decided(lambda: pc.check_logprobs(B2, "sglang.chat_completion", content, reasoning, tokens, "choice 0"))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["logprobs_cover_other_text"], ds
    assert "reasoning span" in ds[0].note and "markers" in ds[0].note and ds[0].chosen.value.content == len("".join(tokens))
    out, ds = decided(lambda: pc.check_logprobs(B2, "sglang.chat_completion", content, None, ["The", " answer", " is", " 4", "."], "choice 0"))
    assert ds == [] and pc.stats(B2)["checks"] == 2 and pc.stats(B2)["broken"] == 1
    out, ds = decided(lambda: pc.check_logprobs(B2, "sglang.chat_completion", content, None, ["The", " answer", " is", " 4", ".", "\n"], "choice 0"))
    assert ds == [], "whitespace at the ends is not held against the logprobs"
    out, ds = decided(lambda: pc.check_logprobs(B2, "sglang.chat_completion", content, None, ["Something", " else"], "choice 0"))
    assert len(ds) == 1 and "not the message's content" in ds[0].note
    assert pc.check_logprobs(B2, "x", content, None, [], "choice 0") == []
    # the SGLang adapter reads a response object or a dict of the same shape
    response = {"choices": [{"index": 0, "message": {"content": content, "reasoning_content": reasoning},
                             "logprobs": {"content": [{"token": t} for t in tokens]}}]}
    out, ds = decided(lambda: sglang_serve.decide_logprobs(response))
    assert len(ds) == 1 and ds[0].verdict is Verdict.BROKEN and "choice 0" in ds[0].note


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
