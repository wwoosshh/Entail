"""parse_contract: what a parser makes of a model's text on the streaming path against the whole-text path, and
the tool calls it hands on against the tools the request declared (LIBRARY_DESIGN.md 11, M18; ROADMAP M18.3).

A chat server parses the model's text twice over: token by token while streaming (vLLM's parse_delta), and whole
when not (parse). Both are the same declaration - this parser, this request - and must yield the same message: the
same content, the same reasoning, the same tool calls with the same argument values. They did not in vllm#49316
(the streaming path skipped the schema's type coercion: "3" against 3), #49412 (whitespace around tool calls
stripped on one path only), #48217 (a plain answer classified as reasoning while streaming and as content whole),
#42047 (108.2 streamed as 108.02). Nothing in the engine compares the two. This contract does: the deltas of a
streamed response are accumulated, and when the stream finishes the whole text is parsed again by a fresh parser
of the same class, and the two results must agree. And the arguments of a tool call must fit the tool the request
declared (vllm#47986: two calls with the same raw arguments shared one slot and one was unwrapped with the other
tool's schema, so a tool with no `city` parameter received {"city": ...}). The rules are here, once:

  stream_differs_from_full   the streamed message and the whole-text message differ (which part, and how)
  tool_args_outside_schema   a tool call names a tool the request did not declare, lacks a parameter the tool
                             declares required, or carries a key the tool's parameters do not have when the tool
                             forbids additional properties (additionalProperties false; JSON Schema allows them
                             by default, so an extra key under a tool that did not forbid it is a note on a pass)
  logprobs_cover_other_text  the response's logprobs decode to other text than its content (sglang#25055: with
                             separate_reasoning the logprobs covered the <think> span the server had parsed out,
                             so they cannot be aligned with the message)

Exact comparison, no tolerance: the same text through the same parser. Argument strings are compared as parsed
JSON (values, not spacing); when either side is not JSON, as text. Content that differs in surrounding whitespace
only is noted on a pass, not broken: vLLM's tool parsers strip the text around a tool call on one path and not the
other on every healthy tool-call stream, and a newline loses no meaning. Reasoning that the stream sent again as
content (vLLM's fallback when the output ends inside its reasoning) is a note too. An output that did not finish
by itself - the token limit reached, the reasoning block still open, a forced tool whose arguments never became
JSON - is where vLLM documents its two paths to differ on the unfinished part, so a difference there is unknown,
not broken. A difference is `broken` (reported, the run goes on; the client already has the streamed message, so
nothing is repaired). A request without tools decides nothing about schemas, and a request that asked for no
reasoning is compared without it. A silent pass is counted, not recorded (one line per stream would be the
request path's cost); a pass with a note is recorded. Whether an extra key came from the model's text or from the
parser is said when the text is at hand; which of the model and the declared tool is wrong is not decided here.
Nothing here reads a device; an error inside entail never breaks the server (the adapter runs this under
load.safely).
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import tally as _tally

RULE_NAMES = ("stream_differs_from_full", "tool_args_outside_schema")


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class Streamed:
    """What the streaming path handed on, accumulated over its deltas."""
    text: List[str] = field(default_factory=list)          # the model's raw text, delta by delta
    ids: List[int] = field(default_factory=list)
    content: List[str] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    tools: Dict[int, dict] = field(default_factory=dict)   # index -> {"name", "args": [pieces]}
    deltas: int = 0

    def add(self, delta) -> None:
        """Take one delta message (vLLM's DeltaMessage: content, reasoning, tool_calls with index and function)."""
        if delta is None:
            return
        self.deltas += 1
        c = _get(delta, "content")
        if c:
            self.content.append(c)
        r = _get(delta, "reasoning")
        if r is None:
            r = _get(delta, "reasoning_content")
        if r:
            self.reasoning.append(r)
        for tc in _get(delta, "tool_calls", None) or []:
            index = _get(tc, "index", 0)
            slot = self.tools.setdefault(int(index if index is not None else 0), {"name": None, "args": []})
            fn = _get(tc, "function")
            name = _get(fn, "name") if fn is not None else None
            if name and not slot["name"]:
                slot["name"] = name
            args = _get(fn, "arguments") if fn is not None else None
            if args:
                slot["args"].append(args)

    def message(self) -> dict:
        return {"content": "".join(self.content) if self.content else None,
                "reasoning": "".join(self.reasoning) if self.reasoning else None,
                "tool_calls": [(self.tools[i]["name"], "".join(self.tools[i]["args"])) for i in sorted(self.tools)]}


def full_message(reasoning, content, tool_calls) -> dict:
    """The whole-text path's result as the same shape: (name, arguments) per tool call, from vLLM's FunctionCall
    (name, arguments) or a dict or an object with .function."""
    calls = []
    for tc in tool_calls or []:
        fn = _get(tc, "function")
        name = _get(fn, "name") if fn is not None else _get(tc, "name")
        args = _get(fn, "arguments") if fn is not None else _get(tc, "arguments")
        calls.append((name, args if isinstance(args, str) else (json.dumps(args) if args is not None else "")))
    return {"content": content if content else None, "reasoning": reasoning if reasoning else None,
            "tool_calls": calls}


def _value(s: Optional[str]) -> Tuple[str, Any]:
    if s is None:
        return "text", ""
    try:
        return "json", json.loads(s)
    except (ValueError, TypeError):
        return "text", s.strip()


def _show(s: Optional[str], n: int = 60) -> str:
    return repr(s if s is not None else None)[:n]


def differences(streamed: dict, full: dict, compare_reasoning: bool = True) -> Tuple[List[str], List[str]]:
    """How the streamed message differs from the whole-text one: (differences that change what the client gets,
    notes that do not). Content or reasoning that differs in surrounding whitespace only is a note, not a
    difference: vLLM's tool parsers strip the text around a tool call on the whole-text path and stream it as it
    comes, on every tool-call stream of healthy traffic (M18.5: the hermes parser streamed two newlines and parsed
    nothing whole, in two of two tool-call requests), and no meaning is lost in a newline."""
    out, notes = [], []
    for field in ("content",) + (("reasoning",) if compare_reasoning else ()):
        a, b = streamed.get(field) or "", full.get(field) or ""
        if a == b:
            continue
        if a.strip() == b.strip():
            notes.append(f"{field} differs in surrounding whitespace only: streamed {_show(a)}, whole {_show(b)}")
        elif field == "reasoning" and a and not b and (full.get("content") or "").strip() == a.strip():
            if (streamed.get("content") or "").strip() == a.strip():
                notes.append(f"the stream sent its reasoning ({len(a)} characters) again as content, the parser's "
                             f"fallback for an output that ended inside its reasoning; the whole parse calls it content")
            else:
                out.append(f"reasoning differs: streamed {len(a)} characters, whole {len(b)} (streamed as reasoning "
                           f"what the whole parse calls content)")
        elif field == "reasoning":
            out.append(f"reasoning differs: streamed {len(a)} characters, whole {len(b)}")
        else:
            i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            out.append(f"content differs from character {i}: streamed {_show(a[i:i + 40])}, whole "
                       f"{_show(b[i:i + 40])} ({len(a)} against {len(b)} characters)")
    sc, fc = streamed.get("tool_calls") or [], full.get("tool_calls") or []
    if len(sc) != len(fc):
        out.append(f"{len(sc)} tool calls streamed, {len(fc)} in the whole parse")
    for k, ((sn, sa), (fn, fa)) in enumerate(zip(sc, fc)):
        if (sn or "") != (fn or ""):
            out.append(f"tool call {k}: streamed name {sn!r}, whole {fn!r}")
            continue
        (sk, sv), (fk, fv) = _value(sa), _value(fa)
        if sk == "json" and fk == "json":
            if sv != fv:
                out.append(f"tool call {k} ({sn}): arguments differ as values: streamed {_show(json.dumps(sv))}, "
                           f"whole {_show(json.dumps(fv))}")
        elif (sa or "").strip() != (fa or "").strip():
            out.append(f"tool call {k} ({sn}): arguments differ: streamed {_show(sa)}, whole {_show(fa)}")
    return out, notes


def digest_of(msg: dict) -> str:
    calls = []
    for name, args in msg.get("tool_calls") or []:
        kind, v = _value(args)
        calls.append([name, json.dumps(v, sort_keys=True) if kind == "json" else v])
    body = json.dumps([msg.get("content") or "", msg.get("reasoning") or "", calls], ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _tool_schema(t) -> Tuple[Optional[str], Optional[dict]]:
    fn = _get(t, "function")
    if fn is None:
        return _get(t, "name"), _get(t, "parameters")
    params = _get(fn, "parameters")
    if params is not None and not isinstance(params, dict) and hasattr(params, "model_dump"):
        params = params.model_dump()
    return _get(fn, "name"), params if isinstance(params, dict) else None


def _origin(keys: List[str], raw_text: Optional[str]) -> str:
    """Where a key the declared tool does not name came from, when the model's text is at hand."""
    if raw_text is None:
        return ""
    if all(f'"{k}"' in raw_text or f"'{k}'" in raw_text or f"<parameter name=\"{k}\"" in raw_text
           or f"<parameter={k}>" in raw_text for k in keys):
        return " (the model's text carries it: the model's call does not fit the declared tool, or the parser reshaped it)"
    return " (the model's text does not carry it: the parser added it)"


def schema_mismatches(tool_calls: List[Tuple[str, str]], tools, raw_text: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """(breaks, notes): the tool calls whose arguments do not fit the declared tools. Breaks: an undeclared tool
    name, a required parameter missing, a key the tool's parameters do not declare when the tool forbids additional
    properties. Notes: an extra key under a tool that did not forbid it (JSON Schema's default allows it). Non-JSON
    arguments decide nothing, and tools without a properties table only their required list."""
    declared = {}
    for t in tools or []:
        name, params = _tool_schema(t)
        if name:
            declared[name] = params
    if not declared:
        return [], []
    out, notes = [], []
    for k, (name, args) in enumerate(tool_calls or []):
        if name not in declared:
            out.append(f"tool call {k} names {name!r}, which the request did not declare (declared: "
                       f"{', '.join(sorted(declared))})")
            continue
        params = declared[name] or {}
        kind, v = _value(args)
        if kind != "json" or not isinstance(v, dict):
            continue
        required = [r for r in (params.get("required") or []) if isinstance(r, str)]
        missing = [r for r in required if r not in v]
        if missing:
            out.append(f"tool call {k} ({name}) lacks its required {missing} (it carries {sorted(v) or 'nothing'})")
        props = params.get("properties")
        if not isinstance(props, dict) or params.get("additionalProperties") is True:
            continue
        extra = sorted(set(v) - set(props))
        if not extra:
            continue
        named = ', '.join(sorted(props)) or 'none'
        if params.get("additionalProperties") is False:
            out.append(f"tool call {k} ({name}) carries {extra}, which its declared parameters ({named}) do not "
                       f"have and the tool forbids{_origin(extra, raw_text)}")
        else:
            notes.append(f"tool call {k} ({name}) carries {extra} beyond its declared parameters ({named}); the tool "
                         f"does not forbid additional properties{_origin(extra, raw_text)}")
    return out, notes


def unfinished(limit_reached: bool, reasoning_open: bool, forced_tool: bool, streamed: dict) -> Optional[str]:
    """Why the output did not finish by itself, from what the adapter read: the request's token limit reached, the
    reasoning block still open at the end, or a forced tool (tool_choice named or required) whose streamed
    arguments never became JSON - vLLM's whole-text path then returns no call. None when it finished."""
    if limit_reached:
        return "the stream reached the request's token limit"
    if reasoning_open:
        return "the output ended inside its reasoning block"
    if forced_tool and any(_value(a)[0] != "json" for _, a in (streamed.get("tool_calls") or [])):
        return "tool_choice forced a tool and the streamed arguments never became valid JSON"
    return None


def _fact(path: str, msg: dict, where: str, certainty):
    from .facts import Fact, Parse, Source

    return Fact("Parse", Parse(path=path, content=len(msg.get("content") or ""),
                               reasoning=len(msg.get("reasoning") or ""), tool_calls=len(msg.get("tool_calls") or []),
                               digest=digest_of(msg)), Source("engine", where), certainty)


def _record(boundary: str, d, rules, noted: bool = False) -> None:
    """Count a decision and record it - except a silent pass, which is counted only (the request path's cost)."""
    from . import load
    from .contracts import Verdict

    _tally.counts(boundary)["checks"] += 1
    if d.verdict is Verdict.PASS:
        _tally.passed(boundary, rules)
    if d.blocking:
        _tally.refused(boundary)
    elif d.verdict is Verdict.BROKEN:
        _tally.broken(boundary)
    _tally.tick(boundary)
    if d.verdict is not Verdict.PASS or noted:
        load.enforce([d])


def check_stream(boundary: str, consumer: str, streamed: dict, full: dict, where: str, policy=None,
                 compare_reasoning: bool = True, record: bool = True, unfinished_why: Optional[str] = None) -> list:
    """Decide the streamed message against the whole-text message of the same parser on the same text. With
    `unfinished_why` (the output did not finish by itself) a difference is unknown, not broken. A silent pass is
    counted, not recorded; a pass with notes is recorded."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty

    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("Parse",))
    declared = _fact("full", full, f"{where}: the whole text parsed once", Certainty.VERIFIED)
    held = _fact("stream", streamed, f"{where}: the streamed deltas", Certainty.VERIFIED)
    diffs, notes = differences(streamed, full, compare_reasoning)
    if diffs and unfinished_why:
        d = load.cannot_check(boundary, consumer, "Parse",
                              f"{where}: the output did not finish by itself ({unfinished_why}), where vLLM's "
                              f"streaming and whole-text paths are documented to differ on the unfinished part, so "
                              f"the difference is not held against the parser: " + "; ".join(diffs + notes), policy)
    elif diffs:
        verdict, blocking = unrepaired(policy, "Parse")
        d = Decision(contract, "Parse", verdict, RULES["stream_differs_from_full"], declared=declared, chosen=held,
                     blocking=blocking, note=f"{where}: " + "; ".join(diffs + notes))
    else:
        d = Decision(contract, "Parse", Verdict.PASS, RULES["match"], declared=declared, chosen=held,
                     note=f"{where}: the streamed message and the whole-text message agree"
                          + ("; " + "; ".join(notes) if notes else ""))
    if record:
        _record(boundary, d, ["parse"], noted=bool(notes))
    return [d]


def check_schema(boundary: str, consumer: str, tool_calls: List[Tuple[str, str]], tools, where: str, policy=None,
                 record: bool = True, raw_text: Optional[str] = None) -> list:
    """Decide the tool calls handed on against the tools the request declared. No tools, no calls: nothing. A
    call that fits is counted, not recorded; an extra key the tool did not forbid is a note on a recorded pass."""
    from . import policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty

    if not tool_calls or not tools:
        return []
    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("Parse",))
    msg = {"content": None, "reasoning": None, "tool_calls": list(tool_calls)}
    held = _fact("full", msg, where, Certainty.VERIFIED)
    bad, notes = schema_mismatches(tool_calls, tools, raw_text)
    if bad:
        verdict, blocking = unrepaired(policy, "Parse")
        d = Decision(contract, "Parse", verdict, RULES["tool_args_outside_schema"], chosen=held, blocking=blocking,
                     note=f"{where}: " + "; ".join(bad + notes))
    else:
        d = Decision(contract, "Parse", Verdict.PASS, RULES["match"], chosen=held,
                     note=f"{where}: the tool calls fit the declared tools" + ("; " + "; ".join(notes) if notes else ""))
    if record:
        _record(boundary, d, ["schema"], noted=bool(notes))
    return [d] if (bad or notes) else []


def check_logprobs(boundary: str, consumer: str, content: Optional[str], reasoning: Optional[str],
                   tokens: List[str], where: str, policy=None, record: bool = True) -> list:
    """Decide a response's logprobs against its message (M18.4; sglang#25055): the logprob tokens decode to the
    message's content, or the client cannot align them. A response whose logprobs also cover the reasoning span
    (and its markers) that the server parsed out of the content is broken. Whitespace at the ends is not held
    against it. Returns the non-pass decisions (nothing is recorded on a pass)."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty

    if not tokens:
        return []
    joined, c, r = "".join(tokens), content or "", reasoning or ""
    if joined == c or joined.strip() == c.strip():
        if record:                    # a silent pass: counted, not recorded (one line per response otherwise)
            _tally.counts(boundary)["checks"] += 1
            _tally.passed(boundary, ["logprobs"])
            _tally.tick(boundary)
        return []
    if r and r.strip() and r.strip() in joined and c.strip() in joined:
        markers = " and its markers" if "<think>" in joined or "</think>" in joined else ""
        what = (f"the {len(tokens)} logprob tokens cover the reasoning span ({len(r)} characters{markers}) as well "
                f"as the content ({len(c)} characters), so they cannot be aligned with the message's content")
    else:
        what = (f"the {len(tokens)} logprob tokens decode to {len(joined)} characters that are not the message's "
                f"content ({len(c)} characters)")
    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("Parse",))
    declared = _fact("full", {"content": c, "reasoning": r, "tool_calls": []}, f"{where}: the message", Certainty.VERIFIED)
    held = _fact("full", {"content": joined, "reasoning": None, "tool_calls": []}, f"{where}: what the logprobs cover",
                 Certainty.VERIFIED)
    verdict, blocking = unrepaired(policy, "Parse")
    d = Decision(contract, "Parse", verdict, RULES["logprobs_cover_other_text"], declared=declared, chosen=held,
                 blocking=blocking, note=f"{where}: {what}")
    if record:
        _tally.counts(boundary)["checks"] += 1
        if d.blocking:
            _tally.refused(boundary)
        else:
            _tally.broken(boundary)
        _tally.tick(boundary)
        load.enforce([d])
    return [d]


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
