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
  tool_args_outside_schema   a tool call's arguments carry a key the declared tool's parameters do not have
                             (additionalProperties not allowed), or name a tool the request did not declare

Exact comparison, no tolerance: the same text through the same parser. Argument strings are compared as parsed
JSON (values, not spacing); when either side is not JSON, as text. A difference is `broken` (reported, the run goes
on; the client already has the streamed message, so nothing is repaired). A request without tools decides nothing
about schemas, and a request that asked for no reasoning is compared without it. Nothing here reads a device; an
error inside entail never breaks the server (the adapter runs this under load.safely).
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


def differences(streamed: dict, full: dict, compare_reasoning: bool = True) -> List[str]:
    """How the streamed message differs from the whole-text one, in words; empty when they agree."""
    out = []
    if (streamed.get("content") or "") != (full.get("content") or ""):
        a, b = streamed.get("content") or "", full.get("content") or ""
        if a.strip() == b.strip():
            out.append(f"content differs in surrounding whitespace only: streamed {_show(a)}, whole {_show(b)}")
        else:
            i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
            out.append(f"content differs from character {i}: streamed {_show(a[i:i + 40])}, whole "
                       f"{_show(b[i:i + 40])} ({len(a)} against {len(b)} characters)")
    if compare_reasoning and (streamed.get("reasoning") or "") != (full.get("reasoning") or ""):
        a, b = streamed.get("reasoning") or "", full.get("reasoning") or ""
        out.append(f"reasoning differs: streamed {len(a)} characters, whole {len(b)}"
                   + (" (streamed as reasoning what the whole parse calls content)"
                      if a and not b and (full.get("content") or "").strip() == a.strip() else ""))
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
    return out


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


def schema_mismatches(tool_calls: List[Tuple[str, str]], tools) -> List[str]:
    """The tool calls whose arguments do not fit the declared tools: an undeclared tool name, or a key the tool's
    parameters do not declare when additionalProperties is not allowed. Non-JSON arguments and tools without a
    properties table decide nothing."""
    declared = {}
    for t in tools or []:
        name, params = _tool_schema(t)
        if name:
            declared[name] = params
    if not declared:
        return []
    out = []
    for k, (name, args) in enumerate(tool_calls or []):
        if name not in declared:
            out.append(f"tool call {k} names {name!r}, which the request did not declare (declared: "
                       f"{', '.join(sorted(declared))})")
            continue
        params = declared[name] or {}
        props = params.get("properties")
        if not isinstance(props, dict) or params.get("additionalProperties") is True:
            continue
        kind, v = _value(args)
        if kind != "json" or not isinstance(v, dict):
            continue
        extra = sorted(set(v) - set(props))
        if extra:
            out.append(f"tool call {k} ({name}) carries {extra}, which its declared parameters "
                       f"({', '.join(sorted(props)) or 'none'}) do not have")
    return out


def _fact(path: str, msg: dict, where: str, certainty):
    from .facts import Fact, Parse, Source

    return Fact("Parse", Parse(path=path, content=len(msg.get("content") or ""),
                               reasoning=len(msg.get("reasoning") or ""), tool_calls=len(msg.get("tool_calls") or []),
                               digest=digest_of(msg)), Source("engine", where), certainty)


def check_stream(boundary: str, consumer: str, streamed: dict, full: dict, where: str, policy=None,
                 compare_reasoning: bool = True, record: bool = True) -> list:
    """Decide the streamed message against the whole-text message of the same parser on the same text."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty

    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("Parse",))
    declared = _fact("full", full, f"{where}: the whole text parsed once", Certainty.VERIFIED)
    held = _fact("stream", streamed, f"{where}: the streamed deltas", Certainty.VERIFIED)
    diffs = differences(streamed, full, compare_reasoning)
    if diffs:
        verdict, blocking = unrepaired(policy, "Parse")
        d = Decision(contract, "Parse", verdict, RULES["stream_differs_from_full"], declared=declared, chosen=held,
                     blocking=blocking, note=f"{where}: " + "; ".join(diffs))
    else:
        d = Decision(contract, "Parse", Verdict.PASS, RULES["match"], declared=declared, chosen=held,
                     note=f"{where}: the streamed message and the whole-text message agree")
    if record:
        _tally.counts(boundary)["checks"] += 1
        if d.verdict is Verdict.PASS:
            _tally.passed(boundary, ["parse"])
        if d.blocking:
            _tally.refused(boundary)
        elif d.verdict is Verdict.BROKEN:
            _tally.broken(boundary)
        _tally.tick(boundary)
        load.enforce([d])
    return [d]


def check_schema(boundary: str, consumer: str, tool_calls: List[Tuple[str, str]], tools, where: str, policy=None,
                 record: bool = True) -> list:
    """Decide the tool calls handed on against the tools the request declared. No tools, no calls: nothing."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty

    if not tool_calls or not tools:
        return []
    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("Parse",))
    msg = {"content": None, "reasoning": None, "tool_calls": list(tool_calls)}
    held = _fact("full", msg, where, Certainty.VERIFIED)
    bad = schema_mismatches(tool_calls, tools)
    if not bad:
        return []          # fitting the schema is the ordinary case; only a break is said
    verdict, blocking = unrepaired(policy, "Parse")
    d = Decision(contract, "Parse", verdict, RULES["tool_args_outside_schema"], chosen=held, blocking=blocking,
                 note=f"{where}: " + "; ".join(bad))
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
