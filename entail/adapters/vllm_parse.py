"""Adapter v2 for vLLM's chat parsers: the streamed message against the whole-text message, and tool calls against
the declared tools (LIBRARY_DESIGN.md 11 M18; ROADMAP M18.3; parse_contract.py; vllm#49316, #49412, #48217,
#47986).

  hook         vllm.parser.parser_manager.ParserManager.get_parser: the class the server builds a parser from, per
               request (0.30: one Parser with parse_delta for streaming and parse for a whole text). The class is
               returned wrapped: its instances remember how they were built, accumulate what parse_delta hands on,
               and when the stream finishes (finished=True) parse the whole text again on a fresh instance of the
               same class and compare. parse (the whole-text path) checks its tool calls against the request's tools.
  read_choice  the accumulated streamed message and the whole-text message; the rules are in the core.
  handles      none: the client already has the streamed message.
The wrapper binds nothing by position it does not need: every argument is passed through whole (principle 12; the
serve adapter's wrapper of the same classmethod is composed with, whichever was installed first). Parsers built
elsewhere than through get_parser (a test driving a parser class directly) are not seen; a parser class without
parse (before 0.30's unified parsers) is left as it is.
"""
from .. import core, parse_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "request:vllm.parser"
CONSUMER = "vllm.parser"
_ORIG = {}
_CLASSES = {}


def hooks():
    return [Hook("vllm.parser.parser_manager.ParserManager.get_parser", "request")]


def handles():
    return {}


def _active():
    return core.mode() in ("load", "debug")


def read_choice(parser):
    """The streamed state a wrapped parser holds (None outside a stream)."""
    return getattr(parser, "_entail_stream", None)


def _on_delta(parser, base_name, args, kwargs, out):
    names = ("delta_text", "delta_token_ids", "request", "prompt_token_ids")   # parse_delta's parameters, 0.30
    bound = dict(zip(names, args))
    bound.update({k: v for k, v in kwargs.items() if k in names})
    state = parser._entail_stream or parse_contract.Streamed()
    state.text.append(bound.get("delta_text") or "")
    state.ids.extend(int(i) for i in (bound.get("delta_token_ids") or []))
    state.add(out)
    parser._entail_stream = state
    if not kwargs.get("finished", False):
        return
    parser._entail_stream = None
    request = bound.get("request")
    consumer = f"vllm.parser.{base_name}"
    init_args, init_kwargs = getattr(parser, "_entail_init", ((), {}))
    fresh = type(parser)(*init_args, **init_kwargs)
    fresh._entail_reference = True
    reasoning, content, tool_calls = fresh.parse("".join(state.text), request,
                                                 enable_auto_tools=getattr(parser, "_entail_auto_tools", False),
                                                 model_output_token_ids=state.ids)
    streamed, full = state.message(), parse_contract.full_message(reasoning, content, tool_calls)
    where = f"{base_name} on a streamed response of {len(state.ids)} tokens ({state.deltas} deltas)"
    compare_reasoning = getattr(request, "include_reasoning", True) is not False
    parse_contract.check_stream(BOUNDARY, consumer, streamed, full, where, compare_reasoning=compare_reasoning)
    parse_contract.check_schema(BOUNDARY, consumer, streamed["tool_calls"], getattr(request, "tools", None),
                                f"{base_name}'s streamed tool calls")


def _on_full(parser, base_name, args, kwargs, out):
    request = args[1] if len(args) > 1 else kwargs.get("request")
    try:
        reasoning, content, tool_calls = out
    except (TypeError, ValueError):
        return
    full = parse_contract.full_message(reasoning, content, tool_calls)
    parse_contract.check_schema(BOUNDARY, f"vllm.parser.{base_name}", full["tool_calls"],
                                getattr(request, "tools", None), f"{base_name}'s tool calls (whole text)")


def wrapped_class(base, auto_tools=False):
    """The parser class wrapped: instances remember their build, accumulate the stream, and compare at its end.
    A class without both parse and parse_delta is returned as it is."""
    if not isinstance(base, type) or not (callable(getattr(base, "parse", None))
                                          and callable(getattr(base, "parse_delta", None))):
        return base
    key = (base, bool(auto_tools))
    if key in _CLASSES:
        return _CLASSES[key]
    name = base.__name__

    class Entailed(base):
        _entail_auto_tools = bool(auto_tools)

        def __init__(self, *a, **kw):
            self._entail_init = (a, dict(kw))
            self._entail_stream = None
            self._entail_reference = False
            super().__init__(*a, **kw)

        def parse_delta(self, *a, **kw):
            out = super().parse_delta(*a, **kw)
            if not getattr(self, "_entail_reference", False) and _active():
                from .. import load

                load.safely(BOUNDARY, f"vllm.parser.{name}", "Parse", lambda: _on_delta(self, name, a, kw, out))
            return out

        def parse(self, *a, **kw):
            out = super().parse(*a, **kw)
            if not getattr(self, "_entail_reference", False) and _active():
                from .. import load

                load.safely(BOUNDARY, f"vllm.parser.{name}", "Parse", lambda: _on_full(self, name, a, kw, out))
            return out

    Entailed.__name__, Entailed.__qualname__, Entailed.__module__ = name, base.__qualname__, base.__module__
    _CLASSES[key] = Entailed
    return Entailed


def install():
    """Wrap ParserManager.get_parser (composed with whatever wrapper is there). Returns 1, or 0 if installed."""
    try:
        from vllm.parser.parser_manager import ParserManager
    except ImportError:
        return 0
    if "get_parser" in _ORIG:
        return 0
    current = _ORIG["get_parser"] = ParserManager.__dict__["get_parser"]

    names = ("tool_parser_name", "reasoning_parser_name", "enable_auto_tools", "model_name")

    def get_parser(cls, *args, **kwargs):
        parser_cls = current.__func__(cls, *args, **kwargs)
        bound = dict(zip(names, args))
        bound.update({k: kwargs[k] for k in names if k in kwargs})
        if not _active():
            return parser_cls
        try:
            return wrapped_class(parser_cls, bound.get("enable_auto_tools", False))
        except Exception:  # noqa: BLE001 - never the server's problem
            return parser_cls

    ParserManager.get_parser = classmethod(get_parser)
    return 1


def uninstall():
    if "get_parser" not in _ORIG:
        return 0
    from vllm.parser.parser_manager import ParserManager

    ParserManager.get_parser = _ORIG.pop("get_parser")
    _CLASSES.clear()
    return 1


def stats():
    return parse_contract.stats(BOUNDARY)


def reset():
    _CLASSES.clear()
    parse_contract.reset(BOUNDARY)
