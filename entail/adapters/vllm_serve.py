"""Adapter v2 for vLLM's OpenAI-compatible server: the request boundary (LIBRARY_DESIGN.md 4.6, 4.7; ROADMAP M5.3).

  hooks        ParserManager.get_parser: where the server builds its tool and reasoning parsers (for chat, for the
               renderer, for responses). The tool parser against the declared tool call format is a load decision,
               made once per model and parser.
               HfRenderer.render_messages(_async): one request about to be rendered - its messages, and the template
               and settings it will be rendered with (ChatParams). Decided here, before rendering: the template is
               applied later in an executor thread, where the request is no longer known.
               OpenAIServingChat.create_chat_completion: the request itself - its fields (pydantic keeps the ones the
               schema does not know in model_extra, and vLLM logs them at debug level) and its own template settings
               - and where a refusal becomes the server's own error response (400), before anything is generated.
               A render with no request of the server around it (the renderer's warm-up at start, LLM.chat) has its
               template and history checked, not its settings: who gave them cannot be told there.
  read_choice  the template the request is rendered with - vLLM's own resolve_chat_template on the same inputs - and
               whether the request or the server's --chat-template named it; the request's own template settings
               (its chat_template_kwargs, reasoning_effort, documents, tools; not the server's defaults, where vLLM
               puts its own --cohere-format) and the ones that reach the model: the template reads the variable
               (vLLM's own parse of the template), or apply_chat_template acts on it itself; reasoning_effort "none"
               also arrives as enable_thinking false, which vLLM derives from it. Per earlier assistant turn, whether
               its reasoning reaches the template: vLLM passes the `reasoning` field (it renames reasoning_content to
               it), so a turn without it has none - unless the server runs no reasoning parser, or one that leaves
               the reasoning in the content, where it cannot be told.
  handles      switch_tool_parser: build the parsers with the named tool parser.
request_contract decides the request rules, load.tool_parser the tool call format. Installed per module (the parser
manager, the renderer, the chat server), each as soon as it has been imported.
"""
import contextvars

from .. import core, load, policies, request_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
TOOL_PARSER = "load:vllm.tool_parser"
TEMPLATE = "request:vllm.chat_template"
SETTINGS = "request:vllm.template_settings"
HISTORY = "request:vllm.reasoning_history"
FIELDS = "request:vllm.request_fields"
CONSUMER = "vllm.chat_template"
# reasoning parsers that return the reasoning inside the content (vllm/reasoning/minimax_m2_reasoning_parser.py:
# MiniMaxM2AppendThinkReasoningParser.extract_reasoning returns (None, "<think>" + output))
INLINE_REASONING = ("minimax_m2_append_think",)
# apply_chat_template parameters that it hands to the template and does not act on itself
TEMPLATE_ONLY = ("tools", "documents")
_ORIG = {}
_REQUEST = contextvars.ContextVar("entail_vllm_request", default=(None, None))   # (request, policy) in this task
_SERVED = {}    # model path -> the reasoning parser the server runs ("" for none)
_PARSERS = {}   # (model path, tool parser, reasoning parser, auto tools, harmony) -> the tool parser to build


def hooks():
    return [Hook("vllm.parser.parser_manager.ParserManager.get_parser", "load"),
            Hook("vllm.renderers.hf.HfRenderer.render_messages", "request"),
            Hook("vllm.renderers.hf.HfRenderer.render_messages_async", "request"),
            Hook("vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat.create_chat_completion",
                 "request")]


def _get(m, key):
    return m.get(key) if isinstance(m, dict) else getattr(m, key, None)


def read_choice(kind, *args):
    """What the server does with one request, as the core's rules take it:
      "template", tokenizer, params, model_config  -> (the template's text, or None when a named template other than
                                                       the default is used; named by the request or command line)
      "settings", params, request, text            -> (the request's own template settings, the ones that arrive)
      "turns", messages, reasoning parser          -> per earlier assistant turn: True, False, or None (cannot tell)
      "fields", request                            -> (the fields the request sets, the ones the schema knows)"""
    if kind == "template":
        from vllm.renderers import hf

        tokenizer, params, model_config = args
        text = hf.resolve_chat_template(tokenizer, chat_template=params.chat_template,
                                        tools=params.chat_template_kwargs.get("tools"), model_config=model_config)
        ct = getattr(tokenizer, "chat_template", None)
        other = isinstance(ct, dict) and any(t == text for n, t in ct.items() if n != "default")
        return (None if other else text), params.chat_template is not None
    if kind == "settings":
        from vllm.renderers import hf

        params, request, text = args
        handed = {k: v for k, v in params.chat_template_kwargs.items() if v is not None}
        own = {k for k, v in (getattr(request, "chat_template_kwargs", None) or {}).items() if v is not None}
        own |= {k for k in ("reasoning_effort",) + TEMPLATE_ONLY if k in handed}
        reads = hf._cached_resolve_chat_template_kwargs(text) if text else set()
        acts = hf._get_hf_base_chat_template_params() - set(TEMPLATE_ONLY)
        reached = {k for k in own if k in reads or k in acts}
        if handed.get("reasoning_effort") == "none" and handed.get("enable_thinking") is False \
                and "enable_thinking" in reads:
            reached.add("reasoning_effort")
        return sorted(own), sorted(reached & own)
    if kind == "turns":
        messages, parser = args
        roles = [_get(m, "role") for m in messages]
        turns = [m for m, r in zip(messages, roles) if r == "assistant"]
        if roles and roles[-1] == "assistant":
            turns = turns[:-1]   # the turn being continued, not an earlier one
        unreadable = not parser or parser in INLINE_REASONING
        return [True if _get(m, "reasoning") else (None if unreadable else False) for m in turns]
    if kind == "fields":
        request, = args
        extra = set(getattr(request, "model_extra", None) or {})
        known = set(getattr(request, "model_fields_set", ())) - extra
        return sorted(known | extra), sorted(known)
    raise ValueError(f"vllm_serve.read_choice: unknown kind {kind!r}")


def handles():
    return {"switch_tool_parser": lambda target: target}


def _active():
    return core.mode() in ("load", "debug")


# --- deciding, through the core ----------------------------------------------------------------------------------

def _tool_parser(model, name):
    decisions = load.tool_parser(engine, name, request_contract.declared(model), policy=policies.current())
    done = load.resolve(decisions, handles())
    load.enforce(decisions)
    return done.get("switch_tool_parser", name)


def _before_render(renderer, messages, params):
    model = str(renderer.model_config.model)
    tokenizer = renderer.get_tokenizer()
    ct = getattr(tokenizer, "chat_template", None)
    facts = request_contract.declared(model, ct.get("default") if isinstance(ct, dict) else ct)
    request, policy = _REQUEST.get()
    parser, policy = _SERVED.get(model, ""), policy or policies.current()
    request_contract.history(HISTORY, CONSUMER, facts, read_choice("turns", messages, parser),
                             f"vllm request messages (reasoning parser: {parser or 'none'})", policy)
    text, named = read_choice("template", tokenizer, params, renderer.model_config)
    how = "named by the request or --chat-template" if named else "picked by vLLM"
    other = "; a named template other than the default" if text is None else ""
    request_contract.template(TEMPLATE, CONSUMER, facts, text, named, f"vllm chat template, {how}{other}", policy)
    if request is not None:
        given, reached = read_choice("settings", params, request, text)
        request_contract.settings(SETTINGS, CONSUMER, given, reached, "the request's own template settings",
                                  "vllm chat template: a setting arrives when the template reads it or "
                                  "apply_chat_template acts on it; vLLM drops the rest", policy)


def _fields(request, policy):
    given, known = read_choice("fields", request)
    request_contract.settings(FIELDS, "vllm.chat_request", given, known, "the fields the request sets",
                              "vllm ChatCompletionRequest: a field the schema does not know is kept in model_extra "
                              "and ignored (logged at debug level)", policy)


# --- installing ----------------------------------------------------------------------------------------------------

def install_parsers():
    """Wrap ParserManager.get_parser. Returns 1, or 0 if already installed."""
    from vllm.parser.parser_manager import ParserManager

    if "get_parser" in _ORIG:
        return 0
    orig = _ORIG["get_parser"] = ParserManager.__dict__["get_parser"]

    def get_parser(cls, tool_parser_name=None, reasoning_parser_name=None, enable_auto_tools=False, model_name=None,
                   is_harmony=False):
        if model_name is not None:
            _SERVED[str(model_name)] = reasoning_parser_name or ""
        name = tool_parser_name
        if _active() and enable_auto_tools and tool_parser_name and model_name is not None:
            key = (str(model_name), tool_parser_name, reasoning_parser_name, enable_auto_tools, is_harmony)
            if key not in _PARSERS:
                _PARSERS[key] = load.safely(TOOL_PARSER, f"vllm.tool_parser.{tool_parser_name}", "Template",
                                            lambda: _tool_parser(model_name, tool_parser_name), tool_parser_name)
            name = _PARSERS[key]
        return orig.__func__(cls, tool_parser_name=name, reasoning_parser_name=reasoning_parser_name,
                             enable_auto_tools=enable_auto_tools, model_name=model_name, is_harmony=is_harmony)

    ParserManager.get_parser = classmethod(get_parser)
    return 1


def install_render():
    """Wrap HfRenderer.render_messages and render_messages_async. Returns 1 or 0."""
    from vllm.renderers.hf import HfRenderer

    if "render" in _ORIG:
        return 0
    _ORIG.update(render=HfRenderer.render_messages, render_async=HfRenderer.render_messages_async)

    def render_messages(self, messages, params):
        if _active():
            request_contract.guarded(TEMPLATE, CONSUMER, _before_render, self, messages, params)
        return _ORIG["render"](self, messages, params)

    async def render_messages_async(self, messages, params):
        if _active():
            request_contract.guarded(TEMPLATE, CONSUMER, _before_render, self, messages, params)
        return await _ORIG["render_async"](self, messages, params)

    HfRenderer.render_messages = render_messages
    HfRenderer.render_messages_async = render_messages_async
    return 1


def install_serving():
    """Wrap OpenAIServingChat.create_chat_completion. Returns 1 or 0."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    if "create" in _ORIG:
        return 0
    _ORIG["create"] = OpenAIServingChat.create_chat_completion

    async def create_chat_completion(self, request, raw_request=None):
        policy = policies.current() if _active() else None
        token = _REQUEST.set((request, policy))
        try:
            if policy is not None:
                request_contract.guarded(FIELDS, "vllm.chat_request", _fields, request, policy)
            return await _ORIG["create"](self, request, raw_request)
        except core.RoleError as e:   # a refusal: the server's own error response, before anything is generated
            return self.create_error_response(str(e))
        finally:
            _REQUEST.reset(token)

    OpenAIServingChat.create_chat_completion = create_chat_completion
    return 1


def install():
    """All three. Returns the number of parts installed now."""
    return install_parsers() + install_render() + install_serving()


def uninstall():
    n = 0
    if "get_parser" in _ORIG:
        from vllm.parser.parser_manager import ParserManager

        ParserManager.get_parser = _ORIG.pop("get_parser")
        n += 1
    if "render" in _ORIG:
        from vllm.renderers.hf import HfRenderer

        HfRenderer.render_messages = _ORIG.pop("render")
        HfRenderer.render_messages_async = _ORIG.pop("render_async")
        n += 1
    if "create" in _ORIG:
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

        OpenAIServingChat.create_chat_completion = _ORIG.pop("create")
        n += 1
    _PARSERS.clear()
    _SERVED.clear()
    return n


def reset():
    request_contract.reset()
    _PARSERS.clear()
