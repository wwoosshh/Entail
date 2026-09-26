"""Adapter v2 for SGLang's OpenAI-compatible server: the chat template when SGLang renders a conversation with one of
its own conversation templates instead of the model's (LIBRARY_DESIGN.md 4.6, 4.7; ROADMAP M9.3, from M9.1's S1).

  hooks        OpenAIServingChat._apply_conversation_template: where SGLang builds the prompt from a conversation
               template it names itself (`--chat-template <name>` of its own list), not from the model's jinja
               template. A model's jinja template goes through the tokenizer's apply_chat_template and is decided
               there (transformers_template).
  read_choice  the template it renders with: a text that stands for the named conversation template, which is not
               the declared one; named by --chat-template, the user's choice.
  handles      none: nothing repairs a template; the render goes on, reported (broken), or stops where the policy
               stops.
request_contract decides the template, once per request.
"""
from .. import core, parse_contract, policies, request_contract
from .base import Hook

engine = "sglang"
versions = "0.5.20"
TEMPLATE = "request:sglang.chat_template"
CONSUMER = "sglang.chat_template"
LOGPROBS = "request:sglang.logprobs"      # M18.4: a response's logprobs against its message (sglang#25055)
_ORIG = {}


def hooks():
    return [Hook("sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat._apply_conversation_template",
                 "request"),
            Hook("sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat._build_chat_response", "request")]


def read_choice(kind, *args):
    """  "template", serving -> (a text that stands for the conversation template SGLang renders with, True: named by
                                --chat-template)"""
    if kind == "template":
        serving, = args
        name = getattr(getattr(serving, "template_manager", None), "chat_template_name", None)
        return f"SGLang conversation template {name!r}", True
    raise ValueError(f"sglang_serve.read_choice: unknown kind {kind!r}")


def handles():
    return {}


def _decide(serving):
    manager = getattr(serving, "tokenizer_manager", None)
    model = str(getattr(manager, "model_path", "") or "")
    held = getattr(getattr(manager, "tokenizer", None), "chat_template", None)
    facts = request_contract.declared(model, held.get("default") if isinstance(held, dict) else held)
    text, named = read_choice("template", serving)
    request_contract.template(TEMPLATE, CONSUMER, facts, text, named, f"{text}, named by --chat-template",
                              policies.current())


def decide_logprobs(response):
    """M18.4: every choice's logprobs against its message (parse_contract.check_logprobs): the logprob tokens must
    decode to the content the client gets; a response whose logprobs cover the parsed-out reasoning span is
    reported (sglang#25055). Works on the response object or a dict of the same shape."""
    def get(o, k):
        return o.get(k) if isinstance(o, dict) else getattr(o, k, None)

    for choice in get(response, "choices") or []:
        lp = get(choice, "logprobs")
        items = get(lp, "content") if lp is not None else None
        if not items:
            continue
        tokens = [str(get(t, "token") or "") for t in items]
        msg = get(choice, "message")
        parse_contract.check_logprobs(LOGPROBS, "sglang.chat_completion", get(msg, "content"),
                                      get(msg, "reasoning_content"), tokens,
                                      f"choice {get(choice, 'index')} of a chat completion, {len(tokens)} logprob tokens")


def install_logprobs():
    """Wrap OpenAIServingChat._build_chat_response (M18.4). Returns 1, or 0 if already installed."""
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    if "response" in _ORIG or not hasattr(OpenAIServingChat, "_build_chat_response"):
        return 0
    orig = _ORIG["response"] = OpenAIServingChat._build_chat_response

    def _build_chat_response(self, *args, **kwargs):
        response = orig(self, *args, **kwargs)
        if core.mode() in ("load", "debug"):
            from .. import load

            load.safely(LOGPROBS, "sglang.chat_completion", "Parse", lambda: decide_logprobs(response))
        return response

    OpenAIServingChat._build_chat_response = _build_chat_response
    return 1


def install():
    """Wrap OpenAIServingChat._apply_conversation_template (and _build_chat_response, M18.4). Returns 1, or 0 if
    already installed."""
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    install_logprobs()
    if "conversation" in _ORIG:
        return 0
    orig = _ORIG["conversation"] = OpenAIServingChat._apply_conversation_template

    def _apply_conversation_template(self, *args, **kwargs):
        if core.mode() in ("load", "debug"):
            request_contract.guarded(TEMPLATE, CONSUMER, _decide, self)
        return orig(self, *args, **kwargs)

    OpenAIServingChat._apply_conversation_template = _apply_conversation_template
    return 1


def uninstall():
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    if "response" in _ORIG:
        OpenAIServingChat._build_chat_response = _ORIG.pop("response")
    if "conversation" not in _ORIG:
        return 0
    OpenAIServingChat._apply_conversation_template = _ORIG.pop("conversation")
    return 1
