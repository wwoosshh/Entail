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
from .. import core, policies, request_contract
from .base import Hook

engine = "sglang"
versions = "0.5.20"
TEMPLATE = "request:sglang.chat_template"
CONSUMER = "sglang.chat_template"
_ORIG = {}


def hooks():
    return [Hook("sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat._apply_conversation_template",
                 "request")]


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


def install():
    """Wrap OpenAIServingChat._apply_conversation_template. Returns 1, or 0 if already installed."""
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

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
    if "conversation" not in _ORIG:
        return 0
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    OpenAIServingChat._apply_conversation_template = _ORIG.pop("conversation")
    return 1
