"""Adapter v2 for the chat template where transformers applies it: a tokenizer's apply_chat_template (LIBRARY_DESIGN.md
4.6, 4.7; ROADMAP M9.3, from M9.1's S1: a model folder declares its chat template, and transformers rendered with it
without any decision).

  hooks        PreTrainedTokenizerBase.apply_chat_template: where every transformers tokenizer turns a conversation
               into the prompt - a script's, and SGLang's server for a model with a jinja template. A server whose
               own adapter decides its renders (vllm_serve) marks them, and this hook steps aside inside them
               (request_contract.decided_elsewhere).
  read_choice  the template the call renders with - the tokenizer's own choice for the same arguments
               (get_chat_template: a template passed in, a named one, "tool_use" when tools are given, or the
               default) - and whether the caller passed it; per earlier assistant turn, whether its reasoning reaches
               the template (a reasoning_content, reasoning or thinking field, or a <think> block in its content).
  handles      none: nothing repairs a template. One that is not the declared one is reported (broken) and the call
               goes on; where the policy stops, the call raises before anything is rendered.
request_contract decides the template and the reasoning history, once per call. A processor's apply_chat_template
(multimodal models) is not hooked.
"""
from .. import core, policies, request_contract
from .base import Hook

engine = "transformers"
versions = "5.17.0"
TEMPLATE = "request:transformers.chat_template"
HISTORY = "request:transformers.reasoning_history"
CONSUMER = "transformers.apply_chat_template"
REASONING = ("reasoning_content", "reasoning", "thinking")   # the fields a chat template reads earlier reasoning from
_ORIG = {}


def hooks():
    return [Hook("transformers.tokenization_utils_base.PreTrainedTokenizerBase.apply_chat_template", "request")]


def _get(m, key):
    return m.get(key) if isinstance(m, dict) else getattr(m, key, None)


def _conversations(conversation):
    """The conversations of one call: apply_chat_template takes one, or a batch of them."""
    if isinstance(conversation, (list, tuple)) and conversation:
        first = conversation[0]
        if isinstance(first, (list, tuple)):
            return list(conversation)
        if hasattr(first, "messages"):
            return [c.messages for c in conversation]
    return [conversation]


def read_choice(kind, *args):
    """What one call does, as the core's rules take it:
      "template", tokenizer, chat_template, tools -> (the template's text, or None when it is one of several named
                                                       templates other than the default; the caller passed it)
      "turns", conversation                       -> per earlier assistant turn: True (its reasoning reaches the
                                                       template) or False"""
    if kind == "template":
        tokenizer, chat_template, tools = args
        try:
            text = tokenizer.get_chat_template(chat_template, tools)
        except Exception:  # noqa: BLE001 - no template at all: apply_chat_template raises that by itself
            return None, chat_template is not None
        held = getattr(tokenizer, "chat_template", None)
        other = isinstance(held, dict) and any(t == text for n, t in held.items() if n != "default")
        return (None if other else text), chat_template is not None
    if kind == "turns":
        messages, = args
        roles = [_get(m, "role") for m in messages]
        turns = [m for m, r in zip(messages, roles) if r == "assistant"]
        if roles and roles[-1] == "assistant":
            turns = turns[:-1]   # the turn being continued, not an earlier one
        return [any(_get(m, k) for k in REASONING) or "<think>" in str(_get(m, "content") or "") for m in turns]
    raise ValueError(f"transformers_template.read_choice: unknown kind {kind!r}")


def handles():
    return {}


def _decide(tokenizer, conversation, tools, chat_template):
    model = str(getattr(tokenizer, "name_or_path", "") or "")
    held = getattr(tokenizer, "chat_template", None)
    facts = request_contract.declared(model, held.get("default") if isinstance(held, dict) else held)
    policy = policies.current()
    for messages in _conversations(conversation):
        request_contract.history(HISTORY, CONSUMER, facts, read_choice("turns", messages),
                                 "the conversation given to apply_chat_template", policy)
    text, named = read_choice("template", tokenizer, chat_template, tools)
    how = "passed to apply_chat_template" if named else "the tokenizer's own"
    other = "; a named template other than the default" if text is None else ""
    request_contract.template(TEMPLATE, CONSUMER, facts, text, named, f"transformers chat template, {how}{other}",
                              policy)


def install():
    """Wrap PreTrainedTokenizerBase.apply_chat_template. Returns 1, or 0 if already installed."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if "apply" in _ORIG:
        return 0
    orig = _ORIG["apply"] = PreTrainedTokenizerBase.apply_chat_template

    def apply_chat_template(self, conversation, *args, **kwargs):
        if core.mode() in ("load", "debug") and not request_contract.decided_elsewhere():
            tools = kwargs.get("tools", args[0] if len(args) > 0 else None)
            chat_template = kwargs.get("chat_template", args[2] if len(args) > 2 else None)
            request_contract.guarded(TEMPLATE, CONSUMER, _decide, self, conversation, tools, chat_template)
        return orig(self, conversation, *args, **kwargs)

    PreTrainedTokenizerBase.apply_chat_template = apply_chat_template
    return 1


def uninstall():
    if "apply" not in _ORIG:
        return 0
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    PreTrainedTokenizerBase.apply_chat_template = _ORIG.pop("apply")
    return 1


def stats(boundary=TEMPLATE):
    return request_contract.stats(boundary)


def reset():
    request_contract.reset()
