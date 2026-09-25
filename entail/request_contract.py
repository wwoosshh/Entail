"""request_contract: what a request must carry for the model, checked where a server renders it (LIBRARY_DESIGN.md
4.6, 4.7; ROADMAP M5.3, M5.5).

The model declares what a request must look like for it (Template, vocabulary v3): the chat template it was trained
with, whether its earlier reasoning is sent back, the format it writes tool calls in. A server adapter says what the
server does with one request; the rules are here, and they run once per request, on the host (principle 6):

  template  the chat template the request is rendered with, against the declared one. A template the request or the
            server's command line names is the user's choice: never overridden, and broken when it contradicts the
            declaration. One the engine picked by itself (a processor's, a fallback) is broken too: no resolution is
            registered
  history   declared keep: every earlier assistant turn reaches the template with its reasoning (market L13: an
            OpenAI-compatible integration dropped it; Tau² 87 -> 64). A turn the adapter cannot read leaves the
            verdict unknown
  settings  Coverage of what the request sets: a field the server's request schema does not know, or a template
            setting the template does not read, is dropped without a word (market L07: reasoning_effort was
            ignored; AIME25 93.3% -> 80.0%). Nothing repairs a setting nobody reads: broken
  window    the context a request is given against its prompt's length (Valid, with where the context came from:
            Origin; market L05: a default context cut the prompt). A default is extended when the model has room
            (resolved); a context the user set, or no room, leaves the cut: broken (M5.5)

The tool call format is decided once, where the server builds its tool parser (load.tool_parser).
Where a template is applied: a server's adapter (vllm_serve) decides the renders of its server itself - it knows
whether the request named the template - and marks them (`deciding`); the tokenizer's own apply_chat_template, which
it calls inside, is then not decided again (transformers_template, M9.3).
Passes are counted (tally). Anything else is a Decision recorded through load.enforce: a broken rule for every
request it breaks in (like an access log's error line), an unknown that does not block once per boundary and counted
afterwards. Under the default policy the request goes on as it would without entail (M5.4); `reported` gives the
lines for an adapter that also puts them in the response. Where the policy stops, a refusal raises RoleError, which
the adapter turns into the server's own error response before anything is generated.
"""
import threading
from dataclasses import replace
from functools import lru_cache
from typing import Dict, Iterable, Optional, Sequence

from . import load, policies
from . import tally as _tally
from .contracts import Contract, Verdict, decide
from .coverage import Coverage
from .facts import Certainty, Fact, Source, Template
from .readers import sha256_text

_DECLARED: Dict[str, object] = {}   # model path -> load.Declared, read once
_PROJECTED: Dict[tuple, tuple] = {}  # (id of a declaration, Template field) -> (it, its candidates for that field)
_RECORDED = set()                   # boundaries whose non-blocking unknown was recorded once
_DECIDING = threading.local()       # .depth > 0 while a server adapter decides the render it is running


@lru_cache(maxsize=64)
def _sha(text: str) -> str:
    return sha256_text(text)


def declared(model_path, held_template: Optional[str] = None):
    """What is declared about the model a server runs (load.declared, read once per model path). When the files say
    nothing about the chat template - a model served by a hub name rather than a folder - the template the engine's
    tokenizer loaded from the model's files is the declaration, as the engine's config object is when there are no
    files."""
    key = str(model_path)
    facts = _DECLARED.get(key)
    if facts is None:
        facts = _DECLARED[key] = load.declared(key)
    if held_template and not _projected(facts, "chat_template_sha256"):
        facts.facts.setdefault("Template", []).append(
            Fact("Template", Template(chat_template_sha256=_sha(held_template)),
                 Source("config", f"the chat template the engine's tokenizer loaded for {key}"), Certainty.DECLARED))
        _PROJECTED.clear()
    return facts


def _projected(facts, field):
    """The declared Template candidates for one field, projected once per declaration."""
    key = (id(facts), field)
    got = _PROJECTED.get(key)
    if got is None or got[0] is not facts:
        got = _PROJECTED[key] = (facts, tuple(load._projected(facts, "Template", (f"Template.{field}",))))
    return got[1]


@lru_cache(maxsize=64)
def _chosen_template(text: Optional[str], explicit: bool, where: str) -> Fact:
    if text is None:
        return Fact("Template", None, Source("engine", where), Certainty.UNKNOWN)
    return Fact("Template", Template(chat_template_sha256=_sha(text)), Source("user" if explicit else "engine", where),
                Certainty.VERIFIED)


def _settle(boundary: str, check: str, decisions) -> list:
    """Count what held; record and raise the rest. A non-blocking unknown is recorded once per boundary."""
    s = _tally.counts(boundary)
    out = []
    for d in decisions:
        if d.verdict is Verdict.UNKNOWN and not d.blocking:
            s["skipped"] += 1
            if boundary not in _RECORDED:
                _RECORDED.add(boundary)
                out.append(d)
            continue
        s["checks"] += 1
        if d.verdict is Verdict.PASS:
            _tally.passed(boundary, [check])
        else:
            out.append(d)
    if any(d.blocking for d in out):
        _tally.refused(boundary)
    elif any(d.verdict is Verdict.BROKEN for d in out):
        _tally.broken(boundary)
    elif any(d.verdict is Verdict.RESOLVED for d in out):
        s["resolved"] += 1
    _tally.tick(boundary)
    if out:
        load.enforce(out)
    return list(decisions)


_ORIGIN_SOURCE = {"default": "default", "user": "user", "checkpoint": "file", "config": "config",
                  "manifest": "manifest"}


def window(boundary: str, consumer: str, prompt_tokens: int, context: int, origin: str,
           model_context: Optional[int], where: str, policy=None) -> list:
    """The context a server gives a request - `context` tokens, whose value came from `origin` (facts.ORIGINS) -
    against the tokens the request's prompt has (Valid; market L05: Ollama cut a 10983-token prompt to its default
    2048 and said so only in the server's log). A prompt that fits passes. One that does not would be cut:
      - a context that came from anywhere but the user is extended to what the prompt needs when the model declares
        room for it (`model_context`): resolved, handle "extend_context", target the prompt's length
      - a context the user set is theirs, never overridden: broken
      - with no room in the model the cut stays: broken
    """
    from .contracts import Resolution
    from .facts import Valid

    taken = min(int(prompt_tokens), int(context))
    declared_fact = Fact("Valid", Valid(length=int(prompt_tokens)),
                         Source("data", f"{where}: the tokens the request's prompt has"), Certainty.VERIFIED)
    chosen = Fact("Valid", Valid(length=taken),
                  Source(_ORIGIN_SOURCE.get(origin, "engine"),
                         f"{where}: the tokens of it the model is given (context {context}, from {origin})"),
                  Certainty.VERIFIED)
    room = model_context is not None and int(model_context) >= int(prompt_tokens)
    extend = Resolution("give the prompt the context it needs, within what the model declares", "extend_context",
                        when=lambda d, c: room, target=lambda d, c: int(prompt_tokens))
    contract = Contract(boundary, consumer, ("Valid",), ("Valid",))
    return _settle(boundary, "window", decide(contract, {"Valid": declared_fact}, {"Valid": chosen},
                                              policy or policies.current(), resolutions={"Valid": [extend]}))


def pad_type(boundary: str, consumer: str, declared: Optional[int], used: Optional[int], where: str,
             policy=None, repairable: Optional[bool] = None, declared_by: str = "") -> list:
    """The token type id a server gave the padding of a request against the one the tokenizer declares for
    padding (TokenType, M15.1; vllm#58138: a cross-encoder's padding was given the document's segment). `declared`
    is the tokenizer's pad_token_type_id (None when the tokenizer does not say); `used` the id the padding got (None
    when nothing was padded: nothing to decide). The resolution gives the padding the declared id - offered only
    where the consumer can carry it: `repairable`, or, when None, the capability table's row for the consumer
    (vLLM 0.30's scoring keeps the ids as the index of the first 1, so a pad type after the document cannot be
    carried: honours false, measured; the decision is broken there, with the row's note)."""
    from . import caps as _caps
    from .contracts import Resolution
    from .facts import TokenType

    if used is None:
        return []
    note = ""
    if repairable is None:
        row = _caps.lookup(_caps.default_table(), consumer, "TokenType.type_id")
        repairable = row is not None and bool(row.honours)
        if row is not None and not row.honours:
            note = row.note
    by = f"the tokenizer class tokenizer_config.json names ({declared_by}) fixes pad_token_type_id" if declared_by \
        else "the tokenizer's pad_token_type_id"
    d = Fact("TokenType", None if declared is None else TokenType("pad", int(declared)),
             Source("config", f"{where}: {by}"), Certainty.UNKNOWN if declared is None else Certainty.DECLARED)
    c = Fact("TokenType", TokenType("pad", int(used)), Source("engine", f"{where}: the id the padding was given"),
             Certainty.VERIFIED)
    give = Resolution("give the padding the type the tokenizer declares", "set_pad_type",
                      target=lambda dd, cc: dd.value.type_id)
    contract = Contract(boundary, consumer, ("TokenType",), ("TokenType",))
    decisions = decide(contract, {"TokenType": d}, {"TokenType": c}, policy or policies.current(),
                       resolutions={"TokenType": [give] if repairable else []})
    if note:
        decisions = [replace(x, note=note) if x.verdict is not Verdict.PASS and not x.note else x for x in decisions]
    return _settle(boundary, "pad_type", decisions)


def reported(decisions) -> list:
    """The ledger line of each decision that broke and was reported while the request went on, for an adapter that
    also puts them in the response (LIBRARY_DESIGN.md 4.6, M5.4)."""
    from .record import line

    return [line(d) for d in decisions if d.verdict is Verdict.BROKEN]


def _skip(boundary: str) -> list:
    _tally.counts(boundary)["skipped"] += 1
    _tally.tick(boundary)
    return []


def template(boundary: str, consumer: str, facts, text: Optional[str], explicit: bool, where: str,
             policy=None) -> list:
    """The chat template a request is rendered with (its text; None when the adapter cannot name it) against the
    declared one. `explicit`: the request or the server's command line named it, the user's choice."""
    contract = Contract(boundary, consumer, ("Template",), ("Template",))
    return _settle(boundary, "template", decide(contract, {"Template": _projected(facts, "chat_template_sha256")},
                                                {"Template": _chosen_template(text, explicit, where)},
                                                policy or policies.current()))


def history(boundary: str, consumer: str, facts, turns: Sequence[Optional[bool]], where: str, policy=None) -> list:
    """Declared keep: every earlier assistant turn reaches the template with its reasoning. `turns` has one entry per
    earlier assistant turn: True (its reasoning reaches the template), False (it does not), None (the adapter cannot
    tell). Nothing is decided when no source declares keep, or there is no earlier turn."""
    candidates = _projected(facts, "reasoning_history")
    if not turns or not any(f.value.reasoning_history == "keep" for f in candidates):
        return _skip(boundary)
    n, missing, unread = len(turns), sum(t is False for t in turns), sum(t is None for t in turns)
    if missing:
        chosen = Fact("Template", Template(reasoning_history="drop"),
                      Source("data", f"{where}: {missing} of {n} earlier assistant turns reach the template without "
                                     f"their reasoning"), Certainty.VERIFIED)
    elif unread:
        chosen = Fact("Template", None,
                      Source("engine", f"{where}: for {unread} of {n} earlier assistant turns it cannot be told "
                                       f"whether their reasoning reaches the template"), Certainty.UNKNOWN)
    else:
        chosen = Fact("Template", Template(reasoning_history="keep"),
                      Source("data", f"{where}: all {n} earlier assistant turns carry their reasoning"),
                      Certainty.VERIFIED)
    contract = Contract(boundary, consumer, ("Template",), ("Template",))
    return _settle(boundary, "reasoning_history", decide(contract, {"Template": candidates},
                                                         {"Template": chosen}, policy or policies.current()))


def settings(boundary: str, consumer: str, given: Iterable[str], taken: Iterable[str], where: str, what: str,
             policy=None) -> list:
    """Coverage of the settings a request gives (their names) by the ones its consumer reads. `where` says where they
    were given (the request, the server's defaults), `what` who reads them and how that is known. Nothing is decided
    when nothing is given."""
    given = sorted(set(given))
    if not given:
        return _skip(boundary)
    left = tuple(sorted(set(given) - set(taken)))
    shown = ", ".join(given[:8]) + (f" and {len(given) - 8} more" if len(given) > 8 else "")
    declared_fact = Fact("Coverage", Coverage(len(given), len(given), ()), Source("user", f"{where}: {shown}"),
                         Certainty.DECLARED)
    chosen = Fact("Coverage", Coverage(len(given), len(given) - len(left), left), Source("engine", what),
                  Certainty.VERIFIED)
    contract = Contract(boundary, consumer, ("Coverage",), ("Coverage",))
    return _settle(boundary, "settings", decide(contract, {"Coverage": declared_fact}, {"Coverage": chosen},
                                                policy or policies.current()))


class deciding:
    """`with request_contract.deciding():` around the work of a server adapter that decides the template of what it
    renders itself. The tokenizer's apply_chat_template, called inside - in the same thread, where the server applies
    the template - steps aside (decided_elsewhere)."""

    def __enter__(self):
        _DECIDING.depth = getattr(_DECIDING, "depth", 0) + 1
        return self

    def __exit__(self, *exc):
        _DECIDING.depth -= 1
        return False


def decided_elsewhere() -> bool:
    """True inside `deciding`: a server's adapter decides this render."""
    return getattr(_DECIDING, "depth", 0) > 0


def guarded(boundary: str, consumer: str, work, *args, **kwargs):
    """tally.guarded for these rules: an error inside entail never breaks the server (principle 12)."""
    return _tally.guarded(boundary, consumer, "Template", work, *args, **kwargs)


def stats(boundary: Optional[str] = None) -> dict:
    return _tally.stats(boundary)


def reset(boundary: Optional[str] = None) -> None:
    """Forget the counts (of one boundary, or all), what was recorded once, and every model's declaration."""
    _tally.reset(boundary)
    if boundary is None:
        _RECORDED.clear()
        _DECLARED.clear()
        _PROJECTED.clear()
    else:
        _RECORDED.discard(boundary)


__all__ = ["declared", "template", "history", "settings", "window", "reported", "deciding", "decided_elsewhere",
           "guarded", "stats", "reset"]
