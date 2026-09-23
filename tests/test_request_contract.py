"""Tests for the request contract (ROADMAP M5.3): the core's rules for one request, and the tool parser contract that
runs once at start. No engine: a model folder with a tokenizer_config.json and a pinned manifest is the declaration,
and what a server does with a request is given as the adapter would read it.

Run: python tests/test_request_contract.py
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
from entail import caps, core, load, manifest, request_contract, tally  # noqa: E402
from entail.contracts import Verdict  # noqa: E402
from entail.facts import Certainty, Fact, Source, Template  # noqa: E402
from entail.policies import Policy  # noqa: E402
from entail.readers import sha256_text  # noqa: E402

STOPS = dict(on_broken="stop", on_unknown_meaning_changing="require")   # the policy before M5.4

TEMPLATE_TEXT = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if enable_thinking %}<think>{% endif %}"
OTHER_TEXT = "{% for m in messages %}{{ m.content }}{% endfor %}"
B, C = "request:test.chat_template", "test.chat_template"
LOAD, REFUSE = Policy(mode="load", **STOPS), Policy(mode="load", **STOPS, on_mismatch="refuse")


def model_folder(template=TEMPLATE_TEXT, **declared):
    """A model folder whose tokenizer_config.json holds the chat template, and a pinned manifest for the rest of
    Template (tool call format, reasoning history). Returns (folder, manifest folder)."""
    root = tempfile.mkdtemp()
    folder = os.path.join(root, "model")
    os.makedirs(folder)
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "llama", "hidden_size": 8}, f)
    if template is not None:
        with open(os.path.join(folder, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({"chat_template": template}, f)
    manifests = os.path.join(root, "manifests")
    os.makedirs(manifests)
    if declared:
        m = manifest.pin(manifest.Manifest(manifest.sha256_of(folder),
                                           (Fact("Template", Template(**declared), Source("manifest", "test"),
                                                 Certainty.DECLARED),)))
        manifest.save(m, os.path.join(manifests, f"{m.sha256}.json"))
    return folder, manifests


def facts_of(folder, manifests, held=None):
    os.environ[load.ENV_MANIFESTS] = manifests
    try:
        request_contract.reset()
        return request_contract.declared(folder, held)
    finally:
        os.environ.pop(load.ENV_MANIFESTS, None)


def quiet(fn, *a, **kw):
    """(decisions, the refusal's text or None)."""
    try:
        with redirect_stdout(io.StringIO()):
            return fn(*a, **kw), None
    except core.RoleError as e:
        return None, str(e)


def one(ds):
    assert ds is not None and len(ds) == 1, ds
    return ds[0]


# --- template -------------------------------------------------------------------------------------------------

def test_the_declared_template_passes_and_is_counted():
    facts = facts_of(*model_folder())
    d = one(request_contract.template(B, C, facts, TEMPLATE_TEXT, False, "picked by the engine", LOAD))
    assert d.verdict is Verdict.PASS, d
    s = request_contract.stats(B)
    assert s["checks"] == 1 and s["passed"] == {"template": 1} and s["refused"] == 0, s


def test_a_template_the_request_names_is_refused_never_overridden():
    facts = facts_of(*model_folder())
    ds, err = quiet(request_contract.template, B, C, facts, OTHER_TEXT, True, "named by the request", LOAD)
    assert err and "explicit choice contradicts the declaration" in err, err
    assert request_contract.stats(B)["refused"] == 1


def test_a_template_the_engine_picked_by_itself_is_refused():
    facts = facts_of(*model_folder())
    ds, err = quiet(request_contract.template, B, C, facts, OTHER_TEXT, False, "a fallback", LOAD)
    assert err and "no resolution is registered" in err, err


def test_a_template_that_cannot_be_named_is_unknown_and_recorded_once():
    facts = facts_of(*model_folder())
    first = len(load.LEDGER.decisions)
    for _ in range(3):
        d = one(quiet(request_contract.template, B, C, facts, None, False, "a named template", LOAD)[0])
        assert d.verdict is Verdict.UNKNOWN and not d.blocking, d
    assert len(load.LEDGER.decisions) - first == 1, "a non-blocking unknown is recorded once, then counted"
    assert request_contract.stats(B)["skipped"] == 3


def test_nothing_declared_is_refused_under_require():
    facts = facts_of(*model_folder(template=None))
    ds, err = quiet(request_contract.template, B, C, facts, OTHER_TEXT, False, "a fallback", LOAD)
    assert err and "nothing declares it" in err, err


def test_the_engines_own_template_stands_in_when_the_files_declare_none():
    """A model served by a hub name: no folder to read, so the template the engine's tokenizer loaded is the
    declaration (as the engine's config object is when there are no files)."""
    facts = facts_of("org/model-on-the-hub", tempfile.mkdtemp(), held=TEMPLATE_TEXT)
    [t] = facts.get("Template")
    assert t.source.kind == "config" and t.value.chat_template_sha256 == sha256_text(TEMPLATE_TEXT), t
    assert one(request_contract.template(B, C, facts, TEMPLATE_TEXT, False, "picked", LOAD)).verdict is Verdict.PASS
    ds, err = quiet(request_contract.template, B, C, facts, OTHER_TEXT, True, "named", LOAD)
    assert err, "an explicit other template is still refused"


# --- history --------------------------------------------------------------------------------------------------

H = "request:test.reasoning_history"


def test_keep_passes_when_every_earlier_turn_carries_its_reasoning():
    facts = facts_of(*model_folder(reasoning_history="keep"))
    assert one(request_contract.history(H, C, facts, [True, True], "conversation", LOAD)).verdict is Verdict.PASS


def test_keep_refuses_a_turn_without_its_reasoning():
    """market L13: an OpenAI-compatible integration dropped earlier reasoning."""
    facts = facts_of(*model_folder(reasoning_history="keep"))
    ds, err = quiet(request_contract.history, H, C, facts, [True, False, True], "conversation", LOAD)
    assert err and "1 of 3 earlier assistant turns reach the template without their reasoning" in err, err
    ds, err = quiet(request_contract.history, H, C, facts, [False], "conversation", REFUSE)
    assert err and "policy repairs nothing" in err, err


def test_keep_with_a_turn_that_cannot_be_read_is_unknown():
    facts = facts_of(*model_folder(reasoning_history="keep"))
    d = one(quiet(request_contract.history, H, C, facts, [True, None], "conversation", LOAD)[0])
    assert d.verdict is Verdict.UNKNOWN and not d.blocking, d
    ds, err = quiet(request_contract.history, H, C, facts, [None], "conversation", Policy(mode="debug"))
    assert err, "blocking in debug mode"


def test_drop_or_no_declaration_or_no_earlier_turn_decides_nothing():
    for facts in (facts_of(*model_folder(reasoning_history="drop")), facts_of(*model_folder())):
        assert request_contract.history(H, C, facts, [False, False], "conversation", LOAD) == []
    facts = facts_of(*model_folder(reasoning_history="keep"))
    assert request_contract.history(H, C, facts, [], "conversation", LOAD) == []
    assert request_contract.stats(H)["skipped"] == 1 and request_contract.stats(H)["checks"] == 0


# --- settings -------------------------------------------------------------------------------------------------

S = "request:test.settings"


def test_settings_all_read_pass():
    d = one(request_contract.settings(S, C, ["enable_thinking", "add_generation_prompt"],
                                      ["add_generation_prompt", "enable_thinking"], "the request", "the template",
                                      LOAD))
    assert d.verdict is Verdict.PASS and d.name == "Coverage", d


def test_a_setting_nobody_reads_is_refused():
    """market L07: reasoning_effort was ignored and every request ran at the default."""
    ds, err = quiet(request_contract.settings, S, C, ["reasoning_effort", "add_generation_prompt"],
                    ["add_generation_prompt"], "the request", "the template", LOAD)
    assert err and "reasoning_effort" in err, err
    d = [x for x in load.LEDGER.decisions if x.contract.boundary == S][-1]
    assert d.chosen.value.left == ("reasoning_effort",) and d.chosen.value.taken == 1, d.chosen


def test_nothing_given_decides_nothing():
    assert request_contract.settings(S, C, [], [], "the request", "the template", LOAD) == []


# --- window: the context a request is given against its prompt (market L05) ------------------------------------

W = "request:test.window"


def test_a_prompt_that_fits_its_context_passes():
    request_contract.reset()
    d = one(request_contract.window(W, C, 1000, 2048, "default", 32768, "request", LOAD))
    assert d.verdict is Verdict.PASS, d


def test_a_default_context_is_extended_when_the_model_has_room():
    """Ollama cut a 10983-token prompt to its default 2048 (the log line of issue #7043)."""
    request_contract.reset()
    d = one(quiet(request_contract.window, W, C, 10983, 2048, "default", 32768, "request", LOAD)[0])
    assert (d.verdict, d.handle, d.target) == (Verdict.RESOLVED, "extend_context", 10983), d
    assert request_contract.stats(W)["resolved"] == 1


def test_a_context_the_user_set_or_no_room_leaves_the_cut():
    request_contract.reset()
    ds, err = quiet(request_contract.window, W, C, 10983, 2048, "user", 32768, "request", LOAD)
    assert err and "explicit choice" in err, err
    ds, err = quiet(request_contract.window, W, C, 10983, 2048, "default", 8192, "request", LOAD)
    assert err and "no resolution" in err, err


# --- the tool parser, once at start ---------------------------------------------------------------------------

def with_evidence(consumer, evidence):
    table = caps.load_table()
    return caps.from_rows([r if r.consumer != consumer else caps.Capability(**{**r.__dict__, "evidence": evidence})
                           for r in table.rows], dict(table.prefer))


def test_tool_parser_passes_switches_or_refuses():
    facts = facts_of(*model_folder(tool_call_format="hermes"))
    assert one(load.tool_parser("vllm", "hermes", facts, policy=LOAD)).verdict is Verdict.PASS
    # the packaged table has seen vLLM's hermes parser read Qwen3's tool calls (testbed/results/m53)
    d = one(load.tool_parser("vllm", "pythonic", facts, policy=LOAD))
    assert d.verdict is Verdict.RESOLVED and d.target == "hermes" and d.handle == "switch_tool_parser", d
    d = one(load.tool_parser("vllm", "pythonic", facts, policy=REFUSE))
    assert d.verdict is Verdict.REFUSED, d
    d = one(load.tool_parser("vllm", "pythonic", facts, policy=LOAD, can_switch=False))
    assert d.verdict is Verdict.REFUSED, d
    # a parser known only from reading the code is not switched to
    d = one(load.tool_parser("vllm", "pythonic", facts, table=with_evidence("vllm.tool_parser.hermes", "code"),
                             policy=LOAD))
    assert d.verdict is Verdict.REFUSED and "no resolution" in d.rule, d


def test_tool_parser_unknown_parser_or_no_declaration():
    facts = facts_of(*model_folder(tool_call_format="hermes"))
    d = one(load.tool_parser("vllm", "kimi_k2", facts, policy=LOAD))
    assert d.verdict is Verdict.UNKNOWN and "not in the capability table" in d.chosen.source.where, d
    assert load.tool_parser("vllm", "hermes", facts_of(*model_folder()), policy=LOAD) == []
    assert load.tool_parser("vllm", None, facts, policy=LOAD) == []


def test_route_takes_a_consumer_that_reads_the_declared_value():
    rows = [{"consumer": "e.parser.a", "fact": "Template.tool_call_format", "honours": False, "reads": "pythonic",
             "evidence": "measured", "ref": "r"},
            {"consumer": "e.parser.b", "fact": "Template.tool_call_format", "honours": False, "reads": "hermes",
             "evidence": "measured", "ref": "r"},
            {"consumer": "e.parser.c", "fact": "Template.tool_call_format", "honours": False, "reads": "hermes",
             "evidence": "code", "ref": "r"}]
    t = caps.from_rows(rows, {"e.parser": ["a", "c", "b"]})
    assert caps.route(t, "e.parser", Template(tool_call_format="hermes")) == "b"
    assert caps.route(t, "e.parser", Template(tool_call_format="hermes"), measured_only=False) == "c"
    assert caps.route(t, "e.parser", Template(tool_call_format="llama3_json")) is None


def test_an_error_inside_entail_never_breaks_the_server():
    def broken():
        raise KeyError("x")

    with redirect_stdout(io.StringIO()):
        assert request_contract.guarded("request:test.broken", C, broken) is None
        assert request_contract.guarded("request:test.broken", C, lambda: 1) is None   # left alone from then on
    tally.reset("request:test.broken")


if __name__ == "__main__":
    core.set_mode("load")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
