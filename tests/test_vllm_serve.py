"""Tests for the vLLM server adapter (ROADMAP M5.3) without vLLM: what read_choice makes of a request, that deciding
before rendering reaches the core's rules, and the parser hook. Where vLLM is not installed, stand-ins give the three
vLLM functions the adapter reads (resolve_chat_template, the template's variables, apply_chat_template's parameters)
and a ParserManager; where it is, the same tests run against vLLM's own functions (the stand-in tokenizer and model
config carry what those read). The hooks on a real server are measured in testbed/m53_serve.py.

Run: python tests/test_vllm_serve.py
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
from entail import caps, core, load, manifest, policies, request_contract  # noqa: E402
from entail.adapters import vllm_serve as vs  # noqa: E402
from entail.facts import Certainty, Fact, Source, Template  # noqa: E402

REAL_VLLM = importlib.util.find_spec("vllm") is not None
TEXT = ("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
        "{% if tools %}{{ tools | length }}{% endif %}{% if enable_thinking %}<think>{% endif %}")
BASE = frozenset({"add_generation_prompt", "chat_template", "continue_final_message", "conversation", "documents",
                  "max_length", "padding", "return_assistant_tokens_mask", "return_dict", "return_tensors", "self",
                  "tokenize", "tokenizer_kwargs", "tools", "truncation"})   # vllm 0.30 + transformers 5.17


class ParserManager:
    @classmethod
    def get_parser(cls, tool_parser_name=None, reasoning_parser_name=None, enable_auto_tools=False, model_name=None,
                   is_harmony=False):
        return ("built", tool_parser_name, reasoning_parser_name)


def _stand_ins():
    """vllm.renderers.hf and vllm.parser.parser_manager as far as the adapter reads them."""
    import jinja2
    import jinja2.meta

    hf = types.ModuleType("vllm.renderers.hf")

    def resolve_chat_template(tokenizer, chat_template, tools, *, model_config):
        if chat_template is not None:
            return chat_template
        ct = tokenizer.chat_template
        return ct if isinstance(ct, str) else ct["tool_use" if tools else "default"]

    hf.resolve_chat_template = resolve_chat_template
    hf._cached_resolve_chat_template_kwargs = lambda text: jinja2.meta.find_undeclared_variables(
        jinja2.Environment().parse(text))
    hf._get_hf_base_chat_template_params = lambda: BASE
    pm = types.ModuleType("vllm.parser.parser_manager")
    pm.ParserManager = ParserManager
    mods = {n: types.ModuleType(n) for n in ("vllm", "vllm.renderers", "vllm.parser")}
    mods.update({"vllm.renderers.hf": hf, "vllm.parser.parser_manager": pm})
    return mods


if not REAL_VLLM:
    sys.modules.update(_stand_ins())


def model_folder(**declared):
    root = tempfile.mkdtemp()
    folder, manifests = os.path.join(root, "model"), os.path.join(root, "manifests")
    os.makedirs(folder)
    os.makedirs(manifests)
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "llama"}, f)
    with open(os.path.join(folder, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"chat_template": TEXT}, f)
    if declared:
        m = manifest.pin(manifest.Manifest(manifest.sha256_of(folder), (
            Fact("Template", Template(**declared), Source("manifest", "test"), Certainty.DECLARED),)))
        manifest.save(m, os.path.join(manifests, f"{m.sha256}.json"))
    os.environ[load.ENV_MANIFESTS] = manifests
    request_contract.reset()
    return folder


def params(template=None, **kwargs):
    return SimpleNamespace(chat_template=template, chat_template_kwargs={"add_generation_prompt": True,
                                                                         "return_dict": False, **kwargs})


def request(**fields):
    return SimpleNamespace(chat_template_kwargs=fields.pop("chat_template_kwargs", None), **fields)


EMPTY = tempfile.mkdtemp()   # a local folder with no processor in it: vLLM's processor lookup fails there at once


def tokenizer(template, folder=EMPTY):
    """A tokenizer as far as vLLM's resolve_chat_template reads one (transformers' get_chat_template)."""
    def get_chat_template(chat_template=None, tools=None):
        if isinstance(template, dict):
            if chat_template in template:
                return template[chat_template]
            if chat_template is not None:
                return chat_template
            return template["tool_use" if tools and "tool_use" in template else "default"]
        return template if chat_template is None else chat_template

    return SimpleNamespace(chat_template=template, name_or_path=folder, get_chat_template=get_chat_template)


def model_config(folder=EMPTY):
    return SimpleNamespace(model=folder, revision=None, code_revision=None, trust_remote_code=False,
                           hf_config=SimpleNamespace(model_type="llama"))


def renderer(folder, template=TEXT):
    return SimpleNamespace(model_config=model_config(folder), get_tokenizer=lambda: tokenizer(template, folder))


def stops(fn, text):
    try:
        with redirect_stdout(io.StringIO()):
            fn()
    except core.RoleError as e:
        assert text in str(e), (text, str(e))
        return True
    return False


# --- read_choice ----------------------------------------------------------------------------------------------

def test_settings_are_the_requests_own_and_arrive_when_the_template_reads_them():
    tok = tokenizer(TEXT)
    # vLLM's own settings (and a server default such as cohere_format) are not the request's
    p = params(cohere_format="cmd4", enable_thinking=False)
    assert vs.read_choice("settings", p, request(chat_template_kwargs={"enable_thinking": False}), TEXT) == \
        (["enable_thinking"], ["enable_thinking"])
    # a misspelt setting does not arrive
    p = params(enable_thinkng=False)
    assert vs.read_choice("settings", p, request(chat_template_kwargs={"enable_thinkng": False}), TEXT) == \
        (["enable_thinkng"], [])
    # reasoning_effort "none" arrives as the enable_thinking false vLLM derives; a level does not
    assert vs.read_choice("settings", params(reasoning_effort="none", enable_thinking=False),
                          request(reasoning_effort="none"), TEXT) == (["reasoning_effort"], ["reasoning_effort"])
    assert vs.read_choice("settings", params(reasoning_effort="low", enable_thinking=True),
                          request(reasoning_effort="low"), TEXT) == (["reasoning_effort"], [])
    # documents go to the template only, and this one does not read them; tools it reads
    got = vs.read_choice("settings", params(documents=[{"text": "d"}], tools=[{"type": "function"}]),
                         request(documents=[{"text": "d"}]), TEXT)
    assert got == (["documents", "tools"], ["tools"]), got
    # market L07: the request sets it, the server never hands it to the template - read from the request, it is lost
    reads_effort = TEXT + "{% if reasoning_effort %}{{ reasoning_effort }}{% endif %}"
    assert vs.read_choice("settings", params(), request(reasoning_effort="high"), reads_effort) == \
        (["reasoning_effort"], [])
    assert vs.read_choice("settings", params(reasoning_effort="high"), request(reasoning_effort="high"),
                          reads_effort) == (["reasoning_effort"], ["reasoning_effort"])
    assert tok


def test_template_named_or_picked_and_named_templates():
    tok = tokenizer({"default": "A", "tool_use": "B"})
    assert vs.read_choice("template", tok, params(), model_config()) == ("A", False)
    assert vs.read_choice("template", tok, params("C"), model_config()) == ("C", True)
    assert vs.read_choice("template", tok, params(tools=[{"type": "function"}]), model_config()) == (None, False), \
        "a named template besides the default: what it declares cannot be said in the vocabulary"


def test_turns_read_the_reasoning_field_vllm_passes():
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a", "reasoning": "r"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "b"},
            {"role": "tool", "content": "x"}]
    assert vs.read_choice("turns", conv, "qwen3") == [True, False]
    assert vs.read_choice("turns", conv, "") == [True, None], "no reasoning parser: it may be in the content"
    assert vs.read_choice("turns", conv, "minimax_m2_append_think") == [True, None]
    assert vs.read_choice("turns", conv[:4], "qwen3") == [True], "a final assistant turn is being continued"


def test_fields_are_the_ones_the_schema_does_not_know():
    req = SimpleNamespace(model_extra={"guided_json": {}}, model_fields_set={"model", "messages", "guided_json"})
    assert vs.read_choice("fields", req) == (["guided_json", "messages", "model"], ["messages", "model"])


# --- deciding before rendering, through the core ----------------------------------------------------------------

ASK = [{"role": "user", "content": "q"}]


def test_before_render_passes_the_declared_template_and_counts_it():
    core.set_mode("load")
    folder = model_folder(reasoning_history="drop")
    token = vs._REQUEST.set((request(chat_template_kwargs={"enable_thinking": False}), None, None))
    try:
        vs._before_render(renderer(folder), ASK, params(enable_thinking=False))
    finally:
        vs._REQUEST.reset(token)
    assert request_contract.stats(vs.TEMPLATE)["passed"] == {"template": 1}
    assert request_contract.stats(vs.SETTINGS)["passed"] == {"settings": 1}
    assert request_contract.stats(vs.HISTORY)["skipped"] == 1


def test_before_render_refuses_the_planted_requests():
    core.set_mode("load")
    folder = model_folder()
    r = renderer(folder)
    assert stops(lambda: vs._before_render(r, ASK, params("{{ messages }}")), "explicit choice")
    token = vs._REQUEST.set((request(chat_template_kwargs={"enable_thinkng": False}), None, None))
    try:
        assert stops(lambda: vs._before_render(r, ASK, params(enable_thinkng=False)), "enable_thinkng")
    finally:
        vs._REQUEST.reset(token)
    token = vs._REQUEST.set((request(documents=[{"text": "d"}]), None, None))
    try:
        assert stops(lambda: vs._before_render(r, ASK, params(documents=[{"text": "d"}])), "documents")
    finally:
        vs._REQUEST.reset(token)


def test_before_render_refuses_dropped_reasoning_when_the_model_keeps_it():
    core.set_mode("load")
    folder = model_folder(reasoning_history="keep")
    vs._SERVED[folder] = "qwen3"
    conv = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}]
    assert stops(lambda: vs._before_render(renderer(folder), conv, params()), "without their reasoning")
    conv[1]["reasoning"] = "because"
    vs._before_render(renderer(folder), conv, params())
    assert request_contract.stats(vs.HISTORY)["passed"] == {"reasoning_history": 1}


def test_a_render_without_a_request_does_not_decide_settings():
    """vLLM's renderer warms up at start with the server's defaults, where vLLM puts its own cohere_format: who gave
    a setting cannot be told there (found on the real server: the warm-up was refused and vLLM logged it)."""
    core.set_mode("load")
    folder = model_folder()
    vs._before_render(renderer(folder), ASK, params(cohere_format="cmd4"))
    assert request_contract.stats(vs.SETTINGS)["checks"] == 0
    assert request_contract.stats(vs.TEMPLATE)["passed"] == {"template": 1}


def test_fields_refuses_a_field_the_server_ignores():
    core.set_mode("load")
    req = SimpleNamespace(model_extra={"guided_json": {}}, model_fields_set={"model", "messages", "guided_json"})
    assert stops(lambda: vs._fields(req, policies.current()), "guided_json")


# --- the parser hook --------------------------------------------------------------------------------------------

def test_get_parser_checks_the_tool_parser_once_and_switches_to_a_measured_one():
    if REAL_VLLM:
        print("  (vllm is installed: the stand-in ParserManager is not used)")
        return
    pm = sys.modules["vllm.parser.parser_manager"].ParserManager
    table = caps._TABLE
    try:
        core.set_mode("load")
        assert vs.install_parsers() == 1 and vs.install_parsers() == 0
        folder = model_folder(tool_call_format="hermes")
        assert pm.get_parser(tool_parser_name="hermes", reasoning_parser_name="qwen3", enable_auto_tools=True,
                             model_name=folder) == ("built", "hermes", "qwen3")
        assert vs._SERVED[folder] == "qwen3"
        # hermes is measured to read its format: the server is started with it instead, once per model and parser
        with redirect_stdout(io.StringIO()):
            got = pm.get_parser(tool_parser_name="pythonic", enable_auto_tools=True, model_name=folder)
        assert got == ("built", "hermes", None), got
        assert pm.get_parser(tool_parser_name="pythonic", enable_auto_tools=True, model_name=folder) == got
        # no auto tool choice: no tool parser runs, nothing is decided
        assert pm.get_parser(tool_parser_name="pythonic", model_name=folder) == ("built", "pythonic", None)
        # a parser known only from reading the code is not switched to: the start is refused
        real = caps.load_table()
        caps._TABLE = caps.from_rows([r if r.consumer != "vllm.tool_parser.hermes" else
                                      caps.Capability(**{**r.__dict__, "evidence": "code"}) for r in real.rows],
                                     dict(real.prefer))
        vs._PARSERS.clear()
        assert stops(lambda: pm.get_parser(tool_parser_name="pythonic", enable_auto_tools=True, model_name=folder),
                     "no resolution")
        assert vs.uninstall() == 1 and pm.get_parser(tool_parser_name="x") == ("built", "x", None)
    finally:
        caps._TABLE = table
        vs._ORIG.clear()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
