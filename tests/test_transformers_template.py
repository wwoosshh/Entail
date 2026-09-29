"""Tests for the chat template where transformers applies it (adapters/transformers_template.py) and where SGLang's
server renders with a conversation template of its own (adapters/sglang_serve.py), ROADMAP M9.3. A tiny tokenizer is
built in memory and saved to a folder, so its chat template is declared the way a model folder declares it; the
rules are request_contract's (tests/test_request_contract.py). Runs on the CPU with transformers and tokenizers.
Run: python tests/test_transformers_template.py
"""
import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("ENTAIL_LOG_DIR", "off")
from entail import core, load, request_contract  # noqa: E402
from entail.adapters import sglang_serve, transformers_template as tt  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402

DECLARED = "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
OTHER = "{% for m in messages %}[{{ m['role'] }}] {{ m['content'] }}{% endfor %}"
ASK = [{"role": "user", "content": "hi there"}]
# a model's template that writes a line per role and content, and an app's copy that drops earlier reasoning
LINES = "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
LINES_DROP = "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'].split('</think>')[-1] }}\n{% endfor %}"


def tokenizer_folder(template=DECLARED):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hi": 1, "there": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.chat_template = template
    d = tempfile.mkdtemp(prefix="entail_template_")
    fast.save_pretrained(d)
    return d


class Hooked:
    """The hook installed, entail on, fresh counts; everything undone afterwards."""

    def __init__(self, mode="load", template=DECLARED):
        self.mode, self.template = mode, template

    def __enter__(self):
        from transformers import AutoTokenizer

        tt.install()
        request_contract.reset()
        core.set_mode(self.mode)
        self.folder = tokenizer_folder(self.template)
        self.tok = AutoTokenizer.from_pretrained(self.folder)
        self.n = len(load.LEDGER.decisions)
        return self

    def made(self):
        return [d for d in load.LEDGER.decisions[self.n:] if d.contract.boundary.startswith("request:")]

    def __exit__(self, *exc):
        core.set_mode("off")
        tt.uninstall()
        request_contract.reset()
        shutil.rmtree(self.folder, ignore_errors=True)


def quiet(fn):
    with redirect_stdout(io.StringIO()):
        return fn()


def test_the_declared_template_passes_and_is_counted():
    with Hooked() as h:
        text = h.tok.apply_chat_template(ASK, tokenize=False)
        assert text == "<|user|>hi there", text
        s = request_contract.stats(tt.TEMPLATE)
        assert s["checks"] == 1 and s["passed"] == {"template": 1} and not h.made(), (s, h.made())


def test_a_template_passed_in_that_is_not_the_declared_one_is_reported_and_the_call_goes_on():
    """The default policy (M5.4): broken, recorded, and the render is the one the caller asked for."""
    with Hooked() as h:
        text = quiet(lambda: h.tok.apply_chat_template(ASK, chat_template=OTHER, tokenize=False))
        assert text == "[user] hi there", text
        d, = h.made()
        assert d.verdict is Verdict.BROKEN and d.rule == RULES["user_choice"] and d.chosen.source.kind == "user", d


def test_a_template_of_its_own_that_renders_the_same_prompt_passes():
    """Issue #35: Xinference passes its own copy of the model's template to apply_chat_template. Held to the prompt it
    renders for the request, not to its text: the same prompt is what the request means, and the render goes on."""
    with Hooked() as h:
        text = quiet(lambda: h.tok.apply_chat_template(ASK, chat_template="{# the app's copy #}" + DECLARED,
                                                       tokenize=False))
        assert text == "<|user|>hi there", text
        s = request_contract.stats(tt.TEMPLATE)
        assert s["checks"] == 1 and s["passed"] == {"template": 1} and not h.made(), (s, h.made())
        ids = quiet(lambda: h.tok.apply_chat_template(ASK, chat_template="{# the app's copy #}" + DECLARED))
        assert request_contract.stats(tt.TEMPLATE)["passed"] == {"template": 2} and not h.made(), ids


def test_a_template_of_its_own_is_broken_where_its_prompt_parts_from_the_declared_one():
    """The app's copy differs on one kind of conversation only: an earlier assistant turn with a <think> block, which
    the copy drops and the model's template keeps (Xinference's Qwen3 template, #35). The line says where the prompts
    part; a template the caller passed stays the caller's choice, and the call goes on with it."""
    conv = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<think>r</think>b"},
            {"role": "user", "content": "c"}]
    with Hooked(template=LINES) as h:
        quiet(lambda: h.tok.apply_chat_template(ASK, chat_template=LINES_DROP, tokenize=False))
        assert not h.made(), "one user turn: the same prompt"
        text = quiet(lambda: h.tok.apply_chat_template(conv, chat_template=LINES_DROP, tokenize=False))
        assert text == "<|user|>\na\n<|assistant|>\nb\n<|user|>\nc\n", text
        d, = h.made()
        assert d.verdict is Verdict.BROKEN and d.rule == RULES["user_choice"] and d.chosen.source.kind == "user", d
        assert "the prompts part at line 4: the declared template's '<think>r</think>b', this one's 'b'" in \
            d.chosen.source.where, d.chosen.source.where
        batch = quiet(lambda: h.tok.apply_chat_template([ASK, conv], chat_template=LINES_DROP, tokenize=False))
        assert len(batch) == 2 and "conversation 2, line 4" in h.made()[-1].chosen.source.where


def test_where_the_policy_stops_the_call_is_refused_before_it_renders():
    with Hooked() as h:
        os.environ["ENTAIL_ON_BROKEN"] = "stop"
        try:
            quiet(lambda: h.tok.apply_chat_template(ASK, chat_template=OTHER, tokenize=False))
            raise AssertionError("a template the model does not declare was rendered under the stopping policy")
        except core.RoleError as e:
            assert "refused at request:transformers.chat_template" in str(e), e
        finally:
            del os.environ["ENTAIL_ON_BROKEN"]


def test_a_template_changed_after_loading_is_reported():
    """The tokenizer's own template, set after it was loaded, is not the one the folder declares: the engine's
    choice (not the caller's), and nothing repairs it."""
    with Hooked() as h:
        h.tok.chat_template = OTHER
        quiet(lambda: h.tok.apply_chat_template(ASK, tokenize=False))
        d, = h.made()
        assert d.verdict is Verdict.BROKEN and d.rule == RULES["no_resolution"] and d.chosen.source.kind == "engine", d


def test_a_server_that_decides_its_own_renders_is_not_decided_twice():
    """vLLM's server adapter decides before rendering and marks the render; the tokenizer's call inside steps aside."""
    with Hooked() as h:
        with request_contract.deciding():
            h.tok.apply_chat_template(ASK, chat_template=OTHER, tokenize=False)
        assert not request_contract.decided_elsewhere()
        assert request_contract.stats(tt.TEMPLATE)["checks"] == 0 and not h.made()


def test_earlier_reasoning_is_read_per_turn():
    conv = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "reasoning_content": "r"},
            {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"},
            {"role": "user", "content": "e"}, {"role": "assistant", "content": "<think>x</think>f"},
            {"role": "user", "content": "g"}]
    assert tt.read_choice("turns", conv) == [True, False, True]
    assert tt.read_choice("turns", conv[:4]) == [True]   # the last assistant turn is the one being continued
    assert tt._conversations([ASK, ASK]) == [ASK, ASK] and tt._conversations(ASK) == [ASK]


def test_off_mode_decides_nothing():
    with Hooked(mode="off") as h:
        h.tok.apply_chat_template(ASK, chat_template=OTHER, tokenize=False)
        assert request_contract.stats(tt.TEMPLATE)["checks"] == 0 and not h.made()


def test_sglang_s_own_conversation_template_is_reported():
    """SGLang renders with a conversation template of its own name, not the model's: the user's choice when
    --chat-template named it, SGLang's when it chose one from the model path by itself (issue #37: DeepSeek-OCR's
    'deepseek-ocr' was called named by --chat-template)."""
    with Hooked() as h:
        def serving(arg):
            return SimpleNamespace(template_manager=SimpleNamespace(chat_template_name="chatml"),
                                   tokenizer_manager=SimpleNamespace(model_path=h.folder, tokenizer=h.tok,
                                                                     server_args=SimpleNamespace(chat_template=arg)))
        quiet(lambda: sglang_serve._decide(serving("chatml")))
        d, = h.made()
        assert d.verdict is Verdict.BROKEN and d.rule == RULES["user_choice"] and "'chatml'" in str(d.chosen.source), d
        assert "named by --chat-template" in d.chosen.source.where
        quiet(lambda: sglang_serve._decide(serving(None)))
        d = h.made()[-1]
        assert d.verdict is Verdict.BROKEN and d.rule == RULES["no_resolution"] and d.chosen.source.kind == "engine", d
        assert "chosen by SGLang from the model path (no --chat-template given)" in d.chosen.source.where


def installed(name, code):
    """A module installed the way pip installs one (<tmp>/site-packages/<name>/__init__.py), imported."""
    import importlib

    site = os.path.join(tempfile.mkdtemp(prefix="entail_site_"), "site-packages")
    os.makedirs(os.path.join(site, name))
    with open(os.path.join(site, name, "__init__.py"), "w", encoding="utf-8") as f:
        f.write(code)
    sys.path.insert(0, site)
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(site)


def test_a_template_an_installed_package_passes_is_its_choice_not_the_users():
    """Issue #37: Xinference passes its own template to apply_chat_template; the person running it passed nothing,
    and entail called it the user's choice. A template an installed package's code passes is named as that
    package's (the engine's side); one the user's own code passes stays the user's (the tests above)."""
    app = installed("entail_fake_app", "def render(tok, conversation, template):\n"
                                       "    return tok.apply_chat_template(conversation, chat_template=template, "
                                       "tokenize=False)\n")
    with Hooked() as h:
        assert quiet(lambda: app.render(h.tok, ASK, OTHER)) == "[user] hi there"
        d, = h.made()
        assert d.verdict is Verdict.BROKEN and d.rule == RULES["no_resolution"] and d.chosen.source.kind == "engine", d
        assert "passed to apply_chat_template by entail_fake_app (entail_fake_app.render)" in d.chosen.source.where
        quiet(lambda: app.render(h.tok, ASK, "{# the app's copy #}" + DECLARED))
        assert len(h.made()) == 1, "the same prompt passes, whoever passed the template (#35)"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
