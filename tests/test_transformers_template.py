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


def tokenizer_folder():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hi": 1, "there": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.chat_template = DECLARED
    d = tempfile.mkdtemp(prefix="entail_template_")
    fast.save_pretrained(d)
    return d


class Hooked:
    """The hook installed, entail on, fresh counts; everything undone afterwards."""

    def __init__(self, mode="load"):
        self.mode = mode

    def __enter__(self):
        from transformers import AutoTokenizer

        tt.install()
        request_contract.reset()
        core.set_mode(self.mode)
        self.folder = tokenizer_folder()
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
    """SGLang renders with a conversation template of its own name (--chat-template chatml), not the model's."""
    with Hooked() as h:
        serving = SimpleNamespace(template_manager=SimpleNamespace(chat_template_name="chatml"),
                                  tokenizer_manager=SimpleNamespace(model_path=h.folder, tokenizer=h.tok))
        quiet(lambda: sglang_serve._decide(serving))
        d, = h.made()
        assert d.verdict is Verdict.BROKEN and d.rule == RULES["user_choice"] and "'chatml'" in str(d.chosen.source), d


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
