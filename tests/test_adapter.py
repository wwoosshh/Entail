"""Tests for the transformers attention adapter v2 (ROADMAP M3.3): what it reads, that its decisions are the core's
(load.attention) and land in the ledger, and install/uninstall. Run: python tests/test_adapter.py"""
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
# These tests check what stops, so they run under the policy that stopped before M5.4 (ENTAIL_ON_BROKEN=stop,
# unknown meaning-changing facts required); the default, which reports and goes on, is tested in test_report.py.
os.environ["ENTAIL_ON_BROKEN"] = "stop"
os.environ["ENTAIL_UNKNOWN"] = "require"
from entail import core, load  # noqa: E402
from entail.adapters import transformers_adapter as adapter  # noqa: E402
from entail.contracts import Verdict  # noqa: E402

GEMMA = os.path.join(os.path.expanduser(os.environ.get("ENTAIL_TEST_MODELS", "~/models")), "gemma-2-2b-it")


def test_read_choice():
    from types import SimpleNamespace

    assert adapter.read_choice("sdpa") == ("attention", "sdpa")
    assert adapter.read_choice("paged|sdpa") == ("paged_attention", "sdpa")
    assert [h.target.rsplit(".", 1)[1] for h in adapter.hooks()] == ["_check_and_adjust_attn_implementation",
                                                                    "tie_weights"]
    assert adapter.loader_ties(SimpleNamespace(tie_word_embeddings=False)) is False
    assert adapter.loader_ties(SimpleNamespace(text_config=SimpleNamespace(tie_word_embeddings=True))) is True
    assert adapter.loader_ties(SimpleNamespace()) is None


def test_the_model_contracts_run_once_per_model():
    if not os.path.isdir(GEMMA):
        print("skip (no local model)")
        return
    with _On() as on, redirect_stdout(io.StringIO()):
        model = _gemma_on_meta("eager")           # post_init's call on a meta model decides nothing yet (M11.2)
        assert not [d for d in on.new() if d.contract.boundary == "load:transformers.loader"]
        model.tie_weights(missing_keys=set())     # from_pretrained's call, after loading: decides once
        model.tie_weights(missing_keys=set())     # a second call decides nothing again
        loader = [d for d in on.new() if d.contract.boundary == "load:transformers.loader"]
        rope = [d for d in on.new() if d.contract.boundary == "load:transformers.config.rope_parameters"]
    assert len(loader) == 1 and loader[0].verdict is Verdict.PASS          # no lm_head.weight: tied, as declared
    assert len(rope) == 1 and rope[0].verdict is Verdict.PASS


def test_what_the_loader_left_is_read_from_the_model():
    """M11.2: after tie_weights the adapter reads whether the head shares the embedding's tensor, and only at a call
    that has the weights (from_pretrained's passes missing_keys; a meta model's post_init call is skipped)."""
    import torch
    from types import SimpleNamespace

    w = torch.zeros(4, 2)
    emb = lambda t: SimpleNamespace(weight=t)  # noqa: E731
    tied = SimpleNamespace(get_output_embeddings=lambda: emb(w), get_input_embeddings=lambda: emb(w))
    own = SimpleNamespace(get_output_embeddings=lambda: emb(w.clone()), get_input_embeddings=lambda: emb(w))
    none = SimpleNamespace(get_output_embeddings=lambda: None, get_input_embeddings=lambda: emb(w))
    m = torch.empty(2, device="meta")
    meta = SimpleNamespace(get_output_embeddings=lambda: emb(m), get_input_embeddings=lambda: emb(m))
    assert adapter.tied_in_memory(tied) is True and adapter.tied_in_memory(own) is False
    assert adapter.tied_in_memory(none) is None and adapter.tied_in_memory(meta) is None
    real = SimpleNamespace(parameters=lambda: iter([torch.nn.Parameter(w)]))
    on_meta = SimpleNamespace(parameters=lambda: iter([torch.nn.Parameter(m)]))
    assert adapter.weights_there(real, (), {}) and not adapter.weights_there(on_meta, (), {})
    assert adapter.weights_there(on_meta, (), {"missing_keys": set()}) and adapter.weights_there(on_meta, (set(),), {})
    assert not adapter.weights_there(on_meta, (), {"missing_keys": None})


def test_install_is_reversible():
    from transformers import PreTrainedModel
    before = PreTrainedModel._check_and_adjust_attn_implementation
    assert adapter.install() == 1
    assert adapter.install() == 0  # already installed
    assert PreTrainedModel._check_and_adjust_attn_implementation is not before
    assert adapter.uninstall() == 1
    assert adapter.uninstall() == 0
    assert PreTrainedModel._check_and_adjust_attn_implementation is before


class _On:
    def __init__(self, policy="resolve"):
        self.policy = policy

    def __enter__(self):
        adapter.install()
        core.set_mode("load")
        core.set_policy(self.policy)
        self.n = len(load.LEDGER.decisions)
        return self

    def new(self):
        return load.LEDGER.decisions[self.n:]

    def __exit__(self, *exc):
        core.set_policy("resolve")
        core.set_mode("off")
        adapter.uninstall()


def _gemma_on_meta(impl):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(GEMMA), attn_implementation=impl)


def test_resolve_switches_to_an_implementation_that_honours_the_model():
    """Under the default policy a dropped property is repaired, not refused: sdpa becomes eager, and it is said."""
    if not os.path.isdir(GEMMA):
        print("skip (no local model)")
        return
    out = io.StringIO()
    with _On() as on, redirect_stdout(out):
        model = _gemma_on_meta("sdpa")
        assert model.config._attn_implementation == "eager", model.config._attn_implementation
        new = [d for d in on.new() if d.verdict is Verdict.RESOLVED]
    assert new and new[0].contract.consumer == "transformers.attention.sdpa" and new[0].target == "eager"
    assert new[0].declared.source.kind == "config" and "gemma-2-2b-it" in new[0].declared.source.where
    assert "[entail] resolved at load:transformers.attention" in out.getvalue()


def test_refuse_still_refuses():
    if not os.path.isdir(GEMMA):
        print("skip (no local model)")
        return
    with _On("refuse"):
        try:
            with redirect_stdout(io.StringIO()):
                _gemma_on_meta("sdpa")
        except core.RoleError as e:
            assert "policy repairs nothing" in str(e) and "stops here" in str(e), e
        else:
            raise AssertionError("refuse policy let a dropped property through")


def test_nothing_to_resolve_to_is_still_refused():
    """The paged kernels have no alternative that honours softcap: continuous batching is refused, not rerouted."""
    if not os.path.isdir(GEMMA):
        print("skip (no local model)")
        return
    with _On() as on:
        model = _gemma_on_meta("eager")
        try:
            with redirect_stdout(io.StringIO()):
                model.set_attn_implementation("paged|sdpa")
        except core.RoleError as e:
            assert "load:transformers.paged_attention" in str(e) and "no resolution is registered" in str(e), e
        else:
            raise AssertionError("a paged kernel that drops softcap was let through")
        assert any(d.verdict is Verdict.REFUSED for d in on.new())


def test_a_model_that_declares_nothing_for_attention_gets_no_decision():
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig

    tiny = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                       num_key_value_heads=1)
    with _On() as on, torch.device("meta"):
        AutoModelForCausalLM.from_config(tiny, attn_implementation="sdpa")
        assert [d for d in on.new() if d.contract.boundary.startswith("load:transformers.attention")] == []


def test_mode_off_does_nothing():
    """With the mode off the wrapper must not decide, raise or record, even on a violating pair."""
    if not os.path.isdir(GEMMA):
        print("skip test_mode_off_does_nothing (no local model)")
        return
    adapter.install()
    core.set_mode("off")
    n = len(load.LEDGER.decisions)
    try:
        assert _gemma_on_meta("sdpa").config._attn_implementation == "sdpa"
        assert len(load.LEDGER.decisions) == n
    finally:
        adapter.uninstall()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
