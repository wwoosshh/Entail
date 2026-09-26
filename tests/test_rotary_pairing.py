"""Tests for the rotary pairing contract in the core (ROADMAP M17.4; data/rotary_pairing.json): the declaration
from a config key, from the architecture table and from nothing; the kernel row by engine version (the retrospective
on vllm#42016: GLM-OCR on vLLM 0.22.0); the built layers against the declaration with and without the repair; the
vLLM adapter on fake modules. Pure Python, no engine.
Run: python tests/test_rotary_pairing.py"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, rotary_pairing_contract as rpc  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402

B, C = "load:test.rotary_pairing", "test.rotary_embedding"


def decided(fn):
    n = len(load.LEDGER.decisions)
    was = core.mode()
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            out = fn()
    finally:
        core.set_mode(was)
    return out, load.LEDGER.decisions[n:]


def test_the_declaration_comes_from_a_key_then_the_architecture_then_nothing():
    assert rpc.declared({"model_type": "deepseek_v3", "rope_interleave": True})[0] == "interleaved"
    assert rpc.declared({"model_type": "deepseek_v3", "rope_interleave": False})[0] == "split"
    assert rpc.declared({"model_type": "glm4"})[0] == "interleaved"
    assert rpc.declared({"model_type": "glm_ocr", "text_config": {"model_type": "glm_ocr_text"}})[0] == "interleaved"
    assert rpc.declared({"model_type": "qwen3"})[0] == "split"
    p, why = rpc.declared({"model_type": "some_new_arch"})
    assert p is None and "some_new_arch" in why
    assert rpc.declared(None)[0] is None
    assert rpc.declared({"model_type": "x", "text_config": {"is_neox_style": False}})[0] == "interleaved"
    fact = SimpleNamespace(value=SimpleNamespace(pairing="split"), source=SimpleNamespace(where="a Rotary fact"))
    assert rpc.declared({"model_type": "glm4"}, facts={"Rotary": [fact]}.get and None) == ("interleaved",
                                                                                          rpc.declared({"model_type": "glm4"})[1])
    facts = SimpleNamespace(get=lambda name: [fact] if name == "Rotary" else [])
    assert rpc.declared({"model_type": "glm4"}, facts=facts) == ("split", "a Rotary fact")


def test_the_kernel_row_ignores_the_pairing_before_0_27_for_mrope_models_only():
    assert rpc.kernel_ignores("vllm", "0.22.0", mrope=True) is not None
    assert rpc.kernel_ignores("vllm", "0.26.0", mrope=True) is not None
    assert rpc.kernel_ignores("vllm", "0.27.0", mrope=True) is None
    assert rpc.kernel_ignores("vllm", "0.30.0", mrope=True) is None
    assert rpc.kernel_ignores("vllm", "0.22.0", mrope=False) is None
    assert rpc.kernel_ignores("sglang", "0.22.0", mrope=True) is None
    assert rpc._version_tuple("0.30.0rc1") == (0, 30, 0)


def test_the_retrospective_on_vllm_42016_and_the_layers_against_the_declaration():
    pairing, where = rpc.declared({"model_type": "glm_ocr", "text_config": {"model_type": "glm_ocr_text"}})
    held = {"language_model.layers.0.self_attn.rotary_emb": "interleaved"}
    out = rpc.check(B, C, "vllm", pairing, where, held, "the built layers", version="0.22.0", mrope=True, record=False)
    assert len(out) == 1 and out[0].verdict is Verdict.BROKEN and out[0].rule == RULES["rotary_pairing_ignored"], out
    out = rpc.check(B, C, "vllm", pairing, where, held, "the built layers", version="0.30.0", mrope=True, record=False)
    assert [d.verdict for d in out] == [Verdict.PASS], out
    # a layer built with the other convention: broken, or resolved with the repair
    wrong = {"layers.0.rotary_emb": "split", "layers.1.rotary_emb": "split"}
    out = rpc.check(B, C, "vllm", pairing, where, wrong, "the built layers", version="0.30.0", record=False)
    assert out[0].verdict is Verdict.BROKEN and out[0].rule == RULES["rotary_pairing_mismatch"] and "2 rotary" in out[0].note
    out = rpc.check(B, C, "vllm", pairing, where, wrong, "the built layers", version="0.30.0",
                    handles={"set_pairing": lambda p: True}, record=False)
    assert out[0].verdict is Verdict.RESOLVED and out[0].target == "interleaved"
    # nothing declared, or a split model on the old kernel: no decision / pass
    assert rpc.check(B, C, "vllm", None, "nothing", held, "x", version="0.22.0", mrope=True, record=False) == []
    out = rpc.check(B, C, "vllm", "split", "qwen", {"l": "split"}, "x", version="0.22.0", mrope=True, record=False)
    assert [d.verdict for d in out] == [Verdict.PASS]


def test_the_vllm_adapter_reads_the_layers_and_sets_them():
    from entail.adapters import vllm_pairing

    class MRotaryEmbedding:
        def __init__(self, neox):
            self.is_neox_style = neox

    class Model:
        def __init__(self):
            self.mods = [("layers.0.rotary_emb", MRotaryEmbedding(True)), ("layers.1.rotary_emb", MRotaryEmbedding(True)),
                         ("embed", SimpleNamespace())]

        def named_modules(self):
            return list(self.mods)

    model = Model()
    cfg = SimpleNamespace(to_dict=lambda: {"model_type": "glm_ocr", "text_config": {"model_type": "glm_ocr_text"}})
    held, mrope, config, path = vllm_pairing.read_choice(model, SimpleNamespace(hf_config=cfg, model="/m"))
    assert held == {"layers.0.rotary_emb": "split", "layers.1.rotary_emb": "split"} and path == "/m"
    assert mrope is False, "no readable dispatch: the kernel row is not claimed (review finding 13)"
    for _, m in model.mods[:2]:
        m._forward_method = SimpleNamespace(__name__="forward_cuda")
    assert vllm_pairing.read_choice(model, SimpleNamespace(hf_config=cfg, model="/m"))[1] is True
    _, rec = decided(lambda: vllm_pairing._decide(model, SimpleNamespace(hf_config=cfg, model="/m")))
    got = [d for d in rec if d.contract.boundary == vllm_pairing.BOUNDARY]
    assert got and got[0].verdict is Verdict.RESOLVED, got
    assert all(m.is_neox_style is False for _, m in model.mods[:2])


def test_only_the_language_model_of_a_multimodal_model_is_compared():
    """GLM-OCR on vLLM 0.30: the text model's two MRoPE modules pair interleaved as declared; the vision tower's 26
    modules pair split by its own reference. The first real run compared the whole model and flipped the vision
    tower: the declaration is the text model's, so only the language model is read and set."""
    from entail.adapters import vllm_pairing

    class Rot:
        def __init__(self, neox):
            self.is_neox_style = neox

    class LM:
        def __init__(self, neox):
            self.mods = [("layers.0.rotary_emb", Rot(neox)), ("mtp.rotary_emb", Rot(neox))]

        def named_modules(self):
            return list(self.mods)

    class Multimodal:
        def __init__(self, text_neox):
            self.lm = LM(text_neox)
            self.vision = [(f"visual.blocks.{i}.attn.apply_rotary_emb", Rot(True)) for i in range(3)]

        def get_language_model(self):
            return self.lm

        def named_modules(self):
            return [("language_model." + n, m) for n, m in self.lm.mods] + list(self.vision)

    cfg = SimpleNamespace(to_dict=lambda: {"model_type": "glm_ocr", "text_config": {"model_type": "glm_ocr_text"}})
    mc = SimpleNamespace(hf_config=cfg, model="glm-ocr")
    right = Multimodal(text_neox=False)
    held, mrope, _, _ = vllm_pairing.read_choice(right, mc)
    assert held == {"layers.0.rotary_emb": "interleaved", "mtp.rotary_emb": "interleaved"}, held
    _, rec = decided(lambda: vllm_pairing._decide(right, mc))
    got = [d for d in rec if d.contract.boundary == vllm_pairing.BOUNDARY]
    assert got and all(d.verdict is Verdict.PASS for d in got), got
    assert all(m.is_neox_style is True for _, m in right.vision), "the vision tower is not touched"
    wrong = Multimodal(text_neox=True)
    _, rec = decided(lambda: vllm_pairing._decide(wrong, SimpleNamespace(hf_config=cfg, model="glm-ocr-2")))
    got = [d for d in rec if d.contract.boundary == vllm_pairing.BOUNDARY]
    assert got and got[0].verdict is Verdict.RESOLVED and "language model's rotary modules" in got[0].chosen.source.where
    assert got[0].declared.value.pairing == "interleaved" and got[0].chosen.value.pairing == "split"
    assert all(m.is_neox_style is False for _, m in wrong.lm.mods) and all(m.is_neox_style is True for _, m in wrong.vision)


def test_review_findings_deepseek_default_mixed_conventions_indexer_and_a_missing_language_model():
    """M17.4 review: (9) DeepSeek-V3's reference default is interleaved (rope_interleave defaults to True; the
    repo's own config class has no key); (10) modules that pair both ways are said unknown and left alone, and a
    DSA indexer (its own key) is not compared; (12) a multimodal model whose language model cannot be found is said
    unknown, not compared as a whole; (13) the kernel row applies only when the MRoPE module dispatches to the
    kernel."""
    from entail.adapters import vllm_pairing

    assert rpc.declared({"model_type": "deepseek_v3"})[0] == "interleaved"
    assert rpc.declared({"model_type": "deepseek_v3", "rope_interleave": False})[0] == "split"

    class Rot:
        def __init__(self, neox, cls="RotaryEmbedding", fwd="forward_native"):
            self.is_neox_style = neox
            self._forward_method = SimpleNamespace(__name__=fwd)
            self.__class__ = type(cls, (Rot,), {})

    class LM:
        def __init__(self, mods):
            self.mods = mods

        def named_modules(self):
            return list(self.mods)

    cfg = SimpleNamespace(to_dict=lambda: {"model_type": "glm4"})
    # mixed conventions in the language model: unknown, nothing set
    mixed = LM([("layers.0.rotary_emb", Rot(False)), ("layers.1.other_rotary", Rot(True))])
    _, rec = decided(lambda: vllm_pairing._decide(mixed, SimpleNamespace(hf_config=cfg, model="mixed")))
    got = [d for d in rec if d.contract.boundary == vllm_pairing.BOUNDARY]
    assert got and got[0].verdict is Verdict.UNKNOWN and "pair both ways" in got[0].note, got
    assert mixed.mods[0][1].is_neox_style is False and mixed.mods[1][1].is_neox_style is True
    # an indexer with its own key is not compared and not set
    dsa = LM([("layers.0.self_attn.rotary_emb", Rot(False)), ("layers.0.self_attn.indexer.rotary_emb", Rot(True))])
    held, mrope, _, _ = vllm_pairing.read_choice(dsa, SimpleNamespace(hf_config=cfg, model="dsa"))
    assert held == {"layers.0.self_attn.rotary_emb": "interleaved"} and not mrope
    _, rec = decided(lambda: vllm_pairing._decide(dsa, SimpleNamespace(hf_config=cfg, model="dsa")))
    got = [d for d in rec if d.contract.boundary == vllm_pairing.BOUNDARY]
    assert got and all(d.verdict is Verdict.PASS for d in got) and dsa.mods[1][1].is_neox_style is True
    # the kernel row needs the MRoPE module on its kernel path
    m_native = LM([("layers.0.rotary_emb", Rot(False, "MRotaryEmbedding", "forward_native"))])
    m_cuda = LM([("layers.0.rotary_emb", Rot(False, "MRotaryEmbedding", "forward_cuda"))])
    assert vllm_pairing.read_choice(m_native, SimpleNamespace(hf_config=cfg))[1] is False
    assert vllm_pairing.read_choice(m_cuda, SimpleNamespace(hf_config=cfg))[1] is True

    # a multimodal model whose language model cannot be found: unknown, the vision tower untouched
    class Multimodal:
        def __init__(self):
            self.vision = [("visual.blocks.0.attn.apply_rotary_emb", Rot(True))]

        def get_language_model(self):
            raise NotImplementedError

        def named_modules(self):
            return list(self.vision)

    mm = Multimodal()
    assert vllm_pairing.language_model(mm) is None
    _, rec = decided(lambda: vllm_pairing._decide(mm, SimpleNamespace(hf_config=cfg, model="mm")))
    got = [d for d in rec if d.contract.boundary == vllm_pairing.BOUNDARY]
    assert got and got[0].verdict is Verdict.UNKNOWN and "could not be found" in got[0].note, got
    assert mm.vision[0][1].is_neox_style is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
