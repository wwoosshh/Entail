"""Tests for the image facts' sources and contracts (ROADMAP M6.1): what a safetensors header and a LoRA's tensor names
declare (readers.header_facts, lora_modules), and the load contracts prediction, latent_scale and lora on the shapes of
the M6 test problems (fd-m7, fd-lora, market I01 and I04, the single-file VAE). No engine, no GPU, no torch.
Run: python tests/test_image_contracts.py"""
import io
import json
import os
import struct
import sys
import tempfile
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
for k in ("ENTAIL_ON_BROKEN", "ENTAIL_UNKNOWN", "ENTAIL_POLICY", "ENTAIL_FACT_POLICY", "ENTAIL_MANIFESTS"):
    os.environ.pop(k, None)
from entail import load, readers, record, sources  # noqa: E402
from entail.contracts import Verdict  # noqa: E402
from entail.core import RoleError  # noqa: E402
from entail.coverage import Coverage  # noqa: E402
from entail.facts import Certainty, Fact, LatentScale, Prediction, Source  # noqa: E402
from entail.policies import Policy  # noqa: E402

ON = Policy(mode="load")                                   # the default policy: repair, else report and go on
STRICT = Policy(mode="load", on_broken="stop")            # ENTAIL_ON_BROKEN=stop
OBSERVE = Policy(mode="load", on_mismatch="refuse")        # ENTAIL_POLICY=refuse: nothing repaired
REQUIRE = Policy(mode="load", on_unknown_meaning_changing="require")
UNET = ["model.diffusion_model.input_blocks.0.0.weight"]


def facts_of(keys, meta):
    return load.declared(header=(keys, meta, "test.safetensors"))


def write_safetensors(path, keys, meta):
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, k in enumerate(keys)}
    header["__metadata__"] = meta
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"\0" * (4 * len(keys)))


# --- sources ----------------------------------------------------------------------------------------------------

def test_a_header_declares_the_prediction_by_metadata_or_marker_keys():
    (f,) = readers.header_facts(UNET, {"modelspec.prediction_type": "v"}, "astolfo.safetensors").facts
    assert f.value == Prediction("v") and f.source == Source("file", "astolfo.safetensors#__metadata__."
                                                                    "modelspec.prediction_type")
    assert f.certainty is Certainty.DECLARED
    (f,) = readers.header_facts(UNET + ["v_pred", "ztsnr"], {}, "noob.safetensors").facts
    assert f.value == Prediction("v", zsnr=True) and "key v_pred (zsnr from key ztsnr)" in f.source.where
    (f,) = readers.header_facts(UNET, {"ss_v_parameterization": "false"}, "lora.safetensors").facts
    assert f.value == Prediction("eps")
    assert readers.header_facts(UNET, {}, "plain.safetensors").facts == [], "nothing stated, nothing emitted"
    r = readers.header_facts(UNET, {"modelspec.prediction_type": "sigma"}, "odd.safetensors")
    assert r.facts == [] and "not in vocabulary" in r.problems[0]


def test_a_file_that_contradicts_itself_is_a_conflict():
    r = readers.header_facts(UNET + ["v_pred"], {"ss_v_parameterization": "false"}, "merged.safetensors")
    best, conflict = sources.pick(r.facts)
    assert len(r.facts) == 2 and len(conflict) == 2


def test_the_file_reader_and_a_header_in_hand_say_the_same():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "m.safetensors")
    write_safetensors(p, UNET + ["v_pred"], {"modelspec.prediction_type": "v"})
    from_file = [(f.name, f.value) for f in readers.SafetensorsMeta().read(p).facts]
    in_hand = [(f.name, f.value) for f in readers.header_facts(*readers.safetensors_header(p), p).facts]
    assert from_file == in_hand == [("Prediction", Prediction("v")), ("Prediction", Prediction("v"))]
    assert [f.value for f in load.declared(p).get("Prediction")] == [Prediction("v")] * 2


def test_lora_modules_and_what_a_lora_says_it_was_trained_on():
    keys = ["lora_unet_down_blocks_0.lora_down.weight", "lora_unet_down_blocks_0.lora_up.weight",
            "lora_unet_down_blocks_0.alpha", "lora_te1_text_model_encoder.lora_down.weight",
            "transformer.blocks.0.attn.to_q.lora_A.weight", "unet.mid.hada_w1_a"]
    assert readers.lora_modules(keys) == {"lora_unet_down_blocks_0", "lora_te1_text_model_encoder",
                                          "transformer.blocks.0.attn.to_q", "unet.mid"}
    assert readers.lora_modules(UNET) == set(), "a checkpoint's keys are not LoRA keys"
    assert readers.is_text_module("lora_te1_text_model_encoder") and not readers.is_text_module("lora_unet_x")
    assert readers.lora_base({"ss_base_model_version": "sdxl_base_v1-0"}) == "ss_base_model_version=sdxl_base_v1-0"
    assert readers.lora_base({}) is None


# --- Prediction ---------------------------------------------------------------------------------------------------

def test_a_declaration_no_engine_reads_is_repaired_keeping_what_it_leaves_open():
    # fd-m7: the file says v in its metadata, nothing about zero terminal SNR; the sampler was set up as eps
    (d,) = load.prediction("comfyui", "comfyui.sampler", facts_of(UNET, {"modelspec.prediction_type": "v"}),
                           Prediction("eps", zsnr=False), policy=ON)
    assert d.verdict is Verdict.RESOLVED and d.handle == "switch_prediction"
    assert d.target == Prediction("v", zsnr=False), "the open field keeps the sampler's value"
    assert d.contract.boundary == "load:comfyui.prediction" and not d.blocking


def test_marker_keys_declare_zero_terminal_snr_too():
    facts = facts_of(UNET + ["v_pred", "ztsnr"], {})    # market I04: NoobAI-XL V-Pred's markers
    (d,) = load.prediction("diffusers", "diffusers.scheduler", facts, Prediction("v", zsnr=True), policy=ON)
    assert d.verdict is Verdict.PASS
    (d,) = load.prediction("diffusers", "diffusers.scheduler", facts, Prediction("eps", zsnr=False), policy=ON)
    assert d.verdict is Verdict.RESOLVED and d.target == Prediction("v", zsnr=True)
    (d,) = load.prediction("diffusers", "diffusers.scheduler", facts, Prediction("v", zsnr=False), policy=ON)
    assert d.verdict is Verdict.RESOLVED and d.target == Prediction("v", zsnr=True), "zsnr is declared: compared"


def test_an_explicit_choice_is_reported_never_overridden():
    facts = facts_of(UNET, {"modelspec.prediction_type": "v"})
    (d,) = load.prediction("comfyui", "comfyui.sampler", facts, Prediction("eps"), explicit=True, policy=ON)
    assert d.verdict is Verdict.BROKEN and d.handle is None and not d.blocking
    assert d.chosen.source.kind == "user" and "not overridden" in d.rule
    (d,) = load.prediction("comfyui", "comfyui.sampler", facts, Prediction("eps"), explicit=True, policy=STRICT)
    assert d.verdict is Verdict.REFUSED and d.blocking


def test_nothing_to_switch_with_and_the_observing_policy_report():
    facts = facts_of(UNET, {"modelspec.prediction_type": "v"})
    (d,) = load.prediction("x", "x.sampler", facts, Prediction("eps"), can_switch=False, policy=ON)
    assert d.verdict is Verdict.BROKEN and "no resolution" in d.rule
    (d,) = load.prediction("x", "x.sampler", facts, Prediction("eps"), policy=OBSERVE)
    assert d.verdict is Verdict.BROKEN and "repairs nothing" in d.rule


def test_nothing_declared_is_unknown_and_nothing_is_guessed():
    (d,) = load.prediction("comfyui", "comfyui.sampler", facts_of(UNET, {}), Prediction("eps"), policy=ON)
    assert d.verdict is Verdict.UNKNOWN and d.rule == "nothing declares it" and not d.blocking and d.handle is None
    (d,) = load.prediction("comfyui", "comfyui.sampler", facts_of(UNET, {}), Prediction("eps"), policy=REQUIRE)
    assert d.blocking, "ENTAIL_UNKNOWN=require stops at a prediction nobody declares"
    (d,) = load.prediction("comfyui", "comfyui.sampler", facts_of(UNET, {"modelspec.prediction_type": "v"}), None,
                           policy=ON)
    assert d.verdict is Verdict.UNKNOWN and "consumer" in d.rule


def test_a_manifest_declares_what_the_file_does_not():
    user = [Fact("Prediction", Prediction("v", zsnr=True), Source("manifest", "pinned#Prediction"),
                 Certainty.DECLARED)]
    facts = load.declared(header=(UNET, {}, "no-marker.safetensors"), user=user)
    (d,) = load.prediction("comfyui", "comfyui.sampler", facts, Prediction("eps", zsnr=False), policy=ON)
    assert d.verdict is Verdict.RESOLVED and d.target == Prediction("v", zsnr=True)


# --- LatentScale --------------------------------------------------------------------------------------------------

def declared_scale(scale, shift=None):
    return load.Declared({"LatentScale": [Fact("LatentScale", LatentScale(scale, shift=shift),
                                               Source("config", "model/vae/config.json#scaling_factor"),
                                               Certainty.DECLARED)]})


def test_a_vae_with_another_familys_scale_is_given_the_declared_one():
    (d,) = load.latent_scale("diffusers", "diffusers.vae", declared_scale(0.5), LatentScale(0.25), policy=ON)
    assert d.verdict is Verdict.RESOLVED and d.handle == "set_latent_scale" and d.target == LatentScale(0.5)
    (d,) = load.latent_scale("diffusers", "diffusers.vae", declared_scale(0.5), LatentScale(0.5), policy=ON)
    assert d.verdict is Verdict.PASS
    (d,) = load.latent_scale("diffusers", "diffusers.vae", declared_scale(1.5, 0.06), LatentScale(1.5), policy=ON)
    assert d.verdict is Verdict.RESOLVED and d.target == LatentScale(1.5, shift=0.06), "a declared shift is compared"
    (d,) = load.latent_scale("diffusers", "diffusers.vae", declared_scale(0.5), LatentScale(0.25), explicit=True,
                             policy=STRICT)
    assert d.verdict is Verdict.REFUSED and d.blocking


def test_a_single_file_states_no_latent_scale():
    (d,) = load.latent_scale("comfyui", "comfyui.latent_format", facts_of(UNET, {}), LatentScale(0.25), policy=ON)
    assert d.verdict is Verdict.UNKNOWN and not d.blocking


# --- Coverage of a LoRA --------------------------------------------------------------------------------------------

def test_a_lora_that_reaches_nothing_or_part_is_broken_and_says_what_it_was_made_for():
    given = {f"lora_unet_m{i}" for i in range(280)}
    (d,) = load.lora("comfyui", "anima_e10.safetensors", given, [], base="ss_base_model_version=anima", policy=ON)
    assert d.verdict is Verdict.BROKEN and not d.blocking and d.chosen.value.taken == 0
    assert d.note == "the LoRA's metadata says ss_base_model_version=anima"
    text = record.line(d)
    assert "taken=0" in text and "and 275 more" in text and "reported, not stopped" in text and len(text) < 1200
    (d,) = load.lora("comfyui", "x.safetensors", given, list(given)[:100], policy=ON)
    assert d.verdict is Verdict.BROKEN and d.chosen.value.taken == 100
    (d,) = load.lora("comfyui", "x.safetensors", given, given, base="ss_base_model_version=sdxl", policy=ON)
    assert d.verdict is Verdict.PASS and d.note == ""
    (d,) = load.lora("comfyui", "x.safetensors", given, [], policy=STRICT)
    assert d.verdict is Verdict.REFUSED and d.blocking
    assert load.lora("comfyui", "x.safetensors", [], [], policy=ON) == [], "nothing given, nothing to decide"


def test_enforce_reports_a_broken_lora_and_stops_only_when_asked():
    given = ["lora_unet_a", "lora_unet_b"]
    printed = io.StringIO()
    with redirect_stdout(printed):
        load.enforce(load.lora("comfyui", "x.safetensors", given, [], policy=ON))
    assert "[entail] broken at load:comfyui.lora" in printed.getvalue()
    try:
        with redirect_stdout(io.StringIO()):
            load.enforce(load.lora("comfyui", "x.safetensors", given, [], policy=STRICT))
        raise AssertionError("the strict policy stops")
    except RoleError as e:
        assert "refused at load:comfyui.lora" in str(e)


def test_coverage_prints_a_long_left_short():
    assert str(Coverage(1, 1, ())) == "Coverage(total=1, taken=1, left=())"
    assert str(Coverage(3, 1, ("a", "b"))) == "Coverage(total=3, taken=1, left=('a', 'b'))"
    assert str(Coverage(9, 0, tuple("abcdefghi"))).endswith("'e', '... and 4 more'))")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
