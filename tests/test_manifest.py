"""Tests for manifests (ROADMAP M2.3): a draft names what the artifact declares and leaves slots for what it does not,
an unpinned draft declares nothing, pinning makes reviewed facts declarations, and a manifest is found only for the
file it describes. Run: python tests/test_manifest.py"""
import json
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import cli, contracts, manifest, sources  # noqa: E402
from entail.facts import Assumed, Certainty, Layout, Prediction  # noqa: E402


def checkpoint(metadata=None, name="ckpt.safetensors"):
    keys = ["model.diffusion_model.input_blocks.0.weight", "first_stage_model.decoder.conv.weight"]
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, k in enumerate(keys)}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode()
    d = tempfile.mkdtemp()
    p = os.path.join(d, name)
    with open(p, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\0" * 8)
    return p


def raises(fn, text):
    try:
        fn()
    except ValueError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected ValueError: {text}")


def test_draft_has_slots_for_what_is_not_declared():
    p = checkpoint()
    draft = manifest.infer(p)
    assert draft.sha256 == manifest.sha256_of(p) and not draft.pinned and draft.file == "ckpt.safetensors"
    assert {(f.name, f.value, f.certainty) for f in draft.facts} == {
        ("Prediction", None, Certainty.UNKNOWN), ("LatentScale", None, Certainty.UNKNOWN)}
    assert draft.evidence["Prediction"] == ["not declared by the artifact: fill in after review"]


def test_draft_keeps_what_the_file_declares():
    draft = manifest.infer(checkpoint({"modelspec.prediction_type": "v"}))
    [pred] = [f for f in draft.facts if f.name == "Prediction"]
    assert pred.value == Prediction("v") and pred.certainty is Certainty.DECLARED
    assert draft.evidence["Prediction"][0].startswith("the artifact declares it: ")


def test_unpinned_declares_nothing_and_pin_makes_it_count():
    p = checkpoint()
    out = os.path.join(tempfile.mkdtemp(), "m.json")
    data = manifest.to_json(manifest.infer(p))
    data["facts"][0]["value"] = {"kind": "v", "zsnr": True}      # a person fills in the Prediction slot
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f)
    m = manifest.load(out)
    [pred] = [f for f in m.facts if f.name == "Prediction"]
    assert pred.value == Prediction("v", True) and pred.certainty is Certainty.INFERRED
    assert pred.source.kind == "manifest" and pred.source.where == f"{out}#Prediction"
    pinned = manifest.pin(m)
    [pred] = [f for f in pinned.facts if f.name == "Prediction"]
    assert pinned.pinned and pred.certainty is Certainty.DECLARED
    assert [f.certainty for f in pinned.facts if f.name == "LatentScale"] == [Certainty.UNKNOWN]   # empty stays unknown
    manifest.save(pinned, out)
    assert manifest.load(out) == manifest.from_json(json.load(open(out, encoding="utf-8")), out)


def test_found_only_for_its_own_file():
    p = checkpoint()
    other = checkpoint({"x": "1"})
    d = tempfile.mkdtemp()
    m = manifest.pin(manifest.from_json({"schema": 1, "sha256": manifest.sha256_of(p), "pinned": True,
                                         "facts": [{"name": "Prediction", "value": {"kind": "v"}}]}, "t"))
    manifest.save(m, os.path.join(d, f"{m.sha256}.json"))
    assert manifest.find(p, [d]).sha256 == m.sha256 and manifest.find(other, [d]) is None
    manifest.save(m, other + manifest.SIDECAR)                # a sidecar that describes another file is not used
    assert manifest.find(other, []) is None
    # through read_all: the manifest's declaration reaches the contract, the file itself says nothing
    r = sources.read_all(p, manifest_dirs=[d])
    [pred] = r.facts
    assert pred.source.kind == "manifest" and pred.certainty is Certainty.DECLARED
    c = contracts.Contract("load:sampler", "sampler", ("Prediction",), ("Prediction",))
    [dec] = contracts.decide(c, {"Prediction": pred}, {})
    assert dec.rule == contracts.RULES["consumer_unknown"]   # declared now; only the consumer's choice is missing


def test_a_file_no_manifest_could_be_for_is_not_hashed():
    """M6.3: hashing a 6.9 GB checkpoint on every load cost 18 s. A manifest records a quick fingerprint; a file whose
    fingerprint no manifest has is not hashed in full, and the one it is for still is (the key stays its SHA-256)."""
    d = tempfile.mkdtemp()
    mine, other, dirs = os.path.join(d, "a.bin"), os.path.join(d, "b.bin"), os.path.join(d, "manifests")
    os.makedirs(dirs)
    for p, byte in ((mine, b"a"), (other, b"b")):   # the same size, as two checkpoints of one architecture are
        with open(p, "wb") as f:
            f.write(byte * (9 << 20))
    m = manifest.pin(manifest.Manifest(manifest.sha256_of(mine), (), file="a.bin",
                                       fingerprint=manifest.quick_fingerprint(mine)))
    manifest.save(m, os.path.join(dirs, f"{m.sha256}.json"))
    hashed, real = [], manifest.sha256_of
    manifest.sha256_of = lambda p: (hashed.append(p), real(p))[1]
    try:
        assert manifest.find(other, [dirs]) is None and hashed == [], "another file: not hashed"
        assert manifest.find(mine, [dirs]).sha256 == m.sha256 and hashed == [mine], "its own file: hashed to verify"
        old = dict(manifest.to_json(m), fingerprint=None)   # a manifest written before M6.3
        with open(os.path.join(dirs, "old.json"), "w", encoding="utf-8") as f:
            json.dump(old, f)
        hashed.clear()
        assert manifest.find(other, [dirs]) is None and hashed == [other], "without fingerprints every file is a candidate"
    finally:
        manifest.sha256_of = real
    assert manifest.load(os.path.join(dirs, f"{m.sha256}.json")).fingerprint == m.fingerprint
    assert manifest.infer(mine).fingerprint == manifest.quick_fingerprint(mine), "a draft records it"


def test_bad_manifests_are_refused_with_the_reason():
    ok = {"schema": 1, "sha256": "a" * 64, "facts": []}
    raises(lambda: manifest.from_json({**ok, "schema": 2}, "m"), "manifest m: schema 2, this library reads 1")
    raises(lambda: manifest.from_json({**ok, "sha256": "xyz"}, "m"), "sha256 must be 64 lowercase hex digits")
    raises(lambda: manifest.from_json({**ok, "facts": [{"name": "Colour"}]}, "m"), "unknown fact name 'Colour'")
    raises(lambda: manifest.from_json({**ok, "facts": [{"name": "Prediction", "value": {"kind": "v", "speed": 1}}]},
                                      "m"), "manifest: Prediction has no fields ['speed']")
    raises(lambda: manifest.from_json({**ok, "facts": [{"name": "Prediction", "value": {"kind": "velocity"}}]}, "m"),
           "unknown prediction kind 'velocity'")


def test_values_survive_json_with_their_tuples():
    for name, value in (("Layout", Layout("fp8_block", block=(128, 128))),
                        ("Assumed", Assumed((("batch", 4), ("seq", 128))))):
        assert manifest.value_from_json(name, json.loads(json.dumps(manifest.value_to_json(value)))) == value


def test_cli_infer_then_pin():
    p = checkpoint()
    out = os.path.join(tempfile.mkdtemp(), "draft.json")
    assert cli.main(["infer", p, "--out", out]) == 0
    assert json.load(open(out, encoding="utf-8"))["pinned"] is False
    assert cli.main(["pin", out]) == 0
    assert json.load(open(out, encoding="utf-8"))["pinned"] is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
