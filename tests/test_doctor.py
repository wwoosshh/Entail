"""Tests for `entail doctor`'s engine list (field test, entail#6): ComfyUI and diffusers are listed with the versions
their adapters were measured on, a version that is not one of them says so, and the adapters of engines that are not
installed are folded into one line.
Run: python tests/test_doctor.py"""
import contextlib
import io
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from entail import cli  # noqa: E402


def test_a_version_says_whether_it_is_a_tested_one():
    assert cli._tested("vllm", "0.30.0") == "   (tested with 0.30.0)"
    assert cli._tested("comfyui", "0.37.0") == "   (tested with 0.34.1; not this version)"
    assert cli._tested("torch", "2.14.0") == "" and cli._tested("vllm", None) == ""


def test_comfyuis_version_is_read_from_its_folder_and_listed():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "comfyui_version.py"), "w", encoding="utf-8") as f:
        f.write('# written by ComfyUI\n__version__ = "0.37.0"\n')
    here = os.getcwd()
    try:
        os.chdir(d)
        assert cli._comfyui() == "0.37.0"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.doctor(None)
        text = out.getvalue()
        assert "  comfyui       0.37.0   (tested with 0.34.1; not this version)" in text, text
        assert "  diffusers" in text and "(engine not installed here)" not in text, text
        assert "for engines not installed here:" in text or "after " in text, text
    finally:
        os.chdir(here)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
