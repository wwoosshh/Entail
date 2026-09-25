"""The start-up hook's target table (adapters/autoinstall/sitecustomize.py; M15.3): every module key appears once -
a dict literal keeps only the last value of a repeated key, which silently dropped an adapter once - and every
adapter it names exists. Reads the source; imports no engine. Run: python tests/test_autoinstall_targets.py"""
import ast
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
PATH = os.path.join(os.path.dirname(HERE), "entail", "adapters", "autoinstall", "sitecustomize.py")


def targets_literal():
    tree = ast.parse(open(PATH, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TARGETS" for t in node.targets):
            assert isinstance(node.value, ast.Dict)
            return node.value
    raise AssertionError("TARGETS not found")


def test_every_target_module_is_listed_once():
    keys = [ast.literal_eval(k) for k in targets_literal().keys]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    assert not dup, f"repeated keys in TARGETS (only the last survives): {dup}"


def test_every_adapter_named_exists_and_has_its_entry_point():
    for k, v in zip(targets_literal().keys, targets_literal().values):
        for entry in ast.literal_eval(v):
            mod, _, func = entry.partition(":")
            m = importlib.import_module(mod)
            assert callable(getattr(m, func or "install", None)), entry


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
