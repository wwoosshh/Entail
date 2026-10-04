"""Adapter v2: the model loaders of vLLM and SGLang, where a model is made and its weights loaded (ROADMAP M19 L7,
ENTAIL=types; entail/lifetime.py holds the check). Nothing here is written for one model or one bug.

  hooks        load_model of every model loader class the data file names (data/loaders.json: module -> base
               class; every subclass defined when the module has been imported is included): the model is
               constructed, its weights loaded and processed inside it, and it returns the model
  read_choice  the data file's rows
  handles      none: the window marks allocations and, when the outermost load_model returns, the model's
               parameters are read for elements nothing wrote; nothing is changed or decided here
"""
import functools
import sys

from .. import lifetime

engine = "vllm, sglang"
versions = "vLLM 0.30.0, SGLang 0.5.20"
_WRAPPED = {}
_STATS = {}
_TABLE = []


def _table():
    if not _TABLE:
        import json
        import os

        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "loaders.json")
        with open(path, encoding="utf-8") as f:
            _TABLE.append(json.load(f))
    return _TABLE[0]


def hooks():
    from .base import Hook

    return [Hook(f"{mod}.{base}.load_model", "load") for mod, base in (_table().get("loaders") or {}).items()]


def read_choice(kind, obj=None):
    """The data file's rows: kind is "loaders" (module -> the base class of its model loaders)."""
    return _table().get(kind)


def handles():
    return {}


def _count(k):
    _STATS[k] = _STATS.get(k, 0) + 1


def stats():
    return dict(_STATS)


def _subclasses(cls):
    out, todo = [], [cls]
    while todo:
        c = todo.pop()
        out.append(c)
        todo.extend(c.__subclasses__())
    return out


def _make(orig):
    @functools.wraps(orig)
    def load_model(*a, **k):
        with lifetime.load_window():
            model = orig(*a, **k)
        try:
            lifetime.loaded(model)
        except Exception:  # noqa: BLE001 - never the engine's problem
            _count("read_failed")
        return model
    load_model.__entail_types__ = True
    return load_model


def install():
    n = 0
    for modname, base in (read_choice("loaders") or {}).items():
        mod = sys.modules.get(modname)
        root = getattr(mod, base, None) if mod is not None else None
        if root is None:
            continue
        for cls in _subclasses(root):
            orig = cls.__dict__.get("load_model")
            if orig is None or getattr(orig, "__entail_types__", False) or (cls, "load_model") in _WRAPPED:
                continue
            setattr(cls, "load_model", _make(orig))
            _WRAPPED[(cls, "load_model")] = orig
            n += 1
    _count("installed")
    return n


def uninstall():
    for (holder, name), orig in list(_WRAPPED.items()):
        setattr(holder, name, orig)
    _WRAPPED.clear()
