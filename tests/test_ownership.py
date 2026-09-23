"""Tests for ownership.py: state stays with the object that set it. The loader below is the buffer handling of
ComfyUI v0.34.1's ModelPatcherDynamic (load + restore_loaded_backups), nothing else - the defect reported as
Comfy-Org/ComfyUI#16490. Run: python tests/test_ownership.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import ownership  # noqa: E402
from entail.adapters import _shared  # noqa: E402


def _schedule_class():
    """A sampling class shaped like comfy.model_sampling.ModelSamplingDiscrete: set_sigmas registers the schedule."""
    import torch

    class Sampling(torch.nn.Module):
        def __init__(self, sigma_max):
            super().__init__()
            self.set_sigmas(torch.linspace(0.03, sigma_max, 1000))

        def set_sigmas(self, sigmas):
            self.register_buffer("sigmas", sigmas.float())
            self.register_buffer("log_sigmas", sigmas.log().float())

    return Sampling


def _resolve_attr(obj, attr):  # comfy/utils.py
    attrs = attr.split(".")
    for name in attrs[:-1]:
        obj = getattr(obj, name)
    return obj, attrs[-1]


def _set_attr_buffer(obj, attr, value):  # comfy/utils.py
    obj, name = _resolve_attr(obj, attr)
    obj.register_buffer(name, value, persistent=name not in getattr(obj, "_non_persistent_buffers_set", set()))


class _Dynamic:
    """ModelPatcherDynamic's buffer handling: back up by path at load, write back by path at the next load."""

    def __init__(self, model, backups, guarded):
        self.model, self.backup_buffers, self.guarded = model, backups, guarded

    def restore_loaded_backups(self):
        if self.guarded:
            moved = ownership.return_foreign(self.model, self.backup_buffers, _resolve_attr)
            if moved:
                ownership.say_returned(moved, "test")
        for key in list(self.backup_buffers.keys()):
            _set_attr_buffer(self.model, key, self.backup_buffers.pop(key))

    def load(self):
        self.restore_loaded_backups()
        for key, buf in self.model.named_buffers(recurse=True):
            if key not in self.backup_buffers:
                self.backup_buffers[key] = buf
            _set_attr_buffer(self.model, key, buf.clone())  # stands for the copy on the GPU
        if self.guarded:
            ownership.remember_owners(self.model, self.backup_buffers, _resolve_attr)


def _node_then_plain(guarded, node_max=4518.8):
    import torch

    Sampling = _schedule_class()
    model = torch.nn.Module()
    own, node = Sampling(14.6), Sampling(node_max)  # the checkpoint's schedule; a v_prediction+zsnr node's
    backups = {}
    model.model_sampling = node  # the run with the node: its object is put at 'model_sampling'
    _Dynamic(model, backups, guarded).load()
    model.model_sampling = own  # the node is gone: the model's own object is put back
    _Dynamic(model, backups, guarded).load()
    return float(model.model_sampling.sigmas[-1]), float(node.sigmas[-1])


def test_the_loader_moves_a_node_schedule_into_the_model():
    """The defect as ComfyUI v0.34.1 has it: the node's schedule ends up in the model's own object."""
    assert _node_then_plain(guarded=False)[0] > 4518


def test_guard_keeps_each_schedule_with_its_object():
    before = len(_shared.RESOLUTIONS)
    model_max, node_max = _node_then_plain(guarded=True)
    assert abs(model_max - 14.6) < 1e-4 and abs(node_max - 4518.8) < 1e-2
    assert any(r.get("fact") == "ownership" for r in _shared.RESOLUTIONS[before:])


def test_guard_says_nothing_when_the_values_are_the_same():
    """Two objects with the same schedule (a prediction-type switch keeps the schedule): nothing to hand back."""
    before = len(_shared.RESOLUTIONS)
    assert abs(_node_then_plain(guarded=True, node_max=14.6)[0] - 14.6) < 1e-4
    assert not any(r.get("fact") == "ownership" for r in _shared.RESOLUTIONS[before:])


def test_drift_and_put_back():
    import torch

    Sampling = _schedule_class()
    wrapped = ownership.wrap_setters([Sampling], ("set_sigmas",))
    try:
        ms = Sampling(14.6)
        assert ownership.drift(ms) == {}
        ms.register_buffer("sigmas", ms.sigmas.clone())  # the same values (what a device move does)
        assert ownership.drift(ms) == {}
        ms.register_buffer("sigmas", torch.linspace(0.03, 4518.8, 1000))  # another object's schedule written in
        found = ownership.drift(ms)
        assert list(found) == ["sigmas"]
        ownership.put_back(ms, found, "test")
        assert ownership.drift(ms) == {} and abs(float(ms.sigmas[-1]) - 14.6) < 1e-4
        ms.set_sigmas(torch.linspace(0.03, 20.0, 1000))  # a setter call is a legitimate change: recorded anew
        assert ownership.drift(ms) == {}
    finally:
        for cls, name, fn in wrapped:
            setattr(cls, name, fn)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
