"""State belongs to the object that set it (TIME / ownership, RESEARCH_PLAN.md 5.1 principle 6).

Two contracts on the buffers of torch modules, independent of any engine:

  1. What an object's own setter registered is what it holds. `wrap_setters` makes each setter leave a copy;
     `drift` reports a buffer that no longer matches it; `put_back` restores it and says so.
  2. A backup taken from one object goes back to that object. An engine that backs buffers up by attribute path
     and swaps the object at that path (an object patch) would otherwise write one object's state into another.
     `remember_owners` runs after a backup is taken; `return_foreign` runs before the engine writes backups back.

Measured on ComfyUI 0.34.1's dynamic VRAM loader (VPRED_PROTOCOL.md M6, Comfy-Org/ComfyUI#16490): a sampling
node's schedule reached the checkpoint's own object and later runs came out as another image and then black;
with (2) both directions matched a fresh session pixel for pixel, with (1) alone 2-11/255 remained.
"""
import weakref

from .adapters import _shared


def _record(obj):
    obj.__dict__["_entail_registered"] = {k: b.detach().to("cpu", copy=True) for k, b in obj._buffers.items()
                                          if b is not None}


def wrap_setters(classes, names):
    """Wrap the setters `names` defined on these classes so each call leaves a copy of what it registered.
    Returns [(class, name, original)] for unwrapping."""
    wrapped = []
    for cls in classes:
        for name in names:
            fn = cls.__dict__.get(name)
            if fn is None:
                continue

            def setter(self, *a, _fn=fn, **kw):
                out = _fn(self, *a, **kw)
                try:
                    _record(self)
                except Exception:  # noqa: BLE001 - the copy is for checking; never break the setter
                    pass
                return out

            setter.__name__, setter.__qualname__, setter.__doc__ = fn.__name__, fn.__qualname__, fn.__doc__
            setattr(cls, name, setter)
            wrapped.append((cls, name, fn))
    return wrapped


def drift(obj):
    """{buffer name: (now, registered)} for buffers that no longer hold what the object's setter registered. Empty
    when all match or nothing was recorded. The tolerance only keeps a harmless cast from counting as a change."""
    import torch

    rec = obj.__dict__.get("_entail_registered")
    if not rec:
        return {}
    out = {}
    for name, ref in rec.items():
        now = obj._buffers.get(name)
        if now is None:
            continue
        now, ref = now.detach().to("cpu", torch.float32), ref.to(torch.float32)
        if now.shape != ref.shape or not torch.allclose(now, ref, rtol=1e-3, atol=1e-6):
            out[name] = (now, ref)
    return out


def last_value(buf):
    try:
        return f"{float(buf.detach().float().flatten()[-1]):.1f}"
    except Exception:  # noqa: BLE001
        return "?"


def put_back(obj, found, where):
    """Restore what the object's setter registered, on the device and dtype the buffer has now; say so."""
    for name, (_, ref) in found.items():
        cur = obj._buffers[name]
        obj.register_buffer(name, ref.to(device=cur.device, dtype=cur.dtype),
                            persistent=name not in obj._non_persistent_buffers_set)
    first = next(iter(found))
    now, ref = found[first]
    _shared.note({"where": where, "fact": "ownership", "buffers": sorted(found)},
                 f"{where}: {type(obj).__name__}.{first} held a value its own setter did not register (last entry "
                 f"{last_value(now)} instead of {last_value(ref)}); put back the registered one")


def same_values(a, b):
    import torch

    return (a is not None and b is not None and a.shape == b.shape
            and torch.equal(a.detach().to("cpu", torch.float32), b.detach().to("cpu", torch.float32)))


def remember_owners(root, backups, resolve):
    """After backups were taken: which object each backed-up path belonged to (resolve(root, path) -> (obj, name))."""
    owners = root.__dict__.setdefault("_entail_buffer_owners", {})
    for key in list(backups):
        try:
            owners[key] = weakref.ref(resolve(root, key)[0])
        except (AttributeError, TypeError):
            owners.pop(key, None)


def return_foreign(root, backups, resolve):
    """Before backups are written back by path: hand each one back to the object it came from instead of writing it
    into another object now at that path. A backup equal to what the object there holds is left alone: nothing
    would change. Returns [(path, owner, current, backup)] for what was handed back."""
    owners = root.__dict__.get("_entail_buffer_owners") or {}
    moved = []
    for key in list(backups):
        ref = owners.get(key)
        if ref is None:
            continue
        try:
            current, name = resolve(root, key)
        except AttributeError:
            continue
        owner = ref()
        if owner is current or same_values(backups[key], getattr(current, "_buffers", {}).get(name)):
            continue
        buf = backups.pop(key)
        owners.pop(key, None)
        if owner is not None:
            owner.register_buffer(name, buf, persistent=name not in owner._non_persistent_buffers_set)
        moved.append((key, owner, current, buf))
    return moved


def say_returned(moved, where):
    paths = {}
    for key, owner, current, buf in moved:
        path, _, name = key.rpartition(".")
        paths.setdefault(path, []).append((name, current, buf))
    for path, items in paths.items():
        name, current, buf = items[0]
        own = getattr(current, "_buffers", {}).get(name)
        _shared.note({"where": where, "fact": "ownership", "path": path, "buffers": [n for n, *_ in items]},
                     f"{where}: the loader was about to write '{path}.{name}' of another object (last entry "
                     f"{last_value(buf)}) into the one this run uses (last entry {last_value(own)}); handed it back "
                     f"to its own object")
