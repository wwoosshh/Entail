"""ENGINE-SPECIFIC: a repair of ComfyUI's own defect, not a contract of entail (LIBRARY_DESIGN.md 10; ROADMAP M6.2).

ComfyUI 0.34.1's dynamic VRAM loader (ModelPatcherDynamic) backs a model's buffers up by attribute path at load and
writes them back by path at the next load. A sampling node puts its own schedule object at 'model_sampling' for its
run, so the node's schedule is written into the checkpoint's own object afterwards: later runs without the node come
out as another image, then black, and a node used after a plain run is ignored (Comfy-Org/ComfyUI#16490; measured in
issue_track/comfyui_field_test/VPRED_PROTOCOL.md M6). This is a defect of one engine, so it stays out of the core's
verdicts (testbed/PROBLEMS.md fd-leak) and lives here, marked as the engine's; tests/test_adapter_rules.py lists it
apart from the adapters, which hold no rules.

Two rules on the buffers of torch modules, the engine's repair:
  1. What an object's own setter registered is what it holds: the setters of comfy.model_sampling leave a copy
     (install_schedule_record); at the first model call after the buffers were placed, a buffer that no longer
     matches it is put back (install_schedule_check).
  2. A backup taken from one object goes back to that object: after ModelPatcherDynamic.load, which object each backed
     up path belonged to is remembered; before restore_loaded_backups writes by path, a backup is handed back to its
     owner instead of into another object now at that path (install_buffer_guard).
Measured: with (2) both directions matched a fresh session pixel for pixel; with (1) alone 2-11/255 remained (M6).
What it does is said in one line and written to the record file (load.say); it never breaks the run.
"""
import importlib
import weakref

from .. import load

engine = "comfyui"
versions = "0.34.1"
WHERE = "comfyui sampling schedule (engine-specific repair, Comfy-Org/ComfyUI#16490)"
_ORIG = {}      # (class, attribute) -> original, for uninstall
_SETTERS = []   # (class, name, original) from wrap_setters


def _patch(owner, attr, new):
    if (owner, attr) in _ORIG:
        return 0
    _ORIG[(owner, attr)] = getattr(owner, attr)
    setattr(owner, attr, new)
    return 1


# --- 1. what an object's own setter registered --------------------------------------------------------------------

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


def put_back(obj, found, where=WHERE):
    """Restore what the object's setter registered, on the device and dtype the buffer has now; say so."""
    for name, (_, ref) in found.items():
        cur = obj._buffers[name]
        obj.register_buffer(name, ref.to(device=cur.device, dtype=cur.dtype),
                            persistent=name not in obj._non_persistent_buffers_set)
    first = next(iter(found))
    now, ref = found[first]
    load.say(where, f"{type(obj).__name__}.{first} held a value its own setter did not register (last entry "
                    f"{last_value(now)} instead of {last_value(ref)}); put back the registered one")


# --- 2. a backup goes back to its owner ---------------------------------------------------------------------------

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


def say_returned(moved, where=WHERE):
    paths = {}
    for key, owner, current, buf in moved:
        path, _, name = key.rpartition(".")
        paths.setdefault(path, []).append((name, current, buf))
    for path, items in paths.items():
        name, current, buf = items[0]
        own = getattr(current, "_buffers", {}).get(name)
        load.say(where, f"the loader was about to write '{path}.{name}' of another object (last entry "
                        f"{last_value(buf)}) into the one this run uses (last entry {last_value(own)}); handed it "
                        f"back to its own object")


# --- where it hooks in ComfyUI -------------------------------------------------------------------------------------

def install_schedule_record():
    """Every sampling object ComfyUI or a node makes keeps a copy of the schedule its own setter registered."""
    import torch

    ms_mod = importlib.import_module("comfy.model_sampling")
    if _SETTERS:
        return 0
    classes = [c for c in vars(ms_mod).values()
               if isinstance(c, type) and issubclass(c, torch.nn.Module) and c.__module__ == ms_mod.__name__]
    _SETTERS.extend(wrap_setters(classes, ("set_sigmas", "set_parameters")))
    return len(_SETTERS)


def install_buffer_guard():
    """ModelPatcherDynamic backs buffers up by path and writes them back at the next load. Returns 1, or 0 when that
    loader is not there (legacy loading, another ComfyUI) or the guard is already in place."""
    mp = importlib.import_module("comfy.model_patcher")
    resolve = getattr(importlib.import_module("comfy.utils"), "resolve_attr", None)
    cls = getattr(mp, "ModelPatcherDynamic", None)
    if cls is None or not callable(resolve) or not all(callable(getattr(cls, n, None))
                                                       for n in ("load", "restore_loaded_backups")):
        return 0
    orig_load, orig_restore = cls.load, cls.restore_loaded_backups

    def load_(self, *a, **kw):
        out = orig_load(self, *a, **kw)
        try:
            remember_owners(self.model, self.backup_buffers, resolve)
        except Exception:  # noqa: BLE001 - bookkeeping only
            pass
        return out

    def restore_loaded_backups(self):
        try:
            moved = return_foreign(self.model, self.backup_buffers, resolve)
            if moved:
                say_returned(moved)
        except Exception as e:  # noqa: BLE001 - never break loading because the guard could not run
            load.say(WHERE, f"could not guard the buffer restore ({type(e).__name__}: {e})")
        return orig_restore(self)

    return 1 if _patch(cls, "load", load_) + _patch(cls, "restore_loaded_backups", restore_loaded_backups) else 0


def install_schedule_check():
    """At the first model call after a model's schedule buffers were (re)placed, check them against what the
    sampling object registered. The guard should leave nothing to find; this covers loaders it does not know."""
    cls = getattr(importlib.import_module("comfy.model_base"), "BaseModel", None)
    if cls is None or not callable(getattr(cls, "apply_model", None)):
        return 0
    orig = cls.apply_model

    def apply_model(self, *a, **kw):
        try:
            ms = self._modules.get("model_sampling")
            sig = ms._buffers.get("sigmas") if ms is not None else None
            last = self.__dict__.get("_entail_checked")
            if sig is not None and (last is None or last[0]() is not ms or last[1]() is not sig):
                found = drift(ms)
                if found:
                    put_back(ms, found)
                self.__dict__["_entail_checked"] = (weakref.ref(ms), weakref.ref(ms._buffers["sigmas"]))
        except Exception:  # noqa: BLE001 - a check that cannot run must never break sampling
            pass
        return orig(self, *a, **kw)

    return _patch(cls, "apply_model", apply_model)


def uninstall():
    n = 0
    for cls, name, fn in _SETTERS:
        setattr(cls, name, fn)
        n += 1
    _SETTERS.clear()
    for (owner, attr), orig in list(_ORIG.items()):
        setattr(owner, attr, orig)
        n += 1
    _ORIG.clear()
    return n
