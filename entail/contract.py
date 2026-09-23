"""Compare what an artifact declares with what its consumer is about to use; resolve or stop, and always say so.

An engine adapter supplies two things: what the consumer decided (read from the engine), and a way to make it use
something else. Everything else happens here, the same way for every engine:

  reconcile(Prediction, declared, used, where=..., resolve=..., explicit=..., evidence=...)

  declared   the artifact's own statement (a declared.Declared record, or None)
  used       the fact the consumer decided
  resolve    callable(fact) -> whatever the adapter needs to go on with; None when this engine cannot switch here
  explicit   the user set the consumer's value explicitly (a node, an argument): never overridden silently
  evidence   (fact, how) from watching the running model; it decides only when nothing is declared, and it is
             reported when it contradicts a declaration. Behaviour alone misjudged a model that was switching
             objective (VPRED_PROTOCOL.md M7: probe 0.94, file says v), so a declaration comes first.

Outcomes, in order:
  1. nothing declared, no evidence                 -> nothing to check
  2. the source agrees with what is used           -> nothing to do (a contradicting observation is said in one line)
  3. explicit user choice contradicts the source   -> RoleError: what the file or the model says, and the choice
  4. resolve given and policy "resolve"            -> resolve(source fact), one line saying what, why and from where
  5. otherwise                                     -> RoleError
"""
from dataclasses import replace

from . import core
from .adapters import _shared


def agrees(declared, used, compare_zsnr=True):
    """Two facts of the same kind agree. A field the declaration leaves open (None) is not compared."""
    if type(declared) is not type(used):
        return False
    for name, want in vars(declared).items():
        if want is None or (name == "zsnr" and not compare_zsnr):
            continue
        if getattr(used, name) != want:
            return False
    return True


def reconcile(kind, declared, used, where, resolve=None, explicit=False, evidence=None, what=None):
    """Returns (the fact to use, the adapter's result of resolve or None). See the module docstring."""
    name = kind.__name__
    fact = declared.get(kind) if declared else None
    source = declared.source(kind) if declared else None
    if fact is None and evidence is not None:
        fact, source = evidence[0], f"its behaviour ({evidence[1]})"
    if fact is None:
        return used, None
    # An explicit choice is compared on what it contradicts outright; a schedule detail (zsnr) the user set on
    # purpose is theirs to set.
    if agrees(fact, used, compare_zsnr=not explicit):
        if evidence is not None and declared is not None and declared.get(kind) is not None \
                and not agrees(declared.get(kind), evidence[0], compare_zsnr=False):
            print(f"[entail] {where}: {what or name} {used} as declared by {source}, although "
                  f"the model behaves like {evidence[0]} ({evidence[1]})", flush=True)
        return used, None
    said = f"{where}: {what or name} is {used}, but {source} says {fact}"
    if explicit:
        raise core.RoleError(f"{said}. It was set explicitly, so entail does not override it; set it to {fact}, "
                             f"or remove the setting.")
    if resolve is None or core.policy() == "refuse":
        raise core.RoleError(f"{said}. {'Nothing here can switch it.' if resolve is None else ''}".rstrip())
    # What the source leaves open keeps the value the consumer had.
    target = replace(fact, **{k: getattr(used, k) for k, v in vars(fact).items() if v is None and hasattr(used, k)})
    out = resolve(target)
    extra = ""
    if evidence is not None and declared is not None and declared.get(kind) is not None \
            and not agrees(declared.get(kind), evidence[0], compare_zsnr=False):
        extra = f"; note: the model's behaviour looks like {evidence[0]} ({evidence[1]})"
    _shared.note({"where": where, "fact": name, "from": str(used), "to": str(target), "source": source},
                 f"{what or name}: {said.split(': ', 1)[1]}; using {target}{extra}")
    return target, out
