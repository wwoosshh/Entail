"""What every engine adapter needs: read the declared properties off a config, and judge one backend.

Two outcomes, and they are not the same thing:
  violation - the table says this backend drops a property the model declares. Always an error.
  uncovered - the table does not know this backend, so nothing was checked. An error in `debug` mode, a printed
              warning in `load` mode, because a research table will always lag behind new backends.
"""
from ..facts import ModelProps
from ..preflight import CAPS, check


def props_of(config):
    """ModelProps from a HF config object (PretrainedConfig or anything with the same attributes)."""
    text = getattr(config, "text_config", None) or config
    return ModelProps(softcap=getattr(text, "attn_logit_softcapping", None),
                      sliding_window=getattr(text, "sliding_window", None))


def verdict(config, backend, engine):
    """Return (kind, message) for this config and backend, or None when there is nothing to say."""
    props = props_of(config)
    declared = [f"{k}={v}" for k, v in vars(props).items() if v is not None]
    if not declared:
        return None
    table = CAPS[engine]
    entry = table.get(backend) if isinstance(backend, str) else None
    if entry is None:
        return ("uncovered",
                f"{engine} attention backend '{backend}' is not in the capability table, so "
                f"{', '.join(declared)} was not checked\n  known: {', '.join(sorted(table))}")
    missing = check(props, entry[0])
    if not missing:
        return None
    return ("violation",
            f"{engine} attention backend '{backend}' does not honour: {', '.join(missing)}"
            f"\n  evidence: {entry[1]}")


RESOLUTIONS = []  # every resolution made in this process, so a caller or a test can see what was changed


def choose(config, engine, preference):
    """The first backend in `preference` that honours everything this model declares.

    Only capabilities marked as measured count: a resolution routes the value to a consumer we have *seen*
    honour the declaration, never to one we only believe does (the table has been wrong from code reading
    before - vLLM FLEX_ATTENTION, sweep/RESULTS.md).
    """
    props = props_of(config)
    table = CAPS[engine]
    for name in preference:
        entry = table.get(name)
        if entry is None or check(props, entry[0]) or "measured:" not in entry[1]:
            continue
        return name
    return None


def note_resolution(engine, where, frm, to, config):
    """Say what was changed and why. A resolution that nobody hears about is the silent behaviour this
    library exists to remove, so this always prints, whatever the verbosity."""
    missing = check(props_of(config), CAPS[engine][frm][0]) if frm in CAPS[engine] else ["(not in the table)"]
    evidence = CAPS[engine][to][1]
    RESOLUTIONS.append({"engine": engine, "where": where, "from": frm, "to": to, "missing": missing})
    print(f"[entail] resolved {where}: {engine} backend '{frm}' does not honour {', '.join(missing)}; "
          f"using '{to}' instead (evidence: {evidence})", flush=True)


def note(record, message):
    """A resolution that is not a backend switch (a value converted to the form its consumer reads). Recorded
    and always printed, for the same reason as note_resolution."""
    RESOLUTIONS.append(record)
    print(f"[entail] resolved {message}", flush=True)


def report(said, where=""):
    """Raise or warn according to the mode. Returns True when it warned."""
    from .. import core

    if said is None:
        return False
    kind, msg = said
    if where:
        msg = f"{where}: {msg}"
    if kind == "violation" or core.mode() == "debug":
        raise core.RoleError(msg)
    print(f"[entail] {msg}", flush=True)
    return True
