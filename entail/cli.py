"""The `entail` command.

  entail doctor                          what is installed, whether the start-up hook is in place, what would run
  entail hook {install,uninstall,status} manage the start-up hook in this environment
  entail check --model DIR --engine {sglang,transformers,vllm} [--attention NAME] [--manifests DIR] [--json FILE]
                                         the load decisions before anything runs, without a GPU (M3.4)
  entail preflight --model DIR --engine {sglang,transformers,vllm} [--backend NAME | --list]
                                         the 0.3.0 start-up check, kept for its users; `entail check` supersedes it
  entail infer PATH [--out FILE]         a manifest draft: what a model file or folder declares, and empty slots
  entail pin MANIFEST                    mark a reviewed manifest pinned, so its facts count as declarations
  entail probe --engine E --consumer C --fact Name.field --model DIR [--out FILE] [--no-gate]
                                         check one capability-table row with data (runs the engine on the GPU)
  entail version
"""
import argparse
import importlib.metadata as md
import json
import importlib.util
import os
import platform
import sys

from . import __version__, hook

# Versions the adapters were measured against (see README, "Tested with").
TESTED = {"transformers": "5.12.1, 5.16.1, 5.17.0", "vllm": "0.30.0", "sglang": "0.5.20"}


def _installed(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _targets():
    """The adapter table the start-up hook would use, built without installing anything."""
    saved = os.environ.get("ENTAIL")
    os.environ["ENTAIL"] = "off"
    try:
        spec = importlib.util.find_spec("entail.adapters.autoinstall.sitecustomize")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.TARGETS
    finally:
        if saved is None:
            os.environ.pop("ENTAIL", None)
        else:
            os.environ["ENTAIL"] = saved


def doctor(_args):
    print(f"entail {__version__}  python {platform.python_version()}  ({sys.executable})")
    state, p = hook.status()
    note = {"installed": "ENTAIL=load reaches child processes",
            "absent": "run `entail hook install` (editable installs do not place it)",
            "different": "the file exists but is not the one this version writes; run `entail hook install`"}[state]
    print(f"start-up hook: {state}  {p}\n  {note}")
    print("engines in this environment:")
    for name in ("torch", "transformers", "vllm", "sglang"):
        v = _installed(name)
        tested = f"   (tested with {TESTED[name]})" if name in TESTED else ""
        print(f"  {name:13} {v or 'not installed'}{tested if v else ''}")
    mode = os.environ.get("ENTAIL", "off")
    print(f"ENTAIL={mode}  ENTAIL_POLICY={os.environ.get('ENTAIL_POLICY', 'resolve')}"
          f"  ENTAIL_ONLY={os.environ.get('ENTAIL_ONLY', '') or '(all)'}")
    print("adapters that ENTAIL=load installs, once their module is imported:")
    for module, adapters in _targets().items():
        top = module.split(".")[0]
        # ComfyUI is not a pip package: it is importable when you run from its folder, as its launcher does
        present = (importlib.util.find_spec(top) is not None or os.path.isdir(os.path.join(os.getcwd(), top))
                   or os.path.isfile(os.path.join(os.getcwd(), top + ".py")))
        for a in adapters:
            print(f"  {a:48} after {module}{'' if present else '   (engine not installed here)'}")
    return 0


def _hook(args):
    if args.action == "status":
        state, p = hook.status()
        print(f"{state}  {p}")
        return 0 if state == "installed" else 1
    try:
        if args.action == "install":
            print(f"installed {hook.install()}")
        else:
            p = hook.uninstall()
            print(f"removed {p}" if p else "nothing to remove")
    except PermissionError as e:
        print(f"cannot write to site-packages here ({e}). Use a virtual environment, or run with the rights to "
              f"change this Python installation.")
        return 1
    return 0


def _infer(args):
    from . import manifest
    draft = manifest.infer(args.path)
    text = json.dumps(manifest.to_json(draft), ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            out.write(text + "\n")
        print(f"wrote {args.out}: {sum(x.value is None for x in draft.facts)} empty slot(s) to review")
    else:
        print(text)
    return 0


def _check(args):
    """`entail check`: exit 1 when a decision would stop the run, 2 when the named backend is not in the table."""
    from . import load, record, sites
    from .contracts import Verdict

    per_backend, model, notes = sites.check_static(args.model, args.engine, {"attention": args.attention,
                                                                            "manifest_dirs": args.manifests})
    facts = load.declared(args.model)
    print(f"entail check: {args.model} on {args.engine}")
    for name, group in sorted(facts.facts.items()):
        for f in group:
            print(f"  declared {f.value} ({f.source}, {f.certainty.value})")
    for n in notes:
        print(f"  note: {n}")
    if not args.attention:
        print("  attention backends in the capability table:")
        for d in per_backend:
            what = f"{d.verdict.value}" + (f" (to {d.target})" if d.target else "") + ("; stops" if d.blocking else "")
            print(f"    {d.contract.consumer:42} {what}")
        if not per_backend:
            print("    (the model declares nothing the attention backends act on)")
    for d in (per_backend if args.attention else []) + model:
        print(record.line(d))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "engine": args.engine, "notes": notes,
                       "attention": [record.decision_json(d) for d in per_backend],
                       "model_decisions": [record.decision_json(d) for d in model]}, f, ensure_ascii=False, indent=1)
    chosen = per_backend if args.attention else []
    if any(d.blocking for d in chosen + model):
        return 1
    if args.attention and any(d.verdict is Verdict.UNKNOWN and d.rule.startswith("what the consumer uses")
                              for d in chosen):
        return 2
    return 0


def _pin(args):
    from . import manifest
    m = manifest.load(args.manifest)
    empty = [f.name for f in m.facts if f.value is None]
    manifest.save(manifest.pin(m), args.manifest)
    print(f"pinned {args.manifest}" + (f"; still empty (stay unknown): {empty}" if empty else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="entail", description="Keep what a value means intact across LLM "
                                 "inference-stack boundaries.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="show the environment and what entail would do in it")
    h = sub.add_parser("hook", help="manage the start-up hook in this environment")
    h.add_argument("action", choices=["install", "uninstall", "status"])
    sub.add_parser("preflight", help="check a model directory against an engine's backends", add_help=False)
    i = sub.add_parser("infer", help="write a manifest draft for a model file or folder")
    i.add_argument("path")
    i.add_argument("--out", help="write the draft here instead of printing it")
    pn = sub.add_parser("pin", help="mark a reviewed manifest pinned")
    pn.add_argument("manifest")
    ck = sub.add_parser("check", help="the load decisions for a model and an engine, before anything runs")
    ck.add_argument("--model", required=True, help="a local model folder")
    ck.add_argument("--engine", required=True, choices=["transformers", "sglang", "vllm"])
    ck.add_argument("--attention", help="the backend the engine will use (default: every backend in the table)")
    ck.add_argument("--manifests", action="append", default=[], help="a folder to look for manifests in")
    ck.add_argument("--json", help="write the decisions as JSON here")
    pr = sub.add_parser("probe", help="check one capability-table row with data")
    pr.add_argument("--engine", required=True, choices=["transformers", "sglang", "vllm"])
    pr.add_argument("--consumer", required=True, help="short name (sdpa) or full name (transformers.attention.sdpa)")
    pr.add_argument("--fact", required=True, help="vocabulary name and field, e.g. ModelProps.softcap")
    pr.add_argument("--model", required=True, help="a local model folder that declares the fact")
    pr.add_argument("--table", help="a capability table other than the packaged one")
    pr.add_argument("--out", help="write the measurement as JSON here")
    pr.add_argument("--no-gate", action="store_true", help="skip the binding gate for an 'ignores' verdict")
    sub.add_parser("version", help="print the version")
    args, rest = ap.parse_known_args(argv)
    if args.cmd == "preflight":
        from .preflight import main as preflight_main

        return preflight_main(rest)
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.cmd == "version":
        print(__version__)
        return 0
    if args.cmd == "doctor":
        return doctor(args)
    if args.cmd == "infer":
        return _infer(args)
    if args.cmd == "pin":
        return _pin(args)
    if args.cmd == "probe":
        from .probes import main as probe_main

        return probe_main(args)
    if args.cmd == "check":
        return _check(args)
    return _hook(args)


if __name__ == "__main__":
    sys.exit(main())
