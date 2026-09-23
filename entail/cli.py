"""The `entail` command.

  entail doctor                          what is installed, whether the start-up hook is in place, what would run
  entail hook {install,uninstall,status} manage the start-up hook in this environment
  entail preflight --model DIR --engine {sglang,transformers,vllm} [--backend NAME | --list]
  entail version
"""
import argparse
import importlib.metadata as md
import importlib.util
import os
import platform
import sys

from . import __version__, hook

# Versions the adapters were measured against (see README, "Tested with").
TESTED = {"transformers": "5.12.1, 5.17.0", "vllm": "0.30.0", "sglang": "0.5.20"}


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
        present = importlib.util.find_spec(module.split(".")[0]) is not None
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="entail", description="Keep what a value means intact across LLM "
                                 "inference-stack boundaries.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor", help="show the environment and what entail would do in it")
    h = sub.add_parser("hook", help="manage the start-up hook in this environment")
    h.add_argument("action", choices=["install", "uninstall", "status"])
    sub.add_parser("preflight", help="check a model directory against an engine's backends", add_help=False)
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
    return _hook(args)


if __name__ == "__main__":
    sys.exit(main())
