"""The start-up hook: how ENTAIL=load reaches the processes an engine spawns for itself.

vLLM and SGLang run the model in worker processes they start as fresh interpreters, so patching the parent is not
enough. pip installs `entail-autoinstall.pth` into site-packages next to the package, and Python runs its single
line at every start-up. The line reads the environment and does nothing else unless ENTAIL (or one of the testing
switches) is set; only then does it import the hook that installs the adapters.

An editable install (`pip install -e .`) does not place the file. `entail hook install` writes it, `entail hook
uninstall` removes it, and `entail hook status` says where it is.
"""
import os
import sysconfig

PTH_NAME = "entail-autoinstall.pth"
# One line, as .pth files require: Python executes a line that starts with "import".
PTH_LINE = ('import os; (os.environ.get("ENTAIL", "off") not in ("", "off") or os.environ.get("ENTAIL_LEDGER") '
            'or os.environ.get("ENTAIL_SEED")) and __import__("entail.adapters.autoinstall.sitecustomize")\n')


def path():
    """Where the file belongs for this interpreter (the environment's site-packages)."""
    return os.path.join(sysconfig.get_paths()["purelib"], PTH_NAME)


def status():
    """('installed' | 'absent' | 'different', path)."""
    p = path()
    if not os.path.exists(p):
        return "absent", p
    with open(p, encoding="utf-8") as f:
        return ("installed" if f.read() == PTH_LINE else "different"), p


def install():
    p = path()
    with open(p, "w", encoding="utf-8") as f:
        f.write(PTH_LINE)
    return p


def uninstall():
    p = path()
    if os.path.exists(p):
        os.remove(p)
        return p
    return None
