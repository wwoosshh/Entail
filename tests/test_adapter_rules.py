"""S5, adapter thickness (LIBRARY_DESIGN.md 4.8, 8; ROADMAP M3.3): a static check that the v2 adapters hold no rule
logic, and their size. Reads the source only; imports nothing. Run: python tests/test_adapter_rules.py

An adapter gives three things - hooks, read_choice, handles - and install/uninstall. What would make it a place for
rules, and is therefore refused here:
  - importing from entail anything but the entry points: load (the load-time contracts, resolve, enforce),
    kv_contract, epochs and identity_contract (the container contracts, M5.1, M5.2 and M14), request_contract
    (the request contract, M5.3),
    policies.current,
    core.mode, readers.rotary_of / config_dict / prediction_kind / lora_modules / is_text_module / lora_base (to turn
    what it read into a fact value, M6.2), facts (fact classes), base (Hook). Not caps, contracts, sources or
    preflight: they hold tables, verdicts and precedence.
  - raising RoleError itself: stopping is load.enforce's, on a blocking decision.
  - a module-level table (a dict, list, set or tuple literal of more than three entries): tables are data files.
  - reading the mismatch policy (core.policy): the policy is applied by the core's decide.
Every file in adapters/ is either on this interface, listed in LEGACY with the stage that moves it, or listed in
ENGINE_SPECIFIC: a repair of one engine's own defect, which the design keeps out of the core and marks as the engine's
(M6.2: ComfyUI #16490).
"""
import ast
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTERS = os.path.join(os.path.dirname(HERE), "entail", "adapters")
V2 = ("transformers_adapter", "transformers_config", "sglang_adapter", "vllm_attention", "vllm_loader",
      "vllm_source", "rope_alias", "vllm_layout", "cache_contract", "vllm_cache_contract", "sglang_cache_contract",
      "vllm_serve", "comfyui", "diffusers_adapter", "transformers_template", "sglang_serve", "vllm_identity",
      "vllm_scoring", "sglang_fp8_tile", "transformers_tokenizer", "transformers_stops", "vllm_stops",
      "sglang_stops")
# Not yet on the v2 interface: where they move, and why they have not yet. Empty since M9.3, when the research tools
# (fault injection, the layout ledger, a probe) left the package.
LEGACY = {}
ENGINE_SPECIFIC = {   # repairs of one engine's own defect: not contracts of entail, kept out of the core (M6.2)
    "comfyui_repair": "ComfyUI's dynamic VRAM loader writes one object's schedule into another (Comfy-Org/ComfyUI#16490)",
}
NOT_ADAPTERS = ("__init__", "base")
ALLOWED = {"load": None, "kv_contract": None, "epochs": None, "identity_contract": None, "tile_contract": None,
           "vocab_contract": None, "stops_contract": None, "request_contract": None, "policies": {"current"}, "core": {"mode"},
           "readers": {"rotary_of", "config_dict", "prediction_kind", "lora_modules", "is_text_module", "lora_base"},
           "facts": None, "base": {"Hook"}}
REQUIRED = ("hooks", "read_choice", "handles", "install", "engine", "versions")


def _entail_module(node):
    """The entail module a relative import names ('load', 'core' ...), or None for a non-entail import."""
    if not isinstance(node, ast.ImportFrom) or node.level == 0:
        return None
    if node.level == 1:            # from .base import Hook
        return node.module or ""
    return node.module or ""       # from .. import load  /  from ..readers import rotary_of


def violations(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out, bound = [], {}
    for node in ast.walk(tree):
        mod = _entail_module(node)
        if mod is None:
            continue
        if mod == "":              # from .. import a, b
            for a in node.names:
                if a.name not in ALLOWED:
                    out.append(f"imports entail.{a.name}")
                bound[a.asname or a.name] = a.name
        else:
            allowed = ALLOWED.get(mod.split(".")[-1], set())
            if mod.split(".")[-1] not in ALLOWED:
                out.append(f"imports from entail.{mod}")
            for a in node.names:
                if allowed is not None and a.name not in allowed:
                    out.append(f"imports {a.name} from entail.{mod}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bound:
            allowed = ALLOWED[bound[node.value.id]]
            if allowed is not None and node.attr not in allowed and node.attr != "RoleError":
                out.append(f"uses {bound[node.value.id]}.{node.attr}")
        if isinstance(node, ast.Raise) and node.exc is not None and "RoleError" in ast.unparse(node.exc):
            out.append(f"raises RoleError itself (line {node.lineno})")
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, (ast.Dict, ast.List, ast.Set,
                                                                                      ast.Tuple)):
            n = len(node.value.keys) if isinstance(node.value, ast.Dict) else len(node.value.elts)
            if n > 3:
                out.append(f"a module-level table of {n} entries (line {node.lineno}); tables are data files")
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    out += [f"does not define {r}" for r in REQUIRED if r not in defined]
    return out


def code_lines(path):
    """Lines that are neither blank, comments nor docstrings."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    doc = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node) is not None:
            d = node.body[0]
            doc |= set(range(d.lineno, d.end_lineno + 1))
    lines = open(path, encoding="utf-8").read().splitlines()
    return sum(1 for i, t in enumerate(lines, 1) if t.strip() and not t.strip().startswith("#") and i not in doc)


def test_every_adapter_file_is_classified():
    files = {f[:-3] for f in os.listdir(ADAPTERS) if f.endswith(".py")}
    unclassified = files - set(V2) - set(LEGACY) - set(ENGINE_SPECIFIC) - set(NOT_ADAPTERS)
    assert not unclassified, f"adapters neither on v2 nor listed as legacy or engine-specific: {sorted(unclassified)}"
    assert set(V2) <= files and set(LEGACY) <= files and set(ENGINE_SPECIFIC) <= files


def test_an_engine_specific_repair_says_so_first():
    """A repair of one engine's own defect is marked where a reader starts: the first word of its docstring."""
    for name in ENGINE_SPECIFIC:
        doc = ast.get_docstring(ast.parse(open(os.path.join(ADAPTERS, name + ".py"), encoding="utf-8").read()))
        assert doc and doc.startswith("ENGINE-SPECIFIC"), name


def test_v2_adapters_hold_no_rules():
    found = {name: violations(os.path.join(ADAPTERS, name + ".py")) for name in V2}
    assert not any(found.values()), {k: v for k, v in found.items() if v}


def test_the_check_catches_rules():
    """The check itself, on the adapters as they were before M3.3 (their own rule logic)."""
    import tempfile
    old = ("from .. import core\nfrom . import _shared\nfrom ..preflight import CAPS, check\n"
           "PREFERENCE = ['eager', 'flex_attention', 'sdpa', 'x']\n"
           "def install():\n    if core.policy() == 'resolve':\n        raise core.RoleError('x')\n")
    p = os.path.join(tempfile.mkdtemp(), "old_adapter.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(old)
    v = violations(p)
    for text in ("imports entail._shared", "imports from entail.preflight", "uses core.policy",
                 "raises RoleError itself", "a module-level table of 4 entries", "does not define read_choice"):
        assert any(text in x for x in v), (text, v)


def test_thickness_is_recorded():
    """S5 asks for the size of an adapter; it is printed here and written into the M3 report, not bounded."""
    sizes = {name: code_lines(os.path.join(ADAPTERS, name + ".py")) for name in V2}
    print("  adapter code lines:", ", ".join(f"{k} {v}" for k, v in sizes.items()))
    assert all(v > 0 for v in sizes.values())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
