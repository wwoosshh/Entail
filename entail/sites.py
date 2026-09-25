"""sites: where checks run, and what each may cost (LIBRARY_DESIGN.md 4.7, principle 6).

Always on: load (once, at start-up), container (host-side updates of a cache or a scheduler), request (per request).
Debug and CI only: the diagnosis site (diagnose.py): operation-level propagation (propagate.py) and layers compared
with a reference. Its cost is measured in testbed/results/m71 and m73 (target: at most 2x).
Never inside a compiled or captured region: a check placed there cost 47.7x and changed the output
(reinvestigation/feasibility.md 2.3). Before running: `entail check` makes the load decisions without a GPU.
"""
import os

SITES = ("load", "container", "request", "debug")

# Cost targets (LIBRARY_DESIGN.md 8, S4). Off costs nothing at every site.
BUDGET = {
    "load": "at most 5% of start-up time",
    "container": "at most 1.02x decode time, on the CUDA Graph path too",
    "request": "microseconds per request",
    "debug": "at most 2x",
}


def compares_head(engine: str) -> bool:
    """Whether the engine's loader compares a shipped lm_head.weight with the embedding before tying (load.tie,
    M11.2): what each adapter states of its engine version; False for an engine without an adapter."""
    from .adapters import sglang_adapter, transformers_adapter, vllm_loader

    module = {"transformers": transformers_adapter, "vllm": vllm_loader, "sglang": sglang_adapter}.get(engine)
    return bool(getattr(module, "compares_head", False))


def check_static(model_path: str, engine: str, settings: dict):
    """`entail check` (M3.4): the load decisions for a model folder and an engine, before anything runs, without a
    GPU. settings: "attention" (a backend name, or None for every backend of the engine in the capability table),
    "manifest_dirs", "policy". The engine's config class is built with transformers when it is installed (with entail
    off, so its hooks do not decide twice): it gives the class defaults and the key coverage.
    Returns (decisions for the chosen or every backend, decisions about the model itself, notes)."""
    from dataclasses import replace

    from . import caps, core, load, observe, policies
    from .adapters import transformers_config

    policy = settings.get("policy") or replace(policies.from_env(), mode="load")
    table = settings.get("table") or caps.default_table()
    path = os.path.expanduser(model_path)
    notes, config, raw = [], None, None
    cfg_file = os.path.join(path, "config.json")
    if os.path.isfile(cfg_file):
        import json

        with open(cfg_file, encoding="utf-8") as f:
            raw = json.load(f)
        was = core.mode()
        try:
            core.set_mode("off")
            from transformers import AutoConfig

            # a static check runs no code from the model folder and asks no question: a remote-code model fails at
            # once (transformers would otherwise prompt, and wait 15 s where there is no terminal); nothing is fetched
            config = AutoConfig.from_pretrained(path, trust_remote_code=False, local_files_only=True)
        except ImportError:
            notes.append("transformers is not installed: class defaults and config key coverage are not checked")
        except Exception as e:  # noqa: BLE001 - the check reports it and goes on with the files alone
            notes.append(f"transformers could not build the config ({type(e).__name__}: {e}); checked the files alone")
        finally:
            core.set_mode(was)
    facts = load.declared(path, config, settings.get("manifest_dirs", ()))
    notes = facts.problems + notes
    if settings.get("attention"):
        name = settings["attention"]
        role, short = ("paged_attention", name.split("|", 1)[1]) if name.startswith("paged|") else ("attention", name)
        backends = [(role, short)]
    else:
        backends = [tuple(c.split(".", 2)[1:]) for c in caps.consumers(table, f"{engine}.attention")]
        backends += [tuple(c.split(".", 2)[1:]) for c in caps.consumers(table, f"{engine}.paged_attention")]
    per_backend = []
    for role, short in backends:
        per_backend += load.attention(engine, short, facts, table, policy, role=role)
    model = []
    tie = getattr(config, "tie_word_embeddings", None) if config is not None else None
    if tie is None and config is not None:
        tie = getattr(getattr(config, "text_config", None), "tie_word_embeddings", None)

    def keys():
        scopes = transformers_config.read_choice(config, raw)
        if scopes is None:
            return [load.cannot_check(f"load:{engine}.config", f"{engine}.config", "Coverage",
                                      f"{type(config).__name__} cannot be built with defaults", policy)]
        return load.config_keys(engine, scopes, cfg_file, policy)

    checks = [("tie", lambda: load.tie(engine, facts, path, loader_ties=tie if isinstance(tie, bool) else None,
                                       policy=policy, compares_head=compares_head(engine)))]
    if config is not None and isinstance(raw, dict):
        checks.append(("config keys", keys))
    if facts.get("Layout"):
        checks.append(("layout", lambda: load.layout(f"{engine}.linear.unknown", facts, table,
                                                     observe.scale_format(path), policy)))

    def tokenizer():
        """The tokenizer transformers builds for the folder against the model's vocabulary (M15.3). vLLM and SGLang
        build theirs through the same AutoTokenizer, so the static verdict stands for the three engines."""
        from . import vocab_contract

        size = n = None
        where = "no tokenizer built"
        was = core.mode()
        try:
            core.set_mode("off")
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
            size, n = getattr(tok, "vocab_size", None), len(tok)
            where = f"{type(tok).__name__} built by transformers from the folder"
        except ImportError:
            notes.append("transformers is not installed: the tokenizer is not built; its files are read alone")
        except Exception as e:  # noqa: BLE001 - reported; the files are still compared with each other
            notes.append(f"transformers could not build the tokenizer ({type(e).__name__}: {e})")
        finally:
            core.set_mode(was)
        return vocab_contract.check(f"load:{engine}.tokenizer", f"{engine}.tokenizer", path, size, n, where,
                                    policy=policy, record=False)

    checks.append(("tokenizer", tokenizer))
    for label, run in checks:
        try:
            model += run()
        except Exception as e:  # noqa: BLE001 - one check that cannot run does not take the others down
            notes.append(f"could not check the {label}: {type(e).__name__}: {e}")
    return per_backend, model, notes


def at_load(model_path, engine: str, choices: dict, policy=None, config=None, table=None, manifest_dirs=()):
    """The load decisions (load.py) for one model and what one engine chose for it:
      choices["attention"]  the attention backend's name, e.g. "sdpa"
      choices["tie"]        whether the loader ties the head (what the config it holds says), None if unknown
      choices["layout"]     the consumer that reads the stored weights, e.g. "vllm.linear.Fp8LinearMethod"
    `config` is the config object the engine holds (optional). Returns the Decisions; load.enforce records them."""
    from . import load, observe

    facts = load.declared(model_path, config, manifest_dirs)
    out = []
    if choices.get("attention"):
        out += load.attention(engine, choices["attention"], facts, table, policy)
    out += load.tie(engine, facts, model_path, choices.get("tie"), policy, compares_head=compares_head(engine))
    if choices.get("layout"):
        seen = observe.scale_format(os.path.expanduser(model_path)) if model_path else None
        out += load.layout(choices["layout"], facts, table, observed=seen, policy=policy)
    return out


# The container site is kv_contract.py and epochs.py, called by the cache adapters (cache_contract,
# vllm_cache_contract, sglang_cache_contract); the request site is request_contract.py, called by vllm_serve.


def debug_propagation(on_conflict: str = "policy"):
    """Operation-level propagation for diagnosis (M7.1): propagate.RolePropagation in debug mode, a context that does
    nothing in any other mode (diagnose.propagating)."""
    from .diagnose import propagating

    return propagating(on_conflict)
