"""Start-up check: is what the checkpoint declares still true of the server about to run it?

This is the "load" check of entail (DESIGN.md section 2) as a command-line tool. Three facts, all read
before the model is loaded:

  PROPERTY (attention) - attn_logit_softcapping and sliding_window against a capability table for the engine's
      attention backends. The table was written by reading the installed engines (audits/W_property_audit.md)
      and the rows marked measured were confirmed by running them (audits/cap_probe.py).
  MAPPING (config keys) - a key in config.json may be renamed on the way in, but its value has to survive
      somewhere in the resolved config. The naive "every key must be an attribute" rule fires on healthy
      models; measured in audits/W_MORE_FACTS.md.
  PROPERTY (tied embeddings) - tie_word_embeddings against what the checkpoint actually ships. When the value
      comes from the library default rather than the config, the message says so (PRECEDENCE).

Usage:
  entail preflight --model /path/to/gemma-2-2b-it --engine sglang --backend flashinfer
  entail preflight --model /path/to/gemma-2-2b-it --engine sglang --list
Exit code 1 means something declared would be dropped or contradicted silently; 2 means the table does not
cover the backend, so nothing was checked.
"""
import argparse
import json
import os
import sys

from .facts import KernelCaps, ModelProps

# engine -> backend -> (caps, evidence)
CAPS = {
    "transformers": {
        "eager": (KernelCaps(softcap=True, sliding_window=True), "measured: audits/cap_probe"),
        "sdpa": (KernelCaps(softcap=False, sliding_window=True),
                 "integrations/sdpa_attention.py never reads softcap (measured: audits/cap_probe, rolebench 08)"),
        "flex_attention": (KernelCaps(softcap=True, sliding_window=True),
                           "score_mod applies softcap (measured: audits/cap_probe)"),
        "paged|eager": (KernelCaps(softcap=False, sliding_window=True),
                        "continuous batching paged kernels take no softcap (measured: audits/cap_probe,"
                        " audits/tf_paged_softcap)"),
        "paged|sdpa": (KernelCaps(softcap=False, sliding_window=True), "measured: audits/cap_probe"),
    },
    "sglang": {
        "triton": (KernelCaps(softcap=True, sliding_window=True), "measured: audits/sglang_backend_survey"),
        "flashinfer": (KernelCaps(softcap=False, sliding_window=True),
                       "measured: audits/sglang_backend_survey; upstream issue #33915"),
        "flex_attention": (KernelCaps(softcap=False, sliding_window=False),
                           "measured: audits/sglang_backend_survey, sweep/RESULTS.md; block masks are causal-only"),
        "torch_native": (KernelCaps(softcap=False, sliding_window=True),
                         "measured: rolebench case 17; window only on causal layers"),
        "trtllm_mha": (KernelCaps(softcap=False, sliding_window=True), "code read: no logit_cap in the backend"),
    },
    "vllm": {
        "FLASH_ATTN": (KernelCaps(softcap=True, sliding_window=True),
                       "measured: sweep/RESULTS.md (vLLM 0.30.0, sm_89); softcap passed to the kernel"),
        "TRITON_ATTN": (KernelCaps(softcap=True, sliding_window=True), "measured: sweep/RESULTS.md (vLLM 0.30.0)"),
        "FLASHINFER": (KernelCaps(softcap=True, sliding_window=True),
                       "code read: softcap applied except on the TRTLLM/XQA paths (SM100, Hopper XQA)"),
        # A code read said this backend raises on softcap. Run, it honours it without raising (vLLM 0.30.0).
        "FLEX_ATTENTION": (KernelCaps(softcap=True, sliding_window=True), "measured: sweep/RESULTS.md (vLLM 0.30.0)"),
        "ROCM_ATTN": (KernelCaps(softcap=False, sliding_window=True), "code read: softcap stored, not passed"),
    },
}


def read_props(model_dir):
    with open(os.path.join(os.path.expanduser(model_dir), "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    return ModelProps(softcap=text.get("attn_logit_softcapping"), sliding_window=text.get("sliding_window"),
                      tie_word_embeddings=cfg.get("tie_word_embeddings"))


def check(props, caps):
    missing = []
    if props.softcap is not None and not caps.softcap:
        missing.append(f"attn_logit_softcapping={props.softcap}")
    if props.sliding_window is not None and not caps.sliding_window:
        missing.append(f"sliding_window={props.sliding_window}")
    return missing


# --- facts that need no engine: they are about the checkpoint and its config -------------------------------

def _scalars(obj, out=None):
    """Every scalar anywhere in a nested config, so a key that was renamed can still be found by its value."""
    out = set() if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            _scalars(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _scalars(v, out)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out.add((type(obj).__name__, obj))
    return out


# Keys that are legitimately not model-config fields. Each entry says who reads the key instead, so the list
# stays a declaration and not a way to silence the check. Anything not listed here is reported.
ELSEWHERE = {
    "cache_implementation": "generation_config.json holds this one and GenerationConfig reads it "
                            "(checked on a gemma-2-2b-it checkpoint)",
    "_name_or_path": "where the checkpoint was loaded from, not a fact about the model",
    "architectures": "the loader picks the class from it before the config object exists",
    "torch_dtype": "read by from_pretrained, not stored as a config field",
    "dtype": "read by from_pretrained, not stored as a config field",
}
PROVENANCE_SUFFIXES = ("_version",)  # who produced the checkpoint: unsloth_version, transformers_version


def _classify(key):
    if key in ELSEWHERE:
        return "note", ELSEWHERE[key]
    if key.endswith(PROVENANCE_SUFFIXES):
        return "note", "provenance metadata about the tool that wrote the checkpoint"
    return "complaint", None


def _known_fields(cfg):
    """The fields this config class actually declares, as opposed to whatever was stored on the instance.

    transformers keeps an unknown key as a plain attribute and says nothing, so `hasattr` proves nothing:
    that is exactly benchmark case 15. A default instance of the same class is the honest list.
    """
    try:
        return set(type(cfg)().to_dict())
    except Exception:
        return None


def check_config_keys(model_dir):
    """MAPPING: a key in config.json must land in a field the loader knows, or its value must survive in one.

    Three rules were measured (audits/W_MORE_FACTS.md):
      naive ("every key must be an attribute"): 3 false positives on 3 healthy models - transformers 5.17
          moves rope_theta into rope_parameters and keeps the value.
      value-survives-anywhere: 0 false positives, but it also passes case 15, because an unknown key is stored
          as an attribute and therefore "survives" in to_dict().
      this one (unknown to the class AND the value is in no known field): 0 false positives, catches case 15.
    """
    path = os.path.expanduser(model_dir)
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    try:
        from transformers import AutoConfig
    except ImportError:
        return None, "transformers is not installed"
    cfg = AutoConfig.from_pretrained(path)
    lost, notes = [], []
    for where, obj, d in (("", cfg, raw), ("text_config.", getattr(cfg, "text_config", None),
                                           raw.get("text_config", {}))):
        if obj is None:
            continue
        known = _known_fields(obj)
        if known is None:
            return None, f"cannot build a default {type(obj).__name__} to list its fields"
        resolved = _scalars({k: v for k, v in obj.to_dict().items() if k in known})
        for k, v in d.items():
            if k == "text_config" or k in known:
                continue
            if not isinstance(v, (dict, list)) and (type(v).__name__, v) in resolved:
                continue  # renamed on the way in, but the value landed in a field the loader knows
            kind, why = _classify(k)
            if kind == "note":
                notes.append(f"{where}{k}={v!r} ({why})")
            else:
                lost.append(f"{where}{k}={v!r} is not a field of {type(obj).__name__} and its value is in none")
    why = f"{len(raw)} keys in config.json"
    if notes:
        why += f"; {len(notes)} read elsewhere: {'; '.join(notes)}"
    return lost, why


def check_tie(model_dir):
    """PROPERTY: tie_word_embeddings against what the checkpoint actually ships (benchmark case 07)."""
    import struct

    path = os.path.expanduser(model_dir)
    with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    declared = raw.get("tie_word_embeddings", (raw.get("text_config") or {}).get("tie_word_embeddings"))
    source = "declared in config.json"
    if declared is None:
        # PRECEDENCE: the checkpoint said nothing, so the value comes from the library default. That is still
        # checkable, but the message has to say where the value came from.
        try:
            from transformers import AutoConfig
        except ImportError:
            return None, "the config does not declare tie_word_embeddings"
        cfg = AutoConfig.from_pretrained(path)
        text = getattr(cfg, "text_config", cfg)
        declared = getattr(text, "tie_word_embeddings", None)
        if declared is None:
            return None, "neither the config nor the library gives tie_word_embeddings"
        source = "library default, not in config.json"
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            names = set(json.load(f)["weight_map"])
    else:
        single = os.path.join(path, "model.safetensors")
        if not os.path.exists(single):
            return None, "no safetensors checkpoint to compare with"
        with open(single, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            names = set(json.loads(f.read(n))) - {"__metadata__"}
    has_head = any(name.endswith("lm_head.weight") for name in names)
    if declared and has_head:
        return ([f"tie_word_embeddings=true ({source}) but the checkpoint ships lm_head.weight; "
                 "one of them is wrong and nothing compares them"], f"{len(names)} tensors")
    if not declared and not has_head:
        return ([f"tie_word_embeddings=false ({source}) but the checkpoint has no lm_head.weight"],
                f"{len(names)} tensors")
    return [], f"{len(names)} tensors, tie={declared} ({source})"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", required=True, choices=sorted(CAPS))
    ap.add_argument("--backend")
    ap.add_argument("--list", action="store_true", help="show every backend of the engine for this model")
    args = ap.parse_args(argv)
    props = read_props(args.model)
    declared = {k: v for k, v in vars(props).items() if v is not None}
    print(f"model {args.model}: declared {declared or '{}'}")

    # Facts that do not depend on the engine. They are checked first: a checkpoint that disagrees with its own
    # config is wrong on every backend.
    standalone = 0
    for label, fn in (("config keys (MAPPING)", check_config_keys), ("tied embeddings (PROPERTY)", check_tie)):
        try:
            said, why = fn(args.model)
        except Exception as e:  # a check that cannot run says so; it never passes silently
            print(f"  {'could not check':45} {label} ({type(e).__name__}: {e})")
            continue
        if said is None:
            print(f"  {'not checked':45} {label} ({why})")
        elif said:
            print(f"  {'RoleError: ' + '; '.join(said):45} {label} ({why})")
            standalone = 1
        else:
            print(f"  {'ok':45} {label} ({why})")

    table = CAPS[args.engine]
    if args.list or not args.backend:
        worst = 0
        for name, (caps, why) in sorted(table.items()):
            missing = check(props, caps)
            print(f"  {'DROPS ' + ', '.join(missing) if missing else 'ok':45} {args.engine}:{name:16} ({why})")
            worst = max(worst, 1 if missing else 0)
        return 0 if args.list else max(worst, standalone)
    if args.backend not in table:
        print(f"RoleError: the table does not cover {args.engine} backend '{args.backend}', so nothing was checked."
              f"\n  known: {', '.join(sorted(table))}")
        return 2
    caps, why = table[args.backend]
    missing = check(props, caps)
    if missing:
        print(f"RoleError: {args.engine} backend '{args.backend}' does not honour: {', '.join(missing)}\n  evidence: {why}")
        return 1
    print(f"ok: {args.engine} backend '{args.backend}' honours the declared properties\n  evidence: {why}")
    return standalone


if __name__ == "__main__":
    sys.exit(main())
