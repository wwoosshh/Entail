"""Adapter v2, load side of vLLM: does the weight in memory still hold what the checkpoint file holds?
(LIBRARY_DESIGN.md 4.8; ROADMAP M3.3; test problem fd-shift)

A weight that was already wrong when it arrived - read from the wrong offset, fused in the wrong order, a shard
swapped - looks like any other weight: shapes and dtypes are right. The only reference is the file itself. To compare
a fused weight with the checkpoint one has to know which source tensor each row came from, and that mapping lives in
vLLM's loader code rather than in any declaration; fusion_plan below writes it down per model family.

  hook         vllm.model_executor.model_loader.utils.process_weights_after_loading: the weights are all loaded and
               not yet repacked.
  read_choice  for every linear weight: sampled elements compared with the checkpoint rows the fusion plan names;
               the weights whose samples did not land where the plan puts them, and, for a weight that cannot be
               compared (quantised, not in the checkpoint, no plan), the reason.
  handles      none: a wrong weight cannot be repaired here.
load.weights_taken decides (anything that did not land -> refused); load.cannot_check reports what was not compared.
Only bf16/fp16 weights with tensor parallelism 1 are compared. Opt-in (ENTAIL_SOURCE=1): it costs a little I/O.
"""
import os

from .. import core, load, policies
from .base import Hook

engine = "vllm"
versions = "0.30.0"
SAMPLE_ROWS, SAMPLE_COLS = 4, 4  # 16 elements per weight


def hooks():
    return [Hook("vllm.model_executor.model_loader.utils.process_weights_after_loading", "load")]


def handles():
    return {}


# layer suffix -> [(checkpoint suffix, how many rows it contributes)], rows resolved from the HF config.
# "llama-like" covers Qwen 2/3, Llama, Mistral, Gemma: q/k/v fused into qkv_proj, gate/up into gate_up_proj.
def fusion_plan(suffix, cfg, weight_rows):
    hidden = getattr(cfg, "hidden_size", None)
    heads = getattr(cfg, "num_attention_heads", None)
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    head_dim = getattr(cfg, "head_dim", None) or (hidden // heads if hidden and heads else None)
    inter = getattr(cfg, "intermediate_size", None)
    if suffix == "self_attn.qkv_proj" and head_dim:
        return [("self_attn.q_proj.weight", heads * head_dim),
                ("self_attn.k_proj.weight", kv_heads * head_dim),
                ("self_attn.v_proj.weight", kv_heads * head_dim)]
    if suffix == "mlp.gate_up_proj" and inter:
        return [("mlp.gate_proj.weight", inter), ("mlp.up_proj.weight", inter)]
    if suffix in ("self_attn.o_proj", "mlp.down_proj"):
        return [(suffix + ".weight", weight_rows)]
    return None


class Checkpoint:
    """Single elements read straight from the safetensors files, without loading a tensor."""

    def __init__(self, path):
        import json

        self.path = os.path.expanduser(path)
        index = os.path.join(self.path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index, encoding="utf-8") as f:
                self.map = json.load(f)["weight_map"]
        else:
            single = os.path.join(self.path, "model.safetensors")
            if not os.path.exists(single):
                self.map = {}
                return
            from safetensors import safe_open

            with safe_open(single, framework="pt") as f:
                self.map = {k: "model.safetensors" for k in f.keys()}
        self._open = {}

    def element(self, name, row, col):
        from safetensors import safe_open

        file = self.map.get(name)
        if file is None:
            return None
        handle = self._open.get(file)
        if handle is None:
            handle = self._open[file] = safe_open(os.path.join(self.path, file), framework="pt")
        sl = handle.get_slice(name)
        return sl[row:row + 1, col:col + 1].reshape(-1)[0].to("cpu")

    def close(self):
        for h in self._open.values():
            try:
                h.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
        self._open = {}


def read_layer(name, weight, cfg, ckpt):
    """(mismatch, reason) for one loaded weight: mismatch is "" when every sampled element landed where the plan puts
    it, the first one that did not otherwise; None with a reason when the weight could not be compared."""
    if weight is None or weight.ndim != 2:
        return None, "not a matrix"
    parts = name.rsplit(".", 2)
    suffix = ".".join(parts[-2:]) if len(parts) >= 2 else name
    prefix = name[: len(name) - len(suffix)]
    plan = fusion_plan(suffix, cfg, weight.shape[0])
    if plan is None:
        if name.endswith("embed_tokens") or name.endswith("lm_head"):
            plan = [(suffix + ".weight", weight.shape[0])]
        else:
            return None, "no declared source mapping"
    total = sum(rows for _, rows in plan)
    if total != weight.shape[0]:   # with tensor parallelism 1 (the only case compared) this is a wrong fusion
        return (f"{name}: the declared sources add up to {total} rows, the weight has {weight.shape[0]} (sharded or "
                f"fused differently than declared)"), None
    rows = [min(weight.shape[0] - 1, r * max(1, weight.shape[0] // SAMPLE_ROWS)) for r in range(SAMPLE_ROWS)]
    cols = [min(weight.shape[1] - 1, c * max(1, weight.shape[1] // SAMPLE_COLS)) for c in range(SAMPLE_COLS)]
    for r in rows:
        offset, source, local = 0, None, None
        for src, n in plan:
            if r < offset + n:
                source, local = src, r - offset
                break
            offset += n
        for c in cols:
            want = ckpt.element(prefix + source, local, c)
            if want is None:
                return None, f"{prefix + source} is not in the checkpoint"
            if want.dtype != weight.dtype:
                return None, f"dtype changed at load ({want.dtype} -> {weight.dtype})"
            got = weight[r, c].to("cpu")
            if want.item() != got.item():
                return (f"{name} row {r} (from {source} row {local}) column {c}: {got.item():.6g}, the checkpoint "
                        f"holds {want.item():.6g}"), None
    return "", None


def read_choice(model, model_config):
    """What the loader took, weight by weight: (weights compared, [the weights whose sampled elements did not land
    where the plan puts them, with the first such element], {reason: number of weights not compared})."""
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    cfg = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
    ckpt = Checkpoint(model_config.model)
    compared, left, skipped = 0, [], {}
    if not ckpt.map:
        return 0, [], {"no safetensors checkpoint to compare with": 1}
    try:
        for name, module in model.named_modules():
            if not isinstance(getattr(module, "quant_method", None), QuantizeMethodBase):
                continue
            mismatch, why = read_layer(name, getattr(module, "weight", None), cfg, ckpt)
            if why is not None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            compared += 1
            if mismatch:
                left.append(mismatch)
    finally:
        ckpt.close()
    return compared, left, skipped


def install():
    """Wrap process_weights_after_loading and compare the weights with the files before anything repacks them."""
    from vllm.model_executor.model_loader import utils as loader_utils

    orig = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        if core.mode() in ("load", "debug"):
            def decide():
                policy = policies.current()
                compared, left, skipped = read_choice(model, model_config)
                decisions = load.weights_taken(engine, str(model_config.model), compared, left, policy)
                if skipped:
                    why = "; ".join(f"{n} weight(s): {w}" for w, n in sorted(skipped.items()))
                    decisions.append(load.cannot_check(f"load:{engine}.weights", f"{engine}.loader", "Coverage",
                                                       why, policy))
                load.enforce(decisions)

            load.safely(f"load:{engine}.weights", f"{engine}.loader", "Coverage", decide)
        return orig(model, model_config, target_device, *a, **kw)

    loader_utils.process_weights_after_loading = wrapped
    return 1
