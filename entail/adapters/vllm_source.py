"""D-arm check, load side: does the weight in memory still hold what the checkpoint file holds?

adapters/vllm_layout.py checks the weight against the layout its kernel declares, and against a sample taken
before post-processing. Neither can see a weight that was already wrong when it arrived — read from the wrong
offset, fused in the wrong order, a shard swapped. For that the only reference is the file itself.

The awkward part is the reason this check does not exist upstream: to compare a fused weight with the
checkpoint, you have to know which source tensor each row came from, and that mapping lives inside the loader's
code rather than in any declaration. So it is declared here, per model family, and that declaration is the
point: FUSION below is the fact vLLM knows but never writes down.

Only bf16/fp16 weights with tensor parallelism 1 are compared; anything quantised or sharded on the way in is
reported as skipped rather than silently passed.
"""
import os

from .. import core
from ..core import RoleError

SAMPLE_ROWS, SAMPLE_COLS = 4, 4  # 16 elements per weight

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
            except Exception:
                pass
        self._open = {}


def check_layer(name, weight, cfg, ckpt):
    """Compare a few elements of one loaded weight with the checkpoint they came from."""
    if weight is None or weight.ndim != 2:
        return [], "not a matrix"
    parts = name.rsplit(".", 2)
    suffix = ".".join(parts[-2:]) if len(parts) >= 2 else name
    prefix = name[: len(name) - len(suffix)]
    plan = fusion_plan(suffix, cfg, weight.shape[0])
    if plan is None:
        if name.endswith("embed_tokens") or name.endswith("lm_head"):
            plan = [(suffix + ".weight", weight.shape[0])]
        else:
            return [], "no declared source mapping"
    total = sum(rows for _, rows in plan)
    if total != weight.shape[0]:
        return ([f"{name}: the declared sources add up to {total} rows, the weight has {weight.shape[0]} "
                 f"(sharded or fused differently than declared)"], "shape mismatch")
    rows = [min(weight.shape[0] - 1, r * max(1, weight.shape[0] // SAMPLE_ROWS)) for r in range(SAMPLE_ROWS)]
    cols = [min(weight.shape[1] - 1, c * max(1, weight.shape[1] // SAMPLE_COLS)) for c in range(SAMPLE_COLS)]
    out, compared = [], 0
    for r in rows:
        offset, source, local = 0, None, None
        for src, n in plan:
            if r < offset + n:
                source, local = src, r - offset
                break
            offset += n
        if source is None:
            continue
        for c in cols:
            want = ckpt.element(prefix + source, local, c)
            if want is None:
                return [], f"{prefix + source} is not in the checkpoint"
            if want.dtype != weight.dtype:
                return [], f"dtype changed at load ({want.dtype} -> {weight.dtype})"
            got = weight[r, c].to("cpu")
            compared += 1
            if want.item() != got.item():
                out.append(f"{name}: row {r} should come from {source} row {local}, but the value at column "
                           f"{c} is {got.item():.6g} where the checkpoint holds {want.item():.6g}")
                break
        if out:
            break
    return out, f"compared {compared}"


def install():
    """Wrap process_weights_after_loading and check the weights against the files before anything repacks them."""
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader import utils as loader_utils

    orig = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        if core.mode() in ("load", "debug"):
            import time

            t0 = time.perf_counter()
            cfg = getattr(model_config, "hf_text_config", None) or getattr(model_config, "hf_config", None)
            ckpt = Checkpoint(model_config.model)
            complaints, checked, skipped = [], 0, {}
            if ckpt.map:
                for name, module in model.named_modules():
                    qm = getattr(module, "quant_method", None)
                    if not isinstance(qm, QuantizeMethodBase):
                        continue
                    said, why = check_layer(name, getattr(module, "weight", None), cfg, ckpt)
                    complaints += said
                    if why.startswith("compared"):
                        checked += 1
                    else:
                        skipped[why] = skipped.get(why, 0) + 1
                ckpt.close()
            if os.environ.get("ENTAIL_VERBOSE"):
                print(f"[entail] source check: {checked} weights against the checkpoint, "
                      f"{len(complaints)} complaints, skipped {skipped}, "
                      f"{(time.perf_counter() - t0) * 1e3:.0f} ms", flush=True)
            if complaints:
                head = complaints[:5]
                more = f"\n  ... and {len(complaints) - len(head)} more" if len(complaints) > len(head) else ""
                raise RoleError("weights do not match the checkpoint they were loaded from:\n  "
                                + "\n  ".join(head) + more)
        return orig(model, model_config, target_device, *a, **kw)

    loader_utils.process_weights_after_loading = wrapped
    return 1
