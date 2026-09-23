"""Scaffolding, not a check: plant a known layout defect in a real vLLM run so the check can be tested against it.

Switched on with ENTAIL_SEED:
  transpose_square - transpose every square linear weight after the repack. Shapes still line up, so nothing
                     crashes; the answer is simply wrong. This is the silent case, and it is also the case a
                     shape-based check cannot see, which is the point of the experiment.
  transpose_all    - transpose every linear weight (loud: the shapes stop matching)
  corrupt_at_load  - roll one tensor by a row while it is being read from the checkpoint file, before the model
                     ever sees it. Nothing downstream can notice: there is no earlier state to compare with.
                     Only adapters/vllm_source.py, which reads the file back, can catch this one.
  strided_weights  - replace every linear weight with a strided view holding the same values in the same
                     shape. Only the strides differ, so only a stride declaration can see it.
  roll_output      - roll every linear weight by one row along the output axis. The shape, the dtype and the
                     scale all stay as declared, so a check built on shapes has nothing to say. This is what a
                     wrong packing order looks like, and it is the silent case.

Install it before the layout check so the check sees the planted defect (sitecustomize orders the list).
"""
import os

COUNTS = {"seen": 0, "changed": 0, "square": 0, "at_load": 0}
TARGET_AT_LOAD = os.environ.get("ENTAIL_SEED_TENSOR", "model.layers.0.self_attn.q_proj.weight")


def install_blocks():
    """ENTAIL_SEED=short_blocks: the block table reports one block fewer than the request holds."""
    if os.environ.get("ENTAIL_SEED") != "short_blocks":
        return 0
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    orig = KVCacheManager.get_block_ids

    def wrapped(self, request_id):
        return tuple(list(ids)[:-1] if len(ids) > 1 else list(ids) for ids in orig(self, request_id))

    KVCacheManager.get_block_ids = wrapped
    print("[entail-seed] short_blocks: the block table now reports one block fewer", flush=True)
    return 1


def install_loader():
    """Corrupt one tensor as it streams out of the checkpoint file (ENTAIL_SEED=corrupt_at_load)."""
    import torch
    from vllm.model_executor.model_loader import weight_utils

    if os.environ.get("ENTAIL_SEED") != "corrupt_at_load":
        return 0

    def wrap(fn):
        def wrapped(*a, **kw):
            for name, tensor in fn(*a, **kw):
                if name == TARGET_AT_LOAD and tensor.ndim == 2:
                    tensor = torch.roll(tensor, 1, 0).contiguous()
                    COUNTS["at_load"] += 1
                    print(f"[entail-seed] corrupt_at_load: rolled {name} by one row", flush=True)
                yield name, tensor

        return wrapped

    for fname in ("safetensors_weights_iterator", "multi_thread_safetensors_weights_iterator"):
        if hasattr(weight_utils, fname):
            setattr(weight_utils, fname, wrap(getattr(weight_utils, fname)))
    return 1


def install():
    import torch
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
    from vllm.model_executor.model_loader import utils as loader_utils

    mode = os.environ.get("ENTAIL_SEED")
    orig = loader_utils.process_weights_after_loading

    def wrapped(model, model_config, target_device, *a, **kw):
        out = orig(model, model_config, target_device, *a, **kw)
        for name, module in model.named_modules():
            qm = getattr(module, "quant_method", None)
            if not isinstance(qm, QuantizeMethodBase):
                continue
            w = getattr(module, "weight", None)
            if w is None or w.ndim != 2 or type(qm).__name__ == "UnquantizedEmbeddingMethod":
                continue
            COUNTS["seen"] += 1
            square = w.shape[0] == w.shape[1]
            COUNTS["square"] += int(square)
            if mode == "transpose_all" or (mode == "transpose_square" and square):
                with torch.no_grad():
                    module.weight = torch.nn.Parameter(w.data.t().contiguous(), requires_grad=False)
                COUNTS["changed"] += 1
            elif mode == "strided_weights":
                # same shape, same dtype, same values - only the strides differ. Nothing but a stride check
                # can see it, and that is the point of the probe.
                with torch.no_grad():
                    packed = w.data.t().contiguous()
                    module.weight = torch.nn.Parameter(packed.as_strided(tuple(w.shape),
                                                                        (1, w.shape[0])), requires_grad=False)
                COUNTS["changed"] += 1
            elif mode == "roll_output":
                with torch.no_grad():
                    module.weight = torch.nn.Parameter(torch.roll(w.data, 1, 0).contiguous(), requires_grad=False)
                COUNTS["changed"] += 1
        print(f"[entail-seed] {mode}: {COUNTS['seen']} linear weights seen, {COUNTS['square']} square, "
              f"{COUNTS['changed']} changed after post-processing, {COUNTS['at_load']} changed while loading",
              flush=True)
        return out

    loader_utils.process_weights_after_loading = wrapped
    return 1
