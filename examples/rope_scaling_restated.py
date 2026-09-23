"""A model's own rope_scaling, given again after the config is built — with and without entail.

This is what an engine's launch-time override does (vLLM --hf-overrides, SGLang --json-model-override-args).
No download, no GPU: only transformers 5 is needed.

    python examples/rope_scaling_restated.py              # transformers alone: rope_theta is dropped
    ENTAIL=load python examples/rope_scaling_restated.py  # with entail installed: rope_theta is kept
"""
from transformers import LlamaConfig

LLAMA_3_2 = {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
             "original_max_position_embeddings": 8192}

cfg = LlamaConfig(rope_theta=500000.0, rope_scaling=dict(LLAMA_3_2),  # as read from the checkpoint
                  max_position_embeddings=131072)
print("from the checkpoint:   ", cfg.rope_parameters)
cfg.rope_scaling = dict(LLAMA_3_2)  # the same values, given again at launch
print("after restating it:    ", cfg.rope_parameters)
theta = cfg.rope_parameters.get("rope_theta")
print("rope_theta kept:" if theta else "rope_theta DROPPED - the engine will fall back to its default (10000 in vLLM):",
      theta)
