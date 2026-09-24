"""frontend: a role-typed front end for new code - layer B (LIBRARY_DESIGN.md 4.13; ROADMAP M8).

What the library does at the boundaries of engines it does not own, this does for code written with it. Every value
has a type that says what it is - its named dims, dtype, kind (a query, a key, a value, the model's hidden state) and
the facts it carries (storage format, reduction state, the frame of its positions) - and every operation takes its
arguments by keyword, the keyword being the role. A program is traced once from the types of its inputs, without
data: every role, format, frame, range and property is checked then, and a mismatch is refused before anything runs
(trace time: where a compiled language checks at compile time). What runs afterwards is a plain function over
tensors with no checks left in it but what only the data can say (dtype and fixed sizes, once, when bound).

Grown from phase0/week4/rolec.py - the Korean-named prototype that turned eight silent attention mistakes into
refusals or impossibilities (WEEK4_NOTES.md 7.1) - in English, dividing the work as RESEARCH_PLAN.md 5.3 does: role
markers say direction (into=, src=), conversion target (to=) and accompaniment (share=); what a value is - its
format, reduction state, position frame, valid range, model properties - is its type.

    from entail import frontend as fe
    from entail.facts import Positions, Rotary

    def step(*, q, keys, values, positions, until):
        q = fe.rope(x=q, positions=positions, rotary=Rotary("default", 1e6))
        return fe.attend(query=q, keys=keys, values=values, until=until, share=4)

    program = fe.trace(step, {"attention": "triton"}, q=fe.T(("batch", "tokens", "heads", "head_dim"), ...), ...)
    out = program(q=..., keys=..., values=..., positions=..., until=...)     # plain tensors, no checks

Operations: embed, rms_norm, linear, split_features, split_heads, merge_heads, rope, write, attend, add, swiglu,
last, argmax, all_reduce, to_absolute, last_key, advance, copy, dequantize, reorder (ops.py). Attention lowerings: "torch" (SDPA, or float32
attention for a softcap), "flex" (FlexAttention with a block mask made by arithmetic), "triton" (the hand decode
kernel, kernels.py).
"""
from ..core import RoleError
from .graph import Program, T, trace
from .ops import (ATTENTION, ATTENTION_HONOURS, LINEAR_READS, REORDERS, ROPE_TYPES, add, advance, all_reduce, argmax,
                  attend, copy, dequantize, embed, last, last_key, linear, merge_heads, reads, reorder, rms_norm,
                  rope, split_features, split_heads, swiglu, to_absolute, write)

__all__ = ["RoleError", "Program", "T", "trace", "ATTENTION", "ATTENTION_HONOURS", "LINEAR_READS", "REORDERS",
           "ROPE_TYPES", "add", "advance", "all_reduce", "argmax", "attend", "copy", "dequantize", "embed", "last",
           "last_key", "linear", "merge_heads", "reads", "reorder", "rms_norm", "rope", "split_features",
           "split_heads", "swiglu", "to_absolute", "write"]
