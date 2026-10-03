"""Tests for writeguard (ROADMAP M19 L5.4e, approach A): the live ranges of issued bytes - overlap, release, a range
replaced by a new issue of the same memory, release when the tensor is collected - with stand-ins for device tensors
(no GPU). The dispatch mode and the Triton launch check are measured on the GPU by the evaluation harness
(eval/block_fp8_guarantee, cases I1-I8). Run: python tests/test_writeguard.py
"""
import gc
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import writeguard as W  # noqa: E402


class Dev:
    """A device tensor as span() reads it: contiguous bytes [ptr, ptr + n * size)."""

    is_cuda = True

    def __init__(self, ptr, n, size=1, shape=None, stride=None):
        self.ptr, self.n, self.size = ptr, n, size
        self.shape = shape or (n,)
        self._stride = stride or (1,)

    def numel(self):
        return self.n

    def stride(self):
        return self._stride

    def data_ptr(self):
        return self.ptr

    def element_size(self):
        return self.size


def main():
    W.reset()
    W.ensure_mode = lambda: None             # no torch dispatch mode in this test
    a = Dev(1000, 100)                       # [1000, 1100)
    w = Dev(5000, 64, 4)                     # [5000, 5256)
    W.register(a, 1, "activation")
    W.register(w, 2, "weight")
    assert W.live() == 2
    assert W.overlapping(1050, 1051) == (1, "activation")
    assert W.overlapping(990, 1001) == (1, "activation")
    assert W.overlapping(1100, 1200) is None and W.overlapping(900, 1000) is None
    assert W.overlapping(5255, 6000) == (2, "weight")
    assert W.overlapping(0, 10 ** 9) is not None
    print("ok overlap: inside, across an edge, touching (no), the whole space")
    # a strided view: span covers first to last element
    v = Dev(5000, 4, 4, shape=(2, 2), stride=(32, 1))
    assert W.span(v) == (5000, 5000 + (32 + 1 + 1) * 4)
    print("ok span of a strided view:", W.span(v))
    W.release(1)
    assert W.overlapping(1050, 1051) is None and W.live() == 1
    print("ok release")
    # memory issued again while an old range is still listed (a buffer used again): the new issue replaces it
    b = Dev(4990, 20)                         # [4990, 5010) overlaps the weight's range
    W.register(b, 3, "activation")
    assert W.overlapping(5100, 5101) is None and W.overlapping(5000, 5001) == (3, "activation"), W._RANGES
    print("ok a new issue over an old range replaces it")
    # collected: released
    c = Dev(9000, 10)
    W.register(c, 4, "activation_scale")
    assert W.overlapping(9005, 9006) == (4, "activation_scale")
    del c
    gc.collect()
    assert W.overlapping(9005, 9006) is None
    print("ok a collected tensor's range is released")
    # refusals go through guarantee.refuse_write, which raises Refused (kind "write")
    from entail import guarantee as G

    os.environ["ENTAIL_GUARANTEE_RECORD"] = "off"
    try:
        W.check_launch("k", [b])
    except Exception as e:  # noqa: BLE001
        assert type(e).__name__ == "Refused" and e.kind == "write", e
    else:
        raise AssertionError("a launch over a live range was not refused")
    W.check_launch("k", [b], allowed=[b])     # the gate's own operand: allowed
    W.check_launch("k", [Dev(20000, 8)])      # elsewhere: allowed
    print("ok Triton launch check: refused over a live range, allowed for the gate's operand and elsewhere")
    assert G.stats().get("blocked_write", 0) >= 1
    W.reset()
    print("all writeguard tests passed")


if __name__ == "__main__":
    main()
