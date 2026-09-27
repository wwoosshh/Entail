"""An image step checked with a workshop package's validator (entail product track P5): the picture must be the size
that was asked for. The validator `same_size` comes from the package entail-nodes-basics (workshop/basics), found by
name through the entry point group entail.nodes. No model, no GPU: `render` stands in for a pipeline.

    pip install ./workshop/basics
    ENTAIL=load python examples/custom_nodes/image_app.py            # the size holds: nothing broken
    ENTAIL=load python examples/custom_nodes/image_app.py --fault    # a size rounded to 64: broken at image.output
"""
import sys

from entail import nodes

FAULT = "--fault" in sys.argv


def render(width, height):
    """Stands in for a pipeline; returns the (width, height) of what it made."""
    if FAULT:
        return (width // 64 * 64, height // 64 * 64)
    return (width, height)


if __name__ == "__main__":
    if not any(p.get("attached") and "same_size" in p.get("validators", []) for p in nodes.packages()):
        print("the workshop package entail-nodes-basics is not attached: pip install ./workshop/basics "
              "(and run with ENTAIL=load)")
    for w, h in ((832, 1216), (1000, 750)):
        size = nodes.check("image.output", render(w, h), "same_size", expected=(w, h))
        print(f"asked {w}x{h} -> {size[0]}x{size[1]}")
