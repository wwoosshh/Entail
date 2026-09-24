"""What a consumer took from what it was given. Nothing given may be dropped without a word (MAPPING).

The value of a Coverage fact, and the count that makes one; the verdicts are the core's (load.config_keys,
load.weights_taken, load.lora - M6.2 removed the old check here, which raised by itself). The same count serves every
kind of input that a consumer can quietly ignore:
  - the modules of a LoRA that have no counterpart in the model (ComfyUI and diffusers skip them with a log line
    and the run succeeds; measured: an Anima LoRA in an SDXL workflow changed the image by 0.8/255),
  - the keys of a config that the consumer does not recognise (transformers keeps an unknown `rope_scale` as a
    plain attribute and runs with the default RoPE; rolebench case 15).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Coverage:
    total: int
    taken: int
    left: tuple  # what was given and not taken, sorted

    @property
    def none_taken(self):
        return self.total > 0 and self.taken == 0

    @property
    def all_taken(self):
        return self.taken == self.total

    def __str__(self):
        """As the dataclass shows it, with a long `left` cut after five names (a LoRA can leave hundreds, M6.1)."""
        left = self.left if len(self.left) <= 5 else self.left[:5] + (f"... and {len(self.left) - 5} more",)
        return f"Coverage(total={self.total}, taken={self.taken}, left={left!r})"


def count(given, taken):
    """Coverage of `given` by `taken` (both iterables of names)."""
    given, taken = set(given), set(taken)
    return Coverage(len(given), len(given & taken), tuple(sorted(given - taken)))
