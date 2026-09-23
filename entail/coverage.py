"""What a consumer took from what it was given. Nothing given may be dropped without a word (MAPPING).

The same count serves every kind of input that a consumer can quietly ignore:
  - the modules of a LoRA that have no counterpart in the model (ComfyUI and diffusers skip them with a log line
    and the run succeeds; measured: an Anima LoRA in an SDXL workflow changed the image by 0.8/255),
  - the keys of a config that the consumer does not recognise (transformers keeps an unknown `rope_scale` as a
    plain attribute and runs with the default RoPE; rolebench case 15).
"""
from dataclasses import dataclass

from . import core


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


def check(cov, where, what, stop_when="none"):
    """Say what was left out. stop_when="none": stop only when nothing was taken (the input would change nothing and
    there is nothing to convert); stop_when="any": stop when anything was left out (a config key that is not
    recognised). A partial take is one line and the run goes on. Returns the kind of outcome."""
    if cov.total == 0 or cov.all_taken:
        return "all"
    stop = cov.none_taken if stop_when == "none" else True
    shown = ", ".join(cov.left[:5]) + (f" and {len(cov.left) - 5} more" if len(cov.left) > 5 else "")
    if stop:
        raise core.RoleError(f"{where}: {what}: {len(cov.left)} of {cov.total} were not taken ({shown})")
    print(f"[entail] {where}: {what}: {len(cov.left)} of {cov.total} have no counterpart and are left out ({shown})",
          flush=True)
    return "partial"
