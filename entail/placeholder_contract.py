"""placeholder_contract: where an engine binds a multimodal item's placeholder against the markup the model
declares (LIBRARY_DESIGN.md 11, M18; ROADMAP M18.4; vllm#57740).

A vision-language model's prompt carries each image as a run of placeholder tokens that the chat template puts
inside the model's markup: the Qwen-VL family declares `vision_start_token_id` and `image_token_id` in its config,
and the template writes `<|vision_start|><|image_pad|><|vision_end|>`, which the processor expands into the image's
feature length. vLLM binds an image to the first run of placeholder ids it finds in the token ids. A user whose
text contains the literal `<|image_pad|>` gets that text tokenised to the placeholder id, and the image is bound
there - inside the user's words, where no markup declares an image - while the template's own slot keeps a single
raw pad token; the model's answer changes (vllm#57740). The rule is here, once:

  placeholder_outside_markup   an item's placeholder run is not preceded by the declared start-of-markup token

A model whose config declares no markup (no `vision_start_token_id`) decides nothing. Nothing here reads a
device; the adapter runs the check under load.safely, after the engine has placed the placeholders, and reports
(the prompt is the user's; no repair is offered).
"""
from typing import Dict, List, Optional, Tuple

from . import tally as _tally

RULE_NAMES = ("placeholder_outside_markup",)
MARKUP = (("image", "vision_start_token_id", "image_token_id"), ("video", "vision_start_token_id", "video_token_id"))


def _get(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def declared_markup(config) -> Dict[str, dict]:
    """The markup the model's config declares per modality: the id that starts an item's markup and the item's
    placeholder id (top level, else text_config). Empty when the config declares none."""
    out = {}
    for modality, start_key, token_key in MARKUP:
        for holder in (config, _get(config, "text_config")):
            if holder is None:
                continue
            start, token = _get(holder, start_key), _get(holder, token_key)
            if isinstance(start, int) and not isinstance(start, bool) and isinstance(token, int) and not isinstance(token, bool):
                out[modality] = {"start": start, "token": token, "where": f"config {start_key}={start}, {token_key}={token}"}
                break
    return out


def check(boundary: str, consumer: str, prompt_ids: List[int], placeholders: Dict[str, List[Tuple[int, int]]],
          markup: Dict[str, dict], where: str, policy=None, record: bool = True) -> list:
    """Decide each placeholder run (offset, length) per modality against the declared markup: the id just before
    the run must be the declared start token. Returns the non-pass decisions (passes are tallied)."""
    from . import load, policies
    from .contracts import RULES, Contract, Decision, Verdict, unrepaired
    from .facts import Certainty, Fact, Placeholder, Source

    policy = policy or policies.current()
    contract = Contract(boundary, consumer, ("Placeholder",))
    decisions, checked = [], 0
    for modality, runs in (placeholders or {}).items():
        m = markup.get(modality)
        if not m:
            continue
        tokens = {v["token"] for v in markup.values()}
        for offset, length in runs:
            prev: Optional[int] = int(prompt_ids[offset - 1]) if 0 < offset <= len(prompt_ids) else None
            if prev is None or prev in tokens:
                # the engine's own synthetic prompt (vLLM's memory profiling: placeholder runs back to back from
                # token 0, no template), not a request: nothing precedes the run, or another item's placeholder does
                continue
            checked += 1
            if prev == m["start"]:
                continue
            verdict, blocking = unrepaired(policy, "Placeholder")
            declared = Fact("Placeholder", Placeholder(modality=modality, offset=int(offset), length=int(length),
                                                       preceded_by=m["start"]),
                            Source("config", m["where"]), Certainty.DECLARED)
            held = Fact("Placeholder", Placeholder(modality=modality, offset=int(offset), length=int(length),
                                                   preceded_by=prev),
                        Source("engine", where), Certainty.VERIFIED)
            decisions.append(Decision(
                contract, "Placeholder", verdict, RULES["placeholder_outside_markup"], declared=declared, chosen=held,
                blocking=blocking,
                note=f"{where}: the {modality} placeholder bound at tokens {offset}..{offset + length - 1} is preceded "
                     f"by id {prev}, not by the declared start of the markup (id {m['start']}); the run came from the "
                     f"prompt's text, not from the template"))
    if record and checked:
        _tally.counts(boundary)["checks"] += 1
        if not decisions:
            _tally.passed(boundary, ["placeholder"])
        if any(d.blocking for d in decisions):
            _tally.refused(boundary)
        elif decisions:
            _tally.broken(boundary)
        _tally.tick(boundary)
        if decisions:
            load.enforce(decisions)
    return decisions


def stats(boundary: str) -> dict:
    return _tally.stats(boundary)


def reset(boundary: str) -> None:
    _tally.reset(boundary)
