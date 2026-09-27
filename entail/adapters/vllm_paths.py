"""Adapter v2 for vLLM: the engine's own paths held against each other on probe requests, once, as soon as the
engine is up (M19 L3.3c; path_contract.py holds the rule).

  hook         vllm.entrypoints.llm.LLM.__init__: the offline engine is built. The probes run before the caller's
               first request, through the engine's public generate(), so they take the paths the engine will take
               (its CUDA graphs, its prefix cache, its speculative decoding, its batching).
  read_choice  greedy generations of PROBE_TOKENS tokens with their top log-probabilities, along three pairs:
                 decode_prefill  each probe alone (not reading the prefix cache) against the same tokens fed back in
                                 one fresh prefill (the prompt log-probabilities of the generated positions)
                 alone_batched   each probe alone against all probes in one batch
                 cold_cache      the target probe once more, reading the prefix cache the cold runs wrote (a hit on
                                 its own blocks; the other story shares its prefix), against the target alone
               The prefix cache is reset afterwards, so the caller's requests find nothing the probes left. Eight
               requests in all, PROBE_TOKENS decode steps each for five of them (M19 L3.3c measured the cost).
  handles      none: a running engine cannot be given another path; a disagreement is reported (broken), and a stop
               policy refuses to serve.
Not compared, said once as unknown: a model that does not generate (pooling), a context shorter than the probes,
a pair whose requests the engine refuses; the cache pair when prefix caching is off (nothing to compare, not said).
On a vLLM without SamplingParams.skip_reading_prefix_cache the prefix cache is reset before each cold request.
"""
import os

from .. import core, path_contract
from .base import Hook

engine = "vllm"
versions = "0.30.0"
BOUNDARY = "start:vllm.paths"
CONSUMER = "vllm.engine"
_ORIG = {}
_DONE = set()          # id(LLM) probed in this process


def hooks():
    return [Hook("vllm.entrypoints.llm.LLM.__init__", "start")]


def handles():
    return {}


def _lp(d):
    return {str(k): float(v.logprob) for k, v in (d or {}).items()}


def _record(out):
    o = out.outputs[0]
    return {"tokens": [int(t) for t in o.token_ids], "logprobs": [_lp(d) for d in (o.logprobs or [])],
            "prompt_tokens": [int(t) for t in (out.prompt_token_ids or [])]}


def read_choice(llm):
    """The three pairs' records: {"alone", "forced", "batched", "cache"} (cache None when prefix caching is off)."""
    from vllm import SamplingParams

    try:
        SamplingParams(skip_reading_prefix_cache=True)
        skippable = True
    except TypeError:
        skippable = False

    def params(read_cache=False, **kw):
        base = dict(temperature=0.0, max_tokens=path_contract.PROBE_TOKENS, ignore_eos=True,
                    logprobs=path_contract.TOP)
        base.update(kw)
        if skippable:
            base["skip_reading_prefix_cache"] = not read_cache
        return SamplingParams(**base)

    def cold():
        if not skippable:
            llm.reset_prefix_cache()

    def gen(prompts, sp):
        return llm.generate(prompts, sp, use_tqdm=False)

    alone = {}
    for p in path_contract.PROBES:
        cold()
        alone[p["id"]] = _record(gen([p["text"]], params())[0])
    forced = {}
    for pid, a in alone.items():
        cold()
        full = a["prompt_tokens"] + a["tokens"]
        o = gen([{"prompt_token_ids": full}], params(max_tokens=1, prompt_logprobs=path_contract.TOP))[0]
        n0 = len(a["prompt_tokens"])
        forced[pid] = {"steps": [_lp(o.prompt_logprobs[n0 + i]) for i in range(len(a["tokens"]))]}
    cold()
    outs = gen([p["text"] for p in path_contract.PROBES], params())
    batched = {p["id"]: _record(o) for p, o in zip(path_contract.PROBES, outs)}
    cache = None
    if _prefix_caching(llm):
        text = {p["id"]: p["text"] for p in path_contract.PROBES}[path_contract.TARGET]
        cache = {path_contract.TARGET: _record(gen([text], params(read_cache=True))[0])}
    llm.reset_prefix_cache()
    return {"alone": alone, "forced": forced, "batched": batched, "cache": cache}


def _prefix_caching(llm) -> bool:
    try:
        return bool(llm.llm_engine.vllm_config.cache_config.enable_prefix_caching)
    except Exception:  # noqa: BLE001 - not known: the cache pair is left out
        return False


def _max_len(llm):
    try:
        return int(llm.llm_engine.model_config.max_model_len)
    except Exception:  # noqa: BLE001
        return None


def decide(llm) -> list:
    """Run the probes and decide the three pairs (path_contract). Returns the decisions."""
    from .. import load

    pc = path_contract
    model = getattr(getattr(getattr(llm, "llm_engine", None), "model_config", None), "model", "the model")

    def unknown(why):
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "PathAgreement", why)])
        return []

    try:
        runner = getattr(llm.llm_engine.model_config, "runner_type", "generate")
    except Exception:  # noqa: BLE001
        runner = "generate"
    if runner not in ("generate", None):
        return unknown(f"{model} runs as {runner!r}, not a generating model: its paths are not compared")
    limit = _max_len(llm)
    if limit is not None and limit < 400:
        return unknown(f"the context of {model} ({limit} tokens) is shorter than the probes: its paths are not "
                       f"compared")
    try:
        recs = read_choice(llm)
    except Exception as e:  # noqa: BLE001 - a probe the engine refuses is not the engine's fault here
        return unknown(f"the engine refused a probe request ({type(e).__name__}: {str(e)[:160]}): its paths are not "
                       f"compared")
    where = (f"vLLM on {model}, {len(path_contract.PROBES)} probe requests of {path_contract.PROBE_TOKENS} "
             f"greedy tokens at start-up")
    out = []
    out += pc.check(BOUNDARY, CONSUMER, "decode_prefill",
                    pc.verdict(pc.decode_vs_prefill(recs["alone"], recs["forced"])), where)
    out += pc.check(BOUNDARY, CONSUMER, "alone_batched", pc.verdict(pc.pair(recs["alone"], recs["batched"])), where)
    if recs["cache"] is not None:
        base = {path_contract.TARGET: recs["alone"][path_contract.TARGET]}
        out += pc.check(BOUNDARY, CONSUMER, "cold_cache", pc.verdict(pc.pair(base, recs["cache"])), where)
    # the selective safe path (product track P3): which pairs disagreed, for safe_mode to move on
    from .. import safe_mode

    disagree = [d.chosen.value.paths for d in out if d.verdict.value in ("broken", "refused") and d.chosen is not None]
    load.enforce(safe_mode.after_self_check("vllm", disagree, "start:vllm.safe_mode", "vllm.engine_args", where))
    return out


def install():
    try:
        from vllm.entrypoints.llm import LLM
    except ImportError:
        return 0
    if "init" in _ORIG:
        return 0
    orig = _ORIG["init"] = LLM.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if core.mode() in ("load", "debug") and id(self) not in _DONE and not os.environ.get("ENTAIL_NO_PATHS"):
            _DONE.add(id(self))
            from .. import load

            load.safely(BOUNDARY, CONSUMER, "PathAgreement", lambda: decide(self))

    LLM.__init__ = __init__
    return 1


def uninstall():
    if "init" in _ORIG:
        from vllm.entrypoints.llm import LLM

        LLM.__init__ = _ORIG.pop("init")
        return 1
    return 0


def stats():
    return path_contract.stats(BOUNDARY)


def reset():
    _DONE.clear()
    path_contract.reset(BOUNDARY)
