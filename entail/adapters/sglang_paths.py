"""Adapter v2 for SGLang: the engine's own paths held against each other on probe requests, once, as soon as the
offline engine is up (M19 L3.3c; path_contract.py holds the rule, vllm_paths.py is the same check for vLLM).

  hook         sglang.srt.entrypoints.engine.Engine.__init__: the offline engine is built; the probes run through its
               public generate() before the caller's first request.
  read_choice  greedy generations of PROBE_TOKENS tokens with their top log-probabilities, the radix cache flushed
               before each cold request: decode_prefill (each probe alone against the same tokens fed back in one
               fresh prefill), alone_batched (alone against all probes in one batch), cold_cache (the target once
               more, reading the radix cache the batch wrote, against the target alone). The cache is flushed
               afterwards.
  handles      none (a running engine cannot be given another path; a disagreement is reported).
Not compared, said once as unknown: an embedding model, a context shorter than the probes, a probe the engine refuses.
Off unless asked for (ENTAIL_PATHS=1; said once as unknown otherwise; M19 L4): SGLang runs no prefill while it starts,
so the probes would be the engine's first prefills, and a defect on that path kills the engine before the caller's
first request (principle 12). Measured: SGLang 0.5.20 with Phi-3.5-mini-instruct (head_dim 96) picks flashinfer, whose
state merge does not take head_dim 96; a 128-token prompt stops the scheduler, a longer one hits an illegal memory
access - with entail off as well, and the E2 harness's short prompts never reach it
(testbed/results/m19/l4/bisect_phi35/). vLLM's path check stays on: its start-up runs the model's forward (the
profile run) and its attention selector checks the head size.
"""
import os

from .. import core, path_contract
from .base import Hook

engine = "sglang"
versions = "0.5.20"
BOUNDARY = "start:sglang.paths"
CONSUMER = "sglang.engine"
_ORIG = {}
_DONE = set()


def hooks():
    return [Hook("sglang.srt.entrypoints.engine.Engine.__init__", "start")]


def asked():
    """The probes run on SGLang only when asked for (the module docstring says why)."""
    return os.environ.get("ENTAIL_PATHS", "").strip().lower() in ("1", "on", "yes", "true")


def handles():
    return {}


def _pairs(lst):
    return {str(int(t)): float(lp) for (lp, t, *_rest) in (lst or []) if lp is not None}


def _record(out, n0=None):
    """SGLang's meta_info as path_contract reads it; with `n0`, the prompt log-probabilities from position n0 on
    (the generated tokens fed back) as the forced steps."""
    mi = out["meta_info"]
    toks, steps = [], []
    tops = mi.get("output_top_logprobs") or []
    for i, (lp, t, *_r) in enumerate(mi.get("output_token_logprobs") or []):
        toks.append(int(t))
        d = _pairs(tops[i] if i < len(tops) else [])
        if lp is not None:
            d[str(int(t))] = float(lp)
        steps.append(d)
    rec = {"tokens": toks, "logprobs": steps, "prompt_tokens": []}
    if n0 is not None:
        itops = mi.get("input_top_logprobs") or []
        forced = []
        for i, (lp, t, *_r) in enumerate(mi.get("input_token_logprobs") or []):
            d = _pairs(itops[i] if i < len(itops) else [])
            if lp is not None:
                d[str(int(t))] = float(lp)
            forced.append(d)
        rec["forced"] = forced
    return rec


def read_choice(eng):
    sp = {"temperature": 0.0, "max_new_tokens": path_contract.PROBE_TOKENS, "ignore_eos": True}
    lp = {"return_logprob": True, "top_logprobs_num": path_contract.TOP}
    tok = eng.tokenizer_manager.tokenizer
    ids = {p["id"]: list(tok.encode(p["text"])) for p in path_contract.PROBES}
    alone = {}
    for p in path_contract.PROBES:
        eng.flush_cache()
        alone[p["id"]] = _record(eng.generate(input_ids=ids[p["id"]], sampling_params=sp, **lp))
        alone[p["id"]]["prompt_tokens"] = ids[p["id"]]
    forced = {}
    for pid, a in alone.items():
        eng.flush_cache()
        n0 = len(ids[pid])
        o = eng.generate(input_ids=ids[pid] + a["tokens"], sampling_params=dict(sp, max_new_tokens=1),
                         logprob_start_len=n0 - 1, **lp)
        r = _record(o, n0)
        # the input log-probabilities start at position n0-1; entry k scores token n0-1+k given the ones before it:
        # the generated token i is entry i+1
        forced[pid] = {"steps": [r["forced"][i + 1] if i + 1 < len(r["forced"]) else {}
                                 for i in range(len(a["tokens"]))]}
    eng.flush_cache()
    outs = eng.generate(input_ids=[ids[p["id"]] for p in path_contract.PROBES], sampling_params=sp, **lp)
    batched = {p["id"]: _record(o) for p, o in zip(path_contract.PROBES, outs)}
    target = path_contract.TARGET
    cache = {target: _record(eng.generate(input_ids=ids[target], sampling_params=sp, **lp))}
    eng.flush_cache()
    return {"alone": alone, "forced": forced, "batched": batched, "cache": cache}


def decide(eng) -> list:
    from .. import load

    pc = path_contract
    args = getattr(eng, "server_args", None)
    model = getattr(args, "model_path", "the model")

    def unknown(why):
        load.enforce([load.cannot_check(BOUNDARY, CONSUMER, "PathAgreement", why)])
        return []

    if getattr(args, "is_embedding", False):
        return unknown(f"{model} is an embedding model: its paths are not compared")
    limit = getattr(args, "context_length", None)
    if isinstance(limit, int) and limit < 400:
        return unknown(f"the context of {model} ({limit} tokens) is shorter than the probes: its paths are not "
                       f"compared")
    try:
        recs = read_choice(eng)
    except Exception as e:  # noqa: BLE001 - a probe the engine refuses is not the engine's fault here
        return unknown(f"the engine refused a probe request ({type(e).__name__}: {str(e)[:160]}): its paths are not "
                       f"compared")
    where = (f"SGLang on {model}, {len(path_contract.PROBES)} probe requests of {path_contract.PROBE_TOKENS} "
             f"greedy tokens at start-up")
    out = []
    out += pc.check(BOUNDARY, CONSUMER, "decode_prefill",
                    pc.verdict(pc.decode_vs_prefill(recs["alone"], recs["forced"])), where)
    out += pc.check(BOUNDARY, CONSUMER, "alone_batched", pc.verdict(pc.pair(recs["alone"], recs["batched"])), where)
    base = {pc.TARGET: recs["alone"][pc.TARGET]}
    out += pc.check(BOUNDARY, CONSUMER, "cold_cache", pc.verdict(pc.pair(base, recs["cache"])), where)
    return out


def install():
    try:
        from sglang.srt.entrypoints.engine import Engine
    except ImportError:
        return 0
    if "init" in _ORIG:
        return 0
    orig = _ORIG["init"] = Engine.__init__

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if core.mode() in ("load", "debug") and id(self) not in _DONE and not os.environ.get("ENTAIL_NO_PATHS"):
            _DONE.add(id(self))
            from .. import load

            if asked():
                load.safely(BOUNDARY, CONSUMER, "PathAgreement", lambda: decide(self))
            else:
                load.enforce([load.cannot_check(
                    BOUNDARY, CONSUMER, "PathAgreement",
                    "SGLang has run no prefill yet: the probe requests would be the engine's first, and a defect on "
                    "that path would stop it before your first request; set ENTAIL_PATHS=1 to compare its paths")])

    Engine.__init__ = __init__
    return 1


def uninstall():
    if "init" in _ORIG:
        from sglang.srt.entrypoints.engine import Engine

        Engine.__init__ = _ORIG.pop("init")
        return 1
    return 0


def stats():
    return path_contract.stats(BOUNDARY)


def reset():
    _DONE.clear()
    path_contract.reset(BOUNDARY)
