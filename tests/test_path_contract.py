"""Tests for the path contract (ROADMAP M19 L3.3c; path_contract.py): the same request along two of the engine's own
paths - decode against a fresh prefill, alone against batched, cold against a prefix-cache hit - and the vLLM adapter
on a stand-in engine (no vLLM: a stand-in `vllm` module with SamplingParams, and an LLM whose paths can be made to
disagree). Pure Python. Run: python tests/test_path_contract.py"""
import io
import os
import sys
import types
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from entail import core, load, path_contract as pc  # noqa: E402
from entail.contracts import RULES, Verdict  # noqa: E402
from entail.facts import PathAgreement  # noqa: E402


def decided(fn):
    n = len(load.LEDGER.decisions)
    was = core.mode()
    core.set_mode("load")
    try:
        with redirect_stdout(io.StringIO()):
            out = fn()
    finally:
        core.set_mode(was)
    return out, load.LEDGER.decisions[n:]


def run(tokens, confident=3.5, shift=None):
    """A greedy record: each step's chosen token ahead of the runner-up by `confident` in log-probability; `shift`
    {step: probability move} lowers the chosen token's log-probability at those steps."""
    import math

    lps = []
    for i, t in enumerate(tokens):
        top = -0.03
        if shift and i in shift:
            top = math.log(max(1e-9, math.exp(top) - shift[i]))
        lps.append({str(t): top, str(t + 1000): top - confident})
    return {"tokens": list(tokens), "logprobs": lps, "prompt_tokens": [1, 2, 3]}


def test_the_fact_checks_its_fields():
    f = PathAgreement(paths="alone_batched", probes=3, flip_margin=0.0, pdrift=0.01)
    assert f.paths == "alone_batched"
    for bad in (dict(paths="batch", probes=1, flip_margin=0.0, pdrift=0.0),
                dict(paths="cold_cache", probes=-1, flip_margin=0.0, pdrift=0.0),
                dict(paths="cold_cache", probes=1, flip_margin=-1.0, pdrift=0.0)):
        try:
            PathAgreement(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_numeric_noise_passes_and_lost_meaning_does_not():
    a = {"p": run([5, 6, 7, 8])}
    same = pc.verdict(pc.pair(a, {"p": run([5, 6, 7, 8])}))
    assert not same["differs"] and same["margin"] == 0.0 and same["pdrift"] == 0.0
    tie = pc.verdict(pc.pair({"p": run([5, 6, 7, 8], confident=0.4)}, {"p": run([5, 6, 9, 8])}))
    assert not tie["differs"] and abs(tie["margin"] - 0.4) < 1e-9, "a near-tie that flips is numerics"
    small = pc.verdict(pc.pair(a, {"p": run([5, 6, 7, 8], shift={1: 0.1})}))
    assert not small["differs"] and 0.09 < small["pdrift"] < 0.11
    flip = pc.verdict(pc.pair(a, {"p": run([5, 6, 9, 8])}))
    assert flip["differs"] and flip["probe"] == "p" and "generated token 2 changed" in flip["why"][0], flip
    moved = pc.verdict(pc.pair(a, {"p": run([5, 6, 7, 8], shift={2: 0.5})}))
    assert moved["differs"] and "probability moved" in moved["why"][0], moved


def test_decode_is_held_to_a_fresh_prefill_of_its_own_tokens():
    gen = {"p": run([5, 6, 7, 8])}
    forced = {"p": {"steps": [dict(s) for s in gen["p"]["logprobs"]]}}
    assert not pc.verdict(pc.decode_vs_prefill(gen, forced))["differs"]
    forced["p"]["steps"][2] = {"7": -4.0, "4242": -0.02}          # the prefill is confident of another token there
    v = pc.verdict(pc.decode_vs_prefill(gen, forced))
    assert v["differs"] and "changed a confident prediction" in v["why"][0], v


def test_check_records_pass_and_broken():
    B = "start:test.paths"
    pc.reset(B)
    ok = pc.verdict(pc.pair({"p": run([1, 2])}, {"p": run([1, 2])}))
    _, ds = decided(lambda: pc.check(B, "test.engine", "alone_batched", ok, "test"))
    assert ds[0].verdict is Verdict.PASS and ds[0].name == "PathAgreement" and "agree" in ds[0].note
    bad = pc.verdict(pc.pair({"p": run([1, 2])}, {"p": run([1, 3])}))
    _, ds = decided(lambda: pc.check(B, "test.engine", "decode_prefill", bad, "test"))
    assert ds[0].verdict is Verdict.BROKEN and ds[0].rule == RULES["paths_disagree"] and not ds[0].blocking, ds
    assert "disagree on probe 'p'" in ds[0].note and ds[0].chosen.value.flip_margin > 1.0
    assert pc.stats(B)["broken"] == 1 and pc.stats(B)["checks"] == 2


# --- the vLLM adapter on a stand-in engine ----------------------------------------------------------------------

class SamplingParams:
    def __init__(self, temperature=1.0, max_tokens=16, ignore_eos=False, logprobs=None, prompt_logprobs=None,
                 skip_reading_prefix_cache=False):
        self.max_tokens, self.logprobs, self.prompt_logprobs = max_tokens, logprobs, prompt_logprobs
        self.skip_reading_prefix_cache = skip_reading_prefix_cache


def _next(ctx):
    return (sum(ctx) * 31 + len(ctx)) % 97


def _dist(ctx, chosen=None):
    t = _next(ctx) if chosen is None else chosen
    return {t: SimpleNamespace(logprob=-0.03), (t + 1) % 97 + 100: SimpleNamespace(logprob=-3.5)}


class FakeLLM:
    """Greedy: the next token is a function of the context, confidently. `bug` makes one path say otherwise:
    "batched" (step 2 of a batched request), "decode" (the decode path at step 3), "cache" (a request that read the
    prefix cache, at step 1)."""

    def __init__(self, bug=None, caching=True, runner="generate", max_len=4096):
        self.bug, self.warm = bug, set()
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(model="fake/model", runner_type=runner, max_model_len=max_len),
            vllm_config=SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=caching)))
        self.calls = 0

    def reset_prefix_cache(self):
        self.warm.clear()

    def generate(self, prompts, sp, use_tqdm=False):
        self.calls += 1
        outs = []
        warm = set(self.warm)              # blocks computed in this batch are not read by it (as an engine's step)
        for p in prompts:
            ids = p["prompt_token_ids"] if isinstance(p, dict) else [ord(c) % 90 + 3 for c in p[:60]]
            read = not sp.skip_reading_prefix_cache and tuple(ids[:30]) in warm
            self.warm.add(tuple(ids[:30]))
            toks, lps, ctx = [], [], list(ids)
            for i in range(sp.max_tokens):
                t = _next(ctx)
                if (self.bug == "batched" and len(prompts) > 1 and i == 2) or (self.bug == "decode" and i == 3) \
                        or (self.bug == "cache" and read and i == 1):
                    t = (t + 7) % 97
                toks.append(t)
                lps.append(_dist(ctx, t))
                ctx.append(t)
            plp = None
            if sp.prompt_logprobs:
                plp = [None] + [_dist(ids[:i]) for i in range(1, len(ids))]
            outs.append(SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=plp,
                                        outputs=[SimpleNamespace(token_ids=toks, logprobs=lps)]))
        return outs


def _with_fake_vllm(fn):
    was = sys.modules.get("vllm")
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm"].SamplingParams = SamplingParams
    try:
        return fn()
    finally:
        if was is None:
            del sys.modules["vllm"]
        else:
            sys.modules["vllm"] = was


def test_a_healthy_engine_passes_all_three_pairs():
    from entail.adapters import vllm_paths as vp

    vp.reset()
    llm = FakeLLM()
    out, ds = decided(lambda: _with_fake_vllm(lambda: vp.decide(llm)))
    assert [d.verdict for d in ds] == [Verdict.PASS] * 3, ds
    assert [d.chosen.value.paths for d in ds] == ["decode_prefill", "alone_batched", "cold_cache"]
    assert not llm.warm, "the prefix cache is reset after the probes"


def test_each_disagreeing_path_is_named():
    from entail.adapters import vllm_paths as vp

    for bug, paths in (("batched", "alone_batched"), ("decode", "decode_prefill"), ("cache", "cold_cache")):
        vp.reset()
        out, ds = decided(lambda: _with_fake_vllm(lambda: vp.decide(FakeLLM(bug))))
        broken = [d for d in ds if d.verdict is Verdict.BROKEN]
        assert [d.chosen.value.paths for d in broken] == [paths], (bug, ds)
        assert broken[0].rule == RULES["paths_disagree"]


def test_what_is_not_compared_is_said_once_or_left_out():
    from entail.adapters import vllm_paths as vp

    vp.reset()
    out, ds = decided(lambda: _with_fake_vllm(lambda: vp.decide(FakeLLM(runner="pooling"))))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "not a generating model" in ds[0].note, ds
    out, ds = decided(lambda: _with_fake_vllm(lambda: vp.decide(FakeLLM(max_len=256))))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "shorter than the probes" in ds[0].note, ds
    out, ds = decided(lambda: _with_fake_vllm(lambda: vp.decide(FakeLLM(caching=False))))
    assert [d.chosen.value.paths for d in ds] == ["decode_prefill", "alone_batched"], "no cache: nothing to compare"

    class Refusing(FakeLLM):
        def generate(self, *a, **k):
            raise ValueError("prompt_logprobs is not supported")

    out, ds = decided(lambda: _with_fake_vllm(lambda: vp.decide(Refusing())))
    assert len(ds) == 1 and ds[0].verdict is Verdict.UNKNOWN and "refused a probe" in ds[0].note, ds


def test_an_older_vllm_without_the_cache_skip_resets_the_cache_before_cold_requests():
    from entail.adapters import vllm_paths as vp

    class OldParams(SamplingParams):
        def __init__(self, skip_reading_prefix_cache=None, **kw):
            if skip_reading_prefix_cache is not None:
                raise TypeError("unexpected keyword argument 'skip_reading_prefix_cache'")
            super().__init__(**kw)

    vp.reset()
    was = sys.modules.get("vllm")
    sys.modules["vllm"] = types.ModuleType("vllm")
    sys.modules["vllm"].SamplingParams = OldParams
    try:
        out, ds = decided(lambda: vp.decide(FakeLLM("cache")))
    finally:
        if was is None:
            del sys.modules["vllm"]
        else:
            sys.modules["vllm"] = was
    assert [d.verdict for d in ds][:2] == [Verdict.PASS, Verdict.PASS], "cold requests stay cold with the reset"
    assert ds[2].verdict is Verdict.BROKEN and ds[2].chosen.value.paths == "cold_cache", ds


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    core.set_mode("off")
