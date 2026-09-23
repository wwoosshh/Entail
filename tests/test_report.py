"""Tests for the default policy since M5.4: what nothing repairs is reported and the run goes on (the researcher's
decision of 2026-09-24: errors are put out, the run is not stopped, so no failure a user sees is added, and the
problem is still caught exactly). One place per kind of boundary: the verdict function, the load contracts, the KV
container (per step: a repeat is counted, not recorded again), TIME, the request boundary, the vLLM server adapter on
stand-ins (the response is served, and optionally carries what broke), and `entail check`, which still fails like a
type checker in CI. The other test files check what stops, under the policy that stopped before M5.4.

Run: python tests/test_report.py
"""
import asyncio
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
from contextlib import redirect_stdout
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
for k in ("ENTAIL_ON_BROKEN", "ENTAIL_UNKNOWN", "ENTAIL_POLICY", "ENTAIL_FACT_POLICY", "ENTAIL_RESPONSE_NOTE"):
    os.environ.pop(k, None)   # the defaults are what is tested here
from entail import core, epochs, kv_contract, load, manifest, request_contract, tally  # noqa: E402
from entail.contracts import RULES, Contract, Verdict, decide  # noqa: E402
from entail.facts import Certainty, Fact, Layout, Source, Template  # noqa: E402
from entail.kv_contract import KvExtent  # noqa: E402

REAL_VLLM = importlib.util.find_spec("vllm") is not None


def quiet(fn, *a, **kw):
    """(what fn returned, what it printed); a RoleError here is a failure: nothing may stop by default."""
    out = io.StringIO()
    with redirect_stdout(out):
        got = fn(*a, **kw)
    return got, out.getvalue()


def ledger_since(n):
    return load.LEDGER.decisions[n:]


def mode(m):
    core.set_mode(m)
    core.set_policy("resolve")


# --- the verdict and the ledger -------------------------------------------------------------------------------

def test_a_mismatch_nothing_repairs_is_broken_printed_and_recorded_not_raised():
    mode("load")
    c = Contract("load:test.kernel", "test.kernel", ("Layout",), ("Layout",))
    declared = Fact("Layout", Layout("q8_0", packing="interleaved"), Source("file", "m#layout"), Certainty.DECLARED)
    used = Fact("Layout", Layout("q8_0", packing="split"), Source("engine", "kernel"), Certainty.DECLARED)
    path = os.path.join(tempfile.mkdtemp(), "record.jsonl")
    os.environ["ENTAIL_RECORD"] = path
    try:
        n = len(load.LEDGER.decisions)
        ds, printed = quiet(load.enforce, decide(c, {"Layout": declared}, {"Layout": used}))
    finally:
        os.environ.pop("ENTAIL_RECORD")
    [d] = ds
    assert (d.verdict, d.rule, d.blocking) == (Verdict.BROKEN, RULES["no_resolution"], False), d
    assert printed.startswith("[entail] broken at load:test.kernel") and "reported, not stopped" in printed, printed
    assert ledger_since(n) == [d] and load.LEDGER.broken()[-1] is d
    [line] = [json.loads(x) for x in open(path, encoding="utf-8")]
    assert (line["verdict"], line["blocking"]) == ("broken", False), line


def test_an_undeclared_meaning_changing_fact_is_reported_not_required():
    """M2's open question (15 of 17 image checkpoints would stop under `require`) is settled by the default: report."""
    mode("load")
    c = Contract("load:test.sampler", "test.sampler", ("Prediction",), ("Prediction",))
    [d], printed = quiet(load.enforce, decide(c, {}, {}))
    assert (d.verdict, d.blocking) == (Verdict.UNKNOWN, False) and "[entail] unknown at" in printed, d


# --- the KV container: per step, recorded once per cache or request, counted always --------------------------

B = "container:test.kv"


def test_a_request_short_of_slots_is_reported_once_and_counted_every_step():
    mode("load")
    kv_contract.reset()
    n = len(load.LEDGER.decisions)
    for _ in range(3):   # the same request, three steps, still one slot short
        quiet(kv_contract.check, B, "test.kv_cache", "request 7", KvExtent(held=7, needed=8))
    quiet(kv_contract.check, B, "test.kv_cache", "request 8", KvExtent(held=7, needed=8))
    made = ledger_since(n)
    assert [d.verdict for d in made] == [Verdict.BROKEN, Verdict.BROKEN], made
    assert [d.note.split(":")[0] for d in made] == ["request 7", "request 8"]
    s = kv_contract.stats(B)
    assert (s["checks"], s["broken"], s["refused"]) == (4, 4, 0), s


def test_a_cache_that_lost_a_token_is_reported_once_per_cache_not_per_layer():
    mode("load")
    kv_contract.reset()

    class Cache:   # an owner that takes weak references, as a real cache does
        pass

    a, b = Cache(), Cache()
    n = len(load.LEDGER.decisions)
    for owner in (a, b):
        for layer in range(4):
            kv_contract.grew(B, "test.kv_cache", owner, "test", layer, 8, 8, 1)   # held 8, given 1, still 8
    made = ledger_since(n)
    assert len(made) == 2 and {d.rule for d in made} == {RULES["kv_needed"]}, made   # one per cache
    assert kv_contract.stats(B)["broken"] == 8


def test_after_a_request_every_report_is_recorded():
    mode("load")
    kv_contract.reset()
    n = len(load.LEDGER.decisions)
    for _ in range(2):
        quiet(kv_contract.request, B, "test.kv_cache", "after request", {0: (7, None), 1: (8, None)}, 8)
    assert [d.verdict for d in ledger_since(n)] == [Verdict.BROKEN, Verdict.BROKEN]


# --- TIME -----------------------------------------------------------------------------------------------------

def test_a_stale_read_is_reported_and_the_read_goes_on():
    mode("load")
    epochs.reset()

    class Cache:
        pass

    cache = Cache()
    value = object.__new__(type("Mask", (), {}))
    epochs.live(value, cache, "length")
    epochs.advance(cache, "length")
    n = len(load.LEDGER.decisions)
    for _ in range(3):
        quiet(epochs.read, "container:test.read", "test.attention", "attention", value)
    [d] = ledger_since(n)
    assert (d.verdict, d.rule) == (Verdict.BROKEN, RULES["epoch_stale"]), d
    assert epochs.stats("container:test.read")["broken"] == 3


def test_an_artifact_reused_under_other_conditions_is_reported_and_reused_as_it_is():
    mode("load")
    epochs.reset()
    epochs.assume("reuse:test.graph", "graph 1", batch=4)
    n = len(load.LEDGER.decisions)
    got, _ = quiet(epochs.reuse, "reuse:test.graph", "test.replay", "replay", "graph 1", batch=8)
    assert got == "broken" and [d.verdict for d in ledger_since(n)] == [Verdict.BROKEN]


# --- the request boundary: every broken request is recorded, none is refused -----------------------------------

TEXT = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if enable_thinking %}<think>{% endif %}"


def model_folder(**declared):
    root = tempfile.mkdtemp()
    folder, manifests = os.path.join(root, "model"), os.path.join(root, "manifests")
    os.makedirs(folder)
    os.makedirs(manifests)
    with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model_type": "llama"}, f)
    with open(os.path.join(folder, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({"chat_template": TEXT}, f)
    if declared:
        m = manifest.pin(manifest.Manifest(manifest.sha256_of(folder), (
            Fact("Template", Template(**declared), Source("manifest", "test"), Certainty.DECLARED),)))
        manifest.save(m, os.path.join(manifests, f"{m.sha256}.json"))
    os.environ[load.ENV_MANIFESTS] = manifests
    request_contract.reset()
    return folder


def test_every_broken_request_is_recorded_and_the_request_goes_on():
    mode("load")
    folder = model_folder(reasoning_history="keep")
    facts = request_contract.declared(folder)
    S, H, T = "request:test.settings", "request:test.history", "request:test.template"
    n = len(load.LEDGER.decisions)
    for _ in range(2):
        made, _ = quiet(request_contract.settings, S, "test", ["enable_thinkng"], [], "the request", "the template")
        assert [d.verdict for d in made] == [Verdict.BROKEN]
        assert "enable_thinkng" in request_contract.reported(made)[0]
    quiet(request_contract.history, H, "test", facts, [False], "conversation")
    quiet(request_contract.template, T, "test", facts, TEXT + " ", True, "named by the request")
    assert [d.verdict for d in ledger_since(n)] == [Verdict.BROKEN] * 4
    assert request_contract.stats(S)["broken"] == 2 and request_contract.stats(S)["refused"] == 0


# --- the vLLM server adapter on stand-ins: served as without entail, optionally noted --------------------------

def _stand_ins():
    """As much of vLLM as the adapter hooks and reads: the renderer module (with a renderer class) and a chat server
    whose create_chat_completion renders the request, then answers."""
    import jinja2
    import jinja2.meta

    hf = types.ModuleType("vllm.renderers.hf")

    def resolve_chat_template(tokenizer, chat_template, tools, *, model_config):
        return chat_template if chat_template is not None else tokenizer.chat_template

    hf.resolve_chat_template = resolve_chat_template
    hf._cached_resolve_chat_template_kwargs = lambda text: jinja2.meta.find_undeclared_variables(
        jinja2.Environment().parse(text))
    hf._get_hf_base_chat_template_params = lambda: frozenset({"add_generation_prompt", "chat_template", "tools",
                                                             "documents", "tokenize", "return_dict"})

    class HfRenderer:
        def __init__(self, folder):
            self.model_config = SimpleNamespace(model=folder)
            self.tokenizer = SimpleNamespace(chat_template=TEXT, name_or_path=folder)

        def get_tokenizer(self):
            return self.tokenizer

        def render_messages(self, messages, params):
            return messages, "prompt"

        async def render_messages_async(self, messages, params):
            return messages, "prompt"

    hf.HfRenderer = HfRenderer
    serving = types.ModuleType("vllm.entrypoints.openai.chat_completion.serving")

    class OpenAIServingChat:
        def __init__(self, folder):
            self.renderer = HfRenderer(folder)

        async def create_chat_completion(self, request, raw_request=None):
            params = SimpleNamespace(chat_template=None,
                                     chat_template_kwargs={"add_generation_prompt": True,
                                                           **(request.chat_template_kwargs or {})})
            await self.renderer.render_messages_async(request.messages, params)
            if request.stream:
                async def chunks():
                    yield "data: {}\n\n"
                return chunks()
            return SimpleNamespace(choices=["the answer"], model_dump=lambda: {})

        def create_error_response(self, message):
            return SimpleNamespace(error=str(message), code=400)

    serving.OpenAIServingChat = OpenAIServingChat
    mods = {n: types.ModuleType(n) for n in ("vllm", "vllm.renderers", "vllm.entrypoints", "vllm.entrypoints.openai",
                                             "vllm.entrypoints.openai.chat_completion")}
    mods.update({"vllm.renderers.hf": hf, "vllm.entrypoints.openai.chat_completion.serving": serving})
    return mods, OpenAIServingChat


def _request(stream=False, **kwargs):
    return SimpleNamespace(messages=[{"role": "user", "content": "q"}], chat_template_kwargs=kwargs, stream=stream,
                           model_extra={}, model_fields_set={"messages", "model"})


async def _drain(stream):
    return [c async for c in stream]


def test_the_server_answers_a_broken_request_and_can_note_it():
    if REAL_VLLM:
        print("  (vllm is installed: the stand-ins are not used)")
        return
    from entail.adapters import vllm_serve as vs

    mods, chat = _stand_ins()
    sys.modules.update(mods)
    try:
        mode("load")
        folder = model_folder()
        assert vs.install_render() == 1 and vs.install_serving() == 1
        server = chat(folder)
        n = len(load.LEDGER.decisions)
        response, printed = quiet(asyncio.run, server.create_chat_completion(_request(enable_thinkng=False)))
        assert response.choices == ["the answer"] and not hasattr(response, "entail"), "served, with nothing added"
        assert [d.verdict for d in ledger_since(n)] == [Verdict.BROKEN] and "enable_thinkng" in printed
        os.environ["ENTAIL_RESPONSE_NOTE"] = "1"
        response, _ = quiet(asyncio.run, server.create_chat_completion(_request(enable_thinkng=False)))
        assert len(response.entail) == 1 and "enable_thinkng" in response.entail[0], response
        stream, _ = quiet(asyncio.run, server.create_chat_completion(_request(stream=True, enable_thinkng=False)))
        got = asyncio.run(_drain(stream))
        assert got[0].startswith(": entail: [entail] broken at") and got[0].endswith("\n\n") and got[1:] == \
            ["data: {}\n\n"], got
        response, _ = quiet(asyncio.run, server.create_chat_completion(_request(enable_thinking=False)))
        assert not hasattr(response, "entail"), "a request with nothing broken carries nothing"
        os.environ["ENTAIL_ON_BROKEN"] = "stop"   # the strict policy: the server's own error response, as in M5.3
        response, _ = quiet(asyncio.run, server.create_chat_completion(_request(enable_thinkng=False)))
        assert response.code == 400 and "enable_thinkng" in response.error, response
    finally:
        os.environ.pop("ENTAIL_ON_BROKEN", None)
        os.environ.pop("ENTAIL_RESPONSE_NOTE", None)
        vs.uninstall()
        for name in mods:
            sys.modules.pop(name, None)


# --- entail check: a static check fails when meaning would break, like a type checker in CI ------------------

def test_entail_check_fails_when_meaning_would_break():
    from entail import cli

    d = tempfile.mkdtemp()
    cfg = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64,
           "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
           "vocab_size": 128, "max_position_embeddings": 64, "rope_theta": 10000.0}
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    code, _ = quiet(cli.main, ["check", "--model", d, "--engine", "transformers"])
    assert code == 0, code
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(dict(cfg, rope_scale=2.0), f)   # rolebench 15: a key nobody reads
    code, printed = quiet(cli.main, ["check", "--model", d, "--engine", "transformers"])
    assert code == 1 and "broken at" in printed and "rope_scale" in printed, (code, printed)


if __name__ == "__main__":
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok", name)
    finally:
        core.set_mode("off")
        tally.reset()
