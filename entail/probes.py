"""probes: check one capability-table row against the data (`entail probe`; LIBRARY_DESIGN.md 4.4, principle 9).

Ported from entail/audits/cap_probe.py (transformers) and sweep/run_sglang.py, run_vllm.py (the method of
sweep/PROTOCOL.md). The method:
  1. Two model folders share the checkpoint's weights (symlinks) and differ only in config.json: in one the fact
     binds (a value that must change the output if it is honoured), in the other it is removed.
  2. The consumer decodes the same prompts greedily: twice with the fact bound (the control), once without.
  3. The two bound runs must agree, or nothing can be concluded: "inconclusive".
  4. The same output without the fact means the consumer never read it: "ignores". A difference means it did:
     "honours" - some path read it; a drop on one path can hide behind a difference (caps.json says where).
  5. An "ignores" is believed only if the value binds at all: the first preferred consumer of the group must show
     a difference under the same settings (the binding gate, sweep PROTOCOL revision 2). Otherwise "inconclusive".

Runs on the GPU, in the environment that has the engine: transformers here, sglang and vllm in their own venvs,
e.g. ~/venvs/sglang/bin/python -m entail probe --engine sglang --consumer torch_native --fact ModelProps.softcap
--model ~/models/gemma-2-2b-it
"""
import copy
import json
import os
import shutil
import tempfile
import time

from . import caps as _caps
from .readers import ALIASES

PROMPTS = ["Explain in three sentences why the sky is blue.", "Write a haiku about a quiet library.",
           "List five differences between TCP and UDP.", "What is the capital of Australia?",
           "Solve for x: 3x + 7 = 25.", "Describe photosynthesis simply."]
N_NEW = 24
# A value that must change the output if the consumer honours it (sweep/facts.py: 5.0 and 16 bind on Gemma 2 2B).
BIND = {"ModelProps.softcap": 5.0, "ModelProps.sliding_window": 16}


def _key(fact):
    name, fld = fact.split(".")
    keys = ALIASES.get(name, {}).get("hf_config", {}).get(fld)
    if not keys or fact not in BIND:
        raise ValueError(f"probe: no way to bind {fact} in a config.json; probable facts are {sorted(BIND)}")
    return keys[0]


def configs(base, fact):
    """(bound, removed) copies of a config.json dict. The model must declare the fact: a fact it does not have
    cannot be taken away."""
    key = _key(fact)
    text = "text_config" if isinstance(base.get("text_config"), dict) else None
    scope = base[text] if text else base
    if scope.get(key) is None:
        raise ValueError(f"probe: the model does not declare {key}, so there is nothing to remove")
    bound = copy.deepcopy(base)
    (bound[text] if text else bound)[key] = BIND[fact]
    removed = copy.deepcopy(bound)
    (removed[text] if text else removed).pop(key, None)
    return bound, removed


def model_copy(model_dir, cfg):
    """A model folder that shares the weights (symlinks) and carries a different config.json."""
    d = tempfile.mkdtemp(prefix="entail_probe_", dir=os.path.expanduser("~"))
    for name in os.listdir(model_dir):
        src = os.path.join(model_dir, name)
        if name != "config.json" and os.path.isfile(src):
            os.symlink(src, os.path.join(d, name))
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    return d


# --- runners: decode the prompts with one consumer, `runs` times -----------------------------------------------

def _transformers(model_dir, consumer, runs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
               add_special_tokens=False).input_ids for p in PROMPTS]
    group, impl = consumer.rsplit(".", 1)
    paged = group.endswith(".paged_attention")   # continuous batching: generate_batch runs paged|<impl>
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation=impl)
    model = model.cuda().eval()
    out = []
    try:
        for _ in range(runs):
            if paged:   # generate_batch puts the paged| prefix on the implementation itself
                gc = GenerationConfig(max_new_tokens=N_NEW, do_sample=False, eos_token_id=None)
                with torch.no_grad():
                    res = model.generate_batch(inputs=ids, generation_config=gc)
                keys = list(res)
                out.append([list(res[f"req_{i}"].generated_tokens if f"req_{i}" in res else
                                 res[keys[i]].generated_tokens)[:N_NEW] for i in range(len(ids))])
            else:
                seqs = []
                for x in ids:
                    with torch.no_grad():
                        g = model.generate(torch.tensor([x], device="cuda"), max_new_tokens=N_NEW, do_sample=False)
                    seqs.append(g[0, len(x):].tolist())
                out.append(seqs)
    finally:
        del model
        torch.cuda.empty_cache()
    return out


def _sglang(model_dir, consumer, runs):
    import sglang as sgl

    short = consumer.rsplit(".", 1)[1]
    # The prefix cache is off so that the control run repeats the same computation: with it on, the second run
    # reuses the first run's KV for the prompt, a different path (torch_native did not reproduce its own control).
    engine = sgl.Engine(model_path=model_dir, attention_backend=short, mem_fraction_static=0.7,
                        max_total_tokens=4096, log_level="error", disable_cuda_graph=True, disable_radix_cache=True)
    try:   # one prompt per request: the batch a prompt lands in changes its numerics (sweep PROTOCOL revision 1)
        return [[engine.generate(p, {"max_new_tokens": N_NEW, "temperature": 0})["text"] for p in PROMPTS]
                for _ in range(runs)]
    finally:
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            pass


def _vllm(model_dir, consumer, runs):
    os.environ["VLLM_ATTENTION_BACKEND"] = consumer.rsplit(".", 1)[1]
    import gc

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_dir, max_model_len=2048, gpu_memory_utilization=0.88, enforce_eager=True,
              disable_log_stats=True)
    try:
        params = SamplingParams(max_tokens=N_NEW, temperature=0)
        return [[o.outputs[0].text for o in llm.generate(PROMPTS, params)] for _ in range(runs)]
    finally:
        del llm
        gc.collect()


RUNNERS = {"transformers": _transformers, "sglang": _sglang, "vllm": _vllm}


def judge(bound, control, removed):
    """(verdict, prompts that differ without the fact) for three runs of the same prompts."""
    if bound != control:
        return "inconclusive", None
    differing = sum(1 for a, b in zip(bound, removed) if a != b)
    return ("honours" if differing else "ignores"), differing


def _measure(runner, model_dir, consumer, bound_cfg, removed_cfg):
    dirs = []
    try:
        dirs.append(model_copy(model_dir, bound_cfg))
        dirs.append(model_copy(model_dir, removed_cfg))
        a, a2 = runner(dirs[0], consumer, 2)
        (b,) = runner(dirs[1], consumer, 1)
        return judge(a, a2, b)
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)


def full_name(engine, consumer):
    """"sdpa" -> "transformers.attention.sdpa"; "paged|sdpa" (transformers' own spelling) -> the paged group."""
    if consumer.count(".") >= 2:
        return consumer
    if consumer.startswith("paged|"):
        return f"{engine}.paged_attention.{consumer.split('|', 1)[1]}"
    return f"{engine}.attention.{consumer}"


def probe(engine, consumer, fact, model, table=None, gate=True, runner=None):
    """Measure one row. `consumer` is a short name ("sdpa") or a full one ("transformers.attention.sdpa").
    Returns a dict with the measured verdict, what the table says, and whether they agree."""
    table = table or _caps.default_table()
    full = full_name(engine, consumer)
    short, group = full.rsplit(".", 1)[1], _caps.group_of(full)
    model_dir = os.path.expanduser(model)
    runner = runner or RUNNERS.get(engine)
    if runner is None:
        raise ValueError(f"probe: no runner for engine {engine!r}; engines are {sorted(RUNNERS)}")
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        bound_cfg, removed_cfg = configs(json.load(f), fact)
    row = {"engine": engine, "consumer": full, "fact": fact, "model": model_dir, "bind": BIND[fact],
           "prompts": len(PROMPTS), "new_tokens": N_NEW, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    try:
        verdict, differing = _measure(runner, model_dir, full, bound_cfg, removed_cfg)
        row.update(measured=verdict, differing_prompts=differing)
        if verdict == "ignores" and gate:
            # any consumer of this engine that is preferred for routing shows whether the value binds at all
            refs = [f"{g}.{n}" for g in (group, f"{engine}.attention") for n in table.preferred(g)]
            ref = next((r for r in refs if r != full), None)
            if ref is None:
                row.update(measured="inconclusive", why="no reference consumer to show that the value binds")
            else:
                ref_verdict, ref_diff = _measure(runner, model_dir, ref, bound_cfg, removed_cfg)
                row["gate"] = {"consumer": ref, "measured": ref_verdict, "differing_prompts": ref_diff}
                if ref_verdict != "honours":
                    row.update(measured="inconclusive", why=f"the value does not bind: {ref} did not change either")
    except Exception as e:  # noqa: BLE001 - a probe that fails says so; it never reports a verdict it did not see
        row.update(measured="inconclusive", why=f"{type(e).__name__}: {str(e)[:300]}")
    row["seconds"] = round(time.time() - t0, 1)
    cell = _caps.lookup(table, full, fact)
    row["table"] = None if cell is None else {"honours": cell.honours, "evidence": cell.evidence, "ref": cell.ref}
    if row["measured"] in ("honours", "ignores") and cell is not None:
        row["agrees"] = (row["measured"] == "honours") == cell.honours
    else:
        row["agrees"] = None
    return row


def main(args):
    row = probe(args.engine, args.consumer, args.fact, args.model, table=_caps.load_table(args.table) if args.table
                else None, gate=not args.no_gate)
    table = row["table"]
    said = "not in the table" if table is None else \
        f"{'honours' if table['honours'] else 'drops'} ({table['evidence']}: {table['ref']})"
    differing = row.get("differing_prompts")
    shown = "" if differing is None else f" ({differing} of {len(PROMPTS)} prompts differ)"
    why = f"; {row['why']}" if row.get("why") else ""
    print(f"{row['consumer']} {row['fact']}: measured {row['measured']}{shown}; table says {said}; "
          f"agrees: {row['agrees']}{why}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=1)
        print(f"wrote {args.out}")
    return 0 if row["agrees"] else 1
