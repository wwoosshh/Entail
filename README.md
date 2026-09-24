# entail

[![PyPI](https://img.shields.io/pypi/v/entail-ai)](https://pypi.org/project/entail-ai/) [![tests](https://github.com/wwoosshh/entail/actions/workflows/tests.yml/badge.svg)](https://github.com/wwoosshh/entail/actions/workflows/tests.yml)

**Keep what a value means intact across LLM inference-stack boundaries.**

An inference stack is a chain of parts — checkpoint and config, loader, engine, kernels, quantization, cache.
Each part can be correct on its own terms while the *meaning* of a value is lost between two of them: a declared
property the chosen kernel ignores, a setting that arrives under a name nobody reads any more, a cache that
silently loses a token. The output is then wrong, fluently and without a warning.

The goal of entail is to make that meaning explicit, like a type:
- declared where it is produced and carried to where it is used
- checked against the consumer's choice and against the data
- resolved first when they disagree — routed to a consumer that honours it, or converted to the form the consumer
  reads — and, when no fix exists, reported as an error while the run goes on (it stops only if you ask it to)
- said to be "unknown" when nobody declares it, instead of letting a default stand in silently

The name is the logical sense of *entail*: what a checkpoint declares must entail what the engine executes.
(ent·**AI**·**L** — an AI library.)

> **Status: 1.0, measured on one machine.** Everything below was measured on the engines and versions under
> [Tested with](#tested-with), on one RTX 4070 Ti. The 1.0 evaluation is summarised under
> [How it was measured](#how-it-was-measured), and what it found missing under [Known gaps](#known-gaps).
> entail does not look for defects inside a model, a compiler, a kernel or the hardware: when every boundary it
> checked held and the output is still wrong, it says so and narrows where to look.

## Why: a case measured end to end

Restating a model's **own** `rope_scaling` at launch — the route model cards give for enabling YaRN — drops
`rope_theta` under transformers 5, and the engine silently falls back to a RoPE base of 10,000.

Llama-3.2-3B-Instruct, greedy, GSM8K (first 500 on vLLM, first 200 on SGLang):

| | untouched | same `rope_scaling` passed again at launch | with entail |
|---|---|---|---|
| vLLM 0.30.0 `--hf-overrides` | 379 / 500 | **273 / 500**, no warning | 376 / 500 |
| SGLang 0.5.20 `--json-model-override-args` | 161 / 200 | **106 / 200** | 161 / 200 (outputs identical) |

The vLLM row was measured again on 1.0; the SGLang row is from the earlier measurement. The degraded runs are
identical to an explicit `rope_theta = 10000`. Outputs stay fluent; the answers are wrong.
Qwen3 dense models happen to be safe because those model files fill in 1,000,000; Llama, Qwen3-MoE, Gemma and
others do not.

## Install

```bash
pip install entail-ai
```

Install it into the same environment as your engine (vLLM, SGLang or transformers). entail has no dependencies
of its own. The import name is `entail`. `uv pip install entail-ai` works the same way; for the latest commit,
`pip install "git+https://github.com/wwoosshh/entail"`.

Check what it sees:

```bash
entail doctor
```

## Use

Turn it on with one environment variable; nothing else changes.

```bash
ENTAIL=load vllm serve meta-llama/Llama-3.2-3B-Instruct --hf-overrides '{"rope_scaling": {...}}'
ENTAIL=load python -m sglang.launch_server --model-path ... --json-model-override-args '{...}'
ENTAIL=load python your_transformers_script.py
ENTAIL=load python main.py            # ComfyUI, from its folder (on Windows: set ENTAIL=load in the launcher .bat)
```

When it changes something, it says so:

```
[entail] resolved LlamaConfig.rope_scaling was given after the config was built; it replaces rope_parameters and
would have dropped rope_theta=500000.0, ...; kept it, as config.json would
```

From inside a script:

```python
import entail
entail.enable()          # mode="load", policy="resolve"; child processes inherit it
```

### When a model file does not say what it means

Many image checkpoints declare nothing about their prediction type or latent scale, and entail then reports them
as unknown instead of guessing. A manifest declares it from outside, the way a `.d.ts` file types a JavaScript
library:

```bash
entail infer model.safetensors --out model.safetensors.entail.json   # what the file declares, and empty slots
# fill in the slots you know, then mark it reviewed:
entail pin model.safetensors.entail.json
```

A manifest next to the file is found by itself. Manifests kept in a folder (named `<sha256>.json`, the hash the
draft records) are found through `ENTAIL_MANIFESTS`. Only a pinned manifest counts as a declaration.

### How it reaches engine worker processes

vLLM and SGLang run the model in processes they start themselves. `pip install` puts one file,
`entail-autoinstall.pth`, into site-packages; Python reads it at every start-up. Its single line checks the
environment and does nothing unless `ENTAIL` is set. `entail hook status|install|uninstall` shows or manages it
(an editable install does not place it — run `entail hook install`).

## What it does

**Resolves**

| mismatch | resolution | measured |
|---|---|---|
| a RoPE value given under its transformers-4 name after the config is built (`rope_theta`, `rope_scaling` — keyword to `from_pretrained`, attribute, vLLM `--hf-overrides`, SGLang `--json-model-override-args`) | written where `config.json` would have put it, including per-layer-type RoPE (asks the config class) | equal to the `config.json` route on 4 model families × 2 routes × 3 values; GSM8K restored (table above) |
| an attention backend that drops a declared model property (e.g. Gemma 2 logit soft-capping on transformers `sdpa`, SGLang `flashinfer`) | switched to a backend measured to honour it (`eager`, `triton`) | tokens equal the reference run; cost 1.18× (transformers), 1.13× (SGLang, the backend's own price) |
| **ComfyUI, diffusers:** a LoRA that cannot reach the model it is applied to (e.g. an Anima LoRA in an SDXL workflow), or reaches only part of it. ComfyUI skips each module with a console line and the run "succeeds" with the LoRA doing nothing | nothing can convert it: reported with how many of its modules reach the model and what the LoRA declares it was trained for, while the run goes on (`ENTAIL_ON_BROKEN=stop` stops before sampling) | ComfyUI 0.34.1: an Anima LoRA on an SDXL model left the images pixel-identical to no LoRA; entail reported it at the LoRA load, and with `ENTAIL_ON_BROKEN=stop` stopped before sampling (3/3); the right LoRA (15-31/255 of change) passed, with identical images with entail on and off. diffusers 0.40: a LoRA whose keys it does not read loaded and did nothing (pixel-identical images); reported the same way. Earlier, with 0.3.0's check: 22 right pairings passed with no false alarm |
| **ComfyUI, diffusers:** a v-prediction checkpoint that declares it in a way the engine does not read. ComfyUI reads only a `v_pred` key and samples a checkpoint that states `modelspec.prediction_type = v` in its metadata as eps; diffusers' single-file loader reads neither and falls back to epsilon. The images come out broken while the run "succeeds" | the file's own declaration (metadata, marker keys) or a pinned manifest decides: the sampler is set up for it, as a ModelSamplingDiscrete node or a rebuilt scheduler would, and what the declaration leaves open (zero-terminal SNR) keeps the engine's value. A sampling node or scheduler the user set is not overridden: the contradiction is reported. A checkpoint that declares nothing is reported as unknown: how the model behaves is never the basis for a change, so a checkpoint whose marker was lost needs a manifest | ComfyUI 0.34.1, AstolfoCarmix-VPredXL (declares v in its metadata, has no marker key): 83-95/255 from the author's reference setting without entail; with it identical, 0.16 and 0.14/255 over three seeds. diffusers 0.40 single files: NoobAI-XL-Vpred 55-83/255 from its reference without entail, pixel-identical with it; AstolfoCarmix 90-95/255, pixel-identical. entail 0.3.0 judged by the first model call instead and missed AstolfoCarmix (it behaves like eps at the noisiest step); a checkpoint whose marker was removed is now reported as unknown unless a manifest declares it |
| **ComfyUI:** a sampling node's schedule that outlives its workflow. ComfyUI's dynamic VRAM loader backs model buffers up by attribute path, so after a run with a ModelSamplingDiscrete (or similar) node the checkpoint keeps sampling with that node's schedule once the node is gone, and a node used after a plain run silently gets the plain schedule | each sampling object keeps a copy of the schedule its own setter registered; the loader's backup goes back to the object it came from instead of into another; the first model call after the buffers change checks them against that copy and puts them back | ComfyUI 0.34.1: after one run with a ModelSamplingDiscrete(v_prediction, zsnr) node on waiIllustrious, plain runs came out as another image (55.8/255) and then black, with entail on or off, until a restart; with the fix they match a fresh session pixel for pixel (3/3). NoobAI-XL-Vpred with and without a zsnr=false node, both orders: without entail the later runs took the other setting pixel for pixel; with entail all 12 images match a fresh session. The first-call check alone (guard left out) prevents the black images but leaves 2-11/255. An Anima workflow and all other runs are identical with entail on and off, at the same speed |

| **vLLM server:** a tool-call parser that does not read the format the model declares (a Qwen3 model, which emits hermes calls, served with `--tool-call-parser pythonic`): the call comes back as plain text | switched to a parser measured to read the declared format | vLLM 0.30.0, Qwen3-4B: the tool call is returned as a structured call again |
| **diffusers:** a VAE loaded on its own takes another model's latent scale (an SDXL VAE read as SD1.5's) | the scale a manifest declares for the model is applied | 15-16/255 from the reference image without entail, pixel-identical with it (3 seeds) |

**Checks** (and reports what nothing can resolve)

- attention properties against each engine's backends, before any weight is read
- config keys that would be swallowed, and tied-embedding declarations against the checkpoint
- vLLM weights after repacking: layout, stride and a value sample against the declared transform
- weights in memory against the checkpoint file (`ENTAIL_SOURCE=1`)
- the KV cache contract (a request holds what it needs, nothing shrank) on transformers, vLLM's paged cache
  and SGLang
- image models on ComfyUI and diffusers: the prediction type and latent scale a checkpoint, folder or manifest
  declares against the sampler and the VAE, and a LoRA's modules against the model it is applied to
- on vLLM's OpenAI server, per request: a chat template other than the declared one, reasoning history dropped
  where the model declares it is kept, and request fields or template settings that nothing reads
- in debug mode, the boundaries you declare in your own code (`@entail.boundary`: what each argument means);
  a strided layout, a quantized value and chunk-relative positions are converted where the reader needs it

`entail preflight --model /path/to/model --engine sglang --list` runs the start-up checks without starting a server.

### When the output is still wrong

entail records a verdict for every boundary it checks, in `entail_logs/`. `entail locate` reads that record and says
where meaning broke: the first boundary that did not keep it. If every boundary it checked held and the output was
wrong (`entail locate --wrong`), the fault is not in what was handed between layers but inside one - the model, a
compiler, a kernel, the hardware. A boundary entail could not check stays suspect, with the layers on either side.

To narrow it to a layer, run the code in debug mode and compare layers with a reference on the same inputs. Every
layer's first call is compared (more with `calls=`), and a layer whose output its reference does not reproduce is
named:

```python
import torch, entail
from entail import diagnose
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

def mlp_in_float32(self, x):   # the reference: the same MLP, computed in float32
    f = torch.nn.functional
    gate, up = (f.linear(x.float(), p.weight.float()) for p in (self.gate_proj, self.up_proj))
    return f.linear(f.silu(gate) * up, self.down_proj.weight.float()).to(x.dtype)

entail.enable("debug")         # before the model is loaded, so its load is checked too
with diagnose.propagating(), diagnose.watch(Qwen3MLP, "forward", mlp_in_float32, label="mlp"):
    model.generate(**inputs, max_new_tokens=8)
print("\n".join(entail.locate(output_wrong=True).lines()))
```

`diagnose.propagating()` also names an operation that made a declared fact untrue between two boundaries (a
transpose of a value whose layout was declared). With defects planted on Qwen3-4B and gemma-2-2b-it (transformers
5.17) - at a load, cache or code boundary, inside the attention or MLP kernel, behind a boundary entail could not
check - it pointed to the planted place in all 11 cases. The diagnosis cost 1.72x (eager) and 1.90x (sdpa) on a
64-token decode, with the same tokens.

In a test suite, `pytest --entail` runs each test that way: what breaks fails the test, and a failing test's report
says where. A test that takes the `entail_condition` fixture, marked `@pytest.mark.entail_conditions(model="...")`,
runs once per condition the model's declarations put at stake: one token under, at and over its sliding window, around
where a scaled RoPE takes over, a second turn when it declares how earlier reasoning is kept.

### For new code: a role-typed front end (experimental)

`entail.frontend` is for code written from scratch - a model's decode step, the caller of a kernel. Every value has a
type (named dims, dtype, what it is - a query, a key, a value - and the facts it carries), every operation takes its
arguments by keyword, and a program is traced once from the types of its inputs. A key passed as a value, a length
used as the last key index, a weight in a format no kernel reads, a cache read in its version before a write, a sum
reduced twice: each is refused before anything runs. What can be repaired - chunk-relative positions with their
offset, a quantized value with its scale, a softcap the chosen kernel ignores - is repaired then, and said.

```python
from entail.frontend import qwen3

program = qwen3.trace_decode(model.config, batch=8, slots=640, attention="triton")    # every check runs here
step = qwen3.bind(program, model, cache, tokens, positions, until)          # the load contract
logits = step()["logits"]                                                              # no checks left
```

Qwen3-4B's decode step written this way gives transformers' logits bit for bit with the torch lowering. Compiled and
captured in a CUDA graph (int4, batch 8), it takes 1.005-1.007x the time of the same step assembled by hand with the
hand kernel, and 1.020-1.024x with FlexAttention. Of 16 reproduction cases, 9 are refused or repaired while tracing
and 2 more when the program is bound to its tensors; no fixed version is refused.

## How it was measured

For 1.0 every measurement of the development milestones was run again on the final code, on one RTX 4070 Ti.

- **Healthy runs** (Qwen3-4B, Llama-3.2-3B-Instruct and gemma-2-2b-it on transformers, vLLM and SGLang, each
  engine's defaults): no false alarm. Two repairs, both where a backend drops Gemma 2's soft-capping. Every fact
  the model folders declare for loading reached a decision.
- **31 test problems** (16 reproduction cases, 8 field cases, 7 simulated market incidents): each defect was
  repaired; where no repair exists, it was reported at the boundary and fact where it happened while the run
  went on, or stopped with `ENTAIL_ON_BROKEN=stop`. No fixed version was flagged. (The two ComfyUI cases were
  measured before 1.0 and not run again.)
- **Cost:** at load, 0.3-2.6% of the load time. Always on, vLLM's CUDA-graph path 0.999-1.000x (two runs without
  entail: 0.997-0.999x). About 60 us per request on vLLM's server. The diagnosis mode 1.74x (eager) and 1.85x
  (sdpa). With `ENTAIL` unset, 0.27 ms per Python start and no module imported.
- **Next to post-hoc detection:** GSM8K (500 problems, greedy) caught the RoPE loss above and a planted weight
  shift, but not a backend that drops Gemma 2's soft-capping: SGLang `torch_native` 313 against `triton` 316
  (McNemar p = 0.66); measured before 1.0, transformers `sdpa` against `eager` 337 against 339 (2B) and 442
  against 443 (9B).
  Comparing the outputs with a healthy run found it (198 of 500 answers differ, none between two healthy runs),
  where such a run exists. entail repairs it at load.
- **Locating:** 11 of 11 planted defects located.

## Known gaps

- The chat template is compared with the declared one on vLLM's OpenAI server only. transformers'
  `apply_chat_template` and SGLang's server use it without a check.
- RoPE fields outside the vocabulary (Llama 3's `low_freq_factor` and `high_freq_factor`) are read but not
  compared, and nothing says so while the model runs.
- The always-on KV contract on transformers' dynamic cache costs about 4% of decode time (target: 2%). On vLLM's
  CUDA-graph path it is within noise.
- One GPU. The reduction contracts (a value summed twice across ranks) were measured with one process standing
  in for two ranks.

## Configuration

| variable | values | meaning |
|---|---|---|
| `ENTAIL` | `off` (default), `load`, `debug` | `load`: start-up checks and resolvers; `debug`: also every declared boundary, and uncovered cases become errors |
| `ENTAIL_POLICY` | `resolve` (default), `refuse` | `refuse` repairs nothing: every mismatch is only reported |
| `ENTAIL_ON_BROKEN` | `report` (default), `stop` | what nothing repairs: reported in the log (`entail_logs/`) while the run goes on, or stopped before any output (a request then gets the server's own error). `ENTAIL_FACT_POLICY=Layout=stop` stops for one kind of fact only |
| `ENTAIL_UNKNOWN` | `report` (default), `require`, `stop` | a meaning-changing fact nobody declares: reported, or the run waits for a declaration |
| `ENTAIL_LOG_DIR` | a folder, or `off` | where entail keeps what it said. Unset, whenever entail is on: `entail_logs/` in the folder the program was started from, with a log (`entail-<date>.log`, each line with its time and process) and a record (`record-<date>.jsonl`, every decision as JSON) per day, and a `.gitignore` so it stays out of the project's history. Every process an engine starts writes there too |
| `ENTAIL_RECORD` | a file | the JSON record goes to this file instead of `record-<date>.jsonl` |
| `ENTAIL_RESPONSE_NOTE` | `1` | vLLM server: a response also carries what broke for its request (an `entail` field, or SSE comment lines ahead of a stream) |
| `ENTAIL_ONLY` | e.g. `rope_alias,sglang_adapter` | install only these adapters |
| `ENTAIL_SKIP` | e.g. `comfyui_repair:install_buffer_guard` | leave out these entries (a bare name leaves out the whole adapter), to measure the rest without them |
| `ENTAIL_VERBOSE` | `1` | print each adapter as it is installed |
| `ENTAIL_SOURCE` | `1` | also compare loaded weights with the checkpoint file (vLLM, a little I/O at start-up) |
| `ENTAIL_MANIFESTS` | folders, separated by `:` (`;` on Windows) | where to look for manifests (`<sha256>.json`) of model files that do not declare what they mean. A file is hashed only when a manifest could be for it, and its hash is kept in `entail_hashes.json` in the first folder |

## Tested with

transformers 5.12.1, 5.16.1 and 5.17.0, vLLM 0.30.0, SGLang 0.5.20, diffusers 0.40.0, ComfyUI 0.34.1 (Windows),
torch 2.13–2.14,
Python 3.12, one RTX 4070 Ti (12 GB). Other versions may work; `entail doctor` prints what is installed. On transformers 4.x the RoPE resolver
has nothing to do and stays out of the way.

The capability table (which backend honours what) records its evidence per entry, and only entries marked
*measured* are used as resolution targets. The measurement scripts and raw results are kept in the author's
research workspace and are not in this repository yet.

## Development

```bash
git clone https://github.com/wwoosshh/entail && cd entail
pip install -e .
entail hook install          # editable installs do not place the start-up hook
PYTHON=python bash tests/run_all.sh
```

Tests that need a local model look in `ENTAIL_TEST_MODELS` (default `~/models`) and skip when it is absent.

한국어 안내: [README.ko.md](README.ko.md)

## License

See [LICENSE](LICENSE).
