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

> **Status: research prototype (alpha). Read this before relying on it.**
> 0.3.0 is not yet that general mechanism. It is a set of specific checks and resolvers for cases that were
> measured, listed under [What it does](#what-it-does):
> - RoPE settings passed under old names
> - attention backends that drop a declared property
> - a few ComfyUI cases
> - start-up checks and cache contracts on three LLM engines
>
> Anything not listed there is not checked. The general layer is being built for the next major version: facts
> read from model files and configs, contracts at load, cache and request boundaries, and a ledger that shows
> where meaning broke. Measured on one RTX 4070 Ti with the versions under "Tested with".

## Why: a case measured end to end

Restating a model's **own** `rope_scaling` at launch — the route model cards give for enabling YaRN — drops
`rope_theta` under transformers 5, and the engine silently falls back to a RoPE base of 10,000.

Llama-3.2-3B-Instruct, greedy, GSM8K (first 500 on vLLM, first 200 on SGLang):

| | untouched | same `rope_scaling` passed again at launch | with entail |
|---|---|---|---|
| vLLM 0.30.0 `--hf-overrides` | 379 / 500 | **279 / 500**, no warning | 378 / 500 |
| SGLang 0.5.20 `--json-model-override-args` | 161 / 200 | **106 / 200** | 161 / 200 (outputs identical) |

The degraded runs are identical to an explicit `rope_theta = 10000`. Outputs stay fluent; the answers are wrong.
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
| **ComfyUI:** a LoRA that cannot reach the model it is applied to (e.g. an Anima LoRA in an SDXL workflow). ComfyUI skips each module with a console line and the run "succeeds" with the LoRA doing nothing | nothing can convert it, so the workflow stops before sampling with the reason: what the LoRA declares it was trained for, and which model it met. A partial match is reported and the run goes on | on a real ComfyUI 0.34.1: the wrong pairing changed the image by 0.8/255 (the right LoRA: 35.2) behind 840 console lines; with entail both wrong directions stop; 22 right pairings (21 SDXL LoRAs incl. text encoders, 1 Anima) pass with no false alarm; images identical with entail on and off |
| **ComfyUI:** a v-prediction checkpoint whose `v_pred` marker was lost in a merge or conversion. ComfyUI samples it as eps and the images come out as coloured noise or black, while the run "succeeds" | the first model call of the sampling shows how the model really behaves (an eps model returns the noise it was given, a v model does not), with no extra forward pass; the model is then sampled that way, as a ModelSamplingDiscrete node would. A sampling node in the workflow that contradicts the model stops the run instead | NoobAI-XL-Vpred with the marker removed: 67-102/255 from the right images without entail, 12-20 with it (the rest is the zero-terminal-SNR setting, which behaviour cannot reveal). Eps checkpoints measure 0.9997-0.9999, the v one 0.01. Images identical with entail on and off, the first image after start-up included. A `v_prediction` node left on in front of an eps checkpoint stops at the sampler (3/3; without entail a flat grey image, reported as success) |
| **ComfyUI:** a sampling node's schedule that outlives its workflow. ComfyUI's dynamic VRAM loader backs model buffers up by attribute path, so after a run with a ModelSamplingDiscrete (or similar) node the checkpoint keeps sampling with that node's schedule once the node is gone, and a node used after a plain run silently gets the plain schedule | each sampling object keeps a copy of the schedule its own setter registered; the loader's backup goes back to the object it came from instead of into another; the first model call after the buffers change checks them against that copy and puts them back | ComfyUI 0.34.1: after one run with a ModelSamplingDiscrete(v_prediction, zsnr) node on waiIllustrious, plain runs came out as another image (55.8/255) and then black, with entail on or off, until a restart; with the fix they match a fresh session pixel for pixel (3/3). NoobAI-XL-Vpred with and without a zsnr=false node, both orders: without entail the later runs took the other setting pixel for pixel; with entail all 12 images match a fresh session. The first-call check alone (guard left out) prevents the black images but leaves 2-11/255. An Anima workflow and all other runs are identical with entail on and off, at the same speed |

**Checks** (and reports what nothing can resolve; the ComfyUI and diffusers checks above predate this and still stop)

- attention properties against each engine's backends, before any weight is read
- config keys that would be swallowed, and tied-embedding declarations against the checkpoint
- vLLM weights after repacking: layout, stride and a value sample against the declared transform
- weights in memory against the checkpoint file (`ENTAIL_SOURCE=1`)
- the KV cache contract (a request holds what it needs, nothing shrank) on transformers, vLLM's paged cache
  and SGLang

`entail preflight --model /path/to/model --engine sglang --list` runs the start-up checks without starting a server.

## Configuration

| variable | values | meaning |
|---|---|---|
| `ENTAIL` | `off` (default), `load`, `debug` | `load`: start-up checks and resolvers; `debug`: also every declared boundary, and uncovered cases become errors |
| `ENTAIL_POLICY` | `resolve` (default), `refuse` | `refuse` repairs nothing: every mismatch is only reported |
| `ENTAIL_ON_BROKEN` | `report` (default), `stop` | what nothing repairs: reported in the log and `ENTAIL_RECORD` while the run goes on, or stopped before any output (a request then gets the server's own error). `ENTAIL_FACT_POLICY=Layout=stop` stops for one kind of fact only |
| `ENTAIL_UNKNOWN` | `report` (default), `require`, `stop` | a meaning-changing fact nobody declares: reported, or the run waits for a declaration |
| `ENTAIL_RECORD` | a file | every decision as one JSON line, from every process an engine starts |
| `ENTAIL_RESPONSE_NOTE` | `1` | vLLM server: a response also carries what broke for its request (an `entail` field, or SSE comment lines ahead of a stream) |
| `ENTAIL_ONLY` | e.g. `rope_alias,sglang_adapter` | install only these adapters |
| `ENTAIL_SKIP` | e.g. `comfyui:install_buffer_guard` | leave out these entries (a bare name leaves out the whole adapter), to measure the rest without them |
| `ENTAIL_VERBOSE` | `1` | print each adapter as it is installed |
| `ENTAIL_SOURCE` | `1` | also compare loaded weights with the checkpoint file (vLLM, a little I/O at start-up) |
| `ENTAIL_SEED` | test names | fault injection used to test the checks themselves; never set it in production |

## Tested with

transformers 5.12.1, 5.16.1 and 5.17.0, vLLM 0.30.0, SGLang 0.5.20, ComfyUI 0.34.1 (Windows), torch 2.13–2.14,
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
