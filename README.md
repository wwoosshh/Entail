# entail

[![PyPI](https://img.shields.io/pypi/v/entail-ai)](https://pypi.org/project/entail-ai/) [![tests](https://github.com/wwoosshh/entail/actions/workflows/tests.yml/badge.svg)](https://github.com/wwoosshh/entail/actions/workflows/tests.yml)

**Your model files say how they must be run. Your engine doesn't always listen.**

RoPE base and scaling, soft-capping, sliding windows, chat templates, prediction types: when one of these
declarations does not reach the engine, the output is wrong without a warning. entail reads what the files already
declare, checks it where it is used, repairs it before the first token when it can, and logs exactly what broke
when it cannot. Zero configuration; about 1% of request time, and on vLLM about 1.7 s more at load (13-15% of a
small model's load, most of it a start-up check of the engine's own paths that `ENTAIL_NO_PATHS=1` turns off).

- **64 of 180.** Of the 300 most-downloaded LLMs on Hugging Face, 180 take vLLM's launch-time `rope_scaling`
  override (the usual way to turn on long context). It silently changes the RoPE base of 64 of them. With entail
  on, all 180 keep their base. (vLLM 0.30, checked on the model files with vLLM's own config code;
  [E1](https://github.com/wwoosshh/entail-research/blob/main/testbed/results/m10/E1_SUMMARY.md).)
- **379 → 273.** Measured end to end, that is Llama-3.2-3B-Instruct's GSM8K score under that route (376 with entail
  on); Qwen3-4B-Instruct-2507 goes 183 → 175 under YaRN. No warning in either case.
- **Evals miss some of it.** A backend that drops Gemma 2's soft-capping changed 198 of 500 answers while GSM8K
  moved by 3 (p = 0.66). entail routes to a backend that honours it, at load.
- **102 healthy runs, no run broken.** 38 popular models on transformers, vLLM and SGLang: outputs identical to the
  run without entail in 97 of 98 comparisons (the other is the engine's own nondeterminism), and no check added in
  1.3.0 raised an alarm. Six runs report a real tokenizer difference (transformers 5 drops a leading space for two
  Llama-2-era tokenizers). The one false alarm found outside that set (encoder-decoder models on vLLM) is fixed
  in 2.0.0. (1.0.0 got 17 of its first 81 runs wrong; those causes are fixed and in the [changelog](CHANGELOG.md).)

Check your own model in three lines:

```bash
pip install entail-ai
entail preflight --model /path/to/model --engine vllm --list   # what each backend would drop; no GPU needed
ENTAIL=load vllm serve /path/to/model ...                       # then read entail_logs/
```

entail makes the meaning of a value explicit, like a type: declared where it is produced and carried to where it is
used; checked against the consumer's choice and against the data; resolved first when they disagree (routed to a
consumer that honours it, or converted to the form the consumer reads) and, when no fix exists, reported while the
run goes on (it stops only if you ask it to); said to be "unknown" when nobody declares it, instead of letting a
default stand in silently. The name is the logical sense of *entail*: what a checkpoint declares must entail what
the engine executes. (ent·**AI**·**L**: an AI library.)

**Who it is for:** people who build their own AI project on a Python engine - transformers, diffusers, vLLM,
SGLang - or on ComfyUI. entail works inside the Python process that runs the model, so apps that run models in an
engine compiled into the app (Ollama, LM Studio, llama.cpp) give it nothing to attach to. A Python program that
loads an engine entail does not watch (CTranslate2 under faster-whisper, ONNX Runtime, llama.cpp's Python binding
and a few more; `entail/data/unwatched_engines.json`) gets one `unknown` line for it, so a run is not read as fully
checked when the main work ran there. Where to install it for
each tool - your scripts, vLLM behind Open WebUI, ComfyUI (git, portable, Desktop), text-generation-webui, vLLM in
Docker - is in [INSTALL.md](https://github.com/wwoosshh/entail/blob/main/INSTALL.md).

> **Status: 2.1.4, measured on one machine.** 2.1.4 fixes what the field test of 2.1.3 found (a seeded run with vLLM
> inside a trainer samples as without entail; SGLang's KV check no longer flags a request others joined; a serving
> app's own chat template is held to the prompt it renders; an engine's or app's own choices are named as theirs; a
> launch that reached nothing, and an engine entail does not watch, are said). 2.1.3 fixed what the field test of
> 2.1.2 found (diffusers' prediction decided where the pipeline samples, so apps that set up their own sampler are
> left alone; RoPE lines that name the engine and the writer; models that are not LLMs no longer drawn as one). 2.1.2
> fixed what the field test of 2.1.1 found (declaring a value with a manifest from the docs, `entail preflight`'s false alarms on popular models, a page that leads with what entail repaired,
> upgrading on Windows); 2.1.1 fixed what the first field test of 2.1.0 found (clearer console lines and
> page wording, a tokenizer folder given as a path object, doctor's engine list). 2.1.0 remade the page of
> `entail serve` (plain words, a node canvas) and added install steps by tool
> ([INSTALL.md](https://github.com/wwoosshh/entail/blob/main/INSTALL.md)); the checks are 2.0's. 2.0 adds the
> platform - `entail serve`, two safety modes, official DLCs and custom nodes ([The platform](#the-platform-20)) -
> around the same checks: on the 102 healthy runs it made exactly the decisions 1.3.0's frozen code made (no run
> broken, the same six real tokenizer differences reported), requests cost at most about 1% more with entail on (1.000-1.011 over ten
> interleaved rounds, the same as 1.3.0), with or without `entail serve` reading the records, and a wheel installed with no index brings in nothing but entail.
> Everything below was measured on the engines and versions under [Tested with](#tested-with), on one RTX 4070 Ti. The evaluation is summarised under
> [How it was measured](#how-it-was-measured), and what it found missing under [Known gaps](#known-gaps).
> **What it is today:** a light pre-deployment check. It reads what model files declare, holds the engine's own
> kernels, paths and parsers against references where a declaration lives only in code, repairs the classes it
> knows, and says where meaning broke. **What it is not yet:** protection against bugs nobody has seen. In four
> pre-registered replays of real engine bugs, each run with the code frozen, it detected 0 of 7, 0 of 8, 0 of 5 and
> 0 of 6 of the reproduced bugs in its class; most sat where no check of entail's reaches (a processor's settings, a
> scheduler step, a weight loader, a KV connector, a kernel called from C++).
> 1.1.0 added five facts the 1.0 evaluation showed it did not read; 1.2.0 the sites the first replay found unread;
> 1.3.0 adds reference comparisons (a tokenizer run against the declared one, a custom op's kernel against its own
> definition, a parser's stream against its whole-text parse, a placeholder's origin, a completion's logprobs),
> definitions for five engine functions, checks that reach an engine's warm-up (before a CUDA graph captures a
> kernel), and the engine's own paths held against each other at start.
> entail does not look for defects inside a model, a compiler, a kernel or the hardware: when every boundary it
> checked held and the output is still wrong, it says so and narrows where to look.

## What it does to your environment

- `pip install entail-ai` adds one package (import name `entail`, no dependencies) and one line to `site-packages`
  (`entail-autoinstall.pth`), which is how entail reaches the worker processes an engine starts. With `ENTAIL`
  unset that line returns at once: 0.2-0.3 ms per Python start, no module imported. `entail hook status` shows it
  and `entail hook uninstall` removes it.
- With `ENTAIL=load`, everything entail says goes to `entail_logs/` in the folder the program was started from (a
  log and a JSON record per day, with a `.gitignore`). The folder may also hold `said.txt` (the lines this launch
  already printed, so that an engine's processes do not repeat them), small caches of what entail read from tokenizer
  folders (`tokenizer_ids.json`, `vocab_sources.json`) and the page's two settings (`safe_mode.json`, `nodes.json`).
  `ENTAIL_LOG_DIR=off` writes nothing; `ENTAIL_LOG_DIR=<folder>` moves it. At start one line on stderr says entail is
  on and where it writes. `ENTAIL_QUIET=unknown` keeps the non-blocking `unknown` lines off the console,
  `ENTAIL_QUIET=start` that first line and the closing line of a launch that recorded nothing (both:
  `unknown,start`).
- The adapters hook internal functions of the engine versions they were measured on, listed below. On another
  version an adapter that cannot install says so once (`could not install ...`) and stays out; the rest run.
  `entail doctor` prints what is installed and what would hook.

| engine | measured on | what its adapters check |
|---|---|---|
| transformers | 5.12.1, 5.16.1, 5.17.0 | attention backend, tied head, config keys, RoPE names, KV cache, chat template |
| vLLM | 0.30.0 | attention backend, loader, weight layout after repacking, KV cache, OpenAI server, weights against the file, custom-op kernels and four engine functions against their definitions, the engine's own paths at start |
| SGLang | 0.5.20 | attention backends, loader, KV cache, server, the GDN gate against its definition (the paths at start with `ENTAIL_PATHS=1`) |
| diffusers | 0.40.0 | prediction type, VAE scale, LoRA reach |
| ComfyUI | 0.34.1 | prediction type, VAE scale, LoRA reach; the repair of ComfyUI's own #16490 is the official DLC `entail-dlc-comfyui` (`dlc/comfyui`) |

## Why: a case measured end to end

Restating a model's **own** `rope_scaling` at launch — the route model cards give for enabling YaRN — drops
`rope_theta` under transformers 5, and the engine silently falls back to a RoPE base of 10,000.

Llama-3.2-3B-Instruct, greedy, GSM8K (first 500 on vLLM, first 200 on SGLang):

| | untouched | same `rope_scaling` passed again at launch | with entail |
|---|---|---|---|
| vLLM 0.30.0, transformers 5.17.0 (`hf_overrides`) | 379 / 500 | **273 / 500**, no warning | 376 / 500 |
| SGLang 0.5.20, transformers 5.12.1 (`json_model_override_args`) | 161 / 200 | **106 / 200** | 161 / 200 (outputs identical) |

The vLLM row was measured again on 1.0; the SGLang row is from the earlier measurement. Both used the engines'
offline APIs (vLLM's `LLM`, SGLang's `Engine`). The degraded runs are identical to an explicit `rope_theta = 10000`.
On SGLang's server (`launch_server`, the same versions) a retrieval of 5 codes from a 5.4k-token list found 5 of 5
untouched, 0 of 5 with the override, and 5 of 5 with entail
([#26](https://github.com/wwoosshh/entail/issues/26); a field test that had seen no drop had reached one server
three times, because the next two could not bind the same port). Outputs stay fluent; the answers are wrong.
Qwen3 dense models happen to be safe because those model files fill in 1,000,000; Llama, Qwen3-MoE, Gemma and
others do not.

## Install

```bash
pip install entail-ai
```

Install it into the same environment as your engine (vLLM, SGLang or transformers). For ComfyUI (git, portable,
Desktop), text-generation-webui and vLLM in Docker, [INSTALL.md](https://github.com/wwoosshh/entail/blob/main/INSTALL.md)
says where that environment is. entail has no dependencies of its own. The import name is `entail`. `uv pip install entail-ai` works the same way; for the latest commit,
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

Give the model file's full path. Each empty slot in the draft says the form of its value in `form`; a
v-prediction model with zero terminal SNR is filled in as `"value": {"kind": "v", "zsnr": true}` (`kind` is `eps`,
`v`, `x0`, `flow` or `edm`; `zsnr` is `true`, `false` or `null`), and a latent scale as
`"value": {"scale": 0.13025, "shift": null}`. A word such as `"v_prediction"` or `"epsilon"`, or a bare number for a
latent scale, is read too. When a value cannot be read, `entail pin` says how to write it and changes nothing.

A manifest next to the file is found by itself. Manifests kept in a folder (named `<sha256>.json`, the hash the
draft records) are found through `ENTAIL_MANIFESTS`. Only a pinned manifest counts as a declaration.

### How it reaches engine worker processes

vLLM and SGLang run the model in processes they start themselves. `pip install` puts one file,
`entail-autoinstall.pth`, into site-packages; Python reads it at every start-up. Its single line checks the
environment and does nothing unless `ENTAIL` is set. `entail hook status|install|uninstall` shows or manages it
(an editable install does not place it — run `entail hook install`). A process that runs the Python of another
environment does not read the file: Xinference's per-model virtual environments are out of reach unless the model is
launched with `--disable-virtual-env` ([INSTALL.md](INSTALL.md#xinference)). When a launch ends and no process of it
recorded anything, the process that started it says so in one line on stderr.

## The platform (2.0)

entail 2.0 is a local platform that manages the stability of an AI project: the checks above, a page that shows
where meaning held and where it broke, two safety modes, and checks you attach and detach by node - official DLCs for
an engine, custom nodes for your own code. Everything stays on your machine; the core still has no dependencies.

> **Trust:** a DLC or a custom node is Python code you install, running in your program with your program's rights.
> entail keeps it from breaking a run (a failure, a slow or a stuck validator is recorded and left out) and from
> changing entail's own rules; it does not sandbox it. Install only what you trust, and list what may attach
> (`ENTAIL_DLC`, `ENTAIL_NODES`).

### See your runs: `entail serve`

```bash
ENTAIL=load python your_app.py        # (or vllm serve ..., ComfyUI ...) writes entail_logs/ as before
entail serve --open                   # http://127.0.0.1:8765/ - this machine only
```

The page shows each launch as a workflow of nodes (model files and config, tokenizer, weights, attention and
rotary, the engine's self-check, request, cache, kernels, response; checkpoint, prediction, VAE and LoRA for
images; your own code), each in its state - passed, resolved, broken, stopped, or not verified (`unknown`,
`unchecked`, shown apart from a break). The header says in plain words what happened and names the node where meaning
first broke; clicking a node shows the declared value next to the one the engine used, the fields that differ marked,
with the rule, the repair and the note. New record lines update the page live. The page speaks English or Korean
(the browser's language; a switch in the top bar). The server reads the record files from a process of its own: it
adds nothing to the engine's cost.

- On planted faults (M7.3's eleven and a healthy run) the page named the same place the program located in process
  in 12 of 12 - 9 from the records alone, 3 through the diagnosis mode's `located` lines.
- It is bound to 127.0.0.1 and refuses a request whose Host is not its own. It writes two files of the log folder
  and nothing else (the safety mode of the next start, and which custom nodes are off); a write needs the token it
  prints at start (and puts in its own page; a new one after each write) and an Origin that is itself.

### Two safety modes

`ENTAIL_SAFE` (or the page's settings, which write `entail_logs/safe_mode.json` for the next start):

- **auto** (default) - *the selective safe path.* At start the engine's own paths are held against each other (vLLM:
  decode against a fresh prefill, alone against batched, cold against a prefix-cache hit). When they disagree, the
  next starts of that configuration turn the optimizations the disagreement points at off, one per start, until the
  paths agree - that one is the cause and stays off - or none is left (said once as broken). Nothing stops: the run
  that found it goes on as it was.
- **all** - *the explicit safe mode.* Every optimization the engine declares does not change results is off (vLLM:
  CUDA graphs and torch.compile, the prefix cache, speculative decoding, custom kernels - custom ops and the IR ops'
  kernel priority; SGLang: CUDA graphs, the radix cache, speculative decoding). If a fault stays, entail says the
  cause is outside them; if it went away, that it was inside them.
- **off** - neither.

Measured on vLLM 0.30: the two configurations whose paths disagreed (Qwen3.5-4B-NVFP4 and Nemotron-3-Nano-4B with
n-gram speculative decoding) agreed again at the second start, speculative decoding named the cause (throughput
0.775x on Nemotron, 1.318x on Qwen3.5-NVFP4, where speculation had been a loss in eager mode). Planted faults: the
first try called a fault in vLLM's RMSNorm kernel "outside" (that kernel is chosen by the IR ops' priority, which the
table did not yet turn); with the table fixed, 3 of 3 were told apart correctly, and in a pre-registered replication
on another model, 3 of the 3 the path check could see (one planted fault was too small for it). The self-check at start costs 1.9-7.7 s on those models (9-26% of the
load); `ENTAIL_NO_PATHS=1` turns it off.

### Official DLCs

Engine-specific checks and repairs are packages outside the core, found through the entry point group `entail.dlc`
and installed by the core the way it installs its own adapters (a failure is recorded, and the program goes on).
The first is `entail-dlc-comfyui` (`dlc/comfyui`), the repair of ComfyUI's own defect #16490, which lived in the core
until 1.3. With it, M6's measurement gave 12 of 12 images pixel-equal to the core's repair of 1.3; without it the
leak is back; in two normal workflows it changed no image and said nothing. `ENTAIL_DLC=off` leaves every DLC out,
`ENTAIL_DLC=name,name` attaches only those (the rest are not even imported). The DLC is not on PyPI; it installs
from this repository:

```bash
pip install "git+https://github.com/wwoosshh/entail@v2.1.0#subdirectory=dlc/comfyui"
```

### Custom nodes

Put a check of your own on a point of your program:

```python
import json
from entail import nodes

@nodes.validator("json_object")
def json_object(value, keys=()):
    try:
        obj = json.loads(value)
    except ValueError as e:
        return nodes.broken(f"not JSON ({e})")
    missing = [k for k in keys if k not in obj]
    return nodes.broken(f"missing keys {missing}") if missing else nodes.ok()

@nodes.watch("app.answer", json_object, keys=("title", "body"))   # checks what ask() returns
def ask(question): ...
```

What a validator finds is one of entail's decisions (recorded, shown as the node "app.answer" on the page, which can
turn the node off). A check costs about 5 µs. A validator that raises, is slow (50 ms by default; `budget_ms=`) or
never returns (`hard=True` stops waiting at the budget) is recorded and left out; it never breaks your program.
Packages of validators attach through the entry point group `entail.nodes`; `entail-nodes-basics`
(`workshop/basics`) has `json_object`, `max_chars`, `within_context` and `same_size`. Three small apps are in
`examples/custom_nodes`. The package is not on PyPI either:

```bash
pip install "git+https://github.com/wwoosshh/entail@v2.1.0#subdirectory=workshop/basics"
```

entail does not sandbox a DLC or a validator: it is code in your program's process, with its rights. It keeps them
from breaking the run and from changing the core's rules; which code runs is your choice of what to install and list.

## What it does

**Resolves**

| mismatch | resolution | measured |
|---|---|---|
| a RoPE value given under its transformers-4 name after the config is built (`rope_theta`, `rope_scaling` — keyword to `from_pretrained`, attribute, vLLM `--hf-overrides`, SGLang `--json-model-override-args`) | written where `config.json` would have put it, including per-layer-type RoPE (asks the config class) | equal to the `config.json` route on 4 model families × 2 routes × 3 values; GSM8K restored (table above) |
| an attention backend that drops a declared model property (e.g. Gemma 2 logit soft-capping on transformers `sdpa`, SGLang `flashinfer`) | switched to a backend measured to honour it (`eager`, `triton`) | tokens equal the reference run; cost 1.18× (transformers), 1.13× (SGLang, the backend's own price) |
| **ComfyUI, diffusers:** a LoRA that cannot reach the model it is applied to (e.g. an Anima LoRA in an SDXL workflow), or reaches only part of it. ComfyUI skips each module with a console line and the run "succeeds" with the LoRA doing nothing | nothing can convert it: reported with how many of its modules reach the model and what the LoRA declares it was trained for, while the run goes on (`ENTAIL_ON_BROKEN=stop` stops before sampling) | ComfyUI 0.34.1: an Anima LoRA on an SDXL model left the images pixel-identical to no LoRA; entail reported it at the LoRA load, and with `ENTAIL_ON_BROKEN=stop` stopped before sampling (3/3); the right LoRA (15-31/255 of change) passed, with identical images with entail on and off. diffusers 0.40: a LoRA whose keys it does not read loaded and did nothing (pixel-identical images); reported the same way. Earlier, with 0.3.0's check: 22 right pairings passed with no false alarm |
| **ComfyUI, diffusers:** a v-prediction checkpoint that declares it in a way the engine does not read. ComfyUI reads only a `v_pred` key and samples a checkpoint that states `modelspec.prediction_type = v` in its metadata as eps; diffusers' single-file loader reads neither and falls back to epsilon. The images come out broken while the run "succeeds" | the file's own declaration (metadata, marker keys) or a pinned manifest decides: the sampler is set up for it, as a ModelSamplingDiscrete node or a rebuilt scheduler would, and what the declaration leaves open (zero-terminal SNR) keeps the engine's value. On diffusers this is decided when the pipeline samples, against the scheduler it samples with (a program that already set it up as declared is left alone); a program that runs the model in a sampling loop of its own is told, not changed. A sampling node or scheduler the user set is not overridden: the contradiction is reported. A checkpoint that declares nothing is reported as unknown: how the model behaves is never the basis for a change, so a checkpoint whose marker was lost needs a manifest | ComfyUI 0.34.1, AstolfoCarmix-VPredXL (declares v in its metadata, has no marker key): 83-95/255 from the author's reference setting without entail; with it identical, 0.16 and 0.14/255 over three seeds. diffusers 0.40 single files: NoobAI-XL-Vpred 55-83/255 from its reference without entail, pixel-identical with it; AstolfoCarmix 90-95/255, pixel-identical. entail 0.3.0 judged by the first model call instead and missed AstolfoCarmix (it behaves like eps at the noisiest step); a checkpoint whose marker was removed is now reported as unknown unless a manifest declares it |
| **ComfyUI:** a sampling node's schedule that outlives its workflow. ComfyUI's dynamic VRAM loader backs model buffers up by attribute path, so after a run with a ModelSamplingDiscrete (or similar) node the checkpoint keeps sampling with that node's schedule once the node is gone, and a node used after a plain run silently gets the plain schedule | each sampling object keeps a copy of the schedule its own setter registered; the loader's backup goes back to the object it came from instead of into another; the first model call after the buffers change checks them against that copy and puts them back | ComfyUI 0.34.1: after one run with a ModelSamplingDiscrete(v_prediction, zsnr) node on waiIllustrious, plain runs came out as another image (55.8/255) and then black, with entail on or off, until a restart; with the fix they match a fresh session pixel for pixel (3/3). NoobAI-XL-Vpred with and without a zsnr=false node, both orders: without entail the later runs took the other setting pixel for pixel; with entail all 12 images match a fresh session. The first-call check alone (guard left out) prevents the black images but leaves 2-11/255. An Anima workflow and all other runs are identical with entail on and off, at the same speed |

| **vLLM server:** a tool-call parser that does not read the format the model declares (a Qwen3 model, which emits hermes calls, served with `--tool-call-parser pythonic`): the call comes back as plain text | switched to a parser measured to read the declared format | vLLM 0.30.0, Qwen3-4B: the tool call is returned as a structured call again |
| **diffusers:** a VAE loaded on its own takes another model's latent scale (an SDXL VAE read as SD1.5's) | the scale a manifest declares for the model is applied | 15-16/255 from the reference image without entail, pixel-identical with it (3 seeds) |
| **vLLM:** a prefix-cache block hash that no longer stands for the tokens it was made from. A streaming-session update truncates a request's tokens but its block hashes are only ever appended, so a hash chained over discarded tokens survives, and a later request whose prefix matches the OLD tokens is served the NEW tokens' KV (vllm#49377, #49449; live in 0.30.0) | the stale hashes are forgotten from the first stale block on and the engine remakes them from the current tokens | vLLM 0.30.0, SmolLM2-135M-Instruct: without entail the rebuilt session got a false 16-token cache hit and a wrong continuation; with entail the hash is caught at the update, recomputed, and the output is the correct recomputed one |
| **SGLang:** a block-FP8 kernel whose K tile is not a divisor of the weights' quantization block, so the scale steps once per tile and skips blocks (a hand-supplied config, sglang#39626; also one shipped H100 fused-MoE config for E=512, N=256 with BLOCK_SIZE_K 256 over a block of 128) | the tile is clamped to the block, the engine's own default | SGLang 0.5.20: the dense kernel returned 64 where 288 was right, 288 with the clamp; the shipped MoE config gave 256 where 512 was right at the kernel level, 512 with the clamp. All 1,538 other shipped block-FP8 entries divide, so ordinary runs decide nothing |
| a generation that runs past its end because the file the engine reads for its stop ids is not the file that declares them. generation_config.json, config.json and the tokenizer each declare where a generation ends, and transformers reads only the first, vLLM the first plus the tokenizer, SGLang the first two (the April-2024 Llama 3 shape: config.json named one end, the model emitted another) | the ids the other files declare are added to the engine's stop set at load | transformers 5.17, Llama-3.2-3B-Instruct with a generation_config.json that lists only `<\|end_of_text\|>`: without entail all three test answers ran to the 160-token limit past their `<\|eot_id\|>`; with entail the end config.json declares was added at load and the answers stopped at 8, 18 and 37 tokens |
| a LoRA adapter served with the wrong scale, because its `adapter_config.json` declares a setting the engine never reads (`use_rslora`: PEFT scales by `lora_alpha / sqrt(r)`, SGLang always by `lora_alpha / r`, so the adapter is 4x too weak at r=16 and 8x at r=64; the same file can declare `rank_pattern`, `alpha_pattern`, `lora_bias`, `modules_to_save`, which vLLM and SGLang drop as silently) | every key of the file against the keys each engine reads (a table with code lines); a dropped key is reported, and carried where the engine has a place for it: SGLang's adapter scaling is set to what PEFT would use | SGLang 0.5.20, sglang#40835's own script (a PEFT rsLoRA adapter, r=64 alpha=128): without entail SGLang held scaling 2.0 where PEFT holds 16.0; with entail 16.0. A real Engine with a PEFT adapter (r=16 alpha=32) on Qwen2.5-3B-Instruct: resolved to 8.0; the same adapter without `use_rslora`, and both on vLLM 0.30: no decision but pass |
| a request's `chat_template_kwargs` setting honoured by the chat template under one name and read by the reasoning parser under another: `{"enable_thinking": false}` on Kimi K2 with vLLM 0.2x - the template injects no thinking tokens, the parser reads `thinking`, defaults to on, and returns `content: null` with the answer under `reasoning` (vllm#43728) | the names each reasoning parser reads, per vLLM version (a table with code lines), against the request's names and the template's variables; the request's value is handed to the parser under a name it reads | the rule on the 0.22.0 row of the table: resolved (`thinking` set from `enable_thinking`); vLLM 0.30.0 reads both names for Kimi K2, so there the same request passes. Retrospective, not a detection |
| a prefix-cache key that omits a field that shaped the cached block: vLLM 0.30's block hash covers the tokens, the prompt-embeddings digest, multimodal hashes, the LoRA name and the cache salt, but not `prompt_is_token_ids` (which positions take the embeddings), so a request that differs from an earlier one only in that mask is served the earlier request's KV (vllm#56655, fix unmerged) | the request's input fields against the fields the hash reads (a table with code lines); the missing field's per-block digest is added to the hash's extra keys and the request's hashes are remade | vLLM 0.30.0, Qwen3-0.6B, the report's own script: without entail B after A hit 32 cached tokens and produced A's output; with entail B hit 0 and produced its own, while A after A still hit 32. Retrospective |
| a beam search that reorders only the cache it knows by name: transformers 5.12.1 reordered `past_key_values`, so a Mamba model's `cache_params` stayed unmoved and the beams continued from other beams' states (transformers#46612) | the model's cache kwarg names against the names the reorderer touches (per transformers version) | 5.12.1: reported at the beam-search boundary (no repair; the output stays wrong); 5.17.0 reorders every name: pass. Retrospective |
| a tensor strided in its innermost dimension handed to a Triton kernel that reads it as if it were contiguous: SGLang's `fused_gdn_gating` was given the two halves of a split view (stride 2) and computed its gates from interleaved values (sglang#21843) | every `@triton.jit` kernel launched eagerly in the process, once per kernel and stride pattern (a kernel's first eight strided patterns), engine-independent: a strided innermost dimension against the kernel's own arguments. A kernel told the stride under any name passes; one that names strides but was not told this one is `unknown`, said once; one with no stride-like parameter and no equal integer is reported (`broken`, it cannot know) | SGLang 0.5.20, sglang#21843's tensors: the kernel takes row strides, so entail says `unknown` at the kernel's boundary for each tensor and the run goes on. 19 vLLM 0.30 kernels and 5 SGLang kernels in ordinary runs: pass. Retrospective, a report only |
| a rotary embedding applied with the wrong pairing of dimensions: GLM models pair `2i` with `2i+1` (interleaved) where Llama pairs `i` with `i + d/2` (split), and vLLM's Triton MRoPE kernel paired split-wise whatever the layer said until 0.27, so GLM-OCR produced garbage (vllm#42016; also #49290, and a draft model that did not inherit its target's pairing, #53063) | the pairing a config key or the architecture's reference implementation declares (a table with configuration- and modeling-file lines) against the `is_neox_style` of the rotary modules vLLM built for the language model, set to the declared one where they differ (the repair is exercised in unit tests only); a kernel path that ignores the layer (vLLM's MRoPE kernel before 0.27, when the module dispatches to it) is reported. A DSA indexer, which pairs by its own key, and a language model whose modules pair both ways are left alone | vLLM 0.30.0: GLM-OCR's two text rotary modules pair interleaved as declared and its vision tower's 26 pair split by their own reference: pass, nothing touched, output identical with entail on and off; Qwen3-0.6B and Qwen3-4B (split): pass. vLLM 0.22.0 (before the kernel fix), the report's own images: GLM-OCR emits garbage and entail reports `broken` at the load boundary, naming the kernel path; no repair exists there, so the output stays wrong while the run goes on. Retrospective; the first real run compared the vision tower against the text declaration and was corrected before this row was written |
| a tokenizer built from the right file by the wrong class, so the ids it produces are not the ids the folder's tokenizer.json gives: transformers 5.10.2 built deepseek-coder's tokenizer as `LlamaTokenizer` (transformers#46489), 5.8.0 built Granite's as `GPT2Tokenizer` and lost its pre-tokenizer (#45812), 5.4.0 converted Kimi-K2.5's tiktoken tokenizer and gave 18 of its 23 declared added tokens other ids (`</think>` got `<\|media_end\|>`'s; #45356), 5.12.1 replaced DeepSeek-R1-Distill's declared class (#46710); every one passed the vocabulary-size check | reported at the tokenizer's load: the declared tokenizer is run on ten fixed probe texts and its declared added tokens are looked up, and the ids must be the same (no repair: the decision names the class and the first text that differs, so the declared tokenizer can be loaded directly). A difference the folder's own declared flag explains (`legacy: false` next to a legacy-export tokenizer.json, on text after a special token) is said as the sources disagreeing, with both id lists | Retrospective (the rule was written from these four bugs; not a detection rate): at their reported versions all four are `broken` at the tokenizer boundary (9, 4 and 9 of the 10 probe texts; 18 of 23 added tokens); on 5.17.0 three pass and Kimi-K2.5 is `unknown` (tiktoken: its 23 added tokens match, the texts are not compared). 38 popular folders on 5.17.0 (11 distinct tokenizers; 23 share Qwen2's): 36 pass, 2 `broken`, both the one Llama-2-era tokenizer in the set (TinyLlama-1.1B-Chat and a tiny test folder): transformers 5 rebuilds such a tokenizer.json as Metaspace and never doubles the `▁` before text that starts with whitespace, whatever `legacy` says, so `"   leading spaces"` gets other ids than the file and than sentencepiece give. The 300-folder static corpus on 5.17.0: 201 pass, 9 `broken`, 5 `unknown`, 85 with no tokenizer to compare; the 9 are six of that Llama-2 shape (CodeLlama-7b-hf, EuroLLM-22B-Instruct, MiniCPM-SALA, ...) and three folders that declare `LlamaTokenizerFast` over a byte-level BPE tokenizer.json - **DeepSeek-R1-0528-Qwen3-8B**, deepseek-coder-7b-instruct-v1.5 and an MLX export - which 5.17.0 builds as a Llama pipeline: `"How are you doing?"` becomes `How are y oud o ing ?` and decodes to `Howareyoudoing?` (the #46710 class, live on 5.17.0 for these folders; draft report in the research workspace, not posted). Cost: the first process on a folder +241 ms at the median (+894 ms at most: the reference is built), later processes +2 ms at the median, +36 ms at the 90th percentile |
| a custom op's kernel that computes other values than the op's own native definition: vLLM's Triton MRoPE kernel before 0.27.0 paired rotary dimensions split-wise for a model that pairs them interleaved (GLM-OCR, vllm#42016), and the layer carried both a `forward_native` that was right and a `forward_cuda` that was not | decided on the engine's warm-up calls (made-up token values with the engine's own shapes, once per size class) and otherwise at the first real call of each op (per class, configuration and input pattern): the kernel and the definition are run on a 64-row slice of the same input, cut before the kernel touched it, and every value must agree within the definition's own rounding noise at that value plus a few units in the last place. A mismatch is repaired by sending every module of that configuration to its native definition, when no CUDA graph has captured the kernel at a size not yet verified; otherwise it is reported. Ops that hold the engine's state, ops under tensor parallelism or torch.compile, and ops not reached from the model's modules are said `unknown` once | vLLM 0.22.0, GLM-OCR: the kernel differs at `MRotaryEmbedding` (max difference 10.5 at an output scale of 10.9, allowed 0.588), with no architecture table; in eager mode it is resolved and the output then equals the native mode's; vLLM 0.30.0: pass. 8 popular models on vLLM 0.30.0 with `enforce_eager`: 35 decisions, 34 pass and one `unknown` (ops held by helpers that are not modules), every value within a tenth of its allowance; only 13 of the 34 compare an independent kernel (rotary, activations), the other 21 hold the definition against itself and say so (in eager mode vLLM's RMSNorm path is its definition; the fused kernels live under torch.compile, not compared). Retrospective: the rule was written from this bug; the allowance's factors are headroom, to be set from data |
| a chat parser whose streamed message is not what its parse of the same complete text gives, or whose tool calls carry arguments the request's tool schema does not have: vLLM's kimi_k2 parser skipped the schema's type coercion while streaming (`"3"` against `3`, vllm#49316), the qwen3 parser gave different content around tool calls on the two paths (#49412), the deepseek_v4 parser unwrapped one tool's arguments with another tool's schema (#47986) | reported when the stream finishes: the deltas are accumulated and a fresh parser of the same class parses the whole text; the two must agree exactly (a difference on an output that did not finish by itself is `unknown`), and each tool call must name a declared tool, carry its required parameters and, under a tool that forbids additional properties, no other key (no repair: the client already has the stream) | vLLM 0.30.0's own parsers driven as the server drives them: `broken` on 4 of 4, 2 of 3 (the third differs in surrounding whitespace only, which is noted, not broken) and 1 of 1 texts of the three reports, nothing on 6 well-formed texts; a live vLLM server with every adapter on (Qwen3-0.6B, qwen3 reasoning parser, hermes tool parser, 18 streamed and whole requests: plain, thinking on and off, tool calls, `tool_choice: none`, n=2, cut by length): nothing broken. Retrospective: the rule was written from these bugs |
| a multimodal placeholder bound where the model's declared markup puts none: a literal `<\|image_pad\|>` typed in the user's message tokenises to the placeholder id, vLLM binds the image to that first run, and the template's own `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` slot keeps one raw pad token, so the model answers about the wrong span (Qwen2.5-VL, vllm#57740) | reported when the processor has placed the placeholders: the token before each image run must be the `vision_start_token_id` the model's config declares (a model that declares no markup decides nothing; no repair: the prompt is the user's) | vLLM 0.30.0, Qwen2.5-VL-3B-Instruct, the report's two message orders: the attack order `broken` ("the image placeholder bound at tokens 20..275 is preceded by id 220, not by the declared start of the markup"), the control order pass; vLLM's own profiling prompts are not decided. Retrospective |
| a chat completion whose logprobs cover other text than its content: SGLang with `separate_reasoning` returned `logprobs.content` over the whole raw output, `<think>` span and markers included, while `message.content` held the parsed answer, so the two cannot be aligned (sglang#25055) | reported when the response is built: the logprob tokens must decode to the message's content (surrounding whitespace aside) | SGLang 0.5.20, Qwen3-0.6B with the qwen3 reasoning parser, `logprobs: true, separate_reasoning: true`: `broken` ("the 155 logprob tokens cover the reasoning span (545 characters and its markers) as well as the content (12 characters)"). Retrospective |

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
- where transformers applies a chat template - a script's `apply_chat_template`, SGLang's server - the template
  and the reasoning history the same way, and SGLang's own conversation templates (`--chat-template chatml`)
- the tokenizer the engine built against the model's vocabulary: ids past the embedding, and a folder that
  carries two vocabularies (vocab.txt of 100,000 next to a tokenizer.json of 32,000; transformers#48967) where the
  engine built the one that is not the model's - reported at the tokenizer's load, refused before the first id
  under `ENTAIL_ON_BROKEN=stop`, and exit 1 from `entail check`
- the tokenizer the engine built against the tokenizer the folder declares, run: ten fixed probe texts encoded by
  both (tokenizer.json through the `tokenizers` library; a sentencepiece-only folder is `unknown` until that
  reference is measured), and every declared added token looked up, ids compared exactly - a class that rebuilds
  the tokenizer with its own pre-tokenizer is reported at load, refused before the first id under
  `ENTAIL_ON_BROKEN=stop`, and exit 1 from `entail check`; a declaration that cannot be run is said as `unknown`
  once, and a difference that the folder's own declared flag explains (a legacy-mode tokenizer.json next to
  `legacy: false`, on text after a special token) is the sources disagreeing, said with both id lists
- on vLLM, every custom op that dispatches to a kernel against the op's own native definition, once per op class,
  configuration and input pattern, on vLLM's own warm-up calls used as probes (made-up token values, the engine's
  shapes, once per size class) or else on a 64-row slice of its first real input; a mismatch is sent to the
  definition when no CUDA graph holds the kernel (the definition is run in the input dtype and in float32 and the
  op's tensors are put back; every
  value of the kernel's output must be within the definition's own rounding noise at that value plus a few units in
  the last place); ops that override their forward or hold the engine's state, every op under tensor parallelism
  or torch.compile, ops not reached from the model's modules, calls during CUDA-graph capture and definitions that
  refuse the input are not compared, each said once
- on vLLM's chat server, a parser's streamed message against its parse of the same complete text when the stream
  finishes (a difference on an output that did not finish by itself - the token limit reached, a reasoning block
  left open - is `unknown`, not broken), and every tool call's arguments against the tool the request declared (a
  required parameter missing, or an extra key under a tool that forbids them)
- on vLLM, where each multimodal item's placeholder was bound against the markup the model's config declares
  (`vision_start_token_id` before the image tokens); on SGLang's chat server, a completion's logprobs against its
  content
- on vLLM's scoring path, the token type a cross-encoder's padding is given against the tokenizer's declared pad
  type (vllm#58138: the padding got the document's segment and the /rerank scores moved); vLLM 0.30 keeps token
  types in a form that cannot carry the repair, so this is reported, and refused under `ENTAIL_ON_BROKEN=stop`
- statically, per engine, a LoRA adapter folder's `adapter_config.json` against the keys that engine reads of it
  (`entail check <adapter folder>`: a dropped key is reported, and said resolved where the engine's adapter carries
  it at load).
- statically, per engine, the stop set each engine would build from a model folder against every end its files
  declare (`entail check`)
- every Triton kernel launch in the process, engine-independent, once per kernel and tensor layout: a tensor
  strided in its innermost dimension against the kernel's own parameter names (a kernel with no stride parameter
  cannot know: reported; one that takes strides but was not told this one: `unknown`, said once). Such a launch is
  also run twice on copies, as given and with that tensor laid out again; if the kernel writes other values, that
  layout pattern is launched on relaid copies from then on (repaired). Compile-only warm-ups are not looked at, and
  a kernel is looked at for its first eight layouts only
- five engine functions that carry no definition of their own, against entail's: vLLM's `fused_experts`,
  `w8a8_triton_block_scaled_mm`, `prepare_pos_seq_lens` and `BlockTables.compute_slot_mappings` (positions and KV
  slots compared exactly), SGLang's `fused_gdn_gating`; a mismatch is sent to the definition
- at start, the engine's own paths on three fixed probe requests: decode against a fresh prefill of the same
  tokens, a request alone against the same requests batched, a cold run against one that reads the prefix cache (a
  confident prediction that changes, or a probability that moves by more than 0.25, is reported). On vLLM by
  default (`ENTAIL_NO_PATHS=1` turns it off); on SGLang only with `ENTAIL_PATHS=1`, because SGLang has run no
  prefill yet and a probe could be the request that meets an engine defect first
- the pairing convention of a model's rotary embedding (split, `i` with `i + d/2`; or interleaved, `2i` with
  `2i+1`), declared by a config key or by the architecture's reference implementation, against the layers vLLM
  built for its language model, and against a kernel path that pairs split-wise whatever the layer says (vLLM's
  Triton MRoPE kernel before 0.27: GLM-OCR produced garbage, vllm#42016)
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
  the model folders declare reached a decision where it was used, the chat template included.
- **Wider healthy runs, for 1.0.1** (38 popular models that fit one 12 GB card, chosen by download rank, on the
  same three engines; 102 valid runs): 1.0.0 reported something wrong in 17 of the first 81 runs, none of it a
  real loss - the five causes are in the changelog. With 1.0.1: no `broken`, no `refused`, two repairs (Gemma 2's
  soft-capping again, measured rows), every output identical to the run without entail, load cost median
  0.7-0.9%. What 1.0.1 cannot decide it now says as `unknown` (69 lines over the 81 runs, one per boundary).
- **Real bugs, for 1.1.0:** of 8 reported bugs replayed from a random sample of 229 engine issues, 4 were of
  the class entail targets and 1.0 passed all 4; 1.1.0 repairs 2 (vLLM's stale block hashes, SGLang's kernel
  tile) and reports the other 2 at their boundary (the padding token type, the second vocabulary). Those 4 are
  the bugs the 1.1.0 facts were written from, so "4 of 4" is not a detection rate. The 3 out of
  the class (a CUDA-graph weak reference, a parser's streaming logic, a scheduler's arithmetic) are, as designed,
  not flagged. A fifth, the stop-id class, was replayed on transformers (table above).
- **Detection rate on unseen bugs (pre-registered, vocabulary frozen at 1.1.0):** 150 further issues from the
  same four repositories were screened by fixed rules; 17 passed, 15 reproduced here, and two blind raters put 7
  of the 15 in entail's class (agreement kappa 0.86 over seven categories, 0.95 for in-class versus not). entail
  detected **0 of those 7** and raised no false alarm on the 8 outside the class. Every miss was a fact outside
  the vocabulary, and in 5 of the 7 the defect sits where no adapter checks anything yet (a kernel call, a request
  parser, a LoRA config file, a prefix-cache key, beam reordering). Read the class claim accordingly: the five
  facts above are measured on the bugs they came from; a new bug is usually a fact entail does not read yet.
  Protocol, screening log, ratings and case scripts: `testbed/M16_PROTOCOL.md` and `testbed/results/m16/` in the
  research workspace (github.com/wwoosshh/entail-research). The 38 popular models on
  three engines again (102 valid runs): no `broken`, no `refused`, the same two repairs plus two where a declared
  end was added to transformers' stop set (Nemotron-3-Nano-4B as shipped, and a tiny test model), outputs identical
  in 97 of 98 comparisons without a repair (the one difference is an engine's own nondeterminism), 69 `unknown`
  lines in all, the library's share of load time 1.4% at the median and 8.2% at the 90th percentile (measured
  again with the six rows above added: the same runs, the same repairs, the same 69 lines, nothing new said).
  Statically over 230 popular model folders: no false `broken` from the new facts.
- **Second replay (pre-registered, vocabulary frozen at 1.2.0's code, commit `ce79b19`):** the next 150
  issues in the same fixed order were screened by the same rules; 20 passed and 15 reproduced here (six
  environments at the reported versions were built for them). Two blind raters put 8 of the 15 in entail's class
  (agreement kappa 0.72 over seven categories, 0.68 for in-class versus not; a third blind rater settled the 18
  of 86 items they disagreed on). entail detected **0 of those 8** (rule of three: at most 3 of 8 at 95%),
  raised no false alarm on the 7 reproduced outside the class, and broke one run itself (a wrapper with a fixed
  signature on vLLM 0.23.0; fixed in 1.2.0). The misses: three tokenizers whose built class encodes
  differently from the folder's tokenizer.json (the Vocab check compares sizes, not ids; one of the three was said
  `unknown` at the tokenizer boundary), and five at sites with no boundary (a kernel's scale layout, a
  linear-attention kernel's input layout, a tool parser, a multimodal placeholder's binding, a response's
  logprobs). Class share among the 86 rated reports: 31 (36%). The raters were agents run from the research
  session, as in the first replay. Protocol section 7, screening, ratings and cases: `testbed/results/m17/replay2/`
  in the research workspace.
- **Healthy runs, for 1.3.0** (the same 38 models on the same three engines, 102 valid runs, code frozen at
  `2aa975b`): entail broke no run, no check added since 1.2.0 said `broken` or `refused`, and outputs were identical
  in 97 of 98 comparisons without a repair. Six runs say `broken` at the tokenizer boundary, all from two
  Llama-2-era folders on every engine: a real difference, not a false alarm (transformers 5 builds these tokenizers
  so that text starting with a space loses one space, where the folder's tokenizer.json, the model's sentencepiece
  file and transformers 4.57 agree; transformers#47700 describes it). Before the freeze one run broke with entail on
  (SGLang, Phi-3.5-mini-instruct): the start-up probes were the engine's first prefills and met an engine defect
  (flashinfer's state merge does not take head_dim 96; any prompt of 128 tokens or more stops the scheduler, with
  entail off as well). SGLang's path check is opt-in since.
- **Third and fourth replays (pre-registered, code frozen at `07fceac` and at 1.3.0's `2aa975b`):** the third
  screened the next 150 issues of the same order (10 passed, 8 reproduced), the fourth a new population of 437
  issues from the six months before the first (150 screened, 10 passed, 9 reproduced). Two blind raters, a third
  settling their disagreements, put 5 and 6 of the reproduced in entail's class (kappa 0.735 and 0.826 over seven
  categories, 0.776 and 0.916 for in-class versus not); they also rated whether a defect sits in the data and the
  computation itself or around them. entail detected **0 of 5** and **0 of 6** (0 of 2 and 0 of 5 in the data and
  computation), raised no false alarm in the third and one in the fourth (fixed in 2.0.0). Class share among the rated
  reports: 36 of 96 (37.5%) and 25 of 91 (27.5%). Over the four replays: 0 of 26. Protocol sections 8 and 9,
  screening, ratings and cases: `testbed/results/m18/replay3/` and `testbed/results/m19/replay4/`.
- **31 test problems** (16 reproduction cases, 8 field cases, 7 simulated market incidents): each defect was
  repaired; where no repair exists, it was reported at the boundary and fact where it happened while the run
  went on, or stopped with `ENTAIL_ON_BROKEN=stop`. No fixed version was flagged. (The two ComfyUI cases were
  measured before 1.0 and not run again.)
- **Cost, for 1.3.0:** request throughput on vLLM's default path (torch.compile and CUDA graphs, Qwen3-4B) with
  everything on against entail not installed: 1.0007x, 1.0028x and 1.0110x at batch 1, 8 and 32 (two identical
  states differ by up to 0.6%; at batch 32 each of the four rounds was 0.5-2.0% slower with entail on). At load,
  from an installed copy, vLLM +1.6-1.7 s for 0.6-3B models (13-15% of their load), mostly the start-up path check
  (`ENTAIL_NO_PATHS=1` turns it off); the hook alone -0.03 to +0.25 s.
- **Cost, for 1.0:** at load, 0.3-2.6% of the load time. Always on, vLLM's CUDA-graph path 1.008x, 1.003x and 1.006x at
  batch 1, 8 and 32 with every adapter on (control runs without entail: 0.997-1.004x; measured before the
  per-request boundaries were added: 0.999-1.000x), transformers' dynamic KV cache 1.017-1.022x of an eager decode. About 60 us per request on
  vLLM's server. The diagnosis mode 1.74x (eager) and 1.85x (sdpa). With `ENTAIL` unset, 0.2-0.3 ms per Python start
  and no module imported.
- **Next to post-hoc detection:** GSM8K (500 problems, greedy) caught the RoPE loss above and a planted weight
  shift, but not a backend that drops Gemma 2's soft-capping: SGLang `torch_native` 313 against `triton` 316
  (McNemar p = 0.66); measured before 1.0, transformers `sdpa` against `eager` 337 against 339 (2B) and 442
  against 443 (9B).
  Comparing the outputs with a healthy run found it (198 of 500 answers differ, none between two healthy runs),
  where such a run exists. entail repairs it at load.
- **Locating:** 11 of 11 planted defects located.

## Known gaps

- **Unseen bugs.** On the pre-registered replay above, 0 of 7 in-class reproduced bugs were detected. Since then
  the facts and sites it exposed have been added (the last six rows of the repairs table: the adapter config file,
  the reasoning parser's setting names, the prefix-cache key, beam reordering, a Triton launch's strides, the
  rotary pairing), and what a routing weight's row is remains unread. Run again with the new vocabulary in the
  reported versions, the seven come out as: two repaired (the LoRA scale, the prefix-cache key), two reported
  (beam reordering on transformers 5.12.1, GLM-OCR's pairing on vLLM 0.22.0), one `unknown` at its kernel (the
  strides), one resolved by the rule but at a site vLLM 0.22.0 does not have (the parser's setting name), and one
  not read at all (the Marlin MoE row). That is a retrospective, not a detection rate; the second pre-registered
  replay above, with this vocabulary frozen, detected 0 of 8. What it left unread: a kernel's scale layout, a
  linear-attention kernel's input layout, a tool parser's slots, a multimodal placeholder's origin, and what a
  response's logprobs cover. The ids a built tokenizer produces (three of the eight) are read since 1.3.0 by the
  tokenizer check above, which is written from those bugs and is not a detection rate. The third and fourth
  pre-registered replays (code frozen at `07fceac` and at 1.3.0's `2aa975b`) detected 0 of 5 and 0 of 6, and the
  checks added last for 1.3.0 (warm-up probes, the Triton launch run twice, the start-up path check, the exact index
  definitions) reached none of the fourth replay's defect sites. What they left unread: a multimodal processor's merged settings,
  a reasoning parser's unfinished output (said `unknown` at the site, by rule), the rotary layout inside
  transformers' Qwen2.5-Omni DiT (the pairing rule is vLLM-side), a tokenizer whose pre-tokenization differs only on
  characters none of the ten probe texts holds (combining marks), a KV connector's recompute path, a scheduler step
  that does not use its declared timestep spacing, checkpoint weights a tied-weights mapping left unloaded, a LoRA
  adapter's module paths, a C++ kernel's input scale (Marlin), a GGUF tokenizer's declared type and cross-attention
  metadata captured in a CUDA graph. A multimodal model's vision tower is not compared for its rotary pairing (its
  reference pairs on its own terms); only the language model is.
- **Where the comparisons do not reach:** on vLLM's default path (torch.compile) custom ops are compiled and not
  compared with their definitions (the comparison runs in eager mode), and kernels called from C++ (Marlin) are not
  reached at all. SGLang's start-up path check runs only with `ENTAIL_PATHS=1`.
- **Models loaded by hub id** are read from the snapshot the engine downloaded into the local cache (fixed in
  1.2.0: huggingface_hub refused such snapshots as incomplete, so every hub-id load said "could not be checked"
  for Vocab and Stops; GLM-OCR by hub id on vLLM 0.30 now passes both). A model the cache does not hold at all
  is still `unknown` there.
- **Apps that sample in a loop of their own** (field test of 2.1.1): diffusers' prediction is decided when the
  pipeline samples, so a program that runs the model in a sampling loop of its own, with a scheduler it builds
  from its own model settings (InvokeAI), is not repaired: entail does not see that scheduler, says so once as
  `unknown` with the declared prediction, and changes nothing. Until 2.1.2 the repair was made at the load, where
  it changed SD.Next's images ([#21](https://github.com/wwoosshh/entail/issues/21)) and made InvokeAI's default
  sampler stop ([#24](https://github.com/wwoosshh/entail/issues/24)).

- The always-on KV contract on transformers' dynamic cache is at the edge of its target: 1.017-1.022x of an eager
  decode of Qwen3-4B, depending on how the runs are paired (target 1.02x; 1.041x before these fixes). On vLLM's
  CUDA-graph path it is within noise.
- A multimodal processor's `apply_chat_template` is not checked. A request SGLang's server refuses under
  `ENTAIL_ON_BROKEN=stop` gets SGLang's own error (500); vLLM's server answers 400.
- A RoPE declaration the vocabulary does not carry is reported as not compared: over 230 popular folders that is
  a per-layer-type split under names other than full/sliding attention (DeepSeek-V4), a local layer's own
  `partial_rotary_factor` (Laguna) and `attn_factor`, a name no engine reads.
- The padding token type on vLLM's scoring path is reported, not repaired: vLLM 0.30 keeps token types as the
  index of the first 1, which cannot hold a pad type after the document. Tokenizers built outside transformers'
  `PreTrainedTokenizerBase.from_pretrained` (Mistral's own files, tiktoken) are not checked at run time; one built
  from a GGUF file is `unknown` at the tokenizer boundary, because the GGUF file's own tokenizer declaration is not
  read (transformers#41494: a Gemma GGUF built as another tokenizer type is not caught).
- The stop-set check reads the tokenizer's end from tokenizer_config.json (and tokenizer.json) without building a
  tokenizer; a tokenizer whose files name its `eos_token` nowhere as an id is not seen there. A model saved after
  the repair (`save_pretrained`) writes the repaired list into its generation_config.json.
- A mismatch the capability table knows only from reading an engine's code (SGLang flashinfer's sliding window) is
  reported as inferred, not repaired; only measured rows switch a backend. Config keys outside the vocabulary that
  a model's config class does not take are reported as unread, not as lost.
- SGLang's KV contract skips speculative-decoding batches (the scheduler reserves draft slots ahead of the tokens)
  and says so once per process.
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
| `ENTAIL_SKIP` | e.g. `comfyui:install_nodes` | leave out these entries (a bare name leaves out the whole adapter), to measure the rest without them |
| `ENTAIL_VERBOSE` | `1` | print each adapter as it is installed |
| `ENTAIL_QUIET` | `unknown`, `start` (comma-separated) | `unknown`: keep non-blocking `unknown` decisions off the console; they stay in the log and the record, and the console says so once per process. `start`: leave out the line that says entail is on, and the closing line of a launch that recorded nothing |
| `ENTAIL_SOURCE` | `1` | also compare loaded weights with the checkpoint file (vLLM, a little I/O at start-up) |
| `ENTAIL_NO_PATHS` | `1` | vLLM: leave out the start-up comparison of the engine's own paths (most of entail's load cost there) |
| `ENTAIL_PATHS` | `1` | SGLang: run the start-up comparison of the engine's own paths (off by default: its probes would be SGLang's first prefills) |
| `ENTAIL_SAFE` | `auto` (default), `all`, `off` | the safety modes: `auto` narrows a disagreement of the engine's own paths to one optimization over the next starts; `all` turns every optimization the engine declares result-neutral off; unset, `entail_logs/safe_mode.json` (the page's settings) decides |
| `ENTAIL_DLC` | `off`, or names | official DLCs: all installed ones attach unless `off`; a list attaches only those (the rest are not imported) |
| `ENTAIL_NODES` | `off`, or names | custom nodes: `off` runs none; a list names the workshop packages that may attach. One node is turned off from the page (`entail_logs/nodes.json`) |
| `ENTAIL_MANIFESTS` | folders, separated by `:` (`;` on Windows) | where to look for manifests (`<sha256>.json`) of model files that do not declare what they mean. A file is hashed only when a manifest could be for it, and its hash is kept in `entail_hashes.json` in the first folder |

## Tested with

transformers 5.12.1, 5.16.1 and 5.17.0, vLLM 0.30.0, SGLang 0.5.20, diffusers 0.40.0, ComfyUI 0.34.1 (Windows),
torch 2.13–2.14,
Python 3.12, one RTX 4070 Ti (12 GB). Other versions may work; `entail doctor` prints what is installed. On transformers 4.x the RoPE resolver
has nothing to do and stays out of the way.

The capability table (which backend honours what) records its evidence per entry, and only entries marked
*measured* are used as resolution targets. The measurement scripts, the raw results and the documents this
repository's code refers to (`THEORY.md`, `LIBRARY_DESIGN.md`, `ROADMAP.md`) are published as the research
workspace at https://github.com/wwoosshh/entail-research (mostly in Korean; the numbers in this README are traced
to result files there, see its `testbed/results/m10/PUBLIC_CLAIMS.md`).

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
