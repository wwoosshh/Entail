# Changelog

## 1.0.1

Unreleased. False alarms found when 1.0.0 was run on 30 popular models, three engines each (81 runs: it never broke a
run and every output was identical, but 17 runs carried a report that was wrong; `testbed/results/m10/E2_SUMMARY.md`
in the research repository). Nothing in the vocabulary or the verdicts changes; what changes is what counts as
evidence at five boundaries, and how much is said.

**Fixed**
- Config keys (`load.config_keys`): a key the model's config class does not take was `broken` whatever it was, and
  popular models carry keys nobody reads (`swiglu_limit`, `task_specific_params`; 45 of the 81 runs). Now only a
  misspelling of a key entail's vocabulary maps is `broken` (`rope_scale` for `rope_scaling`: nobody can read it, so
  it is lost here). A key spelt right that the class does not take is decided where a consumer of its fact reads
  it; any other unread key is one `unknown` line that names the keys, blocking only in debug mode. On 215 unread
  keys of 92 popular models the new rule flags none; the rolebench 15 case stays `broken`.
- Tied embeddings (`load.tie`): vLLM 0.30 sets `tie_word_embeddings` to False when the checkpoint ships an
  `lm_head.weight`, loads it and re-ties it when it equals the embedding; transformers 5.17 compares the two the same
  way. 1.0.0 read the False as the loader's choice and reported Qwen3 0.6B, 1.7B and quantised exports (a stored copy
  of the tied head) as `broken`. Now the adapters read what the loader left in the model - the head sharing the
  embedding's tensor, or its own - after its step (vLLM after `process_weights_after_loading`, transformers at the
  `tie_weights` call that has the weights), and the checkpoint is only sampled. A copy satisfies a declared tie; a
  different head is what those loaders run, so the declaration is reported false, and with `use_data` the head that
  runs passes as the data used. The static check (`entail check`), which has no model in memory, compares the
  checkpoint's head with the embedding byte for byte (`observe.head`). SGLang 0.5.20 ties regardless, and is
  decided as before.
- Chat template (`readers.HfTemplate`): `chat_template.jinja` takes precedence over the entry in
  `tokenizer_config.json`, as transformers reads them; the entry is no longer a second declaration. A checkpoint
  whose two copies differed by blank lines only was reported `broken` at every request. An entry that differs from
  the file beyond blank lines and trailing spaces is noted as not what runs.
- Evidence for a repair (`load.attention`, `load.tool_parser`): a mismatch the capability table knows only from
  reading code (SGLang flashinfer's sliding window) is reported as `unknown` - "inferred from its code, not
  measured; nothing is switched on it" - instead of switching the backend on it. A declared sliding window that
  is not below `max_position_embeddings` (Phi-3.5, Phi-4-mini: 262144 over 131072) never binds and is not read as a
  requirement.
- SGLang KV contract under speculative decoding: the scheduler reserves draft slots ahead of the tokens, which the
  contract reported as reserved and written slots disagreeing at every step. Speculative batches are now skipped
  and said so once per process.
- diffusers pipelines named by a hub id are checked from the folder in the local huggingface_hub cache; 1.0.0 checked
  only local folders.
- Weights the layout step cannot read (a conv1d in a hybrid model) are one line per reason, not one per weight (42
  lines per load of Nemotron-H).

## 1.0.0

A redesign. 0.3.0 was a set of checks and resolvers for cases that had been measured one by one; 1.0 is one
mechanism for all of them: facts that say what a value means, read from what declares them, compared where they are
used, under one policy and in one record. Everything listed was measured on the engines and versions in the README
(one RTX 4070 Ti); the README's "How it was measured" and "Known gaps" give the results and the limits.

**Facts and verdicts**
- A closed vocabulary of what a value means (version 4): layout and quantization, RoPE (with llama3's frequency
  factors) and position frames, valid ranges and KV extents, model properties, prediction type and latent scale,
  chat template, key coverage, reduction state, epochs, assumptions and precedence. Each fact carries where it came
  from and how certain it is.
- Five verdicts at every boundary: `pass`, `resolved`, `broken`, `refused`, `unknown`.

**Where facts come from**
- Model folders and files: `config.json`, the tokenizer's chat template, safetensors metadata, GGUF keys, diffusers
  configs.
- Manifests for files that declare nothing: `entail infer` writes a draft, `entail pin` marks it reviewed; a manifest
  next to the file, or in a folder named by `ENTAIL_MANIFESTS`.
- Which consumer honours which fact is a table with its evidence; only measured entries are used for repairs.

**Where they are compared**
- At load: attention properties against each backend, RoPE (a declared key the vocabulary cannot carry is reported
  as not compared), config keys nobody reads, tied embeddings against the checkpoint, vLLM's weights after
  repacking against the signatures of the steps that repack them, weights against the checkpoint file
  (`ENTAIL_SOURCE=1`).
- In containers: the KV cache contract on transformers, vLLM and SGLang; buffers read after an in-place write; CUDA
  graphs and compiled code reused for inputs they were not made for.
- Per request, on vLLM's OpenAI server: the chat template, the reasoning history and tool-call format a model
  declares, request fields and template settings nothing reads. Where transformers applies a chat template - a
  script's `apply_chat_template`, SGLang's server - the template and the reasoning history; and SGLang's server
  when it renders with a conversation template of its own. The core also has a rule for the context a prompt
  needs against what the model declares; no engine adapter calls it yet.
- In code, in debug mode: `@entail.boundary` declares what each argument means; strided layouts, quantized values and
  chunk-relative positions are converted where the reader needs them.
- Image models on ComfyUI and diffusers: prediction type and latent scale against the sampler and the VAE, and whether
  a LoRA reaches the model it is applied to.

**Policy and record**
- A mismatch is repaired first. What nothing can repair is reported as `broken` and the run goes on;
  `ENTAIL_ON_BROKEN=stop` (or `Name=stop` in `ENTAIL_FACT_POLICY`) stops before any output. `ENTAIL_POLICY=refuse`
  repairs nothing and reports everything.
- Everything entail says goes to `entail_logs/` in the folder a program starts from: a log and a JSON record of every
  decision, from every process an engine starts.

**When the output is still wrong**
- `entail locate` names the first boundary that did not keep a fact, or says that every checked boundary held.
- `diagnose.watch` and `diagnose.compare` compare a layer with a reference on the same inputs; `diagnose.propagating()`
  follows declared facts through tensor operations and names the one that made a fact untrue.
- `pytest --entail` runs each test that way, and `entail_conditions` runs a test once per condition a model's
  declarations put at stake.

**For new code (experimental)**
- `entail.frontend`: role-typed values and keyword-only operations, checked when a program is traced; a Qwen3 decode
  step written with it, lowered to torch, FlexAttention or a Triton kernel.

**Changed from 0.3.0**
- A mismatch nothing repairs is reported and the run goes on; 0.3.0 stopped. Set `ENTAIL_ON_BROKEN=stop` to stop.
- A checkpoint that declares nothing is reported as unknown. 0.3.0 judged the prediction type from the first model
  call; a checkpoint whose marker was lost now needs a manifest.
- A LoRA that reaches only part of the model is `broken` (0.3.0: a one-line note), and a sampling setting the user
  chose against the checkpoint's declaration is reported instead of passed over.
- The fault injection, the layout ledger and the bookkeeping probe used to measure entail left the package, and with
  them `ENTAIL_SEED`, `ENTAIL_LEDGER` and `ENTAIL_PROBE`.

## 0.3.0

ComfyUI: a check for v-prediction checkpoints that lost their marker; a sampling schedule kept with the object that set
it (ComfyUI #16490).

## 0.2.0

ComfyUI: a check for LoRAs that cannot reach the model.

## 0.1.0

RoPE settings passed under their transformers-4 names, and attention backends that drop a declared model property,
resolved; start-up checks (attention properties, swallowed config keys, tied embeddings, vLLM's weight layout,
weights against the checkpoint) and the KV cache contract on transformers, vLLM and SGLang; the start-up hook that
reaches engine worker processes.
