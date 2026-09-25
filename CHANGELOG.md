# Changelog

## 1.0.2

Released 2026-09-25. What an external review of 1.0.1 found, and what its readers will ask first.

**Fixed**
- Config keys (`load.keys_taken`): a key the class does not take counted as renamed when its value appeared in any
  field the class knows, so an unread key holding a 1 or a `true` was silently taken as read, and a misspelt
  vocabulary key with such a value (`tie_word_embedding: true`) escaped the misspelling rule. Now a misspelling of a
  key the vocabulary maps is lost whatever its value holds, and a value counts as evidence of a rename only when it
  is distinctive (a float, a string of four characters or more, an integer of 256 or more). The names a class
  renames on the way in (`attribute_map`, GPT-2's `hidden_size` for `n_embd`) are taken by name, and the token ids
  and `use_cache` that `GenerationConfig.from_model_config` reads off any config are listed as read elsewhere. The
  stricter rule reports more keys as read by nothing entail knows (`head_dim`, `max_window_layers` on classes that
  keep them as plain attributes, which their model code may read): on the 230 popular configs, 165 carry one such
  `unknown` line instead of 92. `ENTAIL_QUIET=unknown` keeps it off the console.
- `entail check` builds the config with `trust_remote_code=False` and `local_files_only=True`: a model with its own
  code fails at once instead of prompting (which waited 15 s per model where there was no terminal), and nothing is
  fetched.

**Added**
- `ENTAIL_QUIET=unknown`: non-blocking `unknown` decisions go to the log and the record only; the console says so
  once per process. 1.0.1 printed one such line per boundary that could not be decided (69 lines over 81 loads of
  30 popular models).
- README: what the package does to an environment (the start-up hook and how to remove it, the log folder and how to
  turn it off, the engine versions each adapter was measured against, the import name); the Korean README carries
  the 1.0.1 and 1.0.2 changes.
- The research workspace behind the numbers - the measurement scripts, the result files, the theory, design and
  roadmap documents the code cites - is public at https://github.com/wwoosshh/entail-research.
- README (English and Korean) opens with the measured case (64 of 180 models, 379 → 273, 198 of 500 answers, 0 false
  alarms in 102 runs) and a three-line check of your own model; `CONTRIBUTING.md` says how to report a wrong report
  or a miss, add a measured capability row, and support a new engine version.

## 1.0.1

False alarms found when 1.0.0 was run on 30 popular models, three engines each (81 runs: it never broke a
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
