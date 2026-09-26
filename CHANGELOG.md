# Changelog

## Unreleased

**Added**
- A LoRA adapter's `adapter_config.json` as a declaration file (`adapter_config_contract.py`,
  `data/adapter_config_keys.json`): every PEFT key, which ones each consumer reads (PEFT for transformers and
  diffusers; vLLM 0.30's PEFTHelper; SGLang 0.5.20's LoRAConfig, with code lines), and one rule: a key declared with
  a value that changes how the weights apply, that the consumer does not read, is `broken` at that consumer's load
  boundary, or `resolved` where the consumer can carry it. Neutral values, keys the weights carry and training-time
  keys decide nothing; a key the consumer refuses loudly passes with a note. Adapters `sglang_lora` (carries
  `use_rslora` into the adapter's scaling, computed in the core: sglang#40835 served rsLoRA adapters 4-8x too weak)
  and `vllm_lora` (reports what vLLM drops: `rank_pattern`, `alpha_pattern`, `lora_bias`, ...). `entail check` on an
  adapter folder decides it per engine.
- A request's template settings at the reasoning parser (`request_contract.setting_names`,
  `data/request_settings.json`): a setting the template honoured under one name (`enable_thinking`) that the
  parser reads under another (`thinking`) leaves the parser on its default (vllm#43728: `content: null`). The
  table names, per vLLM version and reasoning parser, the names each parser reads (code lines); the vLLM serve
  adapter wraps the parser class the server builds per request and hands the request's value to the parser under
  a name it reads (`resolved`), or reports it. A parser that reads no name of the setting, or a setting the
  template did not read either, decides nothing.
- A store's key against the fields that shaped the item (`cache_key_contract.py`, `data/cache_key_fields.json`):
  vLLM's prefix-cache block hash keys a request by its tokens, embeddings digest, multimodal hashes, LoRA name and
  cache salt, but not by `prompt_is_token_ids` (which positions take the embeddings), so two requests that differ
  only in that mask share a key (vllm#56655, fix unmerged at 0.30.0). The adapter `vllm_cache_key` decides at
  `Request.__init__` and repairs by appending a digest of the block's mask to the hash's extra keys and remaking
  the request's hashes (`resolved`). The same rule covers a permutation: transformers' beam search reorders the
  cache under the names it knows (`transformers_beam`); a model whose cache lives under another name is reported
  (5.12.1 reordered `past_key_values` only: transformers#46612; 5.17.0 reorders every name).
- What a Triton kernel is told about its tensors (`kernel_launch_contract.py`, adapter `triton_launch`, engine-
  independent: one hook on `JITFunction.run`, so every `@triton.jit` kernel launched eagerly by any engine; kernels
  Inductor generates for a compiled forward are not seen): a tensor strided in its innermost dimension handed to a
  kernel that was not told that stride (no integer argument among its value parameters and stride-named
  constexprs equals it) and whose parameters name no stride at all (`stride`, `_s0`, `sxm`, `ld...`) is `broken`
  (the kernel reads it as if contiguous); handed to a kernel that names strides but was not told this one, it is
  `unknown` (said once). Each (kernel, stride pattern) is
  decided once per process, and a kernel is looked at for its first eight strided patterns; compile-only warm-ups
  are not launches. sglang#21843 (fused_gdn_gating read interleaved a/b) is the case the rule comes from; there
  the kernel takes row strides, so the decision is `unknown` at the kernel's boundary.
- `Rotary.pairing` (vocabulary v8): how a rotary embedding pairs the dimensions it rotates, `split` (i with
  i + d/2, the Llama convention) or `interleaved` (2i with 2i+1, GPT-J's). Declared by a config key
  (`rope_interleave`, `rope_interleaved`, `is_neox_style`) or, failing that, by the architecture's reference
  implementation (`data/rotary_pairing.json`: transformers 5.17.0 configuration and modeling files with lines; GLM,
  Cohere, Ernie 4.5, GPT-J and DeepSeek-V3 interleaved, Llama, Qwen and Gemma split). `rotary_pairing_contract.py`
  decides the rotary modules vLLM built for the language model (`is_neox_style`, adapter `vllm_pairing`) against
  the declaration - `resolved` by setting the modules' convention, `broken` where the model's MRoPE module
  dispatches to a kernel that pairs split-wise whatever the layer says (vLLM's Triton MRoPE kernel before 0.27.0
  with the custom op enabled: vllm#42016, #49290). Not compared: a multimodal model's vision tower (its own
  reference), a DSA indexer (its own key), and modules that pair both ways in one language model (`unknown`, nothing
  set). An architecture the table does not know, without a key, decides nothing.

**Fixed**
- The KV contract's `kv_needed` rule breaks only when a sequence holds fewer slots than its tokens. Holding more
  than one allocation unit over its tokens was also `broken`, and under speculative decoding an engine
  legitimately does that: it reserves lookahead slots and keeps the blocks of the drafts it rejected (vLLM 0.30
  with ngram speculation said `broken` on a healthy run; vLLM 0.23 with extract_hidden_states likewise, seen in
  the second replay). Fewer slots than tokens is the loss and is still reported.
- The vLLM serve adapter's wrapper of `ParserManager.get_parser` binds its arguments by name and passes the rest
  through: with a fixed signature it raised `TypeError` on vLLM 0.23.0 (whose `get_parser` takes `is_harmony`)
  and the API server died - the one run entail broke in the second pre-registered replay.
- A model loaded by hub id resolves to its cached snapshot folder again, so the Vocab and Stops checks decide
  instead of saying "no local folder to read": huggingface_hub 1.32 refuses a cached snapshot that lacks files the
  engine never fetched (`.gitattributes`, evaluation results), and every hub-id load on vLLM and transformers
  was "could not be checked". The folder of the cached `config.json` is used when the snapshot lookup refuses;
  nothing is downloaded.

**Changed**
- The log and record files are kept open per process, one write and one flush per line, instead of being opened
  and closed per line: on a 9P mount (a project under WSL's `/mnt/c`) the open and close cost 4.6 ms per line, and a
  boundary that runs per request (the prefix-cache key) made a batch-32 decode 6% slower; kept open it is 0.18 ms
  there and 0.005 ms on ext4. vLLM's CUDA-graph path with every adapter on: 1.005x, 1.009x and 0.999x at batch 1,
  8 and 32 (control runs without entail 0.994-1.003x).

**Docs**
- README (EN/KO): the "4 of 4" sentence is marked as the bugs the facts were written from, and the pre-registered
  replay with the vocabulary frozen at 1.1.0 is reported next to it: 150 issues screened, 17 passed, 15 reproduced,
  7 in the class by two blind raters, 0 of the 7 detected, 0 false alarms on the 8 outside the class. Known gaps
  list the facts and sites it exposed, and the hub-id loads that the Vocab and Stops checks cannot decide.

## 1.1.0

Released 2026-09-26. Five facts from the low-level study (codebook v2): classes of wrong output that 1.0 did not read,
each measured on the real bug it comes from.

**Added**
- Fact vocabulary v5: `Identity` (TIME) — what a stored or cached item stands for, so a store keyed by identity does
  not serve one sequence's KV under another's key. Rule `identity_stale` in `identity_contract.py`; the repair is to
  forget the stale identities and let the store remake them.
- vLLM adapter `vllm_identity`: wraps `Scheduler._update_request_as_session` and checks a request's prefix-cache
  block hashes against the hashes its current tokens give, from the truncation point on. This is the class behind
  vllm#49377 and #49449 (a streaming-session rebuild leaves stale block hashes; the fix PRs are unmerged, so it is
  live in vLLM 0.30.0). Measured end to end on SmolLM2-135M: the stale hash is caught and repaired, and the wrong
  output (a false 16-token cache hit) becomes the correct recomputed output. entail 1.0.0-1.0.2 passed it (E3 miss).

- Fact vocabulary v6: `TokenType` (MAPPING) — the token type id a position is given by its role (padding).
  `request_contract.pad_type` compares the id a server gave the padding with the tokenizer's `pad_token_type_id`.
  vLLM adapter `vllm_scoring`: wraps the scoring processor's padding of token type ids (vllm#58138: a cross-encoder's
  padding was given the document's segment, and /rerank scores moved). The repair (give the padding the declared
  id) is offered only where the consumer can carry it; vLLM 0.30 keeps token types as the index of the first 1, so
  it cannot (capability row, measured): the decision is broken under the default policy and the padded request is
  refused before scoring under `ENTAIL_ON_BROKEN=stop`.

- Fact vocabulary v6: `KernelConfig` (LAYOUT) — the tile a kernel steps K in, against the block the weights are
  quantized in; the tile must be a divisor of the block (`tile_contract.py`, rule `tile_over_block`). SGLang adapter
  `sglang_fp8_tile`: wraps the block-FP8 Triton matmul and checks the config map it picks from, once per map; the
  repair clamps the tile to the block (the engine's own default). sglang#39626 (a hand-supplied K tile of 64 over a
  block of 32 gave 64 where 288 was right): measured on 0.5.20, the tile is clamped and the kernel returns 288.
  SGLang's shipped tuned configs (1,887 entries) all divide, so ordinary runs decide nothing.
- Records name a transformers config by its class and quote the file's own `_name_or_path` as the file's claim,
  since that field can be stale (a checkpoint copied from another model).
- Fact vocabulary v6: `Vocab` (MAPPING) — the tokenizer's base vocabulary. `vocab_contract.py` reads what a model
  folder declares (tokenizer.json, vocab.txt, vocab.json, a sentencepiece model, config vocab_size, the embedding's
  rows) and decides the tokenizer the engine built against it, with two rules and no threshold: the tokenizer's ids
  must fit the embedding, and when the folder carries two vocabularies the engine must hold the model's. Adapter
  `transformers_tokenizer` wraps `PreTrainedTokenizerBase.from_pretrained` (vLLM and SGLang build their tokenizers
  through it too); `entail check` runs the same rule statically. transformers#48967 (a folder with vocab.txt of
  100,000 and a tokenizer.json of 32,000; transformers 5 built the 32,000 one and the ids changed): reported as
  broken, refused before the first id under `ENTAIL_ON_BROKEN=stop`, exit 1 from `entail check`.
- Start-up hook: the target table had the tokenizer module keyed twice, so the second adapter was silently dropped;
  one entry now, and a test guards the table against repeated keys.
- `Rotary` gains optional fields (v6): yarn's `beta_fast`, `beta_slow`, `attention_factor` (also spelt
  `attn_factor`), `mscale`, `mscale_all_dim`, `truncate`; longrope's `long_factor`/`short_factor` as a SHA-256
  digest and their count; `partial_rotary_factor` (a top-level key, or inside `rope_parameters`); `local_theta` and
  `local_factor` for a model that alternates two RoPEs (Gemma 3: `rope_local_base_freq`, or `rope_parameters` split
  into full_attention/sliding_attention - the local layers declare no scaling, so an engine that scales them is
  caught; a stated local RoPE without a factor is `local_factor` 1.0, a definite value); `mrope_section` and
  `mrope_interleaved` (the Qwen-VL family; `mrope` is not a rope type - transformers 5 normalises the old spelling
  `{"type": "mrope"}` to `default`, and the fact of mrope is its section); and the rope type `proportional`
  (Gemma 4). `original_max_position_embeddings` is read from the config's top level when the scaling dict has none
  (Phi). A local `partial_rotary_factor` or scaling type different from the global one is reported as beyond the
  vocabulary, not dropped. Over the 230 most-downloaded models, RoPE declarations outside the vocabulary went from
  34 to 3 (two `attn_factor`, a name no engine reads, and one local `partial_rotary_factor`; all three are said as
  not compared), and a RoPE key the ENGINE's config holds that the vocabulary cannot carry is now reported at the
  RoPE boundary instead of dropped. An alias is added only for a name a consumer reads.
- `sglang_fp8_tile` also wraps the fused-MoE config lookup (`try_get_optimal_moe_config`), which has no sanitiser:
  SGLang 0.5.20 ships an H100 config for E=512, N=256, fp8 block [128, 128] whose BLOCK_SIZE_K is 256; at the
  kernel level that returns 256 where 512 is right, and the clamp restores 512. The dense hot path now costs a
  dict lookup per call and writes no record line once a map is decided.
- Vocab: only a base vocabulary larger than the embedding is broken; an added token past the rows (gemma-3-1b-it's
  image token on the text-only model) is noted on a passing decision. The embedding is the tensor with at least the
  config's vocabulary of rows; a stale second tokenizer source that is not the model's vocabulary is unknown, not
  judged. Tokenizers built outside `PreTrainedTokenizerBase.from_pretrained` (Mistral, tiktoken, GGUF) are not
  checked at run time; `entail check` says so when Mistral files are present.

- Fact vocabulary v7: `Stops` (MAPPING) - the token ids a generation ends with (and begins with, and is padded
  with), as each file states them: generation_config.json, config.json and the tokenizer's eos_token. Every engine
  builds its stop set from a different subset (`data/stops_sources.json`: transformers from generation_config.json
  alone, vLLM from the tokenizer's eos plus generation_config.json, SGLang from config.json, generation_config.json
  and the tokenizer's eos its scheduler matches), so an end one file declares can be one the engine never sees and the model runs past
  the end of its answer (Llama 3, April 2024). `stops_contract.py` takes the union: a consumer whose set lacks a
  declared end is resolved by adding it (adapters `transformers_stops`, `vllm_stops`, `sglang_stops`), a declared
  id past the tokenizer is broken; `entail check` decides the set each engine would build. The tokenizer's own
  declaration (`eos_token` in tokenizer_config.json) is read as an id by that file's added-token table, without
  building a tokenizer. Measured: a Llama-3.2-3B-Instruct copy whose generation_config.json names only
  `<|end_of_text|>` ran every answer to the token limit on transformers 5.17 and stops at the end with entail; and
  nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 as shipped does the same (its config.json and auto-written
  generation_config.json name `</s>`, its tokenizer and chat template `<|im_end|>`): three answers ran to 160
  tokens without entail, and stopped at 46, 63 and 56 with it. Over 230 popular folders, `entail check` finds no
  id past a tokenizer and would add a dropped end on transformers for 9 folders and on vLLM for 2; SGLang's
  scheduler also matches the tokenizer's eos, so nothing is added there.

**Changed**
- Config coverage: a key the class does not take but the vocabulary maps and compares elsewhere (Qwen2.5 and Qwen3
  write `rope_scaling: null`) is a pass that names it, not an unknown; before, it was 240 of the 394 unknown lines
  in 114 healthy runs, on models where nothing was in doubt. The misspelling rule compares a top-level key with
  top-level vocabulary keys only: a field that lives inside `rope_scaling`/`rope_parameters` (`factor`,
  `beta_fast`, `mrope_interleaved` ...) is no target, so SmolLM2's `rope_interleaved` is an unread key, not a
  misspelt `mrope_interleaved` (which 1.1.0.dev had called broken on all three engines). Checked over the 558
  distinct keys of 230 popular configs: no top-level key is within the rule's distance of a vocabulary key.
- A decision recorded once for an owner (`enforce(once_for=...)`) also works when the owner is a value (a folder
  path, a (class, name) pair): before, such an owner was silently not remembered, and the tokenizer's pass was
  recorded at every one of SGLang's tokenizer builds; the same config's coverage decision is now recorded once per
  process instead of at every build (vLLM and SGLang build the same config several times).
- Load cost: a model folder's vocabulary sources are read once per process and once per machine (a stamp of the
  watched files keys an in-process cache and `entail_logs/vocab_sources.json`), tokenizer.json is counted only
  when a second tokenizer source exists to compare with, and the tokenizer's highest id comes from its added-token
  table instead of a full `get_vocab()`. On Qwen3-4B the library's time at load went from 264/872/999 ms
  (transformers/vLLM/SGLang) to 32/185/94 ms once the cache is filled, 150/289/60 ms on a machine's first run; over
  102 healthy runs the share of load time is 1.2% at the median, 7.8% at the 90th percentile and up to 37% on toy
  models that load in under a second (`testbed/results/m15/E2_SUMMARY.md`, `E2_RECOST_SUMMARY.md`).

Older facts and files still read (`READABLE_VERSIONS`).

**Measured for this release** (`testbed/results/m15/SUMMARY.md`, `E2_SUMMARY.md`, `E2_RECOST_SUMMARY.md` in the
research workspace): the four in-class defects of the M10 E3 sample are blocked (2 resolved: vllm#49377/#49449,
sglang#39626) or reported at their boundary (2: vllm#58138, transformers#48967), where 1.0.0 passed all four; the
three out-of-class ones are, as designed, not flagged. On 38 popular models x 3 engines (102 valid runs, each
engine's defaults): no run broken by entail, 0 broken or refused decisions, 2 resolved decisions backed by a
measured row (Gemma 2's softcap) and, once `Stops` was in, 2 more where a file's declared end was added to
transformers' stop set (Nemotron-3-Nano-4B as shipped; a tiny test model), outputs identical to the run without
entail in 97 of 98 comparisons without a repair (the one difference is an engine's own nondeterminism, seen
off-vs-off too), 69 unknown lines in all (from 364 before the Coverage change and the once-per-process record),
the library's share of load time 1.2% at the median and 9.1% at the 90th percentile. Statically over
230 popular configs: Coverage, Vocab and Rotary broken 0 (Rotary pass 196, unknown 6: DeepSeek-V4's per-layer split
and `attn_factor`, both said as not compared). The Identity hook fires only on streaming-session updates, which
ordinary generation never triggers.

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
