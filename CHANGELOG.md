# Changelog

Version numbers (written down with 2.0.1): the first number changes when the design's structure changes; the second
for a release with a purpose - new features or a large scope of work - that leaves the structure as it is; the third
for bug fixes and small corrections of a few hundred lines or fewer. [RELEASING.md](RELEASING.md) has the rule.

## Unreleased

- **diffusers: the prediction is decided where the pipeline samples, not at the load** (#24, #21). entail rebuilt
  the pipeline's scheduler right after `from_single_file`, and programs that set up a sampler of their own after
  the load got a repair that did not fit them: SD.Next already samples a v-prediction checkpoint as v, and its
  images changed (#21); InvokeAI builds its sampler from its own model settings (epsilon), so the repair never
  reached its sampling, said "resolved" anyway, and left a zero-terminal-SNR schedule under Invoke's epsilon, on
  which its default sampler (DPM++ 3M) stopped with an `IndexError` (#24). Now the load leaves the scheduler as
  diffusers set it up; the pipeline's call is where the scheduler it samples with is compared and, when it differs,
  rebuilt (a plain `pipe(...)` gets the same repair as before, and a program that set it up as declared is left
  alone); a program that runs the model in a sampling loop of its own gets one `unknown` line - the declared
  prediction, "nothing was changed", and where to set it (InvokeAI: the model's settings). A non-pass line names
  the scheduler and its `prediction_type` and `rescale_betas_zero_snr`.
- **A RoPE repair names the engine that reads the value** (#26). The line for an old RoPE name written after the
  config was built said `transformers.rotary_embedding uses Rotary(theta=None)` under SGLang too, and a user could
  not tell whether SGLang would have computed with the wrong base. vLLM and SGLang apply their overrides through
  this write and then run their own model code, which reads `rope_parameters` as transformers' does (SGLang 0.5.20's
  Llama: `rope_parameters.get("rope_theta", 10000)`), so the consumer is now `sglang.rotary_embedding` or
  `vllm.rotary_embedding` in their processes, and a lost `rope_theta` is spelled out: the model takes its default
  base, 10,000 for Llama on vLLM and SGLang. README's measured table names the transformers version of each row and
  that both used the engines' offline APIs; a field test on SGLang's server with the same versions saw no drop,
  which is not explained yet (SGLang's config code, run on CPU, does lose the value there too).

## 2.1.2

Released 2026-09-28. Fixes from the field test of 2.1.1 (issues #14, #15, #17 and #18, found using entail from the
docs only). From this release on, each issue a release's pull requests close gets a comment naming the version
that carries the fix (RELEASING.md).

- **Declaring a value with a manifest works from the docs** (#15). A draft's empty slot shows the form of its value
  (`"form": "{\"kind\": \"v\", \"zsnr\": true} ..."`), and a Prediction may also be written as a word (`"v"`,
  `"v_prediction"`, `"epsilon"`, or entail's own "v-prediction with zero terminal SNR"), a LatentScale as a number;
  a value entail cannot read makes `entail pin` say how to write it, not split the string into letters. `entail
  infer` on a path that does not exist says so in one line instead of a traceback. ComfyUI's lines name the
  checkpoint by its full path, so the page's declare box gives commands that run as pasted; the box also says where
  the form is and how `ENTAIL_MANIFESTS` keeps manifests in a folder.
- **`entail preflight` no longer raises false alarms on popular models** (#17). transformers 5.17 keeps
  `rope_scaling` as `rope_parameters` (adding `rope_theta` to it), and the config-key check, which looked only for
  scalars, printed a `RoleError` for Llama 3.2's and Phi-3.5's `rope_scaling`; a dict a known field holds whole now
  counts as kept ("kept under another name: rope_scaling in rope_parameters"), and `auto_map` is a key the Auto
  classes read. The attention properties are read as the load check reads them, so a window the config switches
  off (Qwen2.5's `use_sliding_window: false`) or one that never binds (Phi-3.5's 262144 over 131072 positions) is no
  longer said to be dropped by SGLang's flashinfer and flex_attention. The output shows the RoPE the config declares
  and where it is checked (at load, `ENTAIL=load`); each backend's evidence is a short label ("measured on vllm
  0.30.0", "read in vllm 0.30.0's code") for the properties the model declares, not a path into the research
  workspace; a model that declares neither softcap nor a sliding window gets one line instead of five rows.
- **The page leads with what entail repaired** (#14). A run with a repair was headlined "No mismatch found", the
  repair said last - in the field test, the repair that kept a long-context retrieval at 5 of 5 instead of 0 of 5.
  Now the header says "entail repaired 1 value: Attention and rotary (theta unset → 500000.0)": the node and the
  fields that changed, from what the engine had to the declared value. The run graph carries each repair's two
  values for it.
- **Windows: upgrading while `entail serve` runs** (#18). pip cannot replace a running `entail.exe`; when the
  environment is on another drive than `%TEMP%`, `pip install -U entail-ai` stops half-way and leaves the
  environment without entail, so a tool started with `ENTAIL=load` runs with entail off. INSTALL now says to start
  the page as `python -m entail serve` on Windows (it holds no `entail.exe`), to close an `entail.exe` serve before
  upgrading, and how to recover (close it, delete the `~ntail*` folders in `site-packages`, install again).

**Known issues** (field test of 2.1.1; both were in 2.1.1 as well)
- The diffusers prediction repair sets up the scheduler of the pipeline a v-prediction checkpoint is loaded into,
  and an app that builds its own sampler from its own model settings can disagree with it. SD.Next already samples
  such a checkpoint as v-prediction; with entail on, its image differs slightly from SD.Next alone (#21). InvokeAI
  registers it as epsilon; with entail on, its default sampler (DPM++ 3M) stops with an `IndexError` while Euler
  still gives noise, although the console says the prediction was resolved (#24).

## 2.1.1

Released 2026-09-28. Fixes from the first field test of 2.1.0 (ComfyUI, transformers and text-generation-webui, used
from the docs only; issues #2 to #9). Releases now go through a release pull request, and the maintainer approves
each one (RELEASING.md).

- **A could-not-check line names its model and says what it means.** ComfyUI's decisions name the checkpoint's file,
  so a second model that declares nothing gets its own line; the console used to leave it off as a repeat of the
  first model's (#2). A non-stopping `unknown` line ends "not checked, so neither a pass nor a fault (details: entail
  serve)" (#3), and INSTALL says what these lines are and how `ENTAIL_QUIET=unknown` keeps them off the console.
- **Lines show what differs, not whole values.** A value in a line leaves out its unset fields; a long one shows only
  the fields that differ from the other side, and a repair names the fields it changed (`theta None -> 5000000`).
  The field test's Rotary repair printed 2,197 characters; the same kind of repair now prints about 520 (#9). The
  record keeps the whole values.
- **The page no longer calls an unchecked value "not a problem"** (#2): unknown and skipped read "neither an
  all-clear nor an alarm", and the header says "No mismatch found", with where to start if the output looks wrong.
  An unknown value that a manifest can declare (Prediction, LatentScale, ModelProps, Rotary, Template) shows how to
  declare it, with the model's file when the record names it (#5). On a Korean page the common rule texts and source
  phrases read in Korean, the original on hover (#7).
- **A folder given as a path object is read** (#8): `load.local_folder` takes a `pathlib.Path` like the same path as a
  string. text-generation-webui loads its tokenizers that way, and both tokenizer checks said "no local folder to
  read" there.
- **An image run no longer draws an LLM flow for its text encoder's tokenizer** (#4): ComfyUI builds its CLIP
  tokenizer with transformers, and the tokenizer node is now shared, as the kernels' is: it joins the flow the launch
  has and keeps its place in an LLM launch.
- **`entail doctor` lists diffusers and ComfyUI** (#6), ComfyUI's version read from its `comfyui_version.py`, each with
  the versions its adapters were measured on and "not this version" when it is not one of them. The adapters of
  engines that are not installed fold into one line.
- **Small things** (#7): at start one line on stderr says entail is on and where it writes (`ENTAIL_QUIET=start`
  leaves it out); `entail serve` says its token needs nothing from the user; the README names every file
  `entail_logs/` may hold (`said.txt`, `tokenizer_ids.json`, `vocab_sources.json`, `safe_mode.json`, `nodes.json`).

## 2.1.0

Released 2026-09-28. The page of `entail serve`, made again so that someone who has never read the design can tell
what happened in a run, and install steps by tool. The checks are 2.0's; the API (one addition below), the two writes
and their checks (token, Origin, Host) are as before. A minor version: a large piece of work with a purpose, and the
structure is unchanged.

- **The look:** a dark, ComfyUI-like node canvas in the ClickHouse design system (as written up in
  VoltAgent/awesome-design-md, MIT): near-black surfaces, one yellow accent kept for selection and the main action,
  hairline borders, no shadows. Each flow is a group; its nodes run in rows as wide as the window allows, joined by
  wires; a dotted wire means steps with no records lie in between (they can be shown). Fonts are named, not downloaded.
- **Plain words:** the header says what happened ("A value broke at Tokenizer"; "Nothing broke", with how many points
  could not be checked) and a button per state picks those nodes out. The right panel starts with the run's summary (what is worth
  a look, when, which engines) and, on a node, shows the declared value next to the one the engine used with the
  fields that differ marked, where each came from, the rule and the note.
- **One language at a time:** English or Korean, by the browser's language, with a switch in the top bar (it was
  both at once).
- **Panels:** runs, settings (safe mode, custom nodes turned off - they can be turned back on from there) and help in
  a side bar; on narrower windows the panels become drawers, and on a phone the flow runs top to bottom.
- `/api/runs` also gives, per launch, the node where meaning first broke (`where`), how many nodes are in each state
  (`states`) and the flows drawn (`flows`), for the run list.
- **Install steps by tool:** [INSTALL.md](INSTALL.md) ([INSTALL.ko.md](INSTALL.ko.md)) says where entail goes and how
  to turn it on for your own scripts, vLLM behind Open WebUI (and in Docker), SGLang, ComfyUI (installed with git,
  portable, Desktop) and text-generation-webui, and which of those steps were run on this project's machine and which
  follow the tool's own source only. The page links it when there are no records yet, and from its help.
- **Who it is for (README):** people who build their own AI project on a Python engine or on ComfyUI. Apps that run
  models in an engine compiled into the app (Ollama, LM Studio, llama.cpp) give entail nothing to attach to.

## 2.0.1

Released 2026-09-28. A correction of the README (English and Korean); the code is 2.0.0's, apart from its version
number. The PyPI page shows the README, so the correction needed a release.

**Fixed**
- The known gaps still listed two defects 2.0.0 had fixed - the start-up path check counting log-probabilities that
  are not finite numbers as agreement (vllm#33560), and the false alarm on vLLM's encoder-decoder models
  (whisper-large-v3-turbo) - and the summary at the top still called that false alarm known. Both are gone from the
  list; the summary and the fourth replay's paragraph say it was fixed in 2.0.0.
- Four sentences written for 1.2.0 and 1.3.0 said "this version": the second replay's frozen code (`ce79b19`), the
  wrapper that broke a run on vLLM 0.23.0 and the hub-id fix are 1.2.0's, and the frozen `2aa975b` is 1.3.0's.
  They name their version now.

**Changed**
- The version rule is written down (above, and in RELEASING.md).

## 2.0.0

Released 2026-09-28. entail 2.0: a local platform that manages the stability of an AI project (ROADMAP product track
P0-P6) - the checks of 1.3.0 with two fixes, and around them `entail serve`, two safety modes, official DLCs and
custom nodes. The DLC and the workshop package are not on PyPI; they install from this repository (below).

**Measured for this release** (one RTX 4070 Ti; the research workspace's `testbed/results/p6/SUMMARY.md`)
- 102 healthy runs (38 models on transformers, vLLM and SGLang): exactly the decisions of 1.3.0's frozen code - no
  run broken, the same six real tokenizer differences reported, no unbacked repair, outputs the same in 97 of 98.
- Requests are at most about 1% slower with entail on (vLLM 0.30, Qwen3-4B, CUDA graphs; 1.000-1.011 over ten rounds
  interleaved with 1.3.0, which measured the same), and no slower while `entail serve` reads the records.
- Load: the median share is 9.0%, above the 5% target - mostly vLLM's start-up path check (`ENTAIL_NO_PATHS=1`).
- The page points at the first broken node and why in 12 of 12 planted-fault workflows; it read back the 114
  launches of the healthy runs without an error.
- A wheel installed with no index brings in nothing but entail; `entail serve` answers with its page.

**Added**
- `entail serve`: a local web server (127.0.0.1 only, the standard library) that shows a project's runs from the
  record files as live nodes - the node that first broke and why (P2). Record lines are version 2: `v`, `t`, `run`
  (P1); version-1 lines still read.
- Two safety modes (P3). The selective safe path (`ENTAIL_SAFE=auto`, the default): when the engine's own paths
  disagree at start, the next starts of that configuration turn the optimizations the disagreement points at off,
  one per start, until the paths agree (that one is the cause and stays off) or none is left (said once as broken).
  The explicit safe mode (`ENTAIL_SAFE=all`): every optimization the engine declares does not change results is
  turned off, and entail says whether a fault that stays is outside them or one that went away was inside them.
  vLLM 0.30 (CUDA graphs, prefix cache, speculative decoding, custom kernels - `custom_ops` and the IR ops' kernel
  priority) and SGLang 0.5.20 (CUDA graphs, radix cache, speculative decoding). The platform's one write sets the
  mode for the next start (`POST /api/safe-mode`: a token that changes after each write, Origin and Host checked).
- Official DLCs (P4): packages outside the core that attach through the entry point group `entail.dlc`
  (`entail/dlc.py`). The core checks the DLC's core range, installs its entries itself - a failure is recorded and the
  program goes on - and shows its nodes on the platform; `entail doctor` lists them; `ENTAIL_DLC=off` leaves them out.

- Custom nodes (P5, `entail.nodes`): a developer's own high-level checks at points of their program - a validator
  is a plain function (`nodes.ok()`, `nodes.broken(why)`, `nodes.unknown(why)`), placed with `nodes.check(...)` or
  `@nodes.watch(...)`. What it finds is a core decision (fact `Check`, vocabulary v12) at `node:<node>/<validator>`,
  shown as the node's own on the platform, which can turn a node off (`POST /api/nodes`). A validator that fails, is
  slow (50 ms by default) or returns something else never breaks the program. Workshop packages of validators
  attach through the entry point group `entail.nodes` (`ENTAIL_NODES=off`, or a list); `workshop/basics`
  (`entail-nodes-basics`) is the first, and `examples/custom_nodes` has three small apps.
  `pip install "git+https://github.com/wwoosshh/entail@v2.0.0#subdirectory=workshop/basics"`

**Changed**
- The repair of ComfyUI's own defect (Comfy-Org/ComfyUI#16490) left the core: it is the official DLC
  `entail-dlc-comfyui` (`dlc/comfyui` in this repository). Without it, entail no longer repairs that defect.
  `pip install "git+https://github.com/wwoosshh/entail@v2.0.0#subdirectory=dlc/comfyui"`

**Fixed**
- The start-up path check counts a log-probability that is not a finite number as a disagreement (`paths_disagree`,
  "log-probabilities are not finite numbers (n on the first path, m on the second)"). NaN compared as no move, so a
  model whose every log-probability was NaN passed all three pairs (vllm#33560: vLLM 0.16, NVFP4 with float16
  activations; the fourth pre-registered replay).
- vLLM's KV extent check no longer holds a cross-attention cache group to the decoder's tokens: such a group
  (`CrossAttentionManager` over a `CrossAttentionSpec`, vLLM 0.14 and 0.30) holds the encoder's states, sized by the
  encoder input, and is counted as skipped. whisper-large-v3-turbo was said broken nine times ("kv cache group 1
  holds 0 KV slots for 216 tokens") while its transcription was right. The decoder's own group is decided as before.

## 1.3.0

Released 2026-09-27. Reference comparisons where a declaration lives only in code (a tokenizer run against the
declared one, a custom op's kernel against its own definition, a parser's stream against its whole-text parse, a
multimodal placeholder's origin, a completion's logprobs), definitions for five engine functions that carry none,
checks that reach an engine's warm-up and capture (warm-up probes, a Triton launch's layout run twice), and the
engine's own paths held against each other at start. In two further pre-registered replays of real engine bugs, the
code frozen at `07fceac` and at this version's `2aa975b`, it detected 0 of 5 and 0 of 6 in-class bugs (0 of 26 over
four replays). Read it as a
light pre-deployment check that repairs the classes it knows and says where meaning broke, not as protection against
unseen bugs; see **Measured for this release** and **Known issues**.

**Added**
- Warm-up probes (M19 L3.3a): the calls an engine makes before serving (vLLM's profile run and warm-ups, SGLang's
  capture warm-ups, recognised as a run of calls with one repeated row before the first real decision) decide the
  kernel-against-definition comparison, on made-up token values with the engine's own shapes, dtypes and strides,
  once per power-of-two size class and at the engine's own row count (512 rows at most). A repair can therefore be in
  place before a CUDA graph captures the kernel: it is offered when no graph captured the kernel at a size class not
  yet verified, and a repair now applies to every module of the same configuration, not only the one compared.
- A Triton launch run twice (M19 L3.3b; `kernel_layout_variant`): the first launch of a pattern whose innermost
  stride the kernel is not told runs on copies, as given and relaid (innermost dimension contiguous, outer strides
  kept; wholly contiguous for a kernel that takes no stride), the relaid copy twice for the kernel's own noise. A
  difference is resolved by launching that pattern on relaid copies from then on and copying back only the tensors
  the kernel wrote. Not run twice: launches under capture, a written tensor overlapping another argument, tensors
  past the budget, layouts that cannot be kept, comparisons where every value is zero.
- `PathAgreement` (vocabulary v10) and `path_contract.py` with the adapters `vllm_paths` and `sglang_paths` (M19
  L3.3c): right after vLLM's `LLM` or SGLang's `Engine` is built, three fixed probe texts go through the public
  generate API and three pairs of the engine's own paths are compared - decode against a fresh prefill of the same
  tokens, a request alone against the same requests batched, a cold run against one that reads the prefix cache.
  Rule `paths_disagree` (`broken`): a confident prediction (margin over 1.0) that changes, or a kept token's
  probability that moves by more than 0.25, thresholds set from healthy engines. The cache is cleared afterwards.
  `ENTAIL_NO_PATHS=1` turns it off.
- Exact definitions for index bookkeeping (M19 L3.3d): vLLM 0.30's `prepare_pos_seq_lens` (positions and sequence
  lengths) and `BlockTables.compute_slot_mappings` (the KV slot of every token) held against entail's definitions
  element by element (`compare_exact`), on fresh output buffers so the engine's persistent buffers are written only
  by the real call. Definitions can now declare the arguments they write, compare whole, use fresh buffers, compare
  exactly, and wrap class methods. The two were chosen because every model of a 14-model census ran them.

- Definitions for engine functions that carry none (M19 L3): `entail/definitions.py` writes, in plain PyTorch, what
  vLLM 0.30's `fused_experts` (unquantized or INT8 W8A8: weight scales applied by their own shape, activations
  quantized as the config says), vLLM 0.30's `w8a8_triton_block_scaled_mm` and SGLang 0.5.20's `fused_gdn_gating`
  compute - the functions' own arguments, outputs of the same shape and dtype; a call a definition does not cover
  raises NotImplementedError and is `unknown`. `adapters/function_reference.py` wraps each function's name as soon as
  its module has loaded (start-up shim) and holds it against its definition on the first real call with the kernel
  reference rule (a 64-row slice, the definition in its noise dtype and in float32; calls whose rows are all one row,
  an engine's dummy batch, decide nothing and are not counted towards giving up); a mismatch is resolved by sending
  the name to the definition (float32, the function's output dtype) from that call on, unless the function was
  already called inside a CUDA graph capture in the process; a repaired function captured later puts its definition
  in the graph when the definition is capturable, else the kernel, said once as broken. Measured on an RTX 4070 Ti:
  vllm#58532 (INT8 MoE, per-channel weight scales, static activation scale), vllm#52576 (block FP8, BLOCK_SIZE_K 256)
  and sglang#21843 (GDN gate on inputs with an inner stride of 2) resolved on their first call, the output's error
  against a float64 formula 0.956 -> 0.0020, 0.612 -> 0.0027, 0.937 -> 1.5e-7; seven healthy controls of the same
  functions pass with the kernel's output untouched. Not caught: a bad launch configuration used only by a later,
  larger call (decided once, on the first call). On healthy engines (entail on against off, six greedy probes):
  SGLang Qwen3.5-4B with and without CUDA graphs and vLLM Qwen3-4B-FP8 on the Triton block-FP8 path (Marlin
  disabled; vLLM picks Marlin on sm_89 by default, which never reaches the function), eager and default mode: the
  function is reached, passes, and the output is unchanged (6/6).

- Kernel reference repair (M19 L3): a custom op whose kernel differs from its own native definition is sent to that
  definition (`forward_native`) - from the very call that was compared (the slice is now decided before the real
  input is computed) and for every later call of that module in the process. Resolved in the record, with the
  resolution; offered only when the engine runs without CUDA graphs (captured graphs replay the kernel), and not
  under `ENTAIL_POLICY=refuse` or `KernelReference=refuse`. Measured: vllm#42016 (GLM-OCR, vLLM 0.22.0, eager)
  resolved, eager output now equal to the native mode's; a healthy Llama-3.2-3B changed nothing.
- `Tokenization` (vocabulary v9) and `tokenizer_contract.py` (M18.1): the tokenizer the engine built is run against
  the tokenizer the folder declares, on ten fixed probe texts, and the ids must be the same. The declaration can be
  run: tokenizer.json by the `tokenizers` library (a sentencepiece-only folder is `unknown` until that reference is
  measured); tokenizer_config.json's
  `added_tokens_decoder` (else tokenizer.json's `added_tokens`, else added_tokens.json) names every added token with
  its id, and each is looked up in the built tokenizer. Rules `tokenizer_ids` and `added_token_id`, `broken` (reported,
  the run goes on); a declaration that cannot be run (no file, no library, a tiktoken.model without its
  pre-tokenization pattern: the added tokens are still compared) is `unknown` once per folder. The size check
  (`Vocab`) passed every tokenizer built from the right file by the wrong class; this is the check those bugs
  needed: transformers#46489 (deepseek-coder as LlamaTokenizer, 5.10.2), #45812 (Granite as GPT2Tokenizer, 5.8.0),
  #45356 (Kimi-K2.5's `</think>` given `<|media_end|>`'s id, 5.4.0), #46710 (DeepSeek-R1-Distill's declared class
  replaced, 5.12.1). The `transformers_tokenizer` adapter runs it after the size check; `entail check` runs it
  statically when transformers can build the tokenizer. The declared tokenizer's probe ids are kept per folder in
  `entail_logs/tokenizer_ids.json`, so a process after the first only encodes the probes with the engine's tokenizer.
  A difference that the folder's own declared flag explains - `legacy: false` (or `add_prefix_space`) next to a
  tokenizer.json exported the other way, on the text the flag speaks of (after a special token, or at the start)
  and exactly as the flag's pipeline gives it - is the sources disagreeing (`unknown`, the flag recorded as the
  conflicting source, `ENTAIL_SOURCE_CONFLICT=stop` honoured); every other difference is `broken`. A difference the
  user's own build settings explain (`legacy=`, `add_prefix_space=`, ... given to from_pretrained) is the user's
  choice. Measured (retrospective: the rule was written from these bugs): the four bugs at their reported versions
  are all `broken` at the tokenizer boundary (Kimi by 18 of 23 declared added tokens with other ids, the others by 4
  to 9 of the 10 probe texts); on 5.17.0 three pass and Kimi is `unknown` (tiktoken: added tokens compared, texts
  not). 38 popular folders on 5.17.0 (11 distinct tokenizers): 36 pass, 2 broken - the one Llama-2-era tokenizer in
  the set (TinyLlama-1.1B-Chat, a tiny test folder): transformers 5 rebuilds a legacy-export tokenizer.json as
  Metaspace and never doubles the `▁` before text that starts with whitespace, whatever `legacy` says. The 300-folder
  static corpus on 5.17.0: 201 pass, 9 broken, 5 unknown, 85 without a tokenizer to compare; the 9 are six of that
  Llama-2 shape and three folders that declare `LlamaTokenizerFast` over a byte-level BPE tokenizer.json
  (DeepSeek-R1-0528-Qwen3-8B, deepseek-coder-7b-instruct-v1.5, an MLX export), which 5.17.0 builds as a Llama
  pipeline: "How are you doing?" decodes back as "Howareyoudoing?". Cost: the first process on a folder +241 ms at
  the median and +894 ms at most (the reference is built), later processes +2 ms at the median, +36 ms at the 90th
  percentile. The sentencepiece reference is not compared until it is measured on sentencepiece-only folders
  (special-token strings would differ); tokenizer.json's padding and truncation are cleared in the reference and
  BPE dropout is not compared; every declared added token is looked up; the machine cache is keyed by the library
  version too, and lives per start folder.
- `KernelReference` (vocabulary v9) and `kernel_reference_contract.py` with the vLLM adapter
  `vllm_kernel_reference` (M18.2): a custom op's dispatched kernel against the op's own native definition, run on
  the same input. vLLM's CustomOp carries its meaning as `forward_native` and dispatches to `forward_cuda`; after
  the model is built, every op dispatching to a kernel path is wrapped, and on its first real call per (op class
  and module, configuration, input pattern) the kernel and the definition are run on a 64-row slice of the real
  input (clones cut before the kernel touched its arguments; the engine's tensors are untouched), the definition in
  the input dtype and in float32, and the op's own tensors put back afterwards (a definition may convert its cache
  to the query's dtype). Rule `kernel_reference_mismatch`, decided value by value and per output tensor in its own
  dtype: a value non-finite on one side only, or differing from the float32 definition by more than FACTOR times
  the definition's own rounding noise at that value (backed by the tensor's typical noise) plus ATOL_ULPS units in
  the last place of the output dtype at that value; `broken` (reported, the run goes on). Afterwards the original
  method is put back, so the steady state costs nothing; under a stop policy the decision raises once. Not
  compared, each said `unknown` once: ops that override `forward` (the mamba mixers), ops holding the engine's
  state (a KV cache, an index buffer, a forward that reads the forward context), every op when the process is one
  rank of several, every op enabled under torch.compile (traced, the wrapper hands the call to the kernel), ops in
  vLLM's registry not reached from the model's modules, arguments that share no token dimension or cannot be cut
  and are too large to clone, and definitions that refuse the input. Calls inside vLLM's own dummy runs (profile,
  capture warm-ups) are neither compared nor counted, and an input that decides nothing (zeros, one repeated row,
  an identity such as rotary at position 0) leaves the wrapper on for the next real call (64 such calls at most).
  Measured (retrospective: the rule was written from this bug): vllm#42016 (GLM-OCR on vLLM 0.22.0, the Triton
  MRoPE kernel pairing split-wise for a model that pairs interleaved) is `broken` at `MRotaryEmbedding` on its
  first real input - max |kernel - definition| 10.5 at a scale of 10.9, allowed 0.588 - with no architecture
  table, and passes on 0.30.0; 8 popular models on vLLM 0.30.0 with `enforce_eager`: 35 decisions, 34 pass and one
  `unknown` (144 `quant_fp8` instances held by linear-kernel helpers, not reached from the model's modules), every
  decision on the first real input. Of the 34, 13 compare an independent kernel (rotary 6, activations 7, the
  activations bitwise equal to the definition) and 21 hold the definition against itself, which the record says:
  in eager mode vLLM 0.30's `RMSNorm.forward_cuda` returns `forward_native` (the fused kernels are reached under
  torch.compile, where entail does not compare). The worst value's ratio to its allowance is at most 0.095
  (median 0.062). FACTOR 8 and ATOL_ULPS 4 are headroom, not derived from that distribution (the kernels'
  largest error equals the definition's own largest rounding step there, which any FACTOR of 1 or more admits);
  every decision records that ratio so a later measurement can fix them from data.
- `Parse` (vocabulary v9), `parse_contract.py` and the vLLM adapter `vllm_parse` (M18.3): a chat parser's streamed
  message against its parse of the same complete text, and its tool calls against the tools the request declared.
  The class vLLM's server builds a parser from per request (`ParserManager.get_parser`, 0.30's unified parsers with
  `parse_delta` and `parse`) is returned wrapped: its instances accumulate what `parse_delta` hands on (content,
  reasoning, tool-call names and argument pieces), and when the stream finishes a fresh instance parses the whole
  text and the two must agree exactly (arguments as JSON values): rule `stream_differs_from_full`. A tool call
  that names an undeclared tool, lacks a parameter the declared tool requires, or carries a key the tool's
  `parameters.properties` do not have when the tool forbids additional properties (`additionalProperties: false`;
  JSON Schema allows them by default, so under a tool that did not forbid them an extra key is a note on a pass)
  is `tool_args_outside_schema`, on both paths; whether the key is in the model's text (the model's call does not
  fit the declared tool, or the parser reshaped it) or not (the parser added it) is said. Both `broken` (reported;
  the client already has the streamed message). An output that did not finish by itself - the request's token
  limit reached, the reasoning block still open, a forced tool whose arguments never became JSON - is where vLLM
  documents its two paths to differ, so a difference there is `unknown`; reasoning the stream sent again as
  content (vLLM's fallback) is a note. Deltas are accumulated with nothing recorded, the comparison runs once at
  the end of the stream, and a silent pass is counted, not recorded. Measured on vLLM 0.30.0's own parsers
  (retrospective: the rules were written from these bugs), driven as the server drives them with a stand-in
  tokenizer and no prompt: vllm#49316 (kimi_k2: the streamed path skips the schema's type coercion, 4 of 4 texts),
  #49412 (qwen3: the content around tool calls is dropped on the whole-text path, 2 of 3; the third differs in
  surrounding whitespace only) and #47986 (deepseek_v4: tool_b unwrapped with tool_a's schema, with tool_b
  declared precisely so that a correct parser passes the same rule) are `broken`; the well-formed texts raise
  nothing. Content that differs in surrounding whitespace only is a note on a pass, not broken: on a live vLLM
  server (Qwen3-0.6B, qwen3 reasoning parser, hermes tool parser, 18 streamed and whole requests) every tool-call
  stream streamed two newlines and parsed nothing whole, and a newline loses no meaning.
- `Placeholder` (vocabulary v9), `placeholder_contract.py` and the vLLM adapter `vllm_multimodal` (M18.4): where
  vLLM binds a multimodal item's placeholder against the markup the model's config declares
  (`vision_start_token_id` before `image_token_id`, the Qwen-VL family): an image placeholder run not preceded by
  the declared start token came from the prompt's text, not from the template - a literal `<|image_pad|>` typed by
  the user took the image (vllm#57740). Rule `placeholder_outside_markup`, `broken`; a model that declares no markup
  decides nothing, and vLLM's own profiling prompts (placeholder runs from token 0, no template) are not decided.
  Measured: Qwen2.5-VL-3B-Instruct on vLLM 0.30.0 with the report's two message orders - the attack order is
  `broken` ("the image placeholder bound at tokens 20..275 is preceded by id 220"), the control order passes.
- The SGLang serve adapter also decides a chat completion's logprobs against its message (M18.4,
  `parse_contract.check_logprobs`): the logprob tokens must decode to the content the client gets; with
  `separate_reasoning` SGLang's logprobs covered the whole raw output, `<think>` span and markers included, while
  `message.content` held the parsed answer (sglang#25055). Rule `logprobs_cover_other_text`, `broken`. Measured:
  SGLang 0.5.20, Qwen3-0.6B with the qwen3 reasoning parser, one request with `logprobs` and `separate_reasoning`:
  `broken` ("the 155 logprob tokens cover the reasoning span (545 characters and its markers) as well as the
  content (12 characters)").
- The false-alarm yardstick now covers a live vLLM server with every adapter on (18 streamed and whole chat
  requests through a reasoning parser and a tool parser: nothing broken), ngram speculative decoding (nothing
  broken: the M17.6 narrowing of `kv_needed` holds) and a hybrid Mamba-attention model with several KV groups
  (nothing broken); prefill-decode disaggregation is not measured on one card.

**Changed**
- Kernel reference slices keep their strides (`kernel_reference_contract.kept`): `clone()` made a view with gaps
  contiguous, so a kernel that misreads a layout read the slice right.
- vLLM's dummy runs are marked in the second GPU runner's graph capture and memory profiling too (`capture_model`,
  `profile_cudagraph_memory`, which run outside `_dummy_run`): their warm-up calls had used up TRIES before the first
  real request.
- SGLang's start-up path check runs only when asked for (`ENTAIL_PATHS=1`; otherwise the start boundary says once
  why it did not compare). SGLang runs no prefill while it starts, so the probe requests were the engine's first
  prefills, and a defect on that path stopped the engine before the caller's first request: SGLang 0.5.20 picks
  flashinfer for Phi-3.5-mini-instruct (head_dim 96), whose state merge does not take that head size, and any prompt
  of 128 tokens or more stops the scheduler, with entail off as well. vLLM's path check stays on.
- A kernel comparison records its worst value-to-allowance ratio with a floor for an all-zero allowance (it was
  written as 0 when the float32 allowance underflowed).

**Measured for this release** (one RTX 4070 Ti; the research workspace's `testbed/results/m19/l4/SUMMARY.md`)
- Healthy runs: 38 popular models on transformers 5.17, vLLM 0.30 and SGLang 0.5.20, 102 valid runs: entail broke no
  run; no check added since 1.2.0 said `broken` or `refused`; outputs identical in 97 of 98 comparisons without a
  repair (the one difference is an engine's own nondeterminism). Six runs say `broken` at the tokenizer boundary,
  all from two Llama-2-era folders (TinyLlama-1.1B-Chat and a tiny test folder) on every engine: a real difference -
  transformers 5 builds these tokenizers so that text starting with a space loses one space, where the folder's
  tokenizer.json, the model's sentencepiece file and transformers 4.57 agree (transformers#47700 describes it).
- Request throughput with everything on against entail not installed, vLLM's default path (torch.compile and CUDA
  graphs), Qwen3-4B: 1.0007x, 1.0028x and 1.0110x at batch 1, 8 and 32 (two identical states differ by up to 0.6%).
- Load: from an installed copy, +1.6 to +1.7 s on vLLM for 0.6-3B models (13-15% of their load), mostly the start-up
  path check (`ENTAIL_NO_PATHS=1` turns it off); the hook alone -0.03 to +0.25 s.
- Detection on unseen bugs (pre-registered, code frozen): the third replay (frozen at 1.2.0's successor `07fceac`)
  detected 0 of the 5 reproduced in-class bugs; the fourth (frozen at this version's code, a new population of 437
  issues from the six months before) detected 0 of the 6, 0 of the 5 low-level ones, and raised one false alarm
  (below). Over four replays: 0 of 26.

**Known issues**
- The start-up path check counts non-finite log-probabilities as agreement: a model whose every log-probability was
  NaN (vLLM 0.16, NVFP4 with float16 activations) passed all three pairs.
- False alarm on encoder-decoder models on vLLM: whisper-large-v3-turbo gets nine `broken` lines at the KV cache
  boundary (`container:vllm.allocate_slots`, cache group 1 holds 0 slots for its tokens) although its output is
  right - the rule does not know that a cross-attention cache group follows the encoder, not the decoder's tokens.
  The run goes on (the default policy reports).
- A tokenizer built from a GGUF file is `unknown` at the tokenizer boundary (the GGUF file's own tokenizer
  declaration is not read), so a GGUF tokenizer built as another type than the file declares is not caught
  (transformers#41494).
- On vLLM's default path (torch.compile) custom ops are compiled and not compared with their definitions; the
  comparison runs in eager mode. Kernels called from C++ (Marlin) are not reached.

## 1.2.0

Released 2026-09-26. The sites the first pre-registered replay found unread (a LoRA adapter's settings file, a
request's template settings at the reasoning parser, the prefix-cache key and the beam reorder, Triton kernel
launches, rotary pairing), each written from the real bug and measured on it, and a second pre-registered replay
with the vocabulary frozen at this version.

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
- README (EN/KO): the second pre-registered replay, with the vocabulary frozen at this version's code (`ce79b19`):
  the next 150 issues screened, 20 passed, 15 reproduced, 8 in the class by two blind raters (kappa 0.72 over
  seven categories, 0.68 in-class versus not; 18 of 86 settled by a third), 0 of the 8 detected (rule of three: at
  most 3 of 8), 0 false alarms on the 7 reproduced outside the class, one run broken by entail (the serve wrapper,
  fixed above). Known gaps name what it left unread, first the ids a built tokenizer produces.

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
