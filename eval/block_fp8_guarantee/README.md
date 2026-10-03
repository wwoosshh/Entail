# Block FP8 guarantee: evaluation harness

ROADMAP M19 L5.4a-L5.4e. Design: `docs/semantic-guarantee-design.ko.md`. Protocol:
`docs/semantic-guarantee-gpu-protocol.ko.md`. The profile itself is `entail/guarantee.py` with the adapter
`entail/adapters/vllm_block_fp8_guarantee.py`, turned on by `ENTAIL=guarantee` (the numeric guarantee). From v3
(L5.4c) the structural check experiment runs as its own mode, `ENTAIL=structure`, and is reported apart: it is not
the guarantee. From v5 (L5.4e) its integrity is measured two more ways, as two more modes: `ENTAIL=structure_writes`
(between the operations: writes into what the producers issued are refused before they run, `entail/writeguard.py`)
and `ENTAIL=structure_inkernel` (inside the operation: the consumer kernel adds up the bytes it reads and compares them
on the device with the issue's, `entail/inkernel.py`).

These are research tools, not part of the library: they need vLLM 0.30, a CUDA GPU with FP8 (sm_89 or later) and
Triton.

| file | what |
|---|---|
| `common.py` | the independent oracle (numpy float64, bit-field fp8 decoding), the real vLLM path (`TritonFp8BlockScaledMMKernel` from vLLM's `init_fp8_linear_kernel`), the consumer mutations (vLLM's kernel source with one misread, switched by a device flag; a call-site scale cache), the observers, the environment record |
| `cases.py` | the 24 scenarios (normal 8, consumer defects 4 x 3 timings, admission 4), from v3 two integrity cases (bytes changed between producer and consumer, eager and at a graph replay), from v5 six more (a Triton kernel through a pointer to them, a Triton kernel past its own buffer's end, the weight between calls, an activation scale, a captured Triton write, an operator whose schema says it only reads), and one development case (the known BLOCK_SIZE_K 256 tile defect); dev seeds fixed, holdout seeds drawn from the freeze manifest's sha256 |
| `run_case.py` | one scenario in one mode in a fresh process; in a graph - and, from v5, after an eager call of an integrity case - the next operation copies what it read to pinned host memory and marks that it ran, so a device stop at a gate is observed (the next operation did not run); the integrity writes as the case says (`common.writers`, `common.quiet_flip`) |
| `run_suite.py` | every scenario in off A, off B, load, guarantee, structure and (v5) structure_writes and structure_inkernel, each a fresh process with entail's start-up hook |
| `aggregate.py` | the verdicts from what the next operation read: normal, repaired, blocked (a replay: only when the next operation did not run), wrong escaped (NaN included), refused after use, stopped, unknown, error; passed / failed / incomplete |
| `calibrate.py` | the oracle's own checks (all 256 fp8 codes, an exact power-of-two case) and the normal calibration of both tolerances |
| `freeze_check.py` | before a frozen evaluation: every code file the manifest lists (sha256), the entail commit and the environment record against the manifest; the holdout, cost and engine runs do not start on a mismatch (v4) |
| `selftest.py` | CPU self-test of the bookkeeping: completeness and partial runs in `aggregate.py`, mismatches in `freeze_check.py` |
| `freeze.py` | the freeze manifest (the guarantee's `plan`, the structural experiment's `structure_plan` and, from v5, `structure_writes_plan` and `structure_inkernel_plan`, tolerances, environment, code hashes, holdout rule); it is also the plan file `ENTAIL_GUARANTEE_PLAN` reads |
| `cost.py` | start, first call, normal repeat and reference-path repeat, time and memory, off and the modes `BFG_COST_MODES` names, in rotating fresh processes |
| `engine_smoke.py` | the real engine (vLLM 0.30, one FP8 model): producers and consumer reached in every layer, off against each mode of the manifest (or `BFG_ENGINE_MODES`), RNG and prefix cache |

Order: environment record and dev suite, `calibrate.py`, `freeze.py`, then the holdout suite with `--freeze`, cost and
the engine smoke with `BFG_FREEZE` set. From v4 each of the three checks the freeze first (`freeze_check.py`), and the
engine smoke takes its settings (configurations, modes, gpu_memory_utilization, rounds) from the manifest. Results that are looked at and lead to a change go to a new frozen version;
the first holdout's results are kept as they are.
