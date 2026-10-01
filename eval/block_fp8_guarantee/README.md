# Block FP8 guarantee: evaluation harness

ROADMAP M19 L5.4a. Design: `docs/semantic-guarantee-design.ko.md`. Protocol: `docs/semantic-guarantee-gpu-protocol.ko.md`.
The profile itself is `entail/guarantee.py` with the adapter `entail/adapters/vllm_block_fp8_guarantee.py`, turned on
by `ENTAIL=guarantee`.

These are research tools, not part of the library: they need vLLM 0.30, a CUDA GPU with FP8 (sm_89 or later) and
Triton.

| file | what |
|---|---|
| `common.py` | the independent oracle (numpy float64, bit-field fp8 decoding), the real vLLM path (`TritonFp8BlockScaledMMKernel` from vLLM's `init_fp8_linear_kernel`), the consumer mutations (vLLM's kernel source with one misread, switched by a device flag; a call-site scale cache), the observers, the environment record |
| `cases.py` | the 24 scenarios (normal 8, consumer defects 4 x 3 timings, admission 4) and one development case (the known BLOCK_SIZE_K 256 tile defect); dev seeds fixed, holdout seeds drawn from the freeze manifest's sha256 |
| `run_case.py` | one scenario in one mode in a fresh process |
| `run_suite.py` | every scenario in off A, off B, load and guarantee, each a fresh process with entail's start-up hook |
| `aggregate.py` | the verdicts from what the next operation read: normal, repaired, blocked, wrong escaped, unknown, error; passed / failed / incomplete |
| `calibrate.py` | the oracle's own checks (all 256 fp8 codes, an exact power-of-two case) and the normal calibration of both tolerances |
| `freeze.py` | the freeze manifest (plan, tolerances, environment, code hashes, holdout rule); it is also the plan file `ENTAIL_GUARANTEE_PLAN` reads |
| `cost.py` | start, first call, normal repeat and reference-path repeat, time and memory, off and guarantee in alternating fresh processes |
| `engine_smoke.py` | the real engine (vLLM 0.30, one FP8 model): producers and consumer reached in every layer, off against guarantee, RNG and prefix cache |

Order: environment record and dev suite, `calibrate.py`, `freeze.py`, then the holdout suite with `--freeze`, cost and
the engine smoke with `BFG_FREEZE` set. Results that are looked at and lead to a change go to a new frozen version;
the first holdout's results are kept as they are.
