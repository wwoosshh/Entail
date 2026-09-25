# Contributing

entail is measured software: every rule has a test problem behind it, and every row of the capability table names
its evidence. Contributions keep that shape. The research behind the numbers is public at
https://github.com/wwoosshh/entail-research.

## Reporting a wrong report (a false alarm) or a miss

Open an issue with:

- the engine and its version, and the model (a Hugging Face id, or the `config.json` if it is private);
- the line entail printed, and the matching line from `entail_logs/record-<date>.jsonl` (one JSON object per
  decision: the boundary, the declared fact and where it came from, what the consumer used, the rule);
- the output of `entail doctor` (versions, and which adapters hooked).

If the output was wrong and entail said nothing: say what the model files declare and what the engine ran with.
That pair is what entail compares, and a miss means either a fact outside its vocabulary or a boundary it does not
hook yet. Both are worth knowing.

## Adding a row to the capability table (`entail/data/caps.json`)

A row says whether a consumer (`engine.role.name`) honours one fact, with its evidence:

- `measured`: a result file of `entail probe` (two model folders that differ in one declaration, decoded by the
  consumer; the fact binds or it does not). Only measured rows are used to switch a backend.
- `code`: the file and line that reads (or ignores) the value. A mismatch known only from code is reported as
  inferred and switches nothing.
- `documented`: a document of the engine.

Run the probe in the engine's own environment (it needs the GPU):

```bash
entail probe --engine sglang --consumer triton --fact ModelProps.softcap --model /path/to/gemma-2-2b-it --out row.json
```

and attach `row.json` to the pull request.

## Adapters and engine versions

Adapters hold no rules. Each one has a hook, `read_choice` (what the engine chose) and `handles` (how a resolution
is carried out); `tests/test_adapter_rules.py` fails when a rule appears in one. The rules live in `load.py`,
`contracts.py`, `kv_contract.py` and `request_contract.py`. To support a new engine version: run `entail check`
on a few models and then the engine itself with `ENTAIL=load` on the same models, compare the outputs with the
run without entail, and add the version to the adapter's `versions` string with the result.

## Tests

```bash
PYTHON=python bash tests/run_all.sh
```

CPU torch and `transformers==5.17.0` are enough; tests that need a local model skip without one. CI runs the same
on Python 3.10 and 3.12 (`.github/workflows/tests.yml`).

## Style

Match the surrounding code. A docstring says what a function decides and which test problem or measurement it
came from. Comments that cite "the researcher" or a date refer to the research log in entail-research.
