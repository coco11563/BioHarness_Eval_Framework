# Contributing to BioHarness Eval Framework

Thank you for your interest in contributing.

## Development setup

```bash
git clone https://github.com/coco11563/BioHarness_Eval_Framework.git
cd BioHarness_Eval_Framework
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Running checks locally

```bash
ruff check .
ruff format --check .
mypy
pytest -q
python verify.py --protocol all   # offline reproducibility gate
```

CI runs the same set on every push and refuses to merge if `verify.py`
reports any drift in the golden artefacts or in the paper Table 1 cells.
`pytest` and `verify.py` must pass.

Known pre-existing failures, to be fixed in a separate change: `ruff check`
reports about 47 lint findings and `ruff format --check` about 33 files
that are not formatted (both mostly in modules that predate 0.2.0), and
`mypy` reports two errors in `src/framework_eval/loader/external.py` (with
a recent numpy installed it first stops on a numpy stub that needs
Python 3.12 syntax, because `[tool.mypy]` sets `python_version = "3.10"`). The
pre-commit and type jobs of CI therefore fail until that change lands. Do
not add new findings in the files you touch.

## Adding a new evaluation method

Implement the `framework_eval.plugins.QAClient` protocol and either

1. Register it via the `framework_eval.methods` entry point in your
   `pyproject.toml`, or
2. Pass `--method package.module:ClassName` on the command line for
   ad-hoc use.

See `src/framework_eval/methods/no_context_llm.py` for a reference
implementation and `src/framework_eval/plugins/protocol.py` for the
protocol contract.

## Frozen artefacts

The following files are part of the reproducibility contract and **must not
be modified casually**:

- `MANIFEST.toml`
- `golden/headline.csv`, `golden/headline_per_dataset/*.csv` (legacy protocol)
- `golden/continuous_v2/**/*.csv` (headline protocol)
- `golden/excluded_ids.json`
- `golden/duplicate_source_ids.json`
- `output/pipeline/*.jsonl`, `output/pipeline/summary.json`
- `src/framework_eval/loader/litqa2_ids.jsonl`

If your change legitimately alters scoring, you must:

1. Update the in-tree evaluator and tests with reasoning in the PR.
2. Re-generate the golden CSVs with `python scripts/regenerate_golden.py`
   (after the run files are final).
3. Update `MANIFEST.toml` with the new hashes.
4. Bump the framework version in `pyproject.toml` and add a `CHANGELOG.md`
   entry under a new section.
5. Document the user-visible impact in `docs/scoring-contract.md`.

## Forbidden content

The pre-commit hook and CI both run `scripts/scan_forbidden_strings.py`,
which rejects any code or commit message containing non-release metadata,
local tooling names, or credentials. The full list lives in that script;
if you trip it, rephrase rather than suppress the check.

## Reporting issues

Open an issue at <https://github.com/coco11563/BioHarness_Eval_Framework/issues>
including the failing command, full traceback, OS / Python version, and
manifest id (`python verify.py --print-manifest-id`).
