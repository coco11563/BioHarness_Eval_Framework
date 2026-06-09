# Contributing to bioHarness

Thank you for your interest in contributing.

## Development setup

```bash
git clone https://github.com/coco11563/XCompass_Eval_Framework.git
cd XCompass_Eval_Framework
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
python verify.py        # offline reproducibility gate
```

All four must pass before a pull request is merged. CI runs the same set on
every push. Once `verify.py` lands (A9), CI will additionally refuse to
merge if it reports any drift in the golden artefacts.

## Adding a new evaluation method

Implement the `framework_eval.plugins.QAClient` protocol and either

1. Register it via the `framework_eval.methods` entry point in your
   `pyproject.toml`, or
2. Pass `--method package.module:ClassName` on the command line for
   ad-hoc use.

See `src/framework_eval/methods/no_context_llm.py` for a reference
implementation and `docs/plugins.md` for the full protocol contract.

## Frozen artefacts

The following files are part of the reproducibility contract and **must not
be modified casually**:

- `MANIFEST.toml`
- `golden/overall_continuous.csv`
- `golden/per_dataset/*.csv`
- `golden/excluded_ids.json`
- `golden/duplicate_source_ids.json`
- `output/pipeline/*.jsonl`

If your change legitimately alters scoring, you must:

1. Update the in-tree evaluator and tests with reasoning in the PR.
2. Re-generate the golden CSVs with `python scripts/regenerate_golden.py`.
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

Open an issue at <https://github.com/coco11563/XCompass_Eval_Framework/issues>
including the failing command, full traceback, OS / Python version, and
manifest id (`framework-eval verify --print-manifest-id`).
