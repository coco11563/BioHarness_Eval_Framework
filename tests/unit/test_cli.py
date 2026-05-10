"""Smoke tests for the CLI surface."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from framework_eval.cli.main import main


def test_no_subcommand_prints_help() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main([])
    assert rc == 0
    out = buf.getvalue()
    assert "framework-eval" in out
    assert "run" in out
    assert "score" in out


def test_thresholds_subcommand_emits_json() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["thresholds"])
    assert rc == 0
    table = json.loads(buf.getvalue())
    assert set(table) == {"yesno", "mcq", "mcq_multi", "factoid", "list", "summary", "expression"}


def test_list_methods_runs_when_no_entries() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["list-methods"])
    assert rc == 0
    # output is either "(no methods..." or a real list; either way printable
    assert buf.getvalue()


def test_verify_without_script_returns_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When verify.py isn't present, the subcommand must fail cleanly."""
    # Ensure neither the expected paths exist
    monkeypatch.chdir(tmp_path)
    buf_err = io.StringIO()
    with redirect_stderr(buf_err):
        rc = main(["verify"])
    # Returns 2 only when verify.py is missing; rc=0 if A9 has shipped.
    if rc != 0:
        assert rc == 2
        assert "verify.py is not present" in buf_err.getvalue()


def test_score_rejects_non_directory(tmp_path: Path) -> None:
    buf = io.StringIO()
    with redirect_stderr(buf):
        rc = main(["score", "--run", str(tmp_path / "nope"), "--output", str(tmp_path / "o")])
    assert rc == 2
    assert "not a directory" in buf.getvalue()


def test_run_subcommand_help() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            main(["run", "--help"])
    assert exc.value.code == 0
    assert "--datasets" in buf.getvalue()
    assert "--method" in buf.getvalue()
