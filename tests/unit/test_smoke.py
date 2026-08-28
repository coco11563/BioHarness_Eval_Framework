"""Smoke tests that prove the package, manifest, CLI, and scanner work."""

from __future__ import annotations

import importlib
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parents[2]


def test_package_imports() -> None:
    import framework_eval

    assert framework_eval.__version__


def test_subpackages_import() -> None:
    for name in ("eval", "loader", "runner", "methods", "plugins", "cli"):
        importlib.import_module(f"framework_eval.{name}")


def test_cli_version() -> None:
    """argparse's built-in --version action raises SystemExit(0)."""
    import pytest

    from framework_eval.cli.main import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
    assert exc.value.code == 0
    assert buf.getvalue().strip()


def test_manifest_parses() -> None:
    data = tomllib.loads((ROOT / "MANIFEST.toml").read_text())
    assert data["schema_version"] == "1"
    assert data["dataset"]["hf_repo_id"] == "Shaow/BioHarness_Eval"
    assert len(data["dataset"]["files"]) == 9
    assert len(data["name_map"]) == 9
    assert data["run"]["total_items"] == 21752


def test_manifest_arithmetic() -> None:
    data = tomllib.loads((ROOT / "MANIFEST.toml").read_text())

    ds_lines = sum(f["lines"] for f in data["dataset"]["files"])
    assert ds_lines == data["dataset"]["total_lines"] == 21924

    run_lines = sum(
        f["lines"] for f in data["run"]["files"] if f["config"] != "_summary"
    )
    assert run_lines == data["run"]["total_items"] == 21752

    delta = data["dataset"]["run_subset_delta"]
    assert delta["delta_lines"] == 172
    assert delta["delta_unique_ids"] == 72


def test_pyproject_entry_points_resolve() -> None:
    """Every declared entry point in `framework_eval.methods` must import."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    entry_points = (
        data.get("project", {})
        .get("entry-points", {})
        .get("framework_eval.methods", {})
    )
    for name, target in entry_points.items():
        module_name, _, attr = target.partition(":")
        assert module_name and attr, f"Malformed entry point {name}={target!r}"
        module = importlib.import_module(module_name)
        assert hasattr(module, attr), (
            f"Entry point {name}={target!r} resolves to module but missing attribute"
        )


def test_forbidden_string_scanner_passes_on_repo() -> None:
    """The scanner must report zero hits across the in-tree files."""
    rc = subprocess.run(
        [sys.executable, "scripts/scan_forbidden_strings.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, (
        f"forbidden-string scanner found hits:\n"
        f"stdout:\n{rc.stdout}\nstderr:\n{rc.stderr}"
    )
