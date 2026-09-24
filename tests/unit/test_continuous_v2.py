"""Tests for the headline continuous-v2 row rules and supplementary pooling."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

from framework_eval.cli.main import main
from framework_eval.eval.continuous_v2 import load_run_v2

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _row(iid: str, subtask: str | None, **kw: object) -> dict:
    base = {
        "type": "item",
        "id": iid,
        "dataset": "bioasq",
        "subtask": subtask,
        "predicted": "",
        "ground_truth": "",
        "score": 0.0,
        "correct": False,
    }
    base.update(kw)
    return base


def test_first_record_per_id_wins(tmp_path: Path) -> None:
    rows = load_run_v2(
        _write(
            tmp_path / "r.jsonl",
            [
                _row("a", "yesno", score=1.0, correct=True),
                _row("a", "yesno", score=0.0, correct=False),
            ],
        )
    )
    assert len(rows) == 1
    assert rows[0].score == 1.0 and rows[0].correct is True


def test_null_dataset_or_subtask_skipped(tmp_path: Path) -> None:
    rows = load_run_v2(
        _write(
            tmp_path / "r.jsonl",
            [
                _row("a", "yesno", dataset=None),
                _row("b", None),
                _row("c", "yesno"),
                {"type": "summary", "id": "s", "dataset": "bioasq", "subtask": "yesno"},
            ],
        )
    )
    assert [r.item_id for r in rows] == ["c"]


def test_factoid_rescored_with_token_f1(tmp_path: Path) -> None:
    rows = load_run_v2(
        _write(
            tmp_path / "r.jsonl",
            [
                _row(
                    "a",
                    "factoid",
                    predicted="the BRCA1 gene",
                    ground_truth="BRCA1",
                    score=0.9,
                    correct=True,
                ),
                _row("b", "list", score=0.25, correct=False),
            ],
        )
    )
    assert abs(rows[0].score - 2 / 3) < 1e-12  # stored 0.9 is ignored
    assert rows[0].correct is True  # binary flag passes through
    assert rows[1].score == 0.25


def test_runner_result_rows_are_read(tmp_path: Path) -> None:
    rows = load_run_v2(
        _write(
            tmp_path / "r.jsonl",
            [
                _row("a", "yesno", type="result", score=1.0, correct=True),
            ],
        )
    )
    assert len(rows) == 1


def test_cli_score_excludes_supplementary_from_overall(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    _write(
        run / "pubmedqa_pqal.jsonl",
        [
            _row("p1", "yesno", dataset="pubmedqa", score=1.0, correct=True, method="m"),
            _row("p2", "yesno", dataset="pubmedqa", score=0.0, correct=False, method="m"),
        ],
    )
    _write(
        run / "litqa2.jsonl",
        [
            _row(f"l{i}", "mcq", dataset="litqa2", score=0.0, correct=False, method="m")
            for i in range(4)
        ],
    )
    assert main(["score", "--run", str(run), "--output", str(tmp_path / "o")]) == 0
    got = {
        r["config"]: r
        for r in csv.DictReader(io.StringIO((tmp_path / "o" / "headline.csv").read_text()))
    }
    assert set(got) == {"pubmedqa_pqal_test", "litqa2", "_overall"}
    assert got["_overall"]["n_items"] == "2"
    assert got["_overall"]["continuous_mean"] == "0.500000"
    assert got["litqa2"]["n_items"] == "4"


def test_verify_paper_table1_flags_drift(capsys: pytest.CaptureFixture[str]) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    import verify

    manifest = verify.tomllib.loads(verify.MANIFEST_PATH.read_text(encoding="utf-8"))
    exact = {c["config"]: c["exact"] for c in manifest["paper_table1"]["cells"]}
    assert verify._verify_paper_table1(manifest, dict(exact)) == []
    drifted = dict(exact, bioasq=exact["bioasq"] - 2e-6)
    errors = verify._verify_paper_table1(manifest, drifted)
    assert len(errors) == 1 and "BioASQ" in errors[0]
    capsys.readouterr()
