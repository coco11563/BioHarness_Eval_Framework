"""Tests for the async runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from framework_eval.eval.types import Item, Prediction
from framework_eval.runner import RunnerConfig, run, write_summary


class StaticClient:
    """Minimal QAClient that always returns a fixed answer."""

    name = "static"

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.calls = 0

    async def generate(self, item: Item) -> Prediction:
        self.calls += 1
        return Prediction(item_id=item.id, answer=self._answer)

    async def aclose(self) -> None: ...


class FlakyClient:
    """Fails the first N calls with the given exception."""

    name = "flaky"

    def __init__(self, fail_first: int, then_answer: str) -> None:
        self.fail_first = fail_first
        self._answer = then_answer
        self.calls = 0

    async def generate(self, item: Item) -> Prediction:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError(f"forced failure {self.calls}")
        return Prediction(item_id=item.id, answer=self._answer)

    async def aclose(self) -> None: ...


def _items(*answers: str) -> list[Item]:
    return [
        Item(
            id=f"i{i}", dataset="bioasq", question="?", question_type="yesno",
            answer=ans,
        )
        for i, ans in enumerate(answers)
    ]


@pytest.mark.asyncio
async def test_run_writes_one_row_per_item(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    cfg = RunnerConfig(method_name="static", output_path=out, concurrency=2)
    client = StaticClient("yes")

    await run(_items("yes", "yes", "no"), client, cfg)

    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 3
    assert {r["id"] for r in rows} == {"i0", "i1", "i2"}
    # Two correct (yes vs yes), one wrong (yes vs no).
    n_correct = sum(1 for r in rows if r["correct"])
    assert n_correct == 2


@pytest.mark.asyncio
async def test_run_resumes_skipping_completed(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    out.write_text(
        json.dumps({"id": "i0", "subtask": "yesno", "score": 1.0, "correct": True}) + "\n"
    )
    client = StaticClient("yes")
    cfg = RunnerConfig(method_name="static", output_path=out, concurrency=1)

    await run(_items("yes", "yes"), client, cfg)
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    # Only i1 was generated; i0 row was preserved.
    assert client.calls == 1


@pytest.mark.asyncio
async def test_run_skips_unanswered(tmp_path: Path) -> None:
    items = [
        Item(id="a", dataset="x", question="?", question_type="yesno", answer="yes"),
        Item(id="b", dataset="x", question="?", question_type="yesno", answer=None),
    ]
    out = tmp_path / "run.jsonl"
    cfg = RunnerConfig(method_name="static", output_path=out, concurrency=1)
    await run(items, StaticClient("yes"), cfg)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert {r["id"] for r in rows} == {"a"}


@pytest.mark.asyncio
async def test_retry_succeeds_after_failures(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    client = FlakyClient(fail_first=2, then_answer="yes")
    cfg = RunnerConfig(
        method_name="flaky", output_path=out, concurrency=1,
        max_retries=3, retry_base_seconds=0.0,
    )
    await run(_items("yes"), client, cfg)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["correct"] is True
    assert client.calls == 3   # two failures + one success


@pytest.mark.asyncio
async def test_retry_exhausted_records_error(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    client = FlakyClient(fail_first=99, then_answer="yes")
    cfg = RunnerConfig(
        method_name="flaky", output_path=out, concurrency=1,
        max_retries=2, retry_base_seconds=0.0,
    )
    await run(_items("yes"), client, cfg)
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    assert rows[0]["correct"] is False
    assert "error" in rows[0]["metadata"]


def test_write_summary(tmp_path: Path) -> None:
    out = tmp_path / "run.jsonl"
    out.write_text(
        json.dumps({"id": "a", "subtask": "yesno", "score": 1.0, "correct": True}) + "\n"
        + json.dumps({"id": "b", "subtask": "yesno", "score": 0.0, "correct": False}) + "\n"
    )
    summary_path = tmp_path / "summary.json"
    s = write_summary(
        method_name="m", output_path=out, summary_path=summary_path,
    )
    assert s["total"] == 2
    assert s["correct"] == 1
    assert s["binary_accuracy"] == 0.5
