"""Async runner smoke test that does not require pytest-asyncio.

The unit-level async tests live in ``tests/unit/test_runner.py`` and need
``pytest-asyncio``. This module exercises the same paths via plain
``asyncio.run`` so the runner is at least minimally covered in environments
that ship without the plugin (e.g. minimal CI images).
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from framework_eval.eval.types import Item, Prediction
from framework_eval.runner import RunnerConfig, run, write_summary


class _StaticClient:
    name = "static"

    def __init__(self, answer: str = "yes") -> None:
        self._answer = answer
        self.calls = 0

    async def generate(self, item: Item) -> Prediction:
        self.calls += 1
        return Prediction(item_id=item.id, answer=self._answer)

    async def aclose(self) -> None:
        pass


def _items(*answers: str | None) -> list[Item]:
    return [
        Item(
            id=f"i{i}", dataset="bioasq", question="?",
            question_type="yesno", answer=ans,
        )
        for i, ans in enumerate(answers)
    ]


async def _run_full() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run.jsonl"
        cfg = RunnerConfig(method_name="static", output_path=out, concurrency=2)

        # Initial run
        await run(_items("yes", "yes", "no"), _StaticClient(), cfg)
        rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert len(rows) == 3
        assert sum(1 for r in rows if r["correct"]) == 2

        # Resume: a second run with the same items must skip
        client = _StaticClient()
        await run(_items("yes", "yes", "no"), client, cfg)
        assert client.calls == 0

        # Skip null-answer items
        out2 = Path(d) / "run2.jsonl"
        await run(
            _items(None, "yes"),
            _StaticClient(),
            RunnerConfig(method_name="static", output_path=out2, concurrency=1),
        )
        rows2 = [json.loads(l) for l in out2.read_text().splitlines() if l.strip()]
        assert {r["id"] for r in rows2} == {"i1"}

        # Summary
        s = write_summary(
            method_name="static",
            output_path=out,
            summary_path=Path(d) / "summary.json",
        )
        assert s["total"] == 3
        assert s["correct"] == 2


def test_runner_end_to_end() -> None:
    asyncio.run(_run_full())
