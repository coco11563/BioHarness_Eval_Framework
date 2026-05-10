"""Tests for the NoContextLLM sample method."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from framework_eval.eval.types import Item
from framework_eval.methods.no_context_llm import NoContextLLM, _build_prompt


def _item(question_type: str = "yesno", **kw: Any) -> Item:
    base = dict(
        id="x", dataset="bioasq", question="Does aspirin help?",
        question_type=question_type, answer="yes",
    )
    base.update(kw)
    return Item(**base)


def test_build_prompt_includes_options_for_mcq() -> None:
    item = _item(
        question_type="mcq",
        question="Best treatment?",
        options={"A": "metformin", "B": "insulin"},
    )
    text = _build_prompt(item)
    assert "Best treatment?" in text
    assert "A. metformin" in text
    assert "B. insulin" in text


def test_build_prompt_yesno_has_format_instruction() -> None:
    text = _build_prompt(_item(question_type="yesno"))
    assert "yes, no, or maybe" in text


def test_no_context_llm_satisfies_qa_client_protocol() -> None:
    from framework_eval.plugins import QAClient

    assert isinstance(NoContextLLM(), QAClient)


def test_no_context_llm_extracts_normalised_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a mock httpx transport and assert the strict-mode extraction."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"]
        assert body["messages"][0]["role"] == "system"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "FINAL(yes)"}},
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    method = NoContextLLM(api_base="http://test/v1")
    method._client = httpx.AsyncClient(transport=transport)

    item = _item(question_type="yesno")
    pred = asyncio.run(method.generate(item))
    assert pred.answer == "yes"
    assert pred.extras["response_text"] == "FINAL(yes)"

    asyncio.run(method.aclose())


def test_entry_point_registered() -> None:
    """The pyproject entry point must resolve to NoContextLLM."""
    from framework_eval.plugins import discover_entry_points, load_method

    entries = {e.name: e for e in discover_entry_points()}
    if "no-context-llm" not in entries:
        pytest.skip("entry-point only visible after `pip install -e .`")
    cls = load_method("no-context-llm")
    assert cls is NoContextLLM
