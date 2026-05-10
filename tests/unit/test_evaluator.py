"""Tests for the threshold table, evaluator dispatch, and aggregator."""

from __future__ import annotations

from math import isclose

import pytest

from framework_eval.eval import (
    Aggregate,
    MetricsEvaluator,
    PerTypeAggregate,
    aggregate,
    aggregate_per_type,
    emit_headline_csv,
    emit_per_type_csv,
)
from framework_eval.eval.threshold import TYPE_THRESHOLDS
from framework_eval.eval.types import EvalResult, Item


# ---------- threshold table -----------------------------------------------


def test_thresholds_cover_all_seven_types() -> None:
    expected = {"yesno", "mcq", "mcq_multi", "factoid", "list", "summary", "expression"}
    assert set(TYPE_THRESHOLDS) == expected


def test_threshold_cutoffs_match_paper() -> None:
    assert TYPE_THRESHOLDS["mcq_multi"].cutoff == 0.5
    assert TYPE_THRESHOLDS["factoid"].cutoff == 0.2
    assert TYPE_THRESHOLDS["list"].cutoff == 0.3
    assert TYPE_THRESHOLDS["summary"].cutoff == 0.1
    assert TYPE_THRESHOLDS["expression"].cutoff == 0.3


# ---------- MetricsEvaluator dispatch -------------------------------------


@pytest.fixture()
def ev() -> MetricsEvaluator:
    return MetricsEvaluator()


def test_evaluate_yesno_correct(ev: MetricsEvaluator) -> None:
    r = ev.evaluate("x", "yes", "yes", "yesno")
    assert r.score == 1.0
    assert r.correct is True


def test_evaluate_yesno_wrong(ev: MetricsEvaluator) -> None:
    r = ev.evaluate("x", "yes", "no", "yesno")
    assert r.score == 0.0
    assert r.correct is False


def test_evaluate_mcq_with_options(ev: MetricsEvaluator) -> None:
    r = ev.evaluate("x", "A", "metformin", "mcq", {"A": "metformin", "B": "insulin"})
    assert r.correct is True


def test_evaluate_mcq_multi_partial_below_threshold(ev: MetricsEvaluator) -> None:
    # F1 = 0.5 -> correct (cutoff is inclusive at 0.5)
    r = ev.evaluate("x", "['A', 'B']", "['A', 'C']", "mcq_multi")
    assert isclose(r.score, 0.5)
    assert r.correct is True


def test_evaluate_mcq_multi_just_below_threshold(ev: MetricsEvaluator) -> None:
    # pred=A, gt=A,B,C -> P=1, R=1/3, F1=0.5 -> correct (>= 0.5)
    r = ev.evaluate("x", "['A']", "['A','B','C']", "mcq_multi")
    assert isclose(r.score, 0.5)
    assert r.correct is True


def test_evaluate_mcq_multi_strictly_below(ev: MetricsEvaluator) -> None:
    # pred=A,B, gt=A,C,D -> P=0.5, R=1/3, F1=2*0.5*1/3/(0.5+1/3)=0.4
    r = ev.evaluate("x", "['A','B']", "['A','C','D']", "mcq_multi")
    assert r.score < 0.5
    assert r.correct is False


def test_evaluate_factoid_above_threshold(ev: MetricsEvaluator) -> None:
    r = ev.evaluate("x", "BRCA1", "BRCA1", "factoid")
    assert r.score == 1.0
    assert r.correct is True


def test_evaluate_factoid_below_threshold(ev: MetricsEvaluator) -> None:
    r = ev.evaluate("x", "completely unrelated noise text", "EGFR", "factoid")
    assert r.score < 0.2
    assert r.correct is False


def test_evaluate_summary_self_match(ev: MetricsEvaluator) -> None:
    r = ev.evaluate("x", "FGF21 is hepatic.", "FGF21 is hepatic.", "summary")
    assert isclose(r.score, 1.0)
    assert r.correct is True


def test_evaluate_list_synonym_groups(ev: MetricsEvaluator) -> None:
    pred = '["EGF", "betacellulin"]'
    gt = '[["EGF"], ["betacellulin"], ["TGF-alpha"]]'
    r = ev.evaluate("x", pred, gt, "list")
    expected_f1 = 2 * 1.0 * (2 / 3) / (1.0 + 2 / 3)
    assert isclose(r.score, expected_f1)
    # F1 ~= 0.8 >> 0.3 -> correct
    assert r.correct is True


def test_evaluate_expression_dict(ev: MetricsEvaluator) -> None:
    r = ev.evaluate(
        "x",
        '{"tissue_list": ["liver", "kidney"]}',
        '{"tissue_list": ["liver"]}',
        "expression",
    )
    # F1 = 2/3 -> >= 0.3 -> correct
    assert r.correct is True


def test_evaluate_null_ground_truth_returns_unscoreable(ev: MetricsEvaluator) -> None:
    r = ev.evaluate("x", "yes", None, "yesno")
    assert r.score == 0.0
    assert r.correct is False
    assert r.detail["reason"] == "no_ground_truth"


# Note: passing an unknown `question_type` is impossible without bypassing
# the typed contract — the Item / EvalResult Pydantic models reject any
# value outside the QuestionType literal at construction time, so the
# `unknown_type` branch in MetricsEvaluator.evaluate is unreachable in
# normal use and is kept only as a defensive guard.


def test_thresholds_introspection() -> None:
    table = MetricsEvaluator.thresholds()
    assert {qt for qt in table} == set(TYPE_THRESHOLDS)
    assert table["mcq_multi"]["cutoff"] == 0.5


# ---------- aggregator ---------------------------------------------------


def _result(item_id: str, qtype: str, score: float, correct: bool) -> EvalResult:
    return EvalResult(
        item_id=item_id, question_type=qtype, score=score, correct=correct,
    )


def test_aggregate_basic() -> None:
    rows = [
        _result("a", "yesno", 1.0, True),
        _result("b", "yesno", 0.0, False),
        _result("c", "factoid", 0.5, True),
    ]
    agg = aggregate("my_method", "bioasq", rows)
    assert agg.method == "my_method"
    assert agg.config == "bioasq"
    assert agg.n_items == 3
    assert isclose(agg.binary_accuracy, 2 / 3)
    assert isclose(agg.continuous_mean, (1.0 + 0.0 + 0.5) / 3)
    assert agg.delta_pp == round(100.0 * (0.5 - 2 / 3), 2)


def test_aggregate_empty() -> None:
    agg = aggregate("m", "x", [])
    assert agg == Aggregate("m", "x", 0, 0.0, 0.0, 0.0)


def test_aggregate_per_type_groups() -> None:
    items = [
        Item(id="a", dataset="bioasq", question="?", question_type="yesno", answer="yes"),
        Item(id="b", dataset="bioasq", question="?", question_type="yesno", answer="no"),
        Item(id="c", dataset="bioasq", question="?", question_type="factoid", answer="x"),
    ]
    results = [
        _result("a", "yesno", 1.0, True),
        _result("b", "yesno", 0.0, False),
        _result("c", "factoid", 0.4, True),
    ]
    rows = aggregate_per_type("m", "bioasq", items, results)
    assert {r.question_type for r in rows} == {"yesno", "factoid"}
    yn = next(r for r in rows if r.question_type == "yesno")
    assert yn.n_items == 2
    assert isclose(yn.binary_accuracy, 0.5)


# ---------- CSV emitter ---------------------------------------------------


def test_headline_csv_format_byte_stable() -> None:
    rows = [
        Aggregate("m1", "bioasq", 100, 0.766035, 0.691446, -7.46),
        Aggregate("m2", "bioasq", 100, 0.7, 0.65, -5.0),
    ]
    csv = emit_headline_csv(rows)
    expected = (
        "method,config,n_items,binary_accuracy,continuous_mean,delta_pp\n"
        "m1,bioasq,100,0.766035,0.691446,-7.460000\n"
        "m2,bioasq,100,0.700000,0.650000,-5.000000\n"
    )
    assert csv == expected


def test_per_type_csv_sort() -> None:
    rows = [
        PerTypeAggregate("m1", "bioasq", "yesno", 10, 1.0, 1.0),
        PerTypeAggregate("m1", "bioasq", "factoid", 5, 0.6, 0.55),
    ]
    csv = emit_per_type_csv(rows)
    # Sorted by (method, config, question_type)
    assert csv.splitlines()[1].startswith("m1,bioasq,factoid")
    assert csv.splitlines()[2].startswith("m1,bioasq,yesno")


def test_csv_uses_lf_line_endings() -> None:
    rows = [Aggregate("m", "c", 1, 0.5, 0.5, 0.0)]
    csv = emit_headline_csv(rows)
    assert "\r\n" not in csv
    assert csv.endswith("\n")
