"""Tests for answer extraction."""

from __future__ import annotations

import pytest

from framework_eval.eval.extraction import extract_answer


# ---------- FINAL() unwrap ------------------------------------------------


def test_final_unwrap_basic() -> None:
    assert extract_answer("FINAL(yes)", "yesno") == "yes"


def test_final_unwrap_with_quotes_and_text_around() -> None:
    s = "Some thinking. FINAL(\"yes\") trailing"
    assert extract_answer(s, "yesno") == "yes"


def test_final_unwrap_handles_inner_parens() -> None:
    s = "FINAL(P53 (TP53))"
    assert extract_answer(s, "factoid") == "P53 (TP53)"


def test_final_unwrap_unbalanced_returns_full_text() -> None:
    s = "FINAL(p < 0.05"
    # Unbalanced FINAL falls back to original text; factoid then trims to
    # the first sentence which slices on the period in '0.05'.
    assert extract_answer(s, "factoid") == "FINAL(p < 0"


# ---------- yesno ---------------------------------------------------------


def test_yesno_exact_label() -> None:
    assert extract_answer("yes", "yesno") == "yes"
    assert extract_answer("No", "yesno") == "no"
    assert extract_answer("maybe", "yesno") == "maybe"


def test_yesno_strict_no_signal_returns_empty() -> None:
    # No exact label, no first-word, no recognized polarity phrases.
    assert extract_answer("yesterday's evidence is mixed.", "yesno") == ""


def test_yesno_lenient_falls_back_to_substring() -> None:
    assert extract_answer("yesterday we showed support", "yesno", mode="lenient") == "yes"


def test_yesno_uncertain_phrase() -> None:
    assert extract_answer("Evidence is inconclusive.", "yesno") == "maybe"


# ---------- mcq -----------------------------------------------------------


def test_mcq_single_letter() -> None:
    assert extract_answer("B", "mcq") == "B"


def test_mcq_answer_is_phrase() -> None:
    assert extract_answer("The answer is C.", "mcq") == "C"


def test_mcq_options_match() -> None:
    options = {"A": "metformin", "B": "insulin"}
    assert extract_answer("I would prescribe metformin.", "mcq", options) == "A"


# ---------- mcq_multi -----------------------------------------------------


def test_mcq_multi_json_array() -> None:
    assert extract_answer("['A', 'C']", "mcq_multi") == "['A', 'C']"


def test_mcq_multi_comma_form() -> None:
    assert extract_answer("A, C", "mcq_multi") == "['A', 'C']"


def test_mcq_multi_strict_rejects_prose() -> None:
    """In strict mode, free-form prose is not a structured answer."""
    assert extract_answer("Probably options A and C.", "mcq_multi") == ""


def test_mcq_multi_lenient_negation_filter() -> None:
    out = extract_answer("A and C are correct, D is wrong.", "mcq_multi", mode="lenient")
    assert out == "['A', 'C']"


# ---------- factoid -------------------------------------------------------


def test_factoid_strips_prefix_and_first_sentence() -> None:
    assert extract_answer("The answer is EGFR. It is a kinase.", "factoid") == "EGFR"


# ---------- list ----------------------------------------------------------


def test_list_bullets() -> None:
    s = "- EGF\n- betacellulin\n- TGF-alpha"
    assert extract_answer(s, "list") == "EGF, betacellulin, TGF-alpha"


def test_list_numbered() -> None:
    s = "1. EGF\n2. betacellulin"
    assert extract_answer(s, "list") == "EGF, betacellulin"


# ---------- summary -------------------------------------------------------


def test_summary_strips_prefix() -> None:
    assert extract_answer("In summary: FGF21 is hepatic.", "summary") == "FGF21 is hepatic."


# ---------- expression ----------------------------------------------------


def test_expression_strips_prefix_and_lists() -> None:
    s = "Expressed in: liver, kidney, heart"
    assert extract_answer(s, "expression") == "liver, kidney, heart"


def test_unknown_question_type_returns_text() -> None:
    # Out-of-spec question types pass through unchanged.
    assert extract_answer("hello", "factoid") == "hello"


def test_empty_input_returns_empty() -> None:
    assert extract_answer("", "yesno") == ""
