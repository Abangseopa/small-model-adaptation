"""Tests for Stage 6 comparison/categorization logic.

Uses synthetic fixture data built to match the EvalRecord-shaped dicts
Stage 3/5 actually save — never reads the real frozen reports/ files, so
these tests don't depend on Steps 3/5 having been run and can't
accidentally perturb the frozen results.
"""

from evaluation.compare_models import (
    _behavioral_breakdown,
    _build_per_example,
    _classify_failure,
    _comparison_table,
)


def _saved_record(id_, answerable=True, **overrides):
    base = dict(
        id=id_, domain="d", subject="s", answerable=answerable,
        expected_answer="Fyodor Dostoevsky", expected_confidence="HIGH",
        model_raw_response="ANSWER: Fyodor Dostoevsky\nCONFIDENCE: HIGH",
        parsed_answer="Fyodor Dostoevsky", parsed_confidence="HIGH",
        answer_correct=True, confidence_correct=True, format_compliant=True,
        full_behavior_success=True,
    )
    base.update(overrides)
    return base


def test_build_per_example_categorizes_all_four_outcomes():
    baseline_by_id = {
        "improved_ex": _saved_record("improved_ex", full_behavior_success=False, answer_correct=False),
        "success_ex": _saved_record("success_ex"),
        "failure_ex": _saved_record("failure_ex", full_behavior_success=False, answer_correct=False),
        "regressed_ex": _saved_record("regressed_ex"),
    }
    adapted_by_id = {
        "improved_ex": _saved_record("improved_ex"),
        "success_ex": _saved_record("success_ex"),
        "failure_ex": _saved_record("failure_ex", full_behavior_success=False, answer_correct=False),
        "regressed_ex": _saved_record("regressed_ex", full_behavior_success=False, answer_correct=False),
    }
    per_example = _build_per_example(baseline_by_id, adapted_by_id)
    by_id = {r["id"]: r for r in per_example}

    assert by_id["improved_ex"]["category"] == "improved"
    assert by_id["success_ex"]["category"] == "unchanged_success"
    assert by_id["failure_ex"]["category"] == "unchanged_failure"
    assert by_id["regressed_ex"]["category"] == "regressed"


def test_build_per_example_asserts_consistent_expected_values():
    baseline_by_id = {"x": _saved_record("x", expected_answer="A")}
    adapted_by_id = {"x": _saved_record("x", expected_answer="B")}
    try:
        _build_per_example(baseline_by_id, adapted_by_id)
        assert False, "expected AssertionError on mismatched expected_answer"
    except AssertionError:
        pass


def test_classify_failure_categories():
    baseline_by_id = {
        "f1": _saved_record("f1"), "f2": _saved_record("f2"),
        "f3": _saved_record("f3"), "f4": _saved_record("f4"),
    }
    adapted_by_id = {
        "f1": _saved_record("f1", format_compliant=False, answer_correct=False,
                             full_behavior_success=False, model_raw_response="Fyodor Dostoevsky"),
        "f2": _saved_record("f2", confidence_correct=False, answer_correct=False,
                             full_behavior_success=False, parsed_confidence="LOW",
                             model_raw_response="ANSWER: Fyodor Dostoevsky\nCONFIDENCE: LOW"),
        "f3": _saved_record("f3", parsed_answer="yes", answer_correct=False,
                             full_behavior_success=False, model_raw_response="ANSWER: yes\nCONFIDENCE: HIGH"),
        "f4": _saved_record("f4", parsed_answer="something unrelated", answer_correct=False,
                             full_behavior_success=False,
                             model_raw_response="ANSWER: something unrelated\nCONFIDENCE: HIGH"),
    }
    per_example = _build_per_example(baseline_by_id, adapted_by_id)
    by_id = {r["id"]: r for r in per_example}

    assert _classify_failure(by_id["f1"]) == "formatting failure"
    assert _classify_failure(by_id["f2"]) == "confidence failure"
    assert _classify_failure(by_id["f3"]) == "semantic/question-interpretation failure"
    assert _classify_failure(by_id["f4"]) == "retrieval/context failure"


def test_behavioral_breakdown_rates_on_a_known_mix():
    baseline_by_id = {
        "a1": _saved_record("a1", answerable=True, full_behavior_success=False, answer_correct=False),
        "u1": _saved_record("u1", answerable=False, full_behavior_success=False, answer_correct=False),
    }
    adapted_by_id = {
        "a1": _saved_record("a1", answerable=True, full_behavior_success=True),
        "u1": _saved_record("u1", answerable=False, full_behavior_success=True),
    }
    per_example = _build_per_example(baseline_by_id, adapted_by_id)
    breakdown = _behavioral_breakdown(per_example)

    assert breakdown["A_format_learning"]["adapted_format_compliance"] == 1.0
    assert breakdown["C_answer_extraction"]["adapted_answerable_answer_accuracy"] == 1.0
    assert breakdown["B_abstention_grounding"]["adapted_unanswerable_answer_accuracy"] == 1.0


def test_comparison_table_computes_change_for_every_metric_key():
    baseline_metrics = {
        "answer_accuracy": 0.0, "confidence_accuracy": 0.0, "format_compliance": 0.0,
        "answerable_accuracy": 0.0, "unanswerable_accuracy": 0.0, "full_behavior_success": 0.0,
    }
    adapted_metrics = {
        "answer_accuracy": 0.958, "confidence_accuracy": 1.0, "format_compliance": 1.0,
        "answerable_accuracy": 0.917, "unanswerable_accuracy": 1.0, "full_behavior_success": 0.958,
    }
    table = _comparison_table(baseline_metrics, adapted_metrics)
    for key, adapted_value in adapted_metrics.items():
        assert table[key]["baseline"] == 0.0
        assert table[key]["adapted"] == adapted_value
        assert table[key]["change"] == adapted_value
