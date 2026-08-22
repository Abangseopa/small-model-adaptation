"""Tests for Stage 3's pure parsing/matching/scoring functions.

No model is loaded here — these are plain string/regex functions and a
dataclass, so they run in milliseconds.
"""

from baseline_evaluation.run_baseline_eval import (
    STRICT_FORMAT_RE,
    EvalRecord,
    _answer_matches,
    _compute_metrics,
    _parse_expected,
    _parse_model_output,
    _split_context_question,
)


def test_split_context_question():
    prompt = "Context: Paris is in France.\nQuestion: Where is Paris?\nInstruction: blah"
    context, question = _split_context_question(prompt)
    assert context == "Paris is in France."
    assert question == "Where is Paris?"


def test_parse_expected_extracts_answer_and_confidence():
    answer, confidence = _parse_expected("ANSWER: Paris\nCONFIDENCE: HIGH")
    assert answer == "Paris"
    assert confidence == "HIGH"


def test_parse_expected_rejects_malformed_response():
    try:
        _parse_expected("Paris is the answer.")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_parse_model_output_extracts_from_clean_format():
    answer, confidence = _parse_model_output("ANSWER: Nairobi\nCONFIDENCE: HIGH")
    assert answer == "Nairobi"
    assert confidence == "HIGH"


def test_parse_model_output_tolerates_preamble_and_trailing_text():
    text = "Sure! Here you go.\nANSWER: Oslo\nCONFIDENCE: HIGH\nHope that helps!"
    answer, confidence = _parse_model_output(text)
    assert answer == "Oslo"
    assert confidence == "HIGH"


def test_parse_model_output_returns_none_when_format_absent():
    answer, confidence = _parse_model_output("Oslo is the capital of Norway.")
    assert answer is None
    assert confidence is None


def test_answer_matches_is_case_and_punctuation_tolerant():
    assert _answer_matches("Paris", "paris.")
    assert _answer_matches("Paris", "**Paris**")
    assert not _answer_matches("Paris", "Oslo")
    assert not _answer_matches("Paris", None)


def test_strict_format_regex_requires_exact_two_line_shape():
    assert STRICT_FORMAT_RE.fullmatch("ANSWER: Paris\nCONFIDENCE: HIGH")
    assert not STRICT_FORMAT_RE.fullmatch("Paris\nCONFIDENCE: HIGH")
    assert not STRICT_FORMAT_RE.fullmatch("ANSWER: Paris\nCONFIDENCE: MAYBE")


def _record(**overrides):
    base = dict(
        id="x", domain="d", subject="s", answerable=True, context="c", question="q",
        expected_answer="Paris", expected_confidence="HIGH",
        model_raw_response="ANSWER: Paris\nCONFIDENCE: HIGH",
        parsed_answer="Paris", parsed_confidence="HIGH",
        answer_correct=True, confidence_correct=True, format_compliant=True,
        full_behavior_success=True,
    )
    base.update(overrides)
    return EvalRecord(**base)


def test_compute_metrics_on_a_known_mix_of_outcomes():
    records = [
        _record(id="a1", answerable=True, answer_correct=True, confidence_correct=True,
                format_compliant=True, full_behavior_success=True),
        _record(id="a2", answerable=True, answer_correct=False, confidence_correct=True,
                format_compliant=True, full_behavior_success=False),
        _record(id="u1", answerable=False, answer_correct=True, confidence_correct=True,
                format_compliant=True, full_behavior_success=True),
        _record(id="u2", answerable=False, answer_correct=False, confidence_correct=False,
                format_compliant=False, full_behavior_success=False),
    ]
    metrics = _compute_metrics(records)
    assert metrics["n_examples"] == 4
    assert metrics["n_answerable"] == 2
    assert metrics["n_unanswerable"] == 2
    assert metrics["answer_accuracy"] == 0.5
    assert metrics["confidence_accuracy"] == 0.75
    assert metrics["format_compliance"] == 0.75
    assert metrics["answerable_accuracy"] == 0.5
    assert metrics["unanswerable_accuracy"] == 0.5
    assert metrics["full_behavior_success"] == 0.5


def test_compute_metrics_handles_all_pass():
    records = [_record(id=f"r{i}") for i in range(3)]
    metrics = _compute_metrics(records)
    assert metrics["full_behavior_success"] == 1.0
    assert metrics["answer_accuracy"] == 1.0
