"""Confirms Stage 5 (inference.evaluate_adapted) reuses — rather than
redefines — Stage 3's scoring logic and Stage 4's fingerprint helpers.

This is an architectural invariant test: if someone later "forks" the
scoring/fingerprint code into evaluate_adapted.py instead of importing it,
these tests catch the duplication even though the duplicated logic might
otherwise look correct.
"""

import baseline_evaluation.run_baseline_eval as baseline
import inference.evaluate_adapted as adapted
import training.train as train_module


def test_evaluate_adapted_imports_baseline_scoring_functions_directly():
    assert adapted._compute_metrics is baseline._compute_metrics
    assert adapted._parse_expected is baseline._parse_expected
    assert adapted._parse_model_output is baseline._parse_model_output
    assert adapted._answer_matches is baseline._answer_matches
    assert adapted._split_context_question is baseline._split_context_question
    assert adapted.STRICT_FORMAT_RE is baseline.STRICT_FORMAT_RE
    assert adapted.EvalRecord is baseline.EvalRecord
    assert adapted.MODEL_ID == baseline.MODEL_ID
    assert adapted.MAX_NEW_TOKENS == baseline.MAX_NEW_TOKENS


def test_evaluate_adapted_reuses_training_fingerprint_helpers():
    assert adapted._fingerprint is train_module._fingerprint
    assert adapted._normalize_param_name is train_module._normalize_param_name
