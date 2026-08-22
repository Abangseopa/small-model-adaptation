"""Tests for Stage 2 dataset generation (data_preparation.prepare_dataset).

Operate entirely in-memory on build_dataset()'s output — never read or
write data/processed/*.jsonl, so these pass on a completely fresh
checkout with no generated artifacts present.
"""

import dataclasses

from data_preparation.prepare_dataset import (
    RESPONSE_PATTERN,
    build_dataset,
    validate_dataset,
)


def test_dataset_has_expected_split_sizes():
    examples = build_dataset()
    train = [e for e in examples if e.split == "train"]
    test = [e for e in examples if e.split == "test"]
    assert len(train) == 72
    assert len(test) == 24


def test_train_and_test_have_no_subject_or_prompt_overlap():
    examples = build_dataset()
    train_subjects = {(e.domain, e.subject) for e in examples if e.split == "train"}
    test_subjects = {(e.domain, e.subject) for e in examples if e.split == "test"}
    assert train_subjects.isdisjoint(test_subjects)

    train_prompts = {e.prompt for e in examples if e.split == "train"}
    test_prompts = {e.prompt for e in examples if e.split == "test"}
    assert train_prompts.isdisjoint(test_prompts)


def test_every_example_response_matches_answer_confidence_format():
    examples = build_dataset()
    for ex in examples:
        assert RESPONSE_PATTERN.fullmatch(ex.response), ex.response


def test_both_answerable_and_unanswerable_present_in_each_split():
    examples = build_dataset()
    for split in ("train", "test"):
        split_examples = [e for e in examples if e.split == split]
        assert any(e.answerable for e in split_examples)
        assert any(not e.answerable for e in split_examples)


def test_no_duplicate_ids():
    examples = build_dataset()
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))


def test_validate_dataset_accepts_the_generated_dataset():
    examples = build_dataset()
    stats = validate_dataset(examples)
    assert stats["total"] == 96
    assert stats["train"] == 72
    assert stats["test"] == 24
    assert stats["train_test_prompt_overlap"] == 0


def test_validate_dataset_rejects_duplicate_ids():
    examples = build_dataset()
    corrupted = examples + [examples[0]]
    try:
        validate_dataset(corrupted)
        assert False, "expected AssertionError for duplicate ids"
    except AssertionError:
        pass


def test_validate_dataset_rejects_train_test_overlap():
    examples = build_dataset()
    # Same prompt/subject as an existing train example, but mislabeled test.
    leaked = dataclasses.replace(examples[0], id="leaked_id", split="test")
    corrupted = examples + [leaked]
    try:
        validate_dataset(corrupted)
        assert False, "expected AssertionError for train/test overlap"
    except AssertionError:
        pass


def test_validate_dataset_rejects_bad_response_format():
    examples = build_dataset()
    corrupted = list(examples)
    corrupted[0] = dataclasses.replace(corrupted[0], response="Paris is the answer.")
    try:
        validate_dataset(corrupted)
        assert False, "expected AssertionError for malformed response"
    except AssertionError:
        pass
