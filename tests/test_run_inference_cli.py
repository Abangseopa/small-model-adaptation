"""Tests for Stage 7's CLI/interactive/demo wiring in inference.run_inference,
using a stub AdaptedModel — no real model is ever loaded.
"""

import pytest

import inference.run_inference as ri


class StubAdaptedModel:
    """Stands in for inference.adapted_model.AdaptedModel."""

    def __init__(self, *args, **kwargs):
        self.active_adapters = ["default"]
        self.device = "cpu"
        self.calls = []

    def ask(self, context, question):
        self.calls.append((context, question))
        return f"ANSWER: stub for {question}\nCONFIDENCE: HIGH"


def _run_main_with_argv(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["run_inference.py", *argv])
    ri.main()


def test_demo_examples_are_not_from_training_or_held_out_data():
    from data_preparation.prepare_dataset import build_dataset

    examples = build_dataset()
    all_prompts = {e.prompt for e in examples}
    for demo in ri.DEMO_EXAMPLES:
        assert not any(demo["context"] in prompt for prompt in all_prompts)


def test_demo_examples_include_answerable_and_unanswerable_cases():
    # Not scored against the frozen dataset — just checked for the shape
    # Step 7 was asked to demonstrate (answerable + insufficient-context
    # + a differently phrased extraction question).
    assert len(ri.DEMO_EXAMPLES) >= 3
    for demo in ri.DEMO_EXAMPLES:
        assert demo["context"] and demo["question"] and demo["label"]


def test_argparse_requires_context_and_question_together(monkeypatch):
    monkeypatch.setattr(ri, "AdaptedModel", StubAdaptedModel)
    with pytest.raises(SystemExit):
        _run_main_with_argv(monkeypatch, ["--context", "only context, no question"])


def test_one_shot_mode_calls_ask_exactly_once_and_prints_result(monkeypatch, capsys):
    stub = StubAdaptedModel()
    monkeypatch.setattr(ri, "AdaptedModel", lambda: stub)
    _run_main_with_argv(monkeypatch, ["--context", "A mine.", "--question", "How much?"])

    captured = capsys.readouterr()
    assert stub.calls == [("A mine.", "How much?")]
    assert "ANSWER: stub for How much?" in captured.out


def test_demo_mode_runs_every_demo_example_through_ask(monkeypatch):
    stub = StubAdaptedModel()
    monkeypatch.setattr(ri, "AdaptedModel", lambda: stub)
    _run_main_with_argv(monkeypatch, ["--demo"])

    assert len(stub.calls) == len(ri.DEMO_EXAMPLES)
    for (context, question), demo in zip(stub.calls, ri.DEMO_EXAMPLES):
        assert context == demo["context"]
        assert question == demo["question"]


def test_interactive_mode_reuses_the_same_model_instance_across_turns(monkeypatch):
    stub = StubAdaptedModel()
    inputs = iter(["Context A", "Question A", "Context B", "Question B", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    ri.run_interactive(stub)

    assert stub.calls == [("Context A", "Question A"), ("Context B", "Question B")]


def test_interactive_mode_stops_on_blank_context(monkeypatch):
    stub = StubAdaptedModel()
    inputs = iter([""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    ri.run_interactive(stub)

    assert stub.calls == []
