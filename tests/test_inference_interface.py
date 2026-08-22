"""Tests for the Stage 7 AdaptedModel wrapper, using stub tokenizer/model
classes — this never loads the real ~2.9GB base model or the real
adapter weights.
"""

import torch

import inference.adapted_model as am


class _FakeBatchEncoding(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "CHAT_TEXT"

    def __call__(self, text, return_tensors):
        assert text == "CHAT_TEXT"
        assert return_tensors == "pt"
        return _FakeBatchEncoding(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
        )

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens is True
        return "ANSWER: Fake Answer\nCONFIDENCE: HIGH"


class FakeBaseModel:
    def to(self, device):
        return self


class FakeAdaptedModel:
    def __init__(self):
        self.active_adapters = ["default"]
        self.generate_kwargs = None

    def to(self, device):
        return self

    def eval(self):
        pass

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        # 3 "input" tokens + 4 newly generated tokens.
        return torch.tensor([[1, 2, 3, 4, 5, 6, 7]])


class _FakeAutoTokenizerFactory:
    def __init__(self, fake):
        self._fake = fake

    def from_pretrained(self, *args, **kwargs):
        return self._fake


class _FakeAutoModelFactory:
    def from_pretrained(self, *args, **kwargs):
        return FakeBaseModel()


class _FakePeftModelFactory:
    def __init__(self, fake):
        self._fake = fake

    def from_pretrained(self, base_model, adapter_dir):
        return self._fake


def _install_fakes(monkeypatch, tokenizer, model):
    monkeypatch.setattr(am, "AutoTokenizer", _FakeAutoTokenizerFactory(tokenizer))
    monkeypatch.setattr(am, "AutoModelForCausalLM", _FakeAutoModelFactory())
    monkeypatch.setattr(am, "PeftModel", _FakePeftModelFactory(model))
    monkeypatch.setattr(am, "_pick_device", lambda: "cpu")


def test_build_prompt_matches_training_prompt_shape():
    from data_preparation.prepare_dataset import RESPONSE_FORMAT_INSTRUCTION

    prompt = am.build_prompt("Some context.", "Some question?")
    assert prompt == (
        "Context: Some context.\nQuestion: Some question?\n"
        f"Instruction: {RESPONSE_FORMAT_INSTRUCTION}"
    )


def test_adapted_model_raises_when_adapter_dir_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    try:
        am.AdaptedModel(adapter_dir=missing)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_adapted_model_asserts_default_adapter_active(monkeypatch, tmp_path):
    tokenizer = FakeTokenizer()
    model = FakeAdaptedModel()
    model.active_adapters = ["not-default"]
    _install_fakes(monkeypatch, tokenizer, model)

    adapter_dir = tmp_path / "fake-adapter"
    adapter_dir.mkdir()
    try:
        am.AdaptedModel(adapter_dir=adapter_dir)
        assert False, "expected AssertionError for non-default active adapter"
    except AssertionError:
        pass


def test_adapted_model_ask_builds_prompt_and_uses_deterministic_generation(monkeypatch, tmp_path):
    tokenizer = FakeTokenizer()
    model = FakeAdaptedModel()
    _install_fakes(monkeypatch, tokenizer, model)

    adapter_dir = tmp_path / "fake-adapter"
    adapter_dir.mkdir()

    wrapped = am.AdaptedModel(adapter_dir=adapter_dir)
    assert wrapped.active_adapters == ["default"]

    result = wrapped.ask("A mine produces 300 tonnes.", "How much does the mine produce?")

    assert result == "ANSWER: Fake Answer\nCONFIDENCE: HIGH"
    assert tokenizer.last_messages == [
        {"role": "user", "content": am.build_prompt(
            "A mine produces 300 tonnes.", "How much does the mine produce?"
        )}
    ]
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["num_beams"] == 1
    assert model.generate_kwargs["max_new_tokens"] == am.MAX_NEW_TOKENS
