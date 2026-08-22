"""Tests for Stage 4's pure helper functions.

Uses the real tokenizer (a ~10MB vocab/merges load) where needed for a
faithful round-trip check, but never AutoModelForCausalLM — that's what
actually loads the multi-GB model weights, and nothing here does that.
"""

import pytest
import torch
from transformers import AutoTokenizer

import inference.adapted_model as am
import training.train as train_module
from training.train import (
    ADAPTER_DIR,
    MODEL_ID,
    _build_supervised_example,
    _collate,
    _fingerprint,
    _normalize_param_name,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)


def test_normalize_param_name_strips_peft_wrapping():
    wrapped = "base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight"
    assert _normalize_param_name(wrapped) == "model.layers.0.self_attn.q_proj.weight"


def test_normalize_param_name_filters_out_lora_params():
    lora_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    lora_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    assert _normalize_param_name(lora_a) is None
    assert _normalize_param_name(lora_b) is None


def test_normalize_param_name_leaves_unwrapped_names_alone():
    # Modules PEFT never touches still just get the "base_model.model." prefix stripped.
    assert _normalize_param_name("base_model.model.model.embed_tokens.weight") == "model.embed_tokens.weight"


def test_fingerprint_is_order_independent_and_content_sensitive():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0])
    fp1 = _fingerprint([("a", a), ("b", b)])
    fp2 = _fingerprint([("b", b), ("a", a)])
    assert fp1 == fp2  # sorted internally by name -> order shouldn't matter

    c = torch.tensor([9.0, 9.0])
    fp3 = _fingerprint([("a", a), ("b", c)])
    assert fp3 != fp1  # different content -> different fingerprint


def test_collate_pads_and_masks_correctly():
    batch = [
        {"id": "x1", "input_ids": [1, 2, 3], "labels": [-100, -100, 3]},
        {"id": "x2", "input_ids": [4, 5], "labels": [-100, 5]},
    ]
    out = _collate(batch, pad_token_id=0)
    assert out["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["labels"].tolist() == [[-100, -100, 3], [-100, 5, -100]]


def test_build_supervised_example_masks_prompt_and_targets_response_only(tokenizer):
    ex = {
        "id": "t1",
        "prompt": "Context: Paris is in France.\nQuestion: Where is Paris?\nInstruction: Answer.",
        "response": "ANSWER: Paris\nCONFIDENCE: HIGH",
    }
    built = _build_supervised_example(tokenizer, ex)
    assert built["id"] == "t1"
    assert len(built["input_ids"]) == len(built["labels"])

    n_masked = sum(1 for label in built["labels"] if label == -100)
    assert 0 < n_masked < len(built["input_ids"])

    unmasked_ids = [tid for tid, label in zip(built["input_ids"], built["labels"]) if label != -100]
    decoded_tail = tokenizer.decode(unmasked_ids, skip_special_tokens=True)
    assert decoded_tail.strip() == ex["response"]


def test_adapter_dir_and_model_id_consistent_across_modules():
    assert am.ADAPTER_DIR == train_module.ADAPTER_DIR
    assert am.MODEL_ID == train_module.MODEL_ID
    assert ADAPTER_DIR.parent.name == "models"
    assert "lora" in ADAPTER_DIR.name


def test_verify_base_model_cache_complete_raises_when_cache_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        train_module._verify_base_model_cache_complete()
