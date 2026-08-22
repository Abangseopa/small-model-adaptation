"""Stage 4: LoRA adaptation of Qwen2.5-1.5B-Instruct on the Stage 2 training set.

Trains a small LoRA adapter (base weights frozen) on the 72 Stage 2 TRAIN
examples only, to shift the model's *behavior* toward: answering only from
supplied context, abstaining when the context is insufficient, and always
emitting the fixed

    ANSWER: <answer>
    CONFIDENCE: <HIGH|LOW>

format. The 24 held-out TEST examples are never loaded by this script.

Run directly:

    python3 training/train.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.jsonl"
ADAPTER_DIR = REPO_ROOT / "models" / "qwen2.5-1.5b-instruct-lora-behavior-adapter"
TRAIN_CONFIG_PATH = ADAPTER_DIR / "training_config.json"
EXPECTED_TRAIN_EXAMPLES = 72

# --- LoRA config -----------------------------------------------------------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- Training config ---------------------------------------------------------
LEARNING_RATE = 2e-4
NUM_EPOCHS = 6
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0

HYPERPARAM_EXPLANATIONS = {
    "lora_r": "Small rank (8) — enough capacity for a narrow behavioral shift "
              "(format + context-adherence), not a new capability, so it keeps "
              "trainable parameters minimal and limits overfitting on just 72 examples.",
    "lora_alpha": "alpha=16 with r=8 gives a 2x scaling factor, a standard default that "
                  "keeps the LoRA update's magnitude moderate relative to the frozen "
                  "pretrained weights.",
    "lora_dropout": "Light dropout (0.05) — mild regularization appropriate for a very "
                    "small dataset, without meaningfully slowing convergence.",
    "target_modules": "Attention projections only (q/k/v/o), not the MLP blocks. Changing "
                       "how the model attends to/weighs the supplied context is the core "
                       "behavior being targeted; leaving the (much larger) MLP layers "
                       "untouched keeps the adapter small and the change conservative.",
    "learning_rate": "2e-4 is a typical LoRA learning rate for small instruction-tuned "
                      "models — high enough to fit 72 examples in a handful of epochs, "
                      "safe because the frozen base weights can't be destabilized.",
    "num_epochs": "6 epochs over 72 examples (54 total optimizer steps) is enough repetition "
                  "for the model to consistently pick up a simple, repeated output "
                  "structure without excessive passes over such a small set.",
    "effective_batch_size": "4 per-device x 2 accumulation steps = 8 — small, matched to a "
                            "72-example dataset (9 optimizer steps/epoch); accumulation is "
                            "used since there's no multi-device parallelism here.",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _verify_base_model_cache_complete() -> Path:
    """Locate the local HF cache snapshot for MODEL_ID and confirm every
    required file is present and resolves to a non-empty blob. Raises if
    anything is missing — we never fall back to downloading."""
    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo_dir = cache_root / ("models--" + MODEL_ID.replace("/", "--"))
    if not repo_dir.exists():
        raise FileNotFoundError(f"Expected cached model directory not found: {repo_dir}")

    snapshots = sorted(p for p in (repo_dir / "snapshots").iterdir() if p.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots found under {repo_dir / 'snapshots'}")
    snapshot_dir = snapshots[-1]

    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        fpath = snapshot_dir / fname
        resolved = fpath.resolve()
        if not fpath.exists() or not resolved.exists() or resolved.stat().st_size == 0:
            raise FileNotFoundError(f"Cached model file missing or incomplete: {fname} -> {resolved}")

    print(f"Verified base model cache complete: {snapshot_dir}")
    return snapshot_dir


def _load_train_examples() -> list[dict]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"{TRAIN_PATH} not found — run data_preparation/prepare_dataset.py first.")
    with TRAIN_PATH.open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    assert len(examples) == EXPECTED_TRAIN_EXAMPLES, (
        f"expected {EXPECTED_TRAIN_EXAMPLES} training examples, found {len(examples)} "
        f"— refusing to train on an unexpected dataset."
    )
    return examples


def _build_supervised_example(tokenizer, ex: dict) -> dict:
    """Tokenize (prompt, response) into a single sequence with labels masked
    (-100) over the prompt, so loss is only computed on the target
    completion — the standard supervised fine-tuning setup."""
    messages = [{"role": "user", "content": ex["prompt"]}]
    prefix_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_text = ex["response"] + tokenizer.eos_token

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prefix_text + target_text, add_special_tokens=False)["input_ids"]
    assert full_ids[: len(prefix_ids)] == prefix_ids, f"tokenization boundary mismatch for {ex['id']}"

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    return {"id": ex["id"], "input_ids": full_ids, "labels": labels}


def _collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _normalize_param_name(name: str) -> Optional[str]:
    """Recover a base model's original parameter name from its PEFT-wrapped
    name (e.g. 'base_model.model.model.layers.0...q_proj.base_layer.weight'
    -> 'model.layers.0...q_proj.weight'). Returns None for LoRA-only params."""
    if ".lora_A." in name or ".lora_B." in name:
        return None
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    return name.replace(".base_layer.", ".")


def _fingerprint(named_params) -> str:
    """SHA-256 over every (name, raw bytes) pair, sorted by name so the
    result doesn't depend on iteration order."""
    h = hashlib.sha256()
    for name, param in sorted(named_params, key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(param.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _print_config(trainable_params: int, total_params: int) -> None:
    print("\n=== LoRA + training configuration (before training) ===")
    print(f"lora_r: {LORA_R}")
    print(f"  -> {HYPERPARAM_EXPLANATIONS['lora_r']}")
    print(f"lora_alpha: {LORA_ALPHA}")
    print(f"  -> {HYPERPARAM_EXPLANATIONS['lora_alpha']}")
    print(f"lora_dropout: {LORA_DROPOUT}")
    print(f"  -> {HYPERPARAM_EXPLANATIONS['lora_dropout']}")
    print(f"target_modules: {LORA_TARGET_MODULES}")
    print(f"  -> {HYPERPARAM_EXPLANATIONS['target_modules']}")
    print(f"learning_rate: {LEARNING_RATE}")
    print(f"  -> {HYPERPARAM_EXPLANATIONS['learning_rate']}")
    print(f"num_epochs: {NUM_EPOCHS}")
    print(f"  -> {HYPERPARAM_EXPLANATIONS['num_epochs']}")
    effective_batch_size = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    print(f"effective_batch_size: {effective_batch_size} "
          f"(per_device={PER_DEVICE_BATCH_SIZE} x grad_accum={GRADIENT_ACCUMULATION_STEPS})")
    print(f"  -> {HYPERPARAM_EXPLANATIONS['effective_batch_size']}")
    print(f"trainable_parameters: {trainable_params:,}")
    print(f"total_parameters: {total_params:,}")
    print(f"trainable_percentage: {100 * trainable_params / total_params:.4f}%")
    print("=========================================================\n")


def train() -> dict:
    _verify_base_model_cache_complete()
    train_examples = _load_train_examples()
    print(f"Loaded {len(train_examples)} training examples from {TRAIN_PATH} "
          f"(test set is never read by this script)")

    _set_seed(SEED)
    device = _pick_device()
    print(f"Loading {MODEL_ID} from local cache only (no download) onto device={device} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, local_files_only=True)
    base_model.to(device)

    fingerprint_before = _fingerprint(list(base_model.named_parameters()))

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    _print_config(trainable_params, total_params)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized = [_build_supervised_example(tokenizer, ex) for ex in train_examples]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

    steps_per_epoch = -(-len(tokenized) // PER_DEVICE_BATCH_SIZE)  # ceil
    optimizer_steps_per_epoch = -(-steps_per_epoch // GRADIENT_ACCUMULATION_STEPS)
    print(f"Training: {len(tokenized)} examples, {steps_per_epoch} micro-batches/epoch, "
          f"{optimizer_steps_per_epoch} optimizer steps/epoch, {NUM_EPOCHS} epochs "
          f"({optimizer_steps_per_epoch * NUM_EPOCHS} optimizer steps total)\n")

    model.train()
    rng = random.Random(SEED)
    epoch_avg_losses: list[float] = []
    global_optimizer_step = 0
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        order = list(range(len(tokenized)))
        rng.shuffle(order)
        micro_batches = [order[i:i + PER_DEVICE_BATCH_SIZE] for i in range(0, len(order), PER_DEVICE_BATCH_SIZE)]

        epoch_losses = []
        optimizer.zero_grad()
        for micro_idx, idx_batch in enumerate(micro_batches, start=1):
            batch = _collate([tokenized[i] for i in idx_batch], pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            epoch_losses.append(outputs.loss.item())

            if micro_idx % GRADIENT_ACCUMULATION_STEPS == 0 or micro_idx == len(micro_batches):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM
                )
                optimizer.step()
                optimizer.zero_grad()
                global_optimizer_step += 1
                print(f"epoch {epoch}/{NUM_EPOCHS} optimizer_step {global_optimizer_step} "
                      f"loss={outputs.loss.item():.4f}")

        epoch_avg = sum(epoch_losses) / len(epoch_losses)
        epoch_avg_losses.append(epoch_avg)
        print(f"--- epoch {epoch}/{NUM_EPOCHS} complete: avg_loss={epoch_avg:.4f} "
              f"(elapsed {time.time() - start:.1f}s) ---\n")

    elapsed = time.time() - start
    decreasing = epoch_avg_losses[-1] < epoch_avg_losses[0]
    pct_change = 100 * (epoch_avg_losses[-1] - epoch_avg_losses[0]) / epoch_avg_losses[0]
    print(f"Training finished in {elapsed:.1f}s. "
          f"Epoch avg losses: {[round(x, 4) for x in epoch_avg_losses]}")
    print(f"Loss decreasing: {decreasing} ({pct_change:+.1f}% from first to last epoch)\n")

    model.eval()
    normalized_after = [
        (norm, p) for n, p in model.named_parameters() if (norm := _normalize_param_name(n)) is not None
    ]
    fingerprint_after = _fingerprint(normalized_after)
    params_unchanged = fingerprint_before == fingerprint_after
    assert params_unchanged, "Base model weights changed during LoRA training — this should never happen."
    print(f"Base model parameters unchanged (before == after LoRA wrap + training): {params_unchanged}")

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)

    training_config = {
        "model_id": MODEL_ID,
        "seed": SEED,
        "lora": {
            "r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT,
            "target_modules": LORA_TARGET_MODULES, "bias": "none", "task_type": "CAUSAL_LM",
        },
        "training": {
            "learning_rate": LEARNING_RATE, "num_epochs": NUM_EPOCHS,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "n_train_examples": len(train_examples),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "trainable_percentage": 100 * trainable_params / total_params,
        "epoch_avg_losses": epoch_avg_losses,
        "loss_decreasing": decreasing,
        "elapsed_seconds": elapsed,
        "device": device,
        "params_unchanged": params_unchanged,
        "fingerprint_before": fingerprint_before,
        "fingerprint_after": fingerprint_after,
    }
    TRAIN_CONFIG_PATH.write_text(json.dumps(training_config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved adapter to {ADAPTER_DIR}")
    print(f"Saved training config to {TRAIN_CONFIG_PATH}")

    adapter_files = list(ADAPTER_DIR.glob("*"))
    print(f"Adapter files: {[f.name for f in adapter_files]}")
    for f in adapter_files:
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")

    smoke_response = _run_smoke_test(train_examples[0], device)

    return {
        "training_config": training_config,
        "smoke_test_example_id": train_examples[0]["id"],
        "smoke_test_response": smoke_response,
    }


def _run_smoke_test(sample_example: dict, device: str) -> str:
    """Reload the base model fresh from cache and load the just-saved adapter
    on top of it (proving the adapter persists and reattaches correctly),
    then run one TRAINING example through it as a sanity check. This is not
    the Stage 5/6 evaluation — that uses the held-out test set and happens
    in a later step."""
    print(f"\n=== Smoke test: reload base model + adapter, run 1 TRAIN example ({sample_example['id']}) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    fresh_base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, local_files_only=True)
    adapted = PeftModel.from_pretrained(fresh_base, ADAPTER_DIR)
    adapted.to(device)
    adapted.eval()

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    messages = [{"role": "user", "content": sample_example["prompt"]}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = adapted.generate(
            **inputs, max_new_tokens=40, do_sample=False, num_beams=1, pad_token_id=pad_token_id
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"Prompt: {sample_example['prompt']}")
    print(f"Expected: {sample_example['response']}")
    print(f"Adapter output: {response!r}")
    return response


def main() -> None:
    train()


if __name__ == "__main__":
    main()
