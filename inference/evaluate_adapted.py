"""Stage 5: Evaluate the ADAPTED model (frozen base + Stage 4 LoRA adapter)
on the exact same 24 Stage 2 held-out test examples used for the Stage 3
baseline.

This is the first look at held-out performance. It deliberately imports
its scoring logic from baseline_evaluation.run_baseline_eval (parsing,
matching, and the six metrics) instead of redefining it, so the adapted
model is scored under exactly the same rules as the baseline — and it
loads the Stage 3 baseline's *saved* results rather than recomputing them,
so the baseline stays the frozen comparison point.

Originally named inference/run_inference.py; renamed (content otherwise
unchanged) in Stage 7 once run_inference.py became the project's
user-facing local inference entry point (see inference/adapted_model.py
and inference/run_inference.py). The frozen results this script produced
(reports/adapted_eval_results.jsonl, reports/adapted_eval_metrics.json)
are unchanged and were not regenerated as part of that rename.

Run directly:

    python3 inference/evaluate_adapted.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_evaluation.run_baseline_eval import (  # noqa: E402
    MAX_NEW_TOKENS,
    MODEL_ID,
    STRICT_FORMAT_RE,
    TEST_PATH,
    EvalRecord,
    _answer_matches,
    _compute_metrics,
    _load_test_examples,
    _param_fingerprint,
    _parse_expected,
    _parse_model_output,
    _pick_device,
    _split_context_question,
)
from training.train import _fingerprint, _normalize_param_name  # noqa: E402

ADAPTER_DIR = REPO_ROOT / "models" / "qwen2.5-1.5b-instruct-lora-behavior-adapter"
REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "adapted_eval_results.jsonl"
METRICS_PATH = REPORTS_DIR / "adapted_eval_metrics.json"
COMPARISON_PATH = REPORTS_DIR / "baseline_vs_adapted.json"
BASELINE_RESULTS_PATH = REPORTS_DIR / "baseline_eval_results.jsonl"
BASELINE_METRICS_PATH = REPORTS_DIR / "baseline_eval_metrics.json"

EXPECTED_TEST_EXAMPLES = 24
METRIC_KEYS = (
    "answer_accuracy",
    "confidence_accuracy",
    "format_compliance",
    "answerable_accuracy",
    "unanswerable_accuracy",
    "full_behavior_success",
)


def _load_baseline() -> tuple[dict[str, dict], dict]:
    """Load Step 3's already-saved results/metrics. Never recomputed here —
    the baseline is the frozen comparison point."""
    if not BASELINE_RESULTS_PATH.exists() or not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Baseline results not found at {BASELINE_RESULTS_PATH} / {BASELINE_METRICS_PATH} "
            "— run baseline_evaluation/run_baseline_eval.py (Step 3) first."
        )
    with BASELINE_RESULTS_PATH.open(encoding="utf-8") as f:
        by_id = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
    metrics = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    return by_id, metrics


def _adapter_file_manifest() -> dict[str, tuple[int, float]]:
    return {
        f.name: (f.stat().st_size, f.stat().st_mtime)
        for f in ADAPTER_DIR.iterdir()
        if f.is_file()
    }


def run_adapted_eval() -> dict:
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"{ADAPTER_DIR} not found — run training/train.py (Step 4) first.")

    torch.manual_seed(0)
    examples = _load_test_examples()
    assert len(examples) == EXPECTED_TEST_EXAMPLES, (
        f"expected {EXPECTED_TEST_EXAMPLES} held-out test examples, found {len(examples)} "
        f"— this must be exactly the Step 3 test set."
    )
    print(f"Loaded {len(examples)} held-out test examples from {TEST_PATH} "
          f"(identical file used for the Step 3 baseline)")

    adapter_manifest_before = _adapter_file_manifest()

    device = _pick_device()
    print(f"Loading base model {MODEL_ID} from local cache only (no download) onto device={device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, local_files_only=True)
    base_model.to(device)

    # Cross-check against Step 3's own frozen fingerprint: same function,
    # same model class/loading call, so an identical digest proves these
    # are the exact same base weights the baseline was scored against.
    fingerprint_vs_baseline = _param_fingerprint(base_model)
    _, baseline_metrics = _load_baseline()
    assert fingerprint_vs_baseline == baseline_metrics["param_fingerprint_before"], (
        "Base model weights differ from the ones used in the Step 3 baseline run."
    )
    print("Base model fingerprint matches the Step 3 baseline's recorded fingerprint exactly.")

    fingerprint_before = _fingerprint(list(base_model.named_parameters()))

    print(f"Attaching LoRA adapter from {ADAPTER_DIR} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.to(device)
    model.eval()

    active_adapters = list(model.active_adapters)
    assert active_adapters == ["default"], f"expected adapter 'default' active, got {active_adapters}"
    n_lora_elements = sum(p.numel() for n, p in model.named_parameters() if ".lora_" in n)
    print(f"Adapter active: {active_adapters} ({n_lora_elements:,} LoRA parameter elements)")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    records: list[EvalRecord] = []
    midpoint = len(examples) // 2
    start = time.time()
    with torch.no_grad():
        for i, ex in enumerate(examples, start=1):
            context, question = _split_context_question(ex["prompt"])
            expected_answer, expected_confidence = _parse_expected(ex["response"])

            messages = [{"role": "user", "content": ex["prompt"]}]
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(chat_text, return_tensors="pt").to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
            )
            new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
            raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            parsed_answer, parsed_confidence = _parse_model_output(raw_response)
            answer_correct = _answer_matches(expected_answer, parsed_answer)
            confidence_correct = parsed_confidence == expected_confidence
            format_compliant = bool(STRICT_FORMAT_RE.fullmatch(raw_response))
            full_success = answer_correct and confidence_correct and format_compliant

            records.append(EvalRecord(
                id=ex["id"], domain=ex["domain"], subject=ex["subject"],
                answerable=ex["answerable"], context=context, question=question,
                expected_answer=expected_answer, expected_confidence=expected_confidence,
                model_raw_response=raw_response, parsed_answer=parsed_answer,
                parsed_confidence=parsed_confidence, answer_correct=answer_correct,
                confidence_correct=confidence_correct, format_compliant=format_compliant,
                full_behavior_success=full_success,
            ))
            print(
                f"[{i}/{len(examples)}] {ex['id']}: "
                f"answer_ok={answer_correct} conf_ok={confidence_correct} format_ok={format_compliant}"
            )

            if i == midpoint:
                normalized_mid = [
                    (norm, p) for n, p in model.named_parameters()
                    if (norm := _normalize_param_name(n)) is not None
                ]
                assert _fingerprint(normalized_mid) == fingerprint_before, (
                    "Base model parameters changed partway through adapted evaluation."
                )
                print(f"  (checkpoint: base parameters unchanged after {i}/{len(examples)} examples)")

    elapsed = time.time() - start
    assert len(records) == EXPECTED_TEST_EXAMPLES
    assert len({r.id for r in records}) == EXPECTED_TEST_EXAMPLES, "duplicate example evaluated"

    normalized_after = [
        (norm, p) for n, p in model.named_parameters() if (norm := _normalize_param_name(n)) is not None
    ]
    fingerprint_after = _fingerprint(normalized_after)
    base_unchanged = fingerprint_after == fingerprint_before
    assert base_unchanged, "Base model parameters changed during adapted evaluation — this should never happen."
    print(f"\nGeneration finished in {elapsed:.1f}s. Base parameters unchanged (before == after): {base_unchanged}")

    adapter_manifest_after = _adapter_file_manifest()
    adapter_unchanged = adapter_manifest_before == adapter_manifest_after
    assert adapter_unchanged, "Adapter files on disk changed during evaluation — this script must not write to them."
    print(f"Adapter files on disk unchanged during evaluation: {adapter_unchanged}")

    metrics = _compute_metrics(records)
    metrics.update({
        "model_id": MODEL_ID,
        "adapter_dir": str(ADAPTER_DIR),
        "device": device,
        "max_new_tokens": MAX_NEW_TOKENS,
        "base_unchanged": base_unchanged,
        "adapter_files_unchanged": adapter_unchanged,
        "base_fingerprint_matches_baseline_run": True,
    })

    _save_results(records, metrics)
    return {"records": records, "metrics": metrics}


def _save_results(records: list[EvalRecord], metrics: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} per-example results to {RESULTS_PATH}")
    print(f"Saved aggregate metrics to {METRICS_PATH}")


def _print_comparison(adapted_metrics: dict, baseline_metrics: dict) -> None:
    print("\n=== Baseline vs Adapted (frozen Step 3 baseline) ===")
    print(f"{'Metric':<24}{'Baseline':>10}{'Adapted':>10}{'Change':>10}")
    comparison = {}
    for key in METRIC_KEYS:
        b, a = baseline_metrics[key], adapted_metrics[key]
        change = a - b
        comparison[key] = {"baseline": b, "adapted": a, "change": change}
        print(f"{key:<24}{100*b:>9.1f}%{100*a:>9.1f}%{100*change:>+9.1f}pp")
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nSaved comparison table to {COMPARISON_PATH}")


def _print_examples(adapted_records: list[EvalRecord], baseline_by_id: dict[str, dict], n: int = 3) -> None:
    def show(r: EvalRecord, label: str) -> None:
        b = baseline_by_id[r.id]
        print(f"--- {r.id} (answerable={r.answerable}) [{label}] ---")
        print(f"Context: {r.context}")
        print(f"Question: {r.question}")
        print(f"Expected: ANSWER: {r.expected_answer} / CONFIDENCE: {r.expected_confidence}")
        print(f"Baseline output: {b['model_raw_response']!r}  (full_success={b['full_behavior_success']})")
        print(f"Adapted output:  {r.model_raw_response!r}  (full_success={r.full_behavior_success})")
        print()

    print("\n=== Representative held-out examples ===")

    answerable = next((r for r in adapted_records if r.answerable), None)
    if answerable:
        show(answerable, "answerable")

    unanswerable = next((r for r in adapted_records if not r.answerable), None)
    if unanswerable:
        show(unanswerable, "unanswerable")

    improved = [
        r for r in adapted_records
        if r.full_behavior_success and not baseline_by_id[r.id]["full_behavior_success"]
    ]
    print(f"Improved over baseline: {len(improved)}/{len(adapted_records)} examples")
    for r in improved[:n]:
        show(r, "improved over baseline")

    remaining_failures = [r for r in adapted_records if not r.full_behavior_success]
    print(f"Remaining failures: {len(remaining_failures)}/{len(adapted_records)} examples")
    for r in remaining_failures[:n]:
        show(r, "still failing")


def _print_generalization_analysis(adapted_metrics: dict) -> None:
    print("\n=== Generalization analysis ===")
    print(
        "MEMORIZATION would mean: the model reproduces ANSWER/CONFIDENCE "
        "correctly only for the 72 exact (context, question) pairs it saw "
        "gradient updates on, and fails on any input outside that set.\n"
        "GENERALIZATION would mean: the model applies the learned policy — "
        "answer from context, abstain when insufficient, use the fixed "
        "format — to the 24 test examples, which involve subjects and (for "
        "half the domains) question phrasing never seen during training.\n"
    )
    print(
        f"These 24 test examples received zero gradient updates in Step 4. "
        f"The adapted model scores {100*adapted_metrics['full_behavior_success']:.1f}% "
        f"full-behavior success on them (vs 0.0% for the untrained baseline on the "
        f"same set), which is evidence *for* generalization of the policy, not just "
        f"memorization of the 72 training pairs — the model was never shown these "
        f"specific inputs.\n"
    )
    print(
        "What this does NOT prove: the dataset is small, synthetic, and templated — "
        "every example follows one of a handful of fixed sentence patterns per domain, "
        "and the test set only varies the subject and one alternate question phrasing "
        "per domain, not the underlying structure. The model may have generalized the "
        "*template* (recognize this sentence shape, emit this two-line format) rather "
        "than a deeper context-grounding policy that would hold on genuinely novel "
        "phrasing, multi-sentence contexts, or adversarial context/question mismatches. "
        "A 24-example test set is also too small to bound the true error rate tightly. "
        "This result supports 'the adapter generalized across novel subjects and "
        "wording within this dataset's format' — it does not support 'the model has "
        "learned a general context-grounding skill that would transfer beyond this "
        "synthetic setup.'"
    )


def main() -> None:
    baseline_by_id, baseline_metrics = _load_baseline()
    result = run_adapted_eval()
    metrics, records = result["metrics"], result["records"]

    print("\n=== Adapted-model metrics ===")
    for key in ("n_examples", "n_answerable", "n_unanswerable", *METRIC_KEYS):
        print(f"{key}: {metrics[key]}")

    _print_comparison(metrics, baseline_metrics)
    _print_examples(records, baseline_by_id)
    _print_generalization_analysis(metrics)


if __name__ == "__main__":
    main()
