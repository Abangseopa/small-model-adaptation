"""Stage 3: Baseline evaluation of the unmodified Qwen2.5-1.5B-Instruct model.

Runs the complete Stage 2 held-out test set (24 examples) through the
original, unmodified base model, zero-shot: the prompt states the task
(answer only from the supplied context, use the ANSWER/CONFIDENCE format,
say so if information is insufficient) but includes no worked examples of
that behavior. This records what the base model does *before* any
adaptation — the reference point Stage 6 will compare the adapted model
against.

Run directly:

    python3 baseline_evaluation/run_baseline_eval.py
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_PATH = REPO_ROOT / "data" / "processed" / "test.jsonl"
REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_PATH = REPORTS_DIR / "baseline_eval_results.jsonl"
METRICS_PATH = REPORTS_DIR / "baseline_eval_metrics.json"

MAX_NEW_TOKENS = 80

# The exact format the dataset itself was generated in (see
# data_preparation/prepare_dataset.py) — used to score strict format
# compliance.
STRICT_FORMAT_RE = re.compile(r"ANSWER: .+\nCONFIDENCE: (HIGH|LOW)")

# Best-effort extraction, tolerant of extra preamble/trailing text the base
# model may add since it has not been taught this format yet.
EXTRACT_RE = re.compile(
    r"ANSWER:\s*(?P<answer>.*?)\s*\n.*?CONFIDENCE:\s*(?P<confidence>HIGH|LOW)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class EvalRecord:
    id: str
    domain: str
    subject: str
    answerable: bool
    context: str
    question: str
    expected_answer: str
    expected_confidence: str
    model_raw_response: str
    parsed_answer: Optional[str]
    parsed_confidence: Optional[str]
    answer_correct: bool
    confidence_correct: bool
    format_compliant: bool
    full_behavior_success: bool


def _load_test_examples() -> list[dict]:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"{TEST_PATH} not found — run data_preparation/prepare_dataset.py first."
        )
    with TEST_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _split_context_question(prompt: str) -> tuple[str, str]:
    context_match = re.search(r"^Context: (.*)$", prompt, re.MULTILINE)
    question_match = re.search(r"^Question: (.*)$", prompt, re.MULTILINE)
    return (
        context_match.group(1) if context_match else "",
        question_match.group(1) if question_match else "",
    )


def _parse_expected(response: str) -> tuple[str, str]:
    """Extract (answer, confidence) from a Stage 2 dataset response.

    Stage 2 already validated every response matches STRICT_FORMAT_RE, so
    failure here indicates the dataset file itself is corrupt.
    """
    stripped = response.strip()
    match = STRICT_FORMAT_RE.fullmatch(stripped)
    if not match:
        raise ValueError(f"dataset response does not match expected format: {response!r}")
    answer_line = stripped.splitlines()[0]
    return answer_line[len("ANSWER: "):], match.group(1)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'^[\s"\'*_-]+', "", text)
    text = re.sub(r'[\s"\'*_.,;:!?]+$', "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _answer_matches(expected: str, parsed: Optional[str]) -> bool:
    if not parsed:
        return False
    exp, got = _normalize(expected), _normalize(parsed)
    return exp == got or exp in got


def _parse_model_output(text: str) -> tuple[Optional[str], Optional[str]]:
    match = EXTRACT_RE.search(text)
    if not match:
        return None, None
    return match.group("answer").strip(), match.group("confidence").upper()


def _param_fingerprint(model) -> str:
    """SHA-256 over every parameter tensor's raw bytes plus its name.

    Used to prove the model's weights are bit-for-bit identical before,
    partway through, and after the evaluation loop.
    """
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode())
        h.update(param.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_baseline_eval() -> dict:
    torch.manual_seed(0)
    examples = _load_test_examples()
    print(f"Loaded {len(examples)} held-out test examples from {TEST_PATH}")

    device = _pick_device()
    print(f"Loading {MODEL_ID} from local cache onto device={device} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    fingerprint_before = _param_fingerprint(model)
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
                fingerprint_mid = _param_fingerprint(model)
                assert fingerprint_mid == fingerprint_before, (
                    "Model parameters changed partway through baseline evaluation."
                )
                print(f"  (checkpoint: parameters unchanged after {i}/{len(examples)} examples)")

    elapsed = time.time() - start
    fingerprint_after = _param_fingerprint(model)
    params_unchanged = fingerprint_before == fingerprint_after
    assert params_unchanged, "Model parameters changed during baseline evaluation — this should never happen."
    print(f"\nGeneration finished in {elapsed:.1f}s. Parameters unchanged (before == after): {params_unchanged}")

    metrics = _compute_metrics(records)
    metrics.update({
        "model_id": MODEL_ID,
        "device": device,
        "max_new_tokens": MAX_NEW_TOKENS,
        "params_unchanged": params_unchanged,
        "param_fingerprint_before": fingerprint_before,
        "param_fingerprint_after": fingerprint_after,
    })

    _save_results(records, metrics)
    return {"records": records, "metrics": metrics}


def _compute_metrics(records: list[EvalRecord]) -> dict:
    n = len(records)
    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]

    def rate(items, pred):
        return sum(1 for r in items if pred(r)) / len(items) if items else 0.0

    return {
        "n_examples": n,
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "answer_accuracy": rate(records, lambda r: r.answer_correct),
        "confidence_accuracy": rate(records, lambda r: r.confidence_correct),
        "format_compliance": rate(records, lambda r: r.format_compliant),
        "answerable_accuracy": rate(answerable, lambda r: r.answer_correct),
        "unanswerable_accuracy": rate(unanswerable, lambda r: r.answer_correct),
        "full_behavior_success": rate(records, lambda r: r.full_behavior_success),
    }


def _save_results(records: list[EvalRecord], metrics: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} per-example results to {RESULTS_PATH}")
    print(f"Saved aggregate metrics to {METRICS_PATH}")


def _print_report(result: dict, n_failures: int = 5) -> None:
    metrics = result["metrics"]
    records = result["records"]

    print("\n=== Baseline metrics ===")
    for key in (
        "n_examples", "n_answerable", "n_unanswerable",
        "answer_accuracy", "confidence_accuracy", "format_compliance",
        "answerable_accuracy", "unanswerable_accuracy", "full_behavior_success",
    ):
        print(f"{key}: {metrics[key]}")
    print(f"params_unchanged: {metrics['params_unchanged']}")

    failures = [r for r in records if not r.full_behavior_success]
    print(f"\n=== {min(n_failures, len(failures))} representative failures (of {len(failures)} total) ===")
    for r in failures[:n_failures]:
        print(f"--- {r.id} (answerable={r.answerable}) ---")
        print(f"Context: {r.context}")
        print(f"Question: {r.question}")
        print(f"Expected: ANSWER: {r.expected_answer} / CONFIDENCE: {r.expected_confidence}")
        print(f"Model raw response: {r.model_raw_response!r}")
        print(f"Parsed: answer={r.parsed_answer!r} confidence={r.parsed_confidence!r}")
        print(
            f"answer_correct={r.answer_correct} "
            f"confidence_correct={r.confidence_correct} "
            f"format_compliant={r.format_compliant}"
        )
        print()


def main() -> None:
    result = run_baseline_eval()
    _print_report(result)


if __name__ == "__main__":
    main()
