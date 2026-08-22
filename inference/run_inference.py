"""Stage 7: user-facing local inference entry point for the adapted model
(frozen Qwen2.5-1.5B-Instruct base + Step 4 LoRA adapter).

This is NOT an evaluation script — it does not touch, rerun, or change the
frozen Step 5/6 held-out results. For those, see
inference/evaluate_adapted.py and evaluation/compare_models.py.

Three usage modes:

  One-shot CLI:
    python3 inference/run_inference.py \\
      --context "Antimony production at the mine is 300 tonnes per month." \\
      --question "What is monthly antimony production?"

  Interactive (repeated Q&A, model loaded once):
    python3 inference/run_inference.py

  Demonstration (fixed, fresh, non-training/non-test examples):
    python3 inference/run_inference.py --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.adapted_model import AdaptedModel  # noqa: E402

# Fresh demonstration examples — not copied from Stage 2's training or
# held-out test data. These illustrate the interface; they are not part
# of, and do not change, the frozen Step 5 score (0.958 full-behavior
# success, 24 held-out examples).
DEMO_EXAMPLES = [
    {
        "label": "Answerable, mining-related, direct extraction",
        "context": "Antimony production at the mine is 300 tonnes per month.",
        "question": "What is monthly antimony production?",
    },
    {
        "label": "Insufficient context, mining-related",
        "context": "The mine's ventilation system was upgraded last quarter.",
        "question": "What is monthly antimony production at the mine?",
    },
    {
        "label": "Differently phrased (polar-shaped) question requiring extraction, not yes/no",
        "context": "The new firmware update for the routers was released by Nova Systems.",
        "question": "Do you happen to know which company released the new firmware update?",
    },
]


def run_demo(model: AdaptedModel) -> None:
    print("\n=== Demonstration: fresh examples, not from Stage 2 training or held-out test data ===")
    print("Illustrative only — these do not add to or change the frozen Step 5 evaluation score.\n")
    for ex in DEMO_EXAMPLES:
        output = model.ask(ex["context"], ex["question"])
        print(f"[{ex['label']}]")
        print(f"Context: {ex['context']}")
        print(f"Question: {ex['question']}")
        print(f"Output:\n{output}\n")


def run_interactive(model: AdaptedModel) -> None:
    print("Interactive mode — enter a context and question for each turn.")
    print("Leave Context blank (or press Ctrl-D) to quit.\n")
    while True:
        try:
            context = input("Context: ").strip()
        except EOFError:
            print()
            break
        if not context:
            break
        try:
            question = input("Question: ").strip()
        except EOFError:
            print()
            break
        if not question:
            break
        output = model.ask(context, question)
        print(f"\n{output}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask the adapted (frozen base + LoRA) model a context-grounded question."
    )
    parser.add_argument("--context", type=str, help="Context text to answer from.")
    parser.add_argument("--question", type=str, help="Question to answer using only that context.")
    parser.add_argument("--demo", action="store_true", help="Run fixed, fresh demonstration examples and exit.")
    args = parser.parse_args()

    if bool(args.context) != bool(args.question):
        parser.error("--context and --question must be provided together for one-shot mode.")

    print("Loading base model + LoRA adapter (once for this session) ...")
    model = AdaptedModel()
    print(f"Adapter active: {model.active_adapters} (device={model.device})\n")

    if args.demo:
        run_demo(model)
        return

    if args.context and args.question:
        print(model.ask(args.context, args.question))
        return

    run_interactive(model)


if __name__ == "__main__":
    main()
