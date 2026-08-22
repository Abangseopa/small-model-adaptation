"""Stage 2: Build the small synthetic instruction-response dataset used to
adapt the base model's *response behavior* — not its factual knowledge.

Every example asks the model to answer a question using only a supplied
context, in a fixed, machine-checkable format:

    ANSWER: <answer>
    CONFIDENCE: <HIGH|LOW>

CONFIDENCE is HIGH when the context supports the answer, and LOW (with a
fixed "insufficient information" answer) when it does not. The dataset is
generated deterministically from fact tables below — no randomness, no
external downloads — so it is fully reproducible.

Run directly to regenerate and validate the dataset:

    python3 data_preparation/prepare_dataset.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

RESPONSE_FORMAT_INSTRUCTION = (
    "Answer using only the information in the context above. If the "
    "context does not contain enough information to answer, say so "
    "instead of guessing."
)

INSUFFICIENT_ANSWER = "Insufficient information to answer."

# Generic "missing detail" context wording, deliberately domain-agnostic so
# it can be reused across every subject. Train and test use different
# phrasing so the test split also exercises novel wording, not just novel
# subjects.
TRAIN_INSUFFICIENT_CONTEXT_TEMPLATE = (
    "Notes reference: {subject}. No further detail was recorded."
)
TEST_INSUFFICIENT_CONTEXT_TEMPLATE = (
    "A record exists for: {subject}. No additional detail is available."
)

RESPONSE_PATTERN = re.compile(r"ANSWER: .+\nCONFIDENCE: (HIGH|LOW)")


@dataclass(frozen=True)
class Example:
    id: str
    domain: str
    split: str  # "train" | "test"
    subject: str
    answerable: bool
    prompt: str
    response: str


@dataclass(frozen=True)
class Domain:
    name: str
    # (subject, answer) pairs. train_facts and test_facts must use disjoint
    # subjects so the test split covers genuinely novel topics.
    train_facts: list[tuple[str, str]]
    test_facts: list[tuple[str, str]]
    context_template: str  # uses {subject}, {answer}
    train_question_template: str  # uses {subject}; distinct wording from test
    test_question_template: str  # uses {subject}; distinct wording from train


DOMAINS: list[Domain] = [
    Domain(
        name="geography",
        train_facts=[
            ("France", "Paris"),
            ("Japan", "Tokyo"),
            ("Brazil", "Brasilia"),
            ("Egypt", "Cairo"),
            ("Canada", "Ottawa"),
            ("Australia", "Canberra"),
        ],
        test_facts=[
            ("Kenya", "Nairobi"),
            ("Norway", "Oslo"),
        ],
        context_template="{subject} is a country. Its capital city is {answer}.",
        train_question_template="What is the capital of {subject}?",
        test_question_template="Can you tell me which city serves as the capital of {subject}?",
    ),
    Domain(
        name="astronomy",
        train_facts=[
            ("Mercury", "first"),
            ("Venus", "second"),
            ("Earth", "third"),
            ("Mars", "fourth"),
            ("Jupiter", "fifth"),
            ("Saturn", "sixth"),
        ],
        test_facts=[
            ("Uranus", "seventh"),
            ("Neptune", "eighth"),
        ],
        context_template="{subject} is the {answer} planet from the Sun.",
        train_question_template="What is the order of {subject} from the Sun?",
        test_question_template="Counting outward from the Sun, which position does {subject} occupy?",
    ),
    Domain(
        name="history",
        train_facts=[
            ("telephone", "1876"),
            ("electric light bulb", "1879"),
            ("printing press", "1440"),
            ("airplane", "1903"),
            ("World Wide Web", "1989"),
            ("telegraph", "1837"),
        ],
        test_facts=[
            ("steam engine", "1769"),
            ("dynamite", "1867"),
        ],
        context_template="The {subject} was introduced in {answer}.",
        train_question_template="In what year was the {subject} introduced?",
        test_question_template="What year did the {subject} first appear?",
    ),
    Domain(
        name="literature",
        train_facts=[
            ("Pride and Prejudice", "Jane Austen"),
            ("1984", "George Orwell"),
            ("Moby-Dick", "Herman Melville"),
            ("War and Peace", "Leo Tolstoy"),
            ("The Odyssey", "Homer"),
            ("Don Quixote", "Miguel de Cervantes"),
        ],
        test_facts=[
            ("Frankenstein", "Mary Shelley"),
            ("Crime and Punishment", "Fyodor Dostoevsky"),
        ],
        context_template='The book "{subject}" was written by {answer}.',
        train_question_template='Who wrote "{subject}"?',
        test_question_template='Do you know who authored "{subject}"?',
    ),
    Domain(
        name="sports",
        train_facts=[
            ("2000 Summer Olympics", "Sydney"),
            ("2004 Summer Olympics", "Athens"),
            ("2008 Summer Olympics", "Beijing"),
            ("2012 Summer Olympics", "London"),
            ("2016 Summer Olympics", "Rio de Janeiro"),
            ("2020 Summer Olympics", "Tokyo"),
        ],
        test_facts=[
            ("2024 Summer Olympics", "Paris"),
            ("1996 Summer Olympics", "Atlanta"),
        ],
        context_template="The {subject} were held in {answer}.",
        train_question_template="Which city hosted the {subject}?",
        test_question_template="Where did the {subject} take place?",
    ),
    Domain(
        name="cooking",
        train_facts=[
            ("Sushi", "Japan"),
            ("Paella", "Spain"),
            ("Tacos", "Mexico"),
            ("Croissant", "France"),
            ("Hummus", "the Levant"),
            ("Pad Thai", "Thailand"),
        ],
        test_facts=[
            ("Baklava", "the Eastern Mediterranean region"),
            ("Kimchi", "Korea"),
        ],
        context_template="{subject} originated in {answer}.",
        train_question_template="Where did {subject} originate?",
        test_question_template="What is the country of origin for {subject}?",
    ),
]


def _slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in ascii_text).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "x"


def _format_prompt(context: str, question: str) -> str:
    return f"Context: {context}\nQuestion: {question}\nInstruction: {RESPONSE_FORMAT_INSTRUCTION}"


def _format_response(answer: str, confidence: str) -> str:
    return f"ANSWER: {answer}\nCONFIDENCE: {confidence}"


def _make_pair(
    domain: str,
    split: str,
    index: int,
    subject: str,
    answer: str,
    context_template: str,
    question_template: str,
    insufficient_context_template: str,
) -> tuple[Example, Example]:
    subj_slug = _slugify(subject)
    base_id = f"{domain}_{split}_{index:02d}_{subj_slug}"
    question = question_template.format(subject=subject)

    answerable_prompt = _format_prompt(context_template.format(subject=subject, answer=answer), question)
    answerable = Example(
        id=f"{base_id}_answerable",
        domain=domain,
        split=split,
        subject=subject,
        answerable=True,
        prompt=answerable_prompt,
        response=_format_response(answer, "HIGH"),
    )

    unanswerable_prompt = _format_prompt(insufficient_context_template.format(subject=subject), question)
    unanswerable = Example(
        id=f"{base_id}_unanswerable",
        domain=domain,
        split=split,
        subject=subject,
        answerable=False,
        prompt=unanswerable_prompt,
        response=_format_response(INSUFFICIENT_ANSWER, "LOW"),
    )
    return answerable, unanswerable


def build_dataset() -> list[Example]:
    """Deterministically generate every example across all domains and splits."""
    examples: list[Example] = []
    for domain in DOMAINS:
        for i, (subject, answer) in enumerate(domain.train_facts, start=1):
            examples.extend(
                _make_pair(
                    domain.name, "train", i, subject, answer,
                    domain.context_template, domain.train_question_template,
                    TRAIN_INSUFFICIENT_CONTEXT_TEMPLATE,
                )
            )
        for i, (subject, answer) in enumerate(domain.test_facts, start=1):
            examples.extend(
                _make_pair(
                    domain.name, "test", i, subject, answer,
                    domain.context_template, domain.test_question_template,
                    TEST_INSUFFICIENT_CONTEXT_TEMPLATE,
                )
            )
    return examples


def validate_dataset(examples: list[Example]) -> dict:
    """Check the dataset's structural invariants. Raises AssertionError on failure."""
    assert examples, "dataset is empty"

    for ex in examples:
        assert ex.id and ex.domain and ex.subject, f"missing required field(s) on {ex}"
        assert ex.split in ("train", "test"), f"invalid split on {ex.id}: {ex.split}"
        assert isinstance(ex.answerable, bool), f"answerable must be bool on {ex.id}"
        assert ex.prompt.strip(), f"empty prompt on {ex.id}"
        assert ex.response.strip(), f"empty response on {ex.id}"
        assert RESPONSE_PATTERN.fullmatch(ex.response), f"response format invalid on {ex.id}: {ex.response!r}"
        expected_confidence = "HIGH" if ex.answerable else "LOW"
        assert f"CONFIDENCE: {expected_confidence}" in ex.response, (
            f"answerable/confidence mismatch on {ex.id}"
        )

    ids = [ex.id for ex in examples]
    assert len(ids) == len(set(ids)), "duplicate example ids found"

    content_keys = [(ex.prompt, ex.response) for ex in examples]
    assert len(content_keys) == len(set(content_keys)), "duplicate (prompt, response) pairs found"

    train = [ex for ex in examples if ex.split == "train"]
    test = [ex for ex in examples if ex.split == "test"]
    assert train, "no training examples"
    assert test, "no test examples"

    train_prompts = {ex.prompt for ex in train}
    test_prompts = {ex.prompt for ex in test}
    prompt_overlap = train_prompts & test_prompts
    assert not prompt_overlap, f"train/test prompt overlap: {prompt_overlap}"

    train_subjects = {(ex.domain, ex.subject) for ex in train}
    test_subjects = {(ex.domain, ex.subject) for ex in test}
    subject_overlap = train_subjects & test_subjects
    assert not subject_overlap, f"train/test subject overlap: {subject_overlap}"

    for split_name, split_examples in (("train", train), ("test", test)):
        n_answerable = sum(1 for ex in split_examples if ex.answerable)
        n_unanswerable = len(split_examples) - n_answerable
        assert n_answerable > 0, f"{split_name} split has no answerable examples"
        assert n_unanswerable > 0, f"{split_name} split has no unanswerable examples"

    return {
        "total": len(examples),
        "train": len(train),
        "test": len(test),
        "train_answerable": sum(1 for ex in train if ex.answerable),
        "train_unanswerable": sum(1 for ex in train if not ex.answerable),
        "test_answerable": sum(1 for ex in test if ex.answerable),
        "test_unanswerable": sum(1 for ex in test if not ex.answerable),
        "train_test_prompt_overlap": len(prompt_overlap),
        "domains": sorted({ex.domain for ex in examples}),
    }


def save_dataset(examples: list[Example]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = [ex for ex in examples if ex.split == "train"]
    test = [ex for ex in examples if ex.split == "test"]

    for name, split_examples in (("train", train), ("test", test)):
        path = OUTPUT_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for ex in split_examples:
                f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")

    info = {
        "response_format": "ANSWER: <answer>\\nCONFIDENCE: <HIGH|LOW>",
        "insufficient_answer_text": INSUFFICIENT_ANSWER,
        "domains": [d.name for d in DOMAINS],
        "counts": validate_dataset(examples),
    }
    (OUTPUT_DIR / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    examples = build_dataset()
    stats = validate_dataset(examples)
    save_dataset(examples)

    train = [ex for ex in examples if ex.split == "train"]
    test = [ex for ex in examples if ex.split == "test"]

    print("=== Dataset summary ===")
    print(f"Train examples: {stats['train']} (answerable={stats['train_answerable']}, unanswerable={stats['train_unanswerable']})")
    print(f"Test examples:  {stats['test']} (answerable={stats['test_answerable']}, unanswerable={stats['test_unanswerable']})")
    print(f"Domains: {', '.join(stats['domains'])}")
    print(f"Train/test prompt overlap: {stats['train_test_prompt_overlap']}")
    print(f"Saved to: {OUTPUT_DIR}")

    def show(ex: Example) -> None:
        print(f"--- [{ex.split}] {ex.id} ---")
        print(ex.prompt)
        print(ex.response)
        print()

    print("\n=== 3 representative TRAIN examples ===")
    train_sample = [
        next(ex for ex in train if ex.domain == "geography" and ex.answerable),
        next(ex for ex in train if ex.domain == "history" and not ex.answerable),
        next(ex for ex in train if ex.domain == "cooking" and ex.answerable),
    ]
    for ex in train_sample:
        show(ex)

    print("=== 2 representative TEST (held-out) examples ===")
    test_sample = [
        next(ex for ex in test if ex.domain == "geography" and ex.answerable),
        next(ex for ex in test if ex.domain == "sports" and not ex.answerable),
    ]
    for ex in test_sample:
        show(ex)


if __name__ == "__main__":
    main()
