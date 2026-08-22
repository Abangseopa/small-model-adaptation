# Small Model Adaptation

Experiments in adapting and evaluating open-source language models.

## Goal

Build a small, understandable experiment showing whether adapting an
open-source language model on a small custom dataset measurably changes
its behavior compared with the original base model.

This is an educational project. Every stage of the pipeline is kept in
its own directory with its own plain Python scripts, rather than hidden
behind a framework, so the whole process stays inspectable end to end.

## Pipeline

The project is built in 8 stages:

1. **Architecture / project scaffolding** — project structure, directories, requirements, `.gitignore` (this step).
2. **Dataset and training examples** — build the small custom dataset used for adaptation.
3. **Base-model evaluation** — run the unmodified base model on the evaluation set and record its outputs.
4. **Model adaptation / training** — fine-tune (adapt) the base model on the custom dataset.
5. **Adapted-model inference** — run the adapted model on the same evaluation set.
6. **Base vs adapted evaluation** — compare outputs from stage 3 and stage 5 side by side.
7. **Analysis of what changed** — summarize and explain the measured differences.
8. **Tests, documentation, and final wrap-up** — test coverage, docs, final report.

## Project structure

```
small-model-adaptation/
├── data/                     # Stage 2 — dataset storage (gitignored contents)
│   ├── raw/                  #   source/seed examples before processing
│   └── processed/            #   final dataset(s) used for training/eval
├── data_preparation/          # Stage 2 — scripts that build the dataset
│   └── prepare_dataset.py
├── baseline_evaluation/        # Stage 3 — run + record the base model's outputs
│   └── run_baseline_eval.py
├── training/                  # Stage 4 — adapt (fine-tune) the base model
│   └── train.py
├── inference/                  # Stage 5 — run the adapted model
│   └── run_inference.py
├── evaluation/                 # Stage 6 — compare base vs adapted outputs
│   └── compare_models.py
├── tests/                      # Stage 8 — automated tests for the pipeline
├── reports/                    # Stage 3/6/7 outputs — metrics, comparisons, analysis (gitignored contents)
├── models/                     # Local model weights / adapter artifacts (gitignored contents)
├── requirements.txt
├── .gitignore
└── README.md
```

Stages 4-6 still contain only stub functions that raise
`NotImplementedError`. `data_preparation/prepare_dataset.py` (Stage 2) and
`baseline_evaluation/run_baseline_eval.py` (Stage 3) are implemented — see
below.

`data/`, `models/`, and `reports/` hold generated or downloaded content
(datasets, model weights, checkpoints, evaluation results). Their
contents are gitignored (aside from a `.gitkeep`) since they are large
and reproducible from the scripts in this repo.

## Local model availability

No model has been downloaded or selected for this project yet. As a
starting point for later stages, the local Hugging Face cache
(`~/.cache/huggingface/hub`) already contains:

- `Qwen/Qwen2.5-1.5B-Instruct` (~2.9 GB) — a small instruction-tuned model, a plausible candidate for base/adapted comparison.
- `sentence-transformers/all-MiniLM-L6-v2` — a small embedding model, potentially useful for evaluation/comparison metrics.

No decision has been made to use either of these; they're noted here so
later stages can reuse what's already local instead of downloading
something new by default.

As of Stage 3, `Qwen/Qwen2.5-1.5B-Instruct` has been selected as the
model under test: its local cache was verified complete (all 7 files —
config, tokenizer, and safetensors weights — resolve to existing,
correctly sized blobs) and it has been loaded for evaluation. It was not
re-downloaded.

## Setup

Dependencies are not installed by default (see `requirements.txt`).
Stages that actually run the model (Stage 3 onward) need `torch` and
`transformers` installed locally, e.g. in a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers
```

`.venv/` is gitignored and not part of the committed project.

## Dataset (Stage 2)

`data_preparation/prepare_dataset.py` deterministically generates a small
synthetic instruction-response dataset targeting **response behavior**,
not factual recall. Every example asks a question against a supplied
context and expects a fixed, machine-checkable output format:

```
ANSWER: <answer>
CONFIDENCE: <HIGH|LOW>
```

`CONFIDENCE: HIGH` is used when the context supports the answer;
`CONFIDENCE: LOW` (with a fixed "Insufficient information to answer."
answer) is used when it does not — the model is meant to learn to say so
rather than invent a plausible-sounding answer, even in cases where it
may already "know" the real-world fact from pretraining.

The dataset spans 6 subject domains (geography, astronomy, history,
literature, sports, cooking) so the experiment can check whether the
behavior generalizes across topics rather than being memorized wording
in one domain. Each domain contributes disjoint subjects (and, for the
test split, different question phrasing) to train vs. test, so the test
set exercises genuinely novel content and wording rather than paraphrases
of what was trained on.

Run `python3 data_preparation/prepare_dataset.py` to regenerate and
validate the dataset. It checks: required fields are present, no
duplicate examples, zero train/test overlap (by prompt and by subject),
every response matches the ANSWER/CONFIDENCE format, and both answerable
and unanswerable examples exist in each split. Output is written to
`data/processed/{train,test}.jsonl` plus a `dataset_info.json` summary
(gitignored, regenerate locally as needed).

Current dataset: 96 examples total — 72 train (36 answerable / 36
unanswerable) and 24 held-out test (12 / 12), evenly split across the 6
domains.

## Baseline evaluation (Stage 3)

`baseline_evaluation/run_baseline_eval.py` loads the unmodified,
locally-cached `Qwen/Qwen2.5-1.5B-Instruct` and runs it, zero-shot, over
all 24 Stage 2 held-out test examples. "Zero-shot" here means the prompt
states the task rules (answer only from the supplied context, use the
ANSWER/CONFIDENCE format, say so if the context is insufficient) but
contains no worked examples of that behavior — the model has to follow
the instructions from a plain description alone, exactly as it would
need to for the adapted model comparison in Stage 6.

Decoding is greedy (`do_sample=False`, `num_beams=1`) for reproducibility.
Parameter integrity is verified with a SHA-256 fingerprint over every
named parameter tensor, taken before, halfway through, and after the run
— all three matched, confirming the base model was never modified.

Run `python3 baseline_evaluation/run_baseline_eval.py` (needs the Stage 3
`venv` from Setup above) to reproduce. Results are written to
`reports/baseline_eval_results.jsonl` (per-example) and
`reports/baseline_eval_metrics.json` (aggregate), both gitignored.

**Baseline results (24/24 test examples):**

| Metric | Value |
|---|---|
| Answer accuracy | 0.0% |
| Confidence accuracy | 0.0% |
| Format compliance | 0.0% |
| Answerable-example accuracy | 0.0% |
| Unanswerable-example accuracy | 0.0% |
| Full-behavior success | 0.0% |

The base model never emits the literal `ANSWER: ...` / `CONFIDENCE: ...`
format, so every strict metric reads 0% — but the raw responses show its
actual behavior is more mixed than that number implies:

- It usually *does* answer correctly, in prose (e.g. "Nairobi serves as
  the capital of Kenya.").
- On some deliberately insufficient-context examples it correctly
  recognizes it can't answer (e.g. Norway/Kenya capital, invention
  dates, dish origins) — but still doesn't use the required format or
  confidence tag.
- On others it **ignores the instruction to use only the supplied
  context** and answers from its own pretrained knowledge instead — e.g.
  given only "Notes reference: Uranus. No further detail was recorded."
  it still answers "Uranus occupies the 7th position..." That's the
  clearest baseline failure: the behavior we actually want to adapt is
  "defer to the supplied context over prior knowledge," and the base
  model doesn't reliably do that.

This baseline must be recorded now, before any training, because Stage 6
can only claim adaptation "changed behavior" by comparing the adapted
model's outputs to this exact frozen reference point on this exact test
set. Stage 4 should specifically target: (1) emitting the literal
ANSWER/CONFIDENCE format, (2) treating context as the sole source of
truth even when it conflicts with the model's own knowledge, and (3)
tagging CONFIDENCE consistently with whether the context actually
supported the answer.

## Status

**Step 3 / 8 — Base-model evaluation: done.**

Steps 1 (scaffolding), 2 (dataset), and 3 (baseline evaluation) are
complete. No training has occurred and no model weights have been
modified — Stage 3 only ran inference on the frozen base model.
`requirements.txt` lists intended dependencies; `torch` and
`transformers` are now installed locally (in `.venv/`, gitignored) to
run the baseline, but no adapter or fine-tuned checkpoint exists yet.
