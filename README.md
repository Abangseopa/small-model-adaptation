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

Stages 2-6 are implemented: `data_preparation/prepare_dataset.py`,
`baseline_evaluation/run_baseline_eval.py`, `training/train.py`,
`inference/run_inference.py`, and `evaluation/compare_models.py` — see
below. Stages 7-8 (analysis write-up, tests/docs) are not yet started.

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
`transformers` installed locally; Stage 4 (LoRA training) also needs
`peft`. E.g. in a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install torch transformers peft
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

## Model adaptation (Stage 4)

`training/train.py` trains a LoRA adapter on the 72 Stage 2 TRAIN examples
only — the 24 held-out test examples are never read by this script. Before
training it re-verifies the base model cache is complete and loads it with
`local_files_only=True` (a hard guarantee against ever downloading a
model). Base model weights are frozen throughout; only small LoRA
matrices are trained.

**Configuration** (conservative, chosen for a 72-example behavioral
dataset on a single Mac):

| Setting | Value | Why |
|---|---|---|
| LoRA rank (r) | 8 | Enough capacity for a narrow behavioral shift, not a new capability — keeps trainable parameters minimal and limits overfitting on 72 examples. |
| LoRA alpha | 16 | 2x-rank scaling (standard default) — a moderate update magnitude relative to the frozen weights. |
| LoRA dropout | 0.05 | Light regularization suited to a very small dataset. |
| Target modules | `q_proj, k_proj, v_proj, o_proj` | Attention projections only, not the MLP blocks — changing how the model weighs supplied context vs. its own knowledge is exactly the attention mechanism's job; skipping MLP keeps the adapter small and the change conservative. |
| Learning rate | 2e-4 | Typical for LoRA SFT on small instruction-tuned models; safe because frozen base weights can't be destabilized. |
| Epochs | 6 (54 optimizer steps) | Enough repetition for the model to pick up a simple, repeated output structure without excessive passes over 72 examples. |
| Effective batch size | 8 (4 per-device x 2 grad-accum steps) | Matched to a 72-example dataset (9 optimizer steps/epoch). |
| Trainable parameters | 2,179,072 / 1,545,893,376 (0.141%) | Confirms this is parameter-efficient fine-tuning, not full-model fine-tuning. |

Training is a plain PyTorch loop (no `Trainer`), so every step — batching,
loss masking (loss is only computed on the `ANSWER:`/`CONFIDENCE:` target
tokens, not the prompt), backward pass, gradient accumulation, optimizer
step — is visible in the script rather than hidden by a framework. Seed
is fixed (42) for reproducibility.

**Training run:** finished in ~593s (~10 min) on this Mac (MPS backend).
Per-epoch average loss fell monotonically every epoch:

```
epoch 1: 1.8687
epoch 2: 0.1258
epoch 3: 0.0041
epoch 4: 0.0006
epoch 5: 0.0002
epoch 6: 0.0001
```

**Post-training verification:**
- Base model parameter fingerprint (SHA-256 over every non-LoRA weight tensor) — identical before and after training. The base model was never modified.
- Adapter saved to `models/qwen2.5-1.5b-instruct-lora-behavior-adapter/` (gitignored, like all generated model artifacts): `adapter_model.safetensors` is **8.75 MB**, vs. the ~2.9 GB base model — roughly **330x smaller**, about 0.3% of the base model's size.
- Smoke test: reloaded a *fresh* copy of the base model from cache, attached the saved adapter to it (proving save/load round-trips correctly), and ran it on one **training** example. Output exactly matched the expected target: `ANSWER: Paris\nCONFIDENCE: HIGH`. (This is a sanity check only — it is a training example, not evidence of generalization, and the 24-example held-out test set was deliberately not touched.)

**Plain-English summary:**

- **What LoRA changed:** a small pair of low-rank matrices bolted onto the attention projections (q/k/v/o) in each of the 28 transformer layers — about 2.2M new parameters (0.14% the size of the base model). These are the only weights that received gradient updates.
- **What remained frozen:** all ~1.546B original Qwen2.5-1.5B-Instruct parameters — every weight the model shipped with, unchanged, verified by an exact fingerprint match before and after training.
- **What gradient descent was doing:** for each training example, computing how far the model's predicted next-token probabilities were from the actual `ANSWER: .../CONFIDENCE: ...` target tokens, then nudging the LoRA matrices (only) to make the correct tokens more likely next time.
- **What training loss means:** the cross-entropy between the model's predicted token distribution and the actual target tokens over the 72 training examples — lower means the model assigns higher probability to reproducing exactly those target sequences.
- **Why a falling loss doesn't prove generalization:** loss here measures how well the model reproduces the *specific 72 training answers* it was shown repeatedly (6 epochs). A model can drive this loss to near zero by memorizing those 72 input→output pairs directly, without having learned the underlying *policy* ("use only the context," "abstain when insufficient," "use this format") in a way that transfers to new subjects and new phrasing. The loss curve alone can't distinguish memorization from generalization.
- **Why we deliberately have not looked at held-out performance yet:** that's the entire point of keeping the 24 test examples untouched through training — it's the only way to test whether the model generalized rather than memorized. Peeking now, even informally, would contaminate the experiment: Stage 5/6 needs a clean first look at test performance to mean anything.
- **How much smaller the adapter is:** ~8.75 MB vs. ~2.9 GB for the base model — about 330x smaller, so the "adaptation" that ships is a tiny patch on top of an unmodified base model, not a new model.

## Adapted-model held-out evaluation (Stage 5)

`inference/run_inference.py` attaches the Stage 4 LoRA adapter to a
freshly loaded copy of the base model and runs it — for the first time —
on the exact same 24 Stage 2 held-out test examples used for the Stage 3
baseline. It deliberately imports its parsing/matching/metric functions
from `baseline_evaluation.run_baseline_eval` rather than redefining them,
and loads Stage 3's *saved* results rather than recomputing them, so the
baseline stays the frozen comparison point and the six metrics are
scored identically for both models.

Before scoring, the script verifies: the loaded base model's parameter
fingerprint matches the one recorded during the Stage 3 baseline run
exactly (same weights); the adapter reports `active_adapters == ['default']`;
a fingerprint over every non-LoRA parameter is identical before, halfway
through, and after inference (base model never touched); the adapter
files on disk are byte-for-byte unchanged after the run (nothing was
merged or overwritten); and all 24 examples were evaluated exactly once
(no duplicates, no reordering-induced skips).

**Results — Baseline vs. Adapted (24/24 held-out test examples):**

| Metric | Baseline | Adapted | Change |
|---|---|---|---|
| Answer accuracy | 0.0% | 95.8% | +95.8pp |
| Confidence accuracy | 0.0% | 100.0% | +100.0pp |
| Format compliance | 0.0% | 100.0% | +100.0pp |
| Answerable-example accuracy | 0.0% | 91.7% | +91.7pp |
| Unanswerable-example accuracy | 0.0% | 100.0% | +100.0pp |
| Full-behavior success | 0.0% | 95.8% | +95.8pp |

23 of 24 test examples improved from failure to full success; 1 remains
a failure. Representative examples:

**Answerable, fixed (Kenya/geography, novel subject + novel phrasing):**
```
Context: Kenya is a country. Its capital city is Nairobi.
Question: Can you tell me which city serves as the capital of Kenya?
Expected: ANSWER: Nairobi / CONFIDENCE: HIGH
Baseline: "Nairobi serves as the capital of Kenya."                    [FAIL — wrong format]
Adapted:  "ANSWER: Nairobi\nCONFIDENCE: HIGH"                          [PASS]
```

**Unanswerable, fixed (same subject, deliberately insufficient context):**
```
Context: A record exists for: Kenya. No additional detail is available.
Question: Can you tell me which city serves as the capital of Kenya?
Expected: ANSWER: Insufficient information to answer. / CONFIDENCE: LOW
Baseline: "The context provided states that there is a record for Kenya but no
           additional details... [long prose, no format]"              [FAIL]
Adapted:  "ANSWER: Insufficient information to answer.\nCONFIDENCE: LOW" [PASS]
```

**Remaining failure (the one example still wrong):**
```
Context: The book "Crime and Punishment" was written by Fyodor Dostoevsky.
Question: Do you know who authored "Crime and Punishment"?
Expected: ANSWER: Fyodor Dostoevsky / CONFIDENCE: HIGH
Baseline: 'Yes, I know that "Crime and Punishment" was written by Fyodor Dostoevsky.' [FAIL — no format]
Adapted:  "ANSWER: yes\nCONFIDENCE: HIGH"                              [FAIL — right format, wrong content]
```
This is the literature domain's alternate test-question phrasing ("Do you
know who authored...?" — a yes/no-shaped question). The adapted model
correctly adopted the format and a HIGH confidence, but literally
answered the yes/no question ("yes") instead of extracting the author
name the underlying question actually wants — a real generalization gap
on a phrasing the model never saw as training data.

**Did it generalize, or memorize?** These 24 examples received zero
gradient updates during Stage 4 training — different subjects than
training (Kenya/Norway/Uranus/Neptune/etc. vs. France/Japan/Mars/etc.),
and for every domain, different question phrasing on top of that. Going
from 0% to 95.8% full-behavior success on inputs the model never trained
on is evidence *for* generalization of the answer/abstain/format policy,
not mere memorization of the 72 training pairs — memorization alone
predicts near-baseline failure here, not near-perfect success.

What this does **not** prove: the dataset is small, synthetic, and
templated — a handful of fixed sentence patterns per domain, with the
test split varying only the subject and one alternate phrasing, not the
underlying structure. The model may have generalized the *template*
(recognize this sentence shape, emit this two-line format, look for a
factoid near the question's keyword) rather than a deep, general
context-grounding skill that would hold up on genuinely novel phrasing
structures, multi-sentence contexts, or adversarial context/question
mismatches — the one remaining failure (misreading a yes/no-shaped
question) is itself a hint of exactly that limitation. A 24-example test
set is also far too small to tightly bound the true error rate. The
honest claim this result supports is: *the LoRA adapter generalized the
target behavior across novel subjects and (mostly) novel phrasing within
this dataset's format* — not that the model acquired a general-purpose
context-grounding capability that would transfer beyond this synthetic
setup.

Run `python3 inference/run_inference.py` (needs Stage 4's adapter to
exist) to reproduce. Results are written to
`reports/adapted_eval_results.jsonl`, `reports/adapted_eval_metrics.json`,
and `reports/baseline_vs_adapted.json` (all gitignored, same as Stage 3).

## Base vs. adapted comparison (Stage 6)

`evaluation/compare_models.py` runs no model and touches no dataset — it
only reads the already-frozen Step 3 (`reports/baseline_eval_results.jsonl`)
and Step 5 (`reports/adapted_eval_results.jsonl`) artifacts and analyzes
them side by side. It never recomputes the evaluation and never redefines
the six scoring metrics.

**Per-example categorization** (24/24 held-out examples):

| Category | Count | Meaning |
|---|---|---|
| Improved | 23 | Baseline failed, adapted passed |
| Unchanged success | 0 | Both passed |
| Unchanged failure | 1 | Both failed |
| Regressed | 0 | Baseline passed, adapted failed |

Zero regressions: nothing the base model already did correctly broke
after adaptation.

**Comparison table:**

| Behavior | Base model | Adapted model | Interpretation |
|---|---|---|---|
| Answer accuracy | 0.0% | 95.8% | Core fact/abstention text extracted, right or wrong |
| Confidence accuracy | 0.0% | 100.0% | HIGH/LOW tagged correctly given actual context sufficiency |
| Format compliance | 0.0% | 100.0% | Literal ANSWER:/CONFIDENCE: structure produced at all |
| Answerable-example accuracy | 0.0% | 91.7% | Answer accuracy where context did support an answer |
| Unanswerable-example accuracy | 0.0% | 100.0% | Correctly abstained instead of guessing |
| Full-behavior success | 0.0% | 95.8% | Answer + confidence + format all correct at once |

**Behavioral breakdown:**

- **A. Format learning** — 0% → 100% format compliance. The adapter fully
  learned to emit the literal two-line structure on every held-out example.
- **B. Abstention / grounding** — on the 12 deliberately insufficient-context
  test examples, both answer accuracy and confidence accuracy went 0% →
  100%. The model learned to defer to the supplied context over its own
  prior knowledge (recall from Step 3 it sometimes answered from pretrained
  knowledge, e.g. Uranus's orbital position, even when the context withheld it) —
  the adapted model consistently abstains instead.
- **C. Answer extraction** — on the 12 answerable examples, answer accuracy
  went 0% → 91.7% (11/12). The one shortfall is the literature failure
  analyzed below.
- **D. Confidence policy** — confidence accuracy went 0% → 100% overall,
  including on the one example where the *answer* was wrong: the model
  correctly tagged `CONFIDENCE: HIGH` for the Crime and Punishment question
  even though it extracted the wrong answer content. This shows the
  confidence policy tracks context sufficiency independently of whether the
  specific answer text was extracted correctly — it's a separate, and more
  robust, sub-skill than answer extraction itself.

**The one remaining failure, analyzed:**

```
Question: Do you know who authored "Crime and Punishment"?
Expected: ANSWER: Fyodor Dostoevsky / CONFIDENCE: HIGH
Adapted:  ANSWER: yes / CONFIDENCE: HIGH
```

Classified as a **semantic/question-interpretation failure**, not a
formatting, confidence, or retrieval failure: the format is exactly
correct, and `CONFIDENCE: HIGH` is exactly correct (the model correctly
judged the context sufficient). What failed is question interpretation —
"Do you know who authored X?" is a polar (yes/no-shaped) question on its
surface, and the model answered that surface shape literally ("yes")
instead of resolving what entity the question is actually requesting. Two
of the three learned sub-skills (format, grounding/confidence) transferred
to this novel phrasing; the third (semantic answer extraction) did not, on
this particular phrasing. Not fixed here — that would require retraining
or dataset changes, out of scope for Step 6.

**Plain-English summary:**

- **What the adapter learned:** a response *policy*, not new facts — always
  emit the ANSWER:/CONFIDENCE: structure, set confidence by whether the
  supplied context (not the model's own knowledge) supports an answer, and
  extract a concise answer or the fixed abstention phrase instead of prose.
- **Evidence for generalization, not memorization:** all 24 test examples
  used subjects never seen during training, and most used question phrasing
  never seen either. Memorizing 72 specific training pairs gives the model
  no mechanism to succeed on inputs it never received gradient updates for
  — yet full-behavior success went from 0/24 to 23/24. That pattern is
  what generalization of a learned policy looks like, not what memorization
  looks like.
- **What the remaining failure teaches us:** the three sub-skills (format,
  grounding/confidence, answer extraction) generalized with different
  robustness. Format and confidence-tagging held up even on the phrasing
  that differs most from training; precise semantic extraction did not, on
  that same input. The adapter looks like it learned "recognize this task
  shape, respond in this structure, gauge context sufficiency" more robustly
  than a deeper natural-language-understanding upgrade.
- **Why 95.8% here doesn't mean 95.8% in the real world:** 24 examples is
  too small to bound a true error rate with any statistical confidence, the
  set is fully synthetic and templated (short single-sentence contexts, 6
  domains, one alternate phrasing each), and it was designed to test one
  narrow behavior change — not to represent real-world question/context
  diversity (multi-sentence contexts, contradictions, adversarial phrasing,
  multi-hop reasoning, none of which this dataset covers).
- **Knowledge vs. behavioral policy:** the base model already "knew" nearly
  every fact in this dataset before training (Step 3 often got the fact
  right in prose). LoRA training didn't add or change any factual knowledge
  — the base model's ~1.546B parameters are verified bit-for-bit unchanged
  throughout. What changed is a separate, tiny policy layer (2.18M LoRA
  parameters, 0.14% of the model) governing *how* the model is willing to
  respond — not *what it knows*.

Run `python3 evaluation/compare_models.py` (no model loading, needs Steps
3 and 5's saved reports) to reproduce. Results are written to
`reports/model_comparison_results.jsonl` (per-example) and
`reports/model_comparison_summary.json` (categorization, breakdown,
failure analysis), both gitignored.

## Status

**Step 6 / 8 — Base vs. adapted comparison: done.**

Steps 1-6 are complete: scaffolding, dataset, frozen baseline, LoRA
training, held-out evaluation of the adapted model, and now a side-by-side
analysis of the two frozen result sets. No retraining, dataset changes, or
adapter changes were made in Step 6 — it is read-only analysis of
already-generated artifacts.
`requirements.txt` lists intended dependencies; `torch`, `transformers`,
and `peft` are installed locally (in `.venv/`, gitignored). Step 6 itself
needs none of them — it only reads JSON.
