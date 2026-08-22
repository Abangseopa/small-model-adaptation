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

Each stage folder currently contains only stub functions that raise
`NotImplementedError` — they exist to establish the shape of the
pipeline before any real code is written.

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

## Status

**Step 1 / 8 — Architecture / project scaffolding: done.**

Nothing has been installed, downloaded, trained, or dataset-generated
yet. `requirements.txt` lists intended dependencies but none are
installed.
