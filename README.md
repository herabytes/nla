# NLA Claim-Drop Experiments

This workspace evaluates a Qwen 3.6 27B Natural Language Autoencoder (NLA) from `ceselder/qwen3.6-27b-nla-rl`.

## Notebook Workflow

`nla_infrastructure.ipynb` provides a staged GPU workflow for a single 80 GB GPU:

1. Downloads the AV base, RL adapter, AR reconstructor, metadata, and sample data into a local Hugging Face cache.
2. Loads the activation verbalizer (AV) and injects a layer-42 activation into the marker-token position.
3. Saves generated verbalizations and unloads AV before loading the activation reconstructor (AR).
4. Uses AR's separately stored `value_head.safetensors` to reconstruct a 5120-dimensional activation and calculate MSE.
5. Splits each verbalization into claims using blank lines and HTML break tags, then measures the effect of removing each claim.

## Claim-Drop Analysis

The analysis records, per dropped claim:

- claim position and total claim count
- original and dropped reconstruction MSE
- absolute and percentage loss change
- a heuristic grammar/completion label
- original prompt text and token count

It checkpoints results as parquet and CSV after each sample and plots claim-position effects alongside grammar/completion versus other claims.

## Data Artifacts

- `example_activations.parquet`: copy of the original 64-example activation dataset.
- `example_verbalizations.json`: saved AV outputs for the 64 original examples.
- `nla_meta.yaml` and `nla_prompts.json`: model metadata and the AV/AR prompt templates.

Large model weights, runtime result files, and the Python environment are excluded from version control via `.gitignore`.

## Safety Evaluation

The notebook includes utilities for generating a custom safety-boundary dataset. It extracts final-token layer-42 activations from a configurable prompt list, writes an NLA-compatible parquet, then runs the same staged AV/AR claim-drop evaluation while preserving prompt token counts for length-aware comparison.