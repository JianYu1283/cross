# Low-level Call-aware Heterogeneous Graph Learning for Smart Contract Reentrancy Detection

Source code for the paper "Low-level Call-aware Heterogeneous Graph Learning for
Smart Contract Reentrancy Detection" (anonymized for review).

## Overview

A heterogeneous-graph pipeline for smart-contract reentrancy detection. The code
implements graph construction (via Slither), three node types (function / stmt /
ext_sink), one-hot / sparse / rich node features, and HGT / GAT / MANDO-HAN /
CodeBERT / GraphCodeBERT classifiers, together with the three-factor ablation,
cross-dataset transfer, class-weighting, noise-node control, and the statistical
tests (paired Wilcoxon / t, bootstrap, clustered bootstrap).

## Layout

- `build_graphs.py`  — Slither-based graph construction (pragma/solc resolution)
- `pipeline.py`      — DGL heterograph conversion + HGT/GAT classifiers + train/eval
- `experiments.py`   — unified three-factor experiment harness (fold x seed CV)
- `feature_ablation.py`   — feature-group decomposition (stmtonly / rronly)
- `cross_dataset_regimes.py`, `grouped_cv.py`, `transfer_multiseed.py` — transfer
- `noise_node_control.py`  — noise-node control for the ext_sink symmetry confound
- `significance.py`, `cluster_bootstrap.py` — statistical tests
- `class_weighted.py` — full-imbalance class-weighting
- `codebert_baseline.py`, `transformer_common.py`, `transformer_full469_metrics.py`,
  `fetch_graphcodebert.py` — transformer baselines
- `mando_baseline.py`, `mando_diag.py` — MANDO/HAN re-implementation
- `slither_baseline.py`, `smartcheck_baseline.py`, `static_baselines_common.py` — static tools
- `inject.py`, `inject_diverse.py` — cross-contract benchmark injection
- `make_figures.py` — figure generation

## Dependencies

Python 3.9, PyTorch, DGL, networkx, scikit-learn, numpy, matplotlib, transformers,
slither-analyzer, solc-select. See `requirements.txt`.

## Datasets

The three public datasets are required and are NOT included here:

- SolidiFI bug-injection benchmark (positive contracts): https://github.com/sbip-sg/SolidiFI-benchmark
- SmartBugs dataset (reentrancy positives): https://github.com/smartbugs/smartbugs
- SmartBugs-wild clean corpus (negatives): https://github.com/smartbugs/smartbugs-wild

Place them under `data/` (the code resolves `data/solidifi_buggy_contracts/...`,
`data/smartbug-dataset/...`, `data/smartbugs-wild-clean-contracts/...`), or set the
paths in the relevant scripts.

## Reproducing the main results

    python experiments.py --model hgt --mode rich --lowcall 1 --seeds 5 --cache data/cache_full.pkl
    python experiments.py --model hgt --mode onehot --lowcall 0 --seeds 5 --cache data/cache_full.pkl
    python noise_node_control.py --seeds 5 --folds 5 --cache data/cache_full.pkl

Build the graph cache first with `python pipeline.py --pos-dirs ... --neg-dirs ...`.
