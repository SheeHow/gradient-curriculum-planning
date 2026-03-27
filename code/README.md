# Gradient-Based Data Valuation for Game-Theoretic Motion Planning

Code for **"Gradient-Based Data Valuation Improves Curriculum Learning for Game-Theoretic Motion Planning"** (CDC 2026).

**Authors:** Shihao Li, Jiachen Li — The University of Texas at Austin

**Project Page:** [https://sheehow.github.io/gradient-curriculum-planning/](https://sheehow.github.io/gradient-curriculum-planning/)

## Overview

This repository provides the scoring, curriculum, and visualization code for reproducing the data-centric curriculum learning pipeline described in the paper. The pipeline has three stages:

1. **Scenario Scoring** — Compute per-sample importance via TracIn gradient similarity or metadata interaction-difficulty
2. **Curriculum Schedule** — Three-phase (warm-up / ramp-up / focus) epoch-dependent weighting
3. **Training** — GameFormer trained with weighted sampling using the curriculum schedule

## Repository Structure

```
code/
  scoring/
    compute_tracin.py        # TracIn gradient dot-product scoring
    score_scenarios.py       # Metadata interaction-difficulty scoring (6 metrics)
    prepare_score_files.py   # Normalize and format scores for curriculum
  curriculum/
    curriculum_sampler.py    # Three-phase curriculum sampler (drop-in for DataLoader)
    curated_dataset.py       # Dataset class with score-based filtering
  configs/
    tracin_curriculum.yaml   # Best config (TracIn curriculum, full data)
    metadata_curriculum.yaml # Metadata curriculum config
    baseline.yaml            # Uniform sampling baseline
  figures/
    generate_ade_comparison.py  # Reproduce Figure 6 (multi-seed ADE comparison)
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- NumPy, Pandas, tqdm, PyYAML, SciPy
- Matplotlib (for figure generation)
- [GameFormer-Planner](https://github.com/MCZhi/GameFormer-Planner) (model architecture)
- [nuPlan](https://www.nuscenes.org/nuplan) dataset (preprocessed to .npz)

```bash
pip install torch numpy pandas tqdm pyyaml scipy matplotlib
```

## Usage

### Step 1: Score scenarios

**Metadata scoring** (interaction-difficulty features):
```bash
python scoring/score_scenarios.py \
    --data_dir /path/to/nuplan_processed/train \
    --output scenario_scores.csv
```

**TracIn scoring** (gradient dot-product):
```bash
python scoring/compute_tracin.py \
    --config configs/tracin_curriculum.yaml \
    --checkpoint /path/to/trained_model.pth \
    --output tracin_scores.csv
```

### Step 2: Train with curriculum

The `CurriculumSampler` is a drop-in replacement for PyTorch's default shuffling:

```python
from curriculum.curriculum_sampler import CurriculumSampler

scores = load_scores("tracin_scores.csv")  # normalized to [0, 1]
sampler = CurriculumSampler(scores, total_epochs=20)
loader = DataLoader(dataset, batch_size=8, sampler=sampler)

for epoch in range(20):
    sampler.set_epoch(epoch)
    for batch in loader:
        # training step
        ...
```

The three curriculum phases:
- **Warm-up** (epochs 1-6): Uniform weights, build general representations
- **Ramp-up** (epochs 7-13): Progressively upweight high-scoring samples
- **Focus** (epochs 14-20): Maximum differentiation, top-20% boosted

### Step 3: Reproduce figures

```bash
python figures/generate_ade_comparison.py
```

## Key Results

| Method | ADE (m) | CV (%) | p-value |
|--------|---------|--------|---------|
| TracIn Curriculum | **1.704 +/- 0.029** | **1.7** | — |
| Hybrid Curriculum | 1.766 +/- 0.069 | 3.9 | 0.266 |
| Baseline (uniform) | 1.772 +/- 0.134 | 7.6 | 0.496 |
| Metadata Curriculum | 1.822 +/- 0.014 | 0.7 | 0.021* |
| Loss SPL | 2.003 +/- 0.391 | 19.5 | 0.379 |

*All comparisons are paired t-tests vs. TracIn curriculum across 3 seeds.

## Citation

```bibtex
@inproceedings{li2026gradient,
  title={Gradient-Based Data Valuation Improves Curriculum Learning
         for Game-Theoretic Motion Planning},
  author={Li, Shihao and Li, Jiachen},
  booktitle={IEEE Conference on Decision and Control (CDC)},
  year={2026}
}
```

## License

This project is released under the MIT License.
