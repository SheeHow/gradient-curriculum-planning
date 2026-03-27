"""
Curriculum Sampler for GameFormer Data-Centric Training.

Implements a 3-phase easy-to-hard curriculum schedule:
  Phase 1 (Warm-up):  Favor easy scenarios for stable gradient learning
  Phase 2 (Ramp):     Gradually shift weight toward hard scenarios
  Phase 3 (Focus):    Emphasize high-interaction scenarios with extra
                      boost on top-20%

The sampler is epoch-aware and produces sampling weights that change
each epoch. It is a drop-in replacement for shuffle=True in DataLoader.

Usage:
    scores = load_scenario_scores(...)  # numpy array of composite scores
    sampler = CurriculumSampler(scores, total_epochs=20)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    for epoch in range(20):
        sampler.set_epoch(epoch)
        for batch in loader:
            ...
"""

import numpy as np
import torch
from torch.utils.data import Sampler


def curriculum_weights(scores, epoch, total_epochs=20):
    """Compute per-scenario sampling weights for a given epoch.

    Parameters
    ----------
    scores : np.ndarray
        Composite interaction-difficulty scores, shape (N,).
    epoch : int
        Current epoch (0-indexed).
    total_epochs : int
        Total number of training epochs.

    Returns
    -------
    weights : np.ndarray
        Sampling probability distribution, shape (N,). Sums to 1.
    """
    # Normalize scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-8:
        return np.ones(len(scores)) / len(scores)
    s = (scores - s_min) / (s_max - s_min)

    progress = (epoch + 1) / total_epochs  # 1/T to 1.0

    if progress <= 0.3:
        # Phase 1: Warm-up (epochs 1–6 for 20 epochs)
        # Favor easy scenarios: weight = 1 - alpha * s
        alpha = 1.0 - progress / 0.3  # 1.0 → 0.0
        weights = np.maximum(1.0 - alpha * s, 0.1)

    elif progress <= 0.65:
        # Phase 2: Ramp (epochs 7–13 for 20 epochs)
        # Uniform → hard-weighted transition
        beta = (progress - 0.3) / 0.35  # 0.0 → 1.0
        weights = 1.0 + beta * s

    else:
        # Phase 3: Focus (epochs 14–20 for 20 epochs)
        # Hard-focused with extra boost on top 20%
        weights = 0.5 + 1.5 * s  # range [0.5, 2.0]
        top_20_threshold = np.percentile(s, 80)
        weights[s >= top_20_threshold] *= 1.5

    # Normalize to probability distribution
    weights = weights / weights.sum()
    return weights


class CurriculumSampler(Sampler):
    """Epoch-dependent weighted sampler for curriculum learning.

    Parameters
    ----------
    scores : np.ndarray
        Composite interaction-difficulty scores, shape (N,).
    total_epochs : int
        Total number of training epochs.
    epoch_size : int or None
        Number of samples per epoch. Defaults to len(scores).
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(self, scores, total_epochs, epoch_size=None, seed=3407):
        super().__init__()
        self.scores = np.asarray(scores, dtype=np.float64)
        self.total_epochs = total_epochs
        self.epoch_size = epoch_size or len(scores)
        self.seed = seed
        self.current_epoch = 0

    def set_epoch(self, epoch):
        """Set current epoch (0-indexed). Must be called before each epoch."""
        self.current_epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.current_epoch)
        weights = curriculum_weights(
            self.scores, self.current_epoch, self.total_epochs
        )
        indices = rng.choice(
            len(self.scores), size=self.epoch_size, replace=True, p=weights
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.epoch_size

    def get_phase_name(self):
        """Return human-readable phase name for logging."""
        progress = (self.current_epoch + 1) / self.total_epochs
        if progress <= 0.3:
            return "warm-up"
        elif progress <= 0.65:
            return "ramp"
        else:
            return "focus"


class UniformSampler(Sampler):
    """Uniform random sampler (baseline comparison).

    Provides the same interface as CurriculumSampler but with
    uniform sampling at every epoch.
    """

    def __init__(self, dataset_size, epoch_size=None, seed=3407):
        super().__init__()
        self.dataset_size = dataset_size
        self.epoch_size = epoch_size or dataset_size
        self.seed = seed
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.current_epoch)
        indices = rng.choice(
            self.dataset_size, size=self.epoch_size, replace=True
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.epoch_size
