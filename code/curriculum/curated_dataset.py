"""
Curated Dataset for GameFormer Data-Centric Training.

Extends GameFormer's DrivingData to support:
1. Subset selection from a pre-computed scenario score file
2. Subset by top-k fraction, random sampling, or explicit file list
"""

import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CuratedDrivingData(Dataset):
    """GameFormer-compatible dataset with scenario curation support.

    Parameters
    ----------
    data_dir : str
        Glob pattern for .npz files (e.g., "/path/to/data/*.npz")
    n_neighbors : int
        Number of neighbor agents to include.
    score_file : str or None
        Path to scenario_scores.csv from score_scenarios.py.
    fraction : float
        Fraction of data to keep (1.0 = all). Used with score_file
        to select top-k% by interaction-difficulty score.
    selection_mode : str
        One of: "top" (highest scores), "bottom" (lowest scores),
        "random" (uniform random), "list" (explicit file list).
    file_list : str or None
        Path to a text file with one filename per line. Used when
        selection_mode="list".
    seed : int
        Random seed for reproducible random selection.
    """

    def __init__(self, data_dir, n_neighbors, score_file=None, fraction=1.0,
                 selection_mode="top", file_list=None, seed=3407):
        self.data_list = sorted(glob.glob(data_dir))
        self._n_neighbors = n_neighbors

        if score_file is not None and fraction < 1.0:
            self.data_list = self._filter_by_score(
                score_file, fraction, selection_mode, seed
            )
        elif file_list is not None:
            self.data_list = self._filter_by_list(file_list)

    def _filter_by_score(self, score_file, fraction, mode, seed):
        """Filter data_list to keep a fraction of scenarios by score."""
        scores = pd.read_csv(score_file)
        total = len(scores)
        k = max(1, int(total * fraction))

        if mode == "top":
            selected = set(
                scores.nlargest(k, "composite_score")["filename"].values
            )
        elif mode == "bottom":
            selected = set(
                scores.nsmallest(k, "composite_score")["filename"].values
            )
        elif mode == "random":
            rng = np.random.RandomState(seed)
            idx = rng.choice(total, size=k, replace=False)
            selected = set(scores.iloc[idx]["filename"].values)
        else:
            raise ValueError(f"Unknown selection_mode: {mode}")

        filtered = [f for f in self.data_list if os.path.basename(f) in selected]
        return filtered

    def _filter_by_list(self, file_list_path):
        """Filter data_list to keep only filenames in the provided list."""
        with open(file_list_path) as f:
            selected = set(line.strip() for line in f if line.strip())
        filtered = [f for f in self.data_list if os.path.basename(f) in selected]
        return filtered

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data["ego_agent_past"]
        neighbors = data["neighbor_agents_past"]
        route_lanes = data["route_lanes"]
        map_lanes = data["lanes"]
        map_crosswalks = data["crosswalks"]
        ego_future_gt = data["ego_agent_future"]
        neighbors_future_gt = data["neighbor_agents_future"][:self._n_neighbors]

        return (ego, neighbors, map_lanes, map_crosswalks, route_lanes,
                ego_future_gt, neighbors_future_gt)

    def get_filenames(self):
        """Return list of scenario filenames (for logging/analysis)."""
        return [os.path.basename(f) for f in self.data_list]


class TypeBalancedDrivingData(CuratedDrivingData):
    """Dataset that balances scenario types to a target total size.

    Samples equally from each scenario type (based on filename prefix
    convention: <map_name>_<token>.npz). Falls back to oversampling
    rare types if needed.
    """

    def __init__(self, data_dir, n_neighbors, fraction=0.2, seed=3407):
        super().__init__(data_dir, n_neighbors)

        rng = np.random.RandomState(seed)
        target_total = max(1, int(len(self.data_list) * fraction))

        # Group files by map name (proxy for scenario type locality)
        groups = {}
        for f in self.data_list:
            basename = os.path.basename(f)
            # filename format: <map_name>_<token>.npz
            parts = basename.rsplit("_", 1)
            group_key = parts[0] if len(parts) > 1 else "unknown"
            groups.setdefault(group_key, []).append(f)

        n_groups = len(groups)
        per_group = max(1, target_total // n_groups)

        selected = []
        for group_files in groups.values():
            if len(group_files) <= per_group:
                selected.extend(group_files)
            else:
                idx = rng.choice(len(group_files), size=per_group, replace=False)
                selected.extend([group_files[i] for i in idx])

        # Trim to exact target if overshot
        if len(selected) > target_total:
            idx = rng.choice(len(selected), size=target_total, replace=False)
            selected = [selected[i] for i in idx]

        self.data_list = selected
