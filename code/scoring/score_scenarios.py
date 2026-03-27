"""
Interaction-Difficulty Scoring for nuPlan Preprocessed Scenarios.

Computes a 6-metric composite interaction-difficulty score for each
preprocessed .npz scenario file. Outputs a CSV with per-scenario metrics,
composite scores, and tier assignments.

Usage:
    python score_scenarios.py --data_dir /path/to/processed_data --output scenario_scores.csv
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Individual Metrics
# ---------------------------------------------------------------------------

def compute_d_min(ego_future_xy, neighbor_futures_xy, valid_mask):
    """Minimum distance between ego and any valid neighbor across all timesteps."""
    if valid_mask.sum() == 0:
        return np.inf

    # ego_future_xy: (T, 2), neighbor_futures_xy: (N, T, 2), valid_mask: (N,)
    dists = np.linalg.norm(
        ego_future_xy[None, :, :] - neighbor_futures_xy[valid_mask], axis=-1
    )  # (V, T)
    return float(np.min(dists))


def compute_ttc_min(ego_future_xy, neighbor_futures_xy, valid_mask, dt=0.1):
    """Minimum time-to-collision proxy across all valid neighbors and timesteps."""
    if valid_mask.sum() == 0:
        return np.inf

    neighbors = neighbor_futures_xy[valid_mask]  # (V, T, 2)
    dists = np.linalg.norm(
        ego_future_xy[None, :, :] - neighbors, axis=-1
    )  # (V, T)

    # Closing speed: positive when approaching
    closing_speed = -np.diff(dists, axis=1) / dt  # (V, T-1)

    # Only consider approaching timesteps
    approaching = closing_speed > 0.1  # minimum closing speed threshold
    if not approaching.any():
        return np.inf

    ttc_values = dists[:, :-1][approaching] / closing_speed[approaching]
    return float(np.min(ttc_values))


def compute_n_conflict(ego_future_xy, neighbor_futures_xy, valid_mask, threshold=3.0):
    """Count timesteps where any neighbor is within threshold of ego."""
    if valid_mask.sum() == 0:
        return 0

    neighbors = neighbor_futures_xy[valid_mask]  # (V, T, 2)
    dists = np.linalg.norm(
        ego_future_xy[None, :, :] - neighbors, axis=-1
    )  # (V, T)

    # For each timestep, check if ANY neighbor is within threshold
    any_close = np.any(dists < threshold, axis=0)  # (T,)
    return int(any_close.sum())


def compute_t_prox(ego_future_xy, neighbor_futures_xy, valid_mask,
                   threshold=10.0, dt=0.1):
    """Total time (seconds) that any neighbor is within threshold of ego."""
    if valid_mask.sum() == 0:
        return 0.0

    neighbors = neighbor_futures_xy[valid_mask]
    dists = np.linalg.norm(
        ego_future_xy[None, :, :] - neighbors, axis=-1
    )
    any_close = np.any(dists < threshold, axis=0)
    return float(any_close.sum() * dt)


def compute_delta_theta_max(ego_future, neighbor_futures, valid_mask):
    """Max heading difference between ego and nearest neighbor."""
    if valid_mask.sum() == 0:
        return 0.0

    ego_xy = ego_future[:, :2]  # (T, 2)
    ego_heading = ego_future[:, 2]  # (T,)
    neighbors_xy = neighbor_futures[valid_mask, :, :2]  # (V, T, 2)
    neighbors_heading = neighbor_futures[valid_mask, :, 2]  # (V, T)

    dists = np.linalg.norm(ego_xy[None, :, :] - neighbors_xy, axis=-1)  # (V, T)
    nearest_idx = np.argmin(dists, axis=0)  # (T,)

    delta_theta = []
    for t in range(len(ego_heading)):
        nh = neighbors_heading[nearest_idx[t], t]
        dtheta = abs(ego_heading[t] - nh)
        dtheta = min(dtheta, 2 * np.pi - dtheta)
        delta_theta.append(dtheta)

    return float(max(delta_theta)) if delta_theta else 0.0


def compute_n_active(neighbor_agents_past, valid_mask, speed_threshold=1.0):
    """Number of neighbors with velocity > speed_threshold at any past timestep."""
    if valid_mask.sum() == 0:
        return 0

    past = neighbor_agents_past[valid_mask]  # (V, 21, 11)
    # Columns 3, 4 are vx, vy in the 11-dim neighbor encoding
    vx = past[:, :, 3]
    vy = past[:, :, 4]
    speeds = np.sqrt(vx ** 2 + vy ** 2)
    active = np.any(speeds > speed_threshold, axis=1)  # (V,)
    return int(active.sum())


# ---------------------------------------------------------------------------
# Composite Score
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "w_d_min": 0.20,
    "w_ttc": 0.25,
    "w_conflict": 0.20,
    "w_prox": 0.15,
    "w_theta": 0.10,
    "w_active": 0.10,
}


def composite_score(d_min, ttc_min, n_conflict, t_prox, delta_theta_max,
                    n_active, weights=None):
    """Compute weighted composite interaction-difficulty score."""
    w = weights or DEFAULT_WEIGHTS

    s_d = 1.0 / max(d_min, 0.5)
    s_ttc = 1.0 / max(ttc_min, 0.5)
    s_conflict = n_conflict / 80.0
    s_prox = t_prox / 8.0
    s_theta = delta_theta_max / np.pi
    s_active = n_active / 20.0

    score = (w["w_d_min"] * s_d
             + w["w_ttc"] * s_ttc
             + w["w_conflict"] * s_conflict
             + w["w_prox"] * s_prox
             + w["w_theta"] * s_theta
             + w["w_active"] * s_active)

    return float(score)


# ---------------------------------------------------------------------------
# Scenario Scorer
# ---------------------------------------------------------------------------

def get_valid_neighbor_mask(neighbor_agents_past):
    """Return boolean mask of valid (non-zero) neighbors."""
    # A neighbor is valid if it has any nonzero position in its past trajectory
    return np.any(neighbor_agents_past[:, :, :2] != 0, axis=(1, 2))


def score_scenario(npz_path):
    """Score a single preprocessed scenario .npz file.

    Returns a dict of metrics + composite score, or None if file is invalid.
    """
    try:
        data = np.load(npz_path)
    except Exception:
        return None

    # Extract arrays
    ego_future = data.get("ego_agent_future")
    neighbor_futures = data.get("neighbor_agents_future")
    neighbor_past = data.get("neighbor_agents_past")

    if ego_future is None or neighbor_futures is None or neighbor_past is None:
        return None

    # Valid neighbor mask
    valid = get_valid_neighbor_mask(neighbor_past)

    # Ego future xy
    ego_xy = ego_future[:, :2]

    # Neighbor future xy
    neighbor_xy = neighbor_futures[:, :, :2]

    # Compute metrics
    d_min = compute_d_min(ego_xy, neighbor_xy, valid)
    ttc_min = compute_ttc_min(ego_xy, neighbor_xy, valid)
    n_conflict = compute_n_conflict(ego_xy, neighbor_xy, valid)
    t_prox = compute_t_prox(ego_xy, neighbor_xy, valid)
    delta_theta_max = compute_delta_theta_max(ego_future, neighbor_futures, valid)
    n_active = compute_n_active(neighbor_past, valid)

    score = composite_score(d_min, ttc_min, n_conflict, t_prox,
                            delta_theta_max, n_active)

    return {
        "filename": os.path.basename(npz_path),
        "d_min": round(d_min, 4),
        "TTC_min": round(ttc_min, 4),
        "n_conflict": n_conflict,
        "t_prox": round(t_prox, 2),
        "delta_theta_max": round(delta_theta_max, 4),
        "n_active": n_active,
        "composite_score": round(score, 6),
    }


def assign_tiers(df):
    """Assign interaction-difficulty tiers based on composite score percentiles."""
    q20 = df["composite_score"].quantile(0.20)
    q50 = df["composite_score"].quantile(0.50)
    q80 = df["composite_score"].quantile(0.80)

    conditions = [
        df["composite_score"] >= q80,
        df["composite_score"] >= q50,
        df["composite_score"] >= q20,
    ]
    choices = [4, 3, 2]
    df["tier"] = np.select(conditions, choices, default=1)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Score nuPlan scenarios by interaction difficulty")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing preprocessed .npz files")
    parser.add_argument("--output", type=str, default="scenario_scores.csv",
                        help="Output CSV path")
    parser.add_argument("--weights", type=str, default=None,
                        help="JSON file with custom weights (optional)")
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if not npz_files:
        print(f"No .npz files found in {args.data_dir}")
        return

    print(f"Scoring {len(npz_files)} scenarios...")

    # Load custom weights if provided
    weights = DEFAULT_WEIGHTS
    if args.weights:
        import json
        with open(args.weights) as f:
            weights = json.load(f)

    records = []
    for path in tqdm(npz_files, desc="Scoring"):
        result = score_scenario(path)
        if result is not None:
            records.append(result)

    df = pd.DataFrame(records)
    df = assign_tiers(df)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.to_csv(args.output, index=False)

    # Summary statistics
    print(f"\nScored {len(df)} scenarios → {args.output}")
    print(f"\nTier distribution:")
    for tier in [4, 3, 2, 1]:
        count = (df["tier"] == tier).sum()
        pct = count / len(df) * 100
        print(f"  Tier {tier}: {count} ({pct:.1f}%)")
    print(f"\nScore range: [{df['composite_score'].min():.4f}, {df['composite_score'].max():.4f}]")
    print(f"Score mean: {df['composite_score'].mean():.4f}, std: {df['composite_score'].std():.4f}")


if __name__ == "__main__":
    main()
