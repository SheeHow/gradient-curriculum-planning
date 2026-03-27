"""
Prepare curriculum-compatible score files from TracIn and loss-based scoring.

Converts tracin_scores.csv to curriculum-compatible format:
1. tracin_scores_for_curriculum.csv — TracIn gradient-similarity scores
2. loss_scores_for_curriculum.csv — loss-based scores (SPL baseline)

Both files have 'filename' and 'composite_score' columns matching the
scenario_scores.csv format, so they work with CurriculumSampler and
CuratedDrivingData without code changes.

Usage:
    python prepare_score_files.py --input tracin_scores.csv
"""

import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="tracin_scores.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples from {args.input}")
    print(f"Columns: {list(df.columns)}")

    # Detect whether this is TracIn or LiSSA influence format
    if "tracin_score" in df.columns:
        score_col = "tracin_score"
        norm_col = "tracin_score_normalized"
        label = "TracIn"
    elif "influence_score" in df.columns:
        score_col = "influence_score"
        norm_col = "influence_score_normalized"
        label = "Influence"
    else:
        raise ValueError(f"Unknown score format. Columns: {list(df.columns)}")

    # --- Gradient-based score file ---
    # Use normalized score as composite_score
    # Higher = more helpful to validation
    score_df = df[["filename", norm_col]].copy()
    score_df.columns = ["filename", "composite_score"]
    score_out = "influence_scores_for_curriculum.csv"
    score_df.to_csv(score_out, index=False)
    print(f"\n{label} scores saved to {score_out}")
    print(f"  Range: [{score_df['composite_score'].min():.4f}, "
          f"{score_df['composite_score'].max():.4f}]")
    print(f"  Mean: {score_df['composite_score'].mean():.4f}")

    # --- Loss-based score file (SPL baseline) ---
    loss = df["training_loss"].values
    loss_norm = (loss - loss.min()) / (loss.max() - loss.min() + 1e-10)
    loss_df = pd.DataFrame({
        "filename": df["filename"],
        "composite_score": loss_norm,
    })
    loss_out = "loss_scores_for_curriculum.csv"
    loss_df.to_csv(loss_out, index=False)
    print(f"\nLoss scores saved to {loss_out}")
    print(f"  Range: [{loss_df['composite_score'].min():.4f}, "
          f"{loss_df['composite_score'].max():.4f}]")
    print(f"  Mean: {loss_df['composite_score'].mean():.4f}")

    # --- Correlation summary ---
    from scipy.stats import spearmanr
    try:
        meta_df = pd.read_csv(
            "/home/sheehow/dr-claw/proj-2026-03-20-10-21-11/"
            "Experiment/datasets/scenario_scores.csv"
        )
        merged = df.merge(meta_df[["filename", "composite_score"]].rename(
            columns={"composite_score": "meta_score"}), on="filename")

        r1, p1 = spearmanr(merged[score_col], merged["meta_score"])
        r2, p2 = spearmanr(merged[score_col], merged["training_loss"])
        r3, p3 = spearmanr(merged["training_loss"], merged["meta_score"])

        print(f"\n--- Rank Correlations ---")
        print(f"Spearman({label}, metadata): r={r1:.4f}, p={p1:.2e}")
        print(f"Spearman({label}, loss):     r={r2:.4f}, p={p2:.2e}")
        print(f"Spearman(loss, metadata):    r={r3:.4f}, p={p3:.2e}")
    except FileNotFoundError:
        print("\nMetadata score file not found, skipping correlation analysis.")


if __name__ == "__main__":
    main()
