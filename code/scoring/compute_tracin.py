"""
TracIn-style Data Influence Scoring for GameFormer-Planner.

Computes per-training-sample influence scores using the gradient dot-product
method (Pruthi et al., "Estimating Training Data Influence by Tracing Gradient
Descent", NeurIPS 2020).

For a single checkpoint:
    tracin(z_i) = -grad_L(z_i; theta*) . grad_L(z_val; theta*)

This avoids the inverse-Hessian approximation entirely, making it:
  - Deterministic (no stochastic estimation noise)
  - Fast (~75 min for 5148 samples vs 3+ hours for LiSSA)
  - Empirically effective for data valuation and sample ranking

Optionally computes multi-checkpoint TracIn by averaging over intermediate
checkpoints from training.

Output: tracin_scores.csv with columns
    [filename, tracin_score, tracin_score_normalized, training_loss,
     rank_tracin, rank_loss]

Usage:
    python compute_tracin.py --config configs/baseline.yaml \
        --checkpoint training_log/E1_baseline_full/model_best.pth \
        --output tracin_scores.csv
"""

import os
import sys
import csv
import yaml
import time
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

GAMEFORMER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "code_references", "gameformer-planner"
)
sys.path.insert(0, GAMEFORMER_DIR)

from GameFormer.predictor import GameFormer
from GameFormer.train_utils import level_k_loss, planning_loss
from curated_dataset import CuratedDrivingData


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def _forward_loss(model, batch, device):
    inputs = {
        "ego_agent_past": batch[0].float().to(device),
        "neighbor_agents_past": batch[1].float().to(device),
        "map_lanes": batch[2].float().to(device),
        "map_crosswalks": batch[3].float().to(device),
        "route_lanes": batch[4].float().to(device),
    }
    ego_future = batch[5].float().to(device)
    neighbors_future = batch[6].float().to(device)
    neighbors_future_valid = torch.ne(neighbors_future[..., :2], 0)

    level_k_outputs, ego_plan = model(inputs)
    loss, _ = level_k_loss(
        level_k_outputs, ego_future, neighbors_future, neighbors_future_valid
    )
    plan_loss = planning_loss(ego_plan, ego_future)
    return loss + plan_loss


def _wrap_sample(sample):
    return tuple(
        torch.tensor(b).unsqueeze(0) if isinstance(b, np.ndarray)
        else b.unsqueeze(0)
        for b in sample
    )


def compute_gradient(model, batch, device):
    model.zero_grad()
    with torch.backends.cudnn.flags(enabled=False):
        loss = _forward_loss(model, batch, device)
    loss.backward()

    grad_list = []
    for p in model.parameters():
        if p.grad is not None:
            grad_list.append(p.grad.detach().flatten())
        else:
            grad_list.append(torch.zeros(p.numel(), device=device))

    return torch.cat(grad_list), loss.item()


def compute_validation_gradient(model, val_loader, device):
    logging.info("Computing validation gradient...")
    total_grad = None
    n_batches = 0

    for batch in tqdm(val_loader, desc="Val gradient"):
        grad, _ = compute_gradient(model, batch, device)
        if total_grad is None:
            total_grad = grad
        else:
            total_grad += grad
        n_batches += 1

    avg_grad = total_grad / n_batches
    logging.info(f"Val gradient computed over {n_batches} batches, "
                 f"||g_val|| = {torch.norm(avg_grad).item():.6f}")
    return avg_grad


def load_model(config, checkpoint_path, device):
    model_cfg = config["model"]
    model = GameFormer(
        encoder_layers=model_cfg.get("encoder_layers", 3),
        decoder_levels=model_cfg.get("decoder_levels", 2),
        modalities=model_cfg.get("modalities", 6),
        neighbors=model_cfg.get("num_neighbors", 10),
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.train()
    return model


def main():
    parser = argparse.ArgumentParser(description="TracIn Data Influence Scoring")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Final checkpoint path")
    parser.add_argument("--extra_checkpoints", type=str, nargs="*", default=[],
                        help="Additional intermediate checkpoints for multi-ckpt TracIn")
    parser.add_argument("--output", type=str, default="tracin_scores.csv")
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    log_path = args.output.replace('.csv', '.log')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s %(asctime)s] %(message)s',
        datefmt='%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode='w'),
        ]
    )

    logging.info(f"Arguments: {args}")
    t_start = time.time()

    cfg = load_config(args.config)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logging.warning("CUDA not available, using CPU")

    # Collect all checkpoint paths
    all_checkpoints = [args.checkpoint] + args.extra_checkpoints
    logging.info(f"Using {len(all_checkpoints)} checkpoint(s) for TracIn")

    # Load datasets
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    n_neighbors = model_cfg.get("num_neighbors", 10)

    train_set = CuratedDrivingData(
        data_dir=data_cfg["train_dir"] + "/*.npz",
        n_neighbors=n_neighbors,
    )
    valid_set = CuratedDrivingData(
        data_dir=data_cfg["valid_dir"] + "/*.npz",
        n_neighbors=n_neighbors,
    )
    logging.info(f"Train: {len(train_set)}, Valid: {len(valid_set)}")

    filenames = train_set.get_filenames()
    n_train = len(train_set)

    # Accumulate TracIn scores across checkpoints
    tracin_scores = np.zeros(n_train)
    all_losses = np.zeros(n_train)  # from final checkpoint

    for ckpt_idx, ckpt_path in enumerate(all_checkpoints):
        logging.info(f"\n{'='*60}")
        logging.info(f"Checkpoint {ckpt_idx+1}/{len(all_checkpoints)}: {ckpt_path}")
        logging.info(f"{'='*60}")

        model = load_model(cfg, ckpt_path, device)
        n_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model: {n_params:,} parameters")

        val_loader = DataLoader(
            valid_set, batch_size=args.val_batch_size,
            shuffle=False, num_workers=4,
        )
        val_grad = compute_validation_gradient(model, val_loader, device)

        logging.info("Scoring training samples...")
        for i in tqdm(range(n_train), desc=f"Scoring (ckpt {ckpt_idx+1})"):
            batch = _wrap_sample(train_set[i])
            grad_i, loss_i = compute_gradient(model, batch, device)

            # TracIn score: -grad_i . g_val
            # Negative means sample reduces validation loss (beneficial)
            score_i = -torch.dot(grad_i, val_grad).item()
            tracin_scores[i] += score_i

            if ckpt_idx == 0:
                all_losses[i] = loss_i

            if (i + 1) % 500 == 0:
                logging.info(f"  Scored {i+1}/{n_train}, "
                             f"mean_tracin={np.mean(tracin_scores[:i+1]):.4f}")

        # Average over checkpoints if using multiple
        if len(all_checkpoints) > 1 and ckpt_idx == len(all_checkpoints) - 1:
            tracin_scores /= len(all_checkpoints)

        del model
        torch.cuda.empty_cache()

    t_score = time.time()
    logging.info(f"\nScoring done in {(t_score - t_start) / 60:.1f} min")

    # Normalize to [0, 1]
    s_min, s_max = tracin_scores.min(), tracin_scores.max()
    scores_norm = (tracin_scores - s_min) / (s_max - s_min + 1e-10)

    # Ranks
    rank_tracin = np.argsort(np.argsort(-tracin_scores))
    rank_loss = np.argsort(np.argsort(-all_losses))

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "tracin_score", "tracin_score_normalized",
            "training_loss", "rank_tracin", "rank_loss"
        ])
        for i in range(n_train):
            writer.writerow([
                filenames[i],
                f"{tracin_scores[i]:.8f}",
                f"{scores_norm[i]:.8f}",
                f"{all_losses[i]:.6f}",
                int(rank_tracin[i]),
                int(rank_loss[i]),
            ])

    logging.info(f"\nTracIn scores saved to {args.output}")
    logging.info(f"Score statistics:")
    logging.info(f"  TracIn: mean={tracin_scores.mean():.6f}, "
                 f"std={tracin_scores.std():.6f}, "
                 f"min={tracin_scores.min():.6f}, max={tracin_scores.max():.6f}")

    from scipy import stats
    spearman_r, spearman_p = stats.spearmanr(tracin_scores, all_losses)
    logging.info(f"  Spearman(TracIn, loss): r={spearman_r:.4f}, p={spearman_p:.2e}")

    # Load metadata for correlation
    try:
        meta_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "datasets", "scenario_scores.csv"
        )
        meta = pd.read_csv(meta_path)
        merged = pd.DataFrame({
            "filename": filenames,
            "tracin": tracin_scores,
            "loss": all_losses,
        }).merge(meta[["filename", "composite_score", "tier"]], on="filename")

        r1, p1 = stats.spearmanr(merged["tracin"], merged["composite_score"])
        r2, p2 = stats.spearmanr(merged["tracin"], merged["loss"])
        r3, p3 = stats.spearmanr(merged["loss"], merged["composite_score"])
        logging.info(f"  Spearman(TracIn, metadata): r={r1:.4f}, p={p1:.2e}")
        logging.info(f"  Spearman(TracIn, loss):     r={r2:.4f}, p={p2:.2e}")
        logging.info(f"  Spearman(loss, metadata):   r={r3:.4f}, p={p3:.2e}")

        logging.info("\n--- Mean TracIn by Tier ---")
        for t in sorted(merged["tier"].unique()):
            tdf = merged[merged["tier"] == t]
            logging.info(f"  Tier {t} (n={len(tdf):4d}): "
                         f"mean={tdf['tracin'].mean():.4f}, "
                         f"std={tdf['tracin'].std():.4f}")
    except Exception as e:
        logging.warning(f"Metadata analysis failed: {e}")

    # Save auxiliary data
    aux_path = args.output.replace('.csv', '_aux.pt')
    torch.save({
        'val_grad': val_grad.cpu(),
        'config': {
            'checkpoints': all_checkpoints,
            'n_train': n_train,
            'n_val': len(valid_set),
            'n_params': n_params,
        }
    }, aux_path)
    logging.info(f"Auxiliary data saved to {aux_path}")

    t_end = time.time()
    logging.info(f"\nTotal time: {(t_end - t_start) / 60:.1f} min")


if __name__ == "__main__":
    import pandas as pd
    main()
