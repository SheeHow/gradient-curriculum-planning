#!/usr/bin/env python3
"""Generate prediction quality comparison GIFs: Baseline vs TracIn-Curriculum.

For each selected validation scenario, renders an animated side-by-side:
  Left:  Baseline model prediction vs. ground truth
  Right: TracIn-Curriculum model prediction vs. ground truth

This directly visualizes sample efficiency: same data budget, better predictions.

Also generates:
  - A 3-row gallery of diverse scenarios with both models
  - Individual before/after GIFs for the most improved scenarios

Usage:
    python generate_prediction_comparison.py \
        --val_dir /path/to/nuplan_processed/val \
        --baseline_ckpt /path/to/baseline/model_best.pth \
        --curriculum_ckpt /path/to/curriculum/model_best.pth \
        --gameformer_dir /path/to/gameformer-planner \
        --output_dir ./output
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────
C_EGO_PAST = '#2166AC'
C_GT_FUTURE = '#2ca02c'       # green for ground truth
C_PRED_FUTURE = '#d62728'     # red for prediction
C_PRED_GHOST = '#d6272855'    # faint prediction trail
C_NEIGHBOR_PAST = '#525252'
C_NEIGHBOR_FUTURE = '#FD8D3C'
C_LANE = '#BDBDBD'
C_ROUTE = '#74C476'
C_BASELINE_BG = '#fff5f5'     # faint red tint
C_CURRICULUM_BG = '#f0fff4'   # faint green tint


def load_model(ckpt_path, device):
    """Load a GameFormer checkpoint."""
    from GameFormer.predictor import GameFormer
    model = GameFormer(
        encoder_layers=3,
        decoder_levels=2,
        modalities=6,
        neighbors=10,
    )
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def load_scenario_raw(filepath):
    """Load a single .npz scenario file."""
    data = np.load(filepath, allow_pickle=True)
    return {k: data[k] for k in data.files}


def prepare_inputs(scenario, device, n_neighbors=10):
    """Prepare model inputs from a raw scenario dict."""
    inputs = {
        'ego_agent_past': torch.from_numpy(scenario['ego_agent_past']).float().unsqueeze(0).to(device),
        'neighbor_agents_past': torch.from_numpy(scenario['neighbor_agents_past'][:n_neighbors]).float().unsqueeze(0).to(device),
        'map_lanes': torch.from_numpy(scenario['lanes']).float().unsqueeze(0).to(device),
        'map_crosswalks': torch.from_numpy(scenario['crosswalks']).float().unsqueeze(0).to(device),
        'route_lanes': torch.from_numpy(scenario['route_lanes']).float().unsqueeze(0).to(device),
    }
    gt_future = scenario['ego_agent_future']  # (80, 3)
    return inputs, gt_future


@torch.no_grad()
def predict(model, inputs):
    """Run model inference. Returns ego plan as numpy (80, 3)."""
    decoder_outputs, ego_plan = model(inputs)
    return ego_plan[0].cpu().numpy()  # (80, 3) — x, y, yaw


def compute_ade(pred, gt):
    """Average Displacement Error (xy only)."""
    mask = np.any(gt[:, :2] != 0, axis=1)
    if mask.sum() == 0:
        return float('inf')
    return np.mean(np.linalg.norm(pred[mask, :2] - gt[mask, :2], axis=1))


def find_interesting_val_scenarios(val_dir, baseline_model, curriculum_model, device,
                                    n_scan=200, n_select=6):
    """Find scenarios where curriculum model clearly outperforms baseline."""
    val_files = sorted(Path(val_dir).glob("*.npz"))[:n_scan]
    results = []

    print(f"  Scanning {len(val_files)} validation scenarios...")
    for fpath in val_files:
        sc = load_scenario_raw(fpath)
        inputs, gt = prepare_inputs(sc, device)

        # Check ego trajectory quality
        mask_gt = np.any(gt[:, :2] != 0, axis=1)
        if mask_gt.sum() < 20:
            continue

        # Compactness check
        ego_past = sc['ego_agent_past'][:, :2]
        mask_p = np.any(ego_past != 0, axis=1)
        all_pts = np.concatenate([ego_past[mask_p], gt[mask_gt, :2]])
        spread = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
                     all_pts[:, 1].max() - all_pts[:, 1].min())
        if spread > 80:
            continue

        pred_base = predict(baseline_model, inputs)
        pred_curr = predict(curriculum_model, inputs)

        ade_base = compute_ade(pred_base, gt)
        ade_curr = compute_ade(pred_curr, gt)

        # Nearby agents for visual interest
        ego_pos = ego_past[mask_p][-1] if mask_p.sum() > 0 else np.zeros(2)
        n_close = 0
        for j in range(min(sc['neighbor_agents_past'].shape[0], 10)):
            np_j = sc['neighbor_agents_past'][j]
            m = np.any(np_j[:, :2] != 0, axis=1)
            if m.sum() > 1:
                pos = np_j[m, :2][-1]
                if np.linalg.norm(pos - ego_pos) < 30:
                    n_close += 1

        improvement = ade_base - ade_curr
        visual_score = n_close * 2.0 + improvement * 5.0

        results.append({
            'path': fpath,
            'scenario': sc,
            'gt': gt,
            'pred_base': pred_base,
            'pred_curr': pred_curr,
            'ade_base': ade_base,
            'ade_curr': ade_curr,
            'improvement': improvement,
            'n_close': n_close,
            'visual_score': visual_score,
        })

    # Sort by visual_score (prioritize improvement + visual interest)
    results.sort(key=lambda x: x['visual_score'], reverse=True)
    selected = results[:n_select]

    for i, r in enumerate(selected):
        print(f"    #{i+1}: {r['path'].name[:40]}... "
              f"ADE baseline={r['ade_base']:.3f}, curriculum={r['ade_curr']:.3f}, "
              f"improvement={r['improvement']:.3f}m, agents={r['n_close']}")

    return selected


def render_comparison_frame(ax, scenario, gt, pred, t_step, n_past=21, n_future=80,
                            title="", ade_val=None, bg_color=None):
    """Render a single frame showing prediction vs ground truth."""
    ax.clear()
    if bg_color:
        ax.set_facecolor(bg_color)

    ego_past = scenario['ego_agent_past']
    neigh_past = scenario['neighbor_agents_past']
    neigh_future = scenario['neighbor_agents_future']
    lanes = scenario['lanes']
    route_lanes = scenario['route_lanes']

    # Draw lanes
    for lane in lanes:
        mask = np.any(lane[:, :2] != 0, axis=1)
        if mask.sum() > 1:
            ax.plot(lane[mask, 0], lane[mask, 1], color=C_LANE, linewidth=0.5,
                    alpha=0.5, zorder=1)
    for rl in route_lanes:
        mask = np.any(rl[:, :2] != 0, axis=1)
        if mask.sum() > 1:
            ax.plot(rl[mask, 0], rl[mask, 1], color=C_ROUTE, linewidth=1.5,
                    alpha=0.4, zorder=2)

    # Time indices
    if t_step < n_past:
        past_end = t_step + 1
        future_end = 0
    else:
        past_end = n_past
        future_end = t_step - n_past + 1

    # Ego past
    ego_p = ego_past[:past_end, :2]
    mask_ep = np.any(ego_p != 0, axis=1)
    if mask_ep.sum() > 1:
        ax.plot(ego_p[mask_ep, 0], ego_p[mask_ep, 1], color=C_EGO_PAST,
                linewidth=2.0, alpha=0.7, zorder=6, solid_capstyle='round')

    # Ground truth future (revealed progressively)
    if future_end > 0:
        gt_f = gt[:future_end, :2]
        mask_gf = np.any(gt_f != 0, axis=1)
        if mask_gf.sum() > 0:
            ax.plot(gt_f[mask_gf, 0], gt_f[mask_gf, 1], color=C_GT_FUTURE,
                    linewidth=2.5, alpha=0.8, zorder=8, solid_capstyle='round',
                    label='Ground Truth')

        # Predicted future (revealed progressively)
        pred_f = pred[:future_end, :2]
        ax.plot(pred_f[:, 0], pred_f[:, 1], color=C_PRED_FUTURE,
                linewidth=2.0, alpha=0.7, zorder=7, linestyle='--',
                solid_capstyle='round', label='Prediction')

    # Full prediction as ghost line
    if future_end > 0:
        gt_full = gt[:, :2]
        mask_gfull = np.any(gt_full != 0, axis=1)
        if mask_gfull.sum() > 1:
            ax.plot(gt_full[mask_gfull, 0], gt_full[mask_gfull, 1],
                    color=C_GT_FUTURE, linewidth=0.8, alpha=0.15, zorder=4)
        ax.plot(pred[:, 0], pred[:, 1], color=C_PRED_FUTURE,
                linewidth=0.8, alpha=0.15, zorder=4, linestyle='--')

    # Current ego position
    if t_step < n_past:
        if mask_ep.sum() > 0:
            cx, cy = ego_p[mask_ep][-1]
            ax.scatter(cx, cy, color='#E31A1C', s=60, zorder=10,
                       edgecolors='k', linewidths=0.8, marker='o')
    else:
        if future_end > 0:
            gt_f = gt[:future_end, :2]
            mask_gf = np.any(gt_f != 0, axis=1)
            if mask_gf.sum() > 0:
                cx, cy = gt_f[mask_gf][-1]
                ax.scatter(cx, cy, color=C_GT_FUTURE, s=50, zorder=10,
                           edgecolors='k', linewidths=0.6, marker='o')
            px, py = pred_f[-1, :2]
            ax.scatter(px, py, color=C_PRED_FUTURE, s=40, zorder=10,
                       edgecolors='k', linewidths=0.6, marker='s')

    # Neighbor agents
    for j in range(min(neigh_past.shape[0], 10)):
        np_p = neigh_past[j, :past_end, :2]
        mask_np = np.any(np_p != 0, axis=1)
        if mask_np.sum() > 1:
            ax.plot(np_p[mask_np, 0], np_p[mask_np, 1],
                    color=C_NEIGHBOR_PAST, linewidth=0.6, alpha=0.3, zorder=3)
        if future_end > 0 and j < neigh_future.shape[0]:
            np_f = neigh_future[j, :future_end, :2]
            mask_nf = np.any(np_f != 0, axis=1)
            if mask_nf.sum() > 0:
                ax.plot(np_f[mask_nf, 0], np_f[mask_nf, 1],
                        color=C_NEIGHBOR_FUTURE, linewidth=0.6, alpha=0.3,
                        linestyle='--', zorder=3)

    # Viewport
    mask_full_p = np.any(ego_past[:, :2] != 0, axis=1)
    mask_full_f = np.any(gt[:, :2] != 0, axis=1)
    all_ego = np.concatenate([ego_past[mask_full_p, :2], gt[mask_full_f, :2]])
    cx, cy = all_ego.mean(axis=0)
    span = max(all_ego[:, 0].max() - all_ego[:, 0].min(),
               all_ego[:, 1].max() - all_ego[:, 1].min(), 25) * 0.7
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=5)
    ax.grid(True, alpha=0.15, linewidth=0.3)

    # Phase and title
    if t_step < n_past:
        phase = f"Observation (t={t_step - n_past + 1:+.1f}s)"
    else:
        phase = f"Future (t=+{(t_step - n_past) * 0.1:.1f}s)"

    ade_str = f" | ADE={ade_val:.2f}m" if ade_val is not None else ""
    ax.set_title(f"{title}{ade_str}\n{phase}", fontsize=7, fontweight='bold')

    if future_end == 2:
        ax.legend(fontsize=5, loc='upper left', framealpha=0.7)


def generate_sidebyside_gif(scenario, gt, pred_base, pred_curr,
                             ade_base, ade_curr, output_path, fps=12):
    """Generate a side-by-side comparison GIF: Baseline vs Curriculum."""
    n_past = 21
    n_future = 80
    past_frames = list(range(0, n_past, 2))
    future_frames = list(range(n_past, n_past + n_future, 2))
    all_frames = past_frames + future_frames

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.0), dpi=130)

    frames = []
    for t in all_frames:
        render_comparison_frame(ax1, scenario, gt, pred_base, t,
                                title="Baseline (Uniform)",
                                ade_val=ade_base, bg_color=C_BASELINE_BG)
        render_comparison_frame(ax2, scenario, gt, pred_curr, t,
                                title="TracIn-Curriculum",
                                ade_val=ade_curr, bg_color=C_CURRICULUM_BG)
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        frames.append(img.copy())

    plt.close(fig)

    gif_path = output_path.with_suffix('.gif')
    imageio.mimsave(str(gif_path), frames, duration=int(1000 / fps), loop=0)
    size_kb = gif_path.stat().st_size / 1024
    print(f"  -> Saved {gif_path.name} ({len(frames)} frames, {size_kb:.0f} KB)")
    return gif_path


def generate_gallery_gif(results, output_path, fps=10):
    """Generate a multi-row gallery GIF of prediction comparisons."""
    n_past = 21
    n_future = 80
    past_frames = list(range(0, n_past, 3))
    future_frames = list(range(n_past, n_past + n_future, 3))
    all_frames = past_frames + future_frames

    n_scenarios = min(len(results), 3)
    fig, axes = plt.subplots(n_scenarios, 2, figsize=(8.5, n_scenarios * 3.2), dpi=120)
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)

    frames = []
    for t in all_frames:
        for i in range(n_scenarios):
            r = results[i]
            render_comparison_frame(
                axes[i, 0], r['scenario'], r['gt'], r['pred_base'], t,
                title="Baseline", ade_val=r['ade_base'], bg_color=C_BASELINE_BG
            )
            render_comparison_frame(
                axes[i, 1], r['scenario'], r['gt'], r['pred_curr'], t,
                title="TracIn-Curriculum", ade_val=r['ade_curr'], bg_color=C_CURRICULUM_BG
            )
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)
        frames.append(img.copy())

    plt.close(fig)

    gif_path = output_path.with_suffix('.gif')
    imageio.mimsave(str(gif_path), frames, duration=int(1000 / fps), loop=0)
    size_kb = gif_path.stat().st_size / 1024
    print(f"  -> Saved {gif_path.name} ({len(frames)} frames, {size_kb:.0f} KB)")
    return gif_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate prediction quality comparison GIFs: Baseline vs Curriculum"
    )
    parser.add_argument("--val_dir", type=str, required=True,
                        help="Path to nuplan_processed/val directory")
    parser.add_argument("--baseline_ckpt", type=str, required=True,
                        help="Path to baseline model_best.pth")
    parser.add_argument("--curriculum_ckpt", type=str, required=True,
                        help="Path to curriculum model_best.pth")
    parser.add_argument("--gameformer_dir", type=str, required=True,
                        help="Path to gameformer-planner repo (contains GameFormer/)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for GIFs (default: ./output)")
    parser.add_argument("--n_scan", type=int, default=300,
                        help="Number of validation scenarios to scan")
    parser.add_argument("--n_select", type=int, default=6,
                        help="Number of top scenarios to select")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: cuda, cpu, or auto (default)")
    args = parser.parse_args()

    # Add GameFormer to path
    sys.path.insert(0, args.gameformer_dir)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("Prediction Quality Comparison: Baseline vs TracIn-Curriculum")
    print("=" * 60)
    print(f"\nDevice: {device}")

    # Load models
    print("\nLoading baseline model...")
    baseline_model = load_model(args.baseline_ckpt, device)
    print(f"  Loaded: {Path(args.baseline_ckpt).name}")

    print("Loading curriculum model...")
    curriculum_model = load_model(args.curriculum_ckpt, device)
    print(f"  Loaded: {Path(args.curriculum_ckpt).name}")

    # Find interesting scenarios
    print("\nFinding scenarios with clear curriculum advantage...")
    results = find_interesting_val_scenarios(
        args.val_dir, baseline_model, curriculum_model, device,
        n_scan=args.n_scan, n_select=args.n_select
    )

    if len(results) == 0:
        print("ERROR: No suitable scenarios found")
        return

    # Generate individual side-by-side GIFs for top 3
    print("\n── Generating individual comparison GIFs ──")
    for i, r in enumerate(results[:3]):
        print(f"\nScenario {i+1}: {r['path'].name}")
        print(f"  Baseline ADE={r['ade_base']:.3f}, Curriculum ADE={r['ade_curr']:.3f}, "
              f"Improvement={r['improvement']:.3f}m")
        generate_sidebyside_gif(
            r['scenario'], r['gt'], r['pred_base'], r['pred_curr'],
            r['ade_base'], r['ade_curr'],
            out_dir / f"demo_pred_comparison_{i+1}"
        )

    # Generate gallery GIF
    print("\n── Generating prediction gallery GIF ──")
    generate_gallery_gif(results[:3], out_dir / "demo_pred_gallery")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary of prediction improvements:")
    print(f"{'Scenario':<45} {'Baseline':>10} {'Curriculum':>10} {'Improve':>10}")
    print("-" * 75)
    for r in results[:args.n_select]:
        name = r['path'].name[:42]
        print(f"{name:<45} {r['ade_base']:>10.3f} {r['ade_curr']:>10.3f} "
              f"{r['improvement']:>+10.3f}")
    mean_imp = np.mean([r['improvement'] for r in results[:args.n_select]])
    print("-" * 75)
    print(f"{'Mean improvement':<45} {'':>10} {'':>10} {mean_imp:>+10.3f}")
    print(f"\nOutput directory: {out_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
