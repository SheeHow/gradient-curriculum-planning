#!/usr/bin/env python3
"""Generate Nature-level main ADE comparison figure.

Replaces the generic bar chart (fig1_multiseed_valADE) with a strip plot
featuring a broken y-axis to handle the Loss SPL outlier:
  - X-axis: methods sorted by mean ADE (best first, left to right)
  - Y-axis: broken axis — lower panel covers the main cluster (1.5–1.95),
    upper panel covers the outlier region (2.5–2.6)
  - Individual seeds as colored markers with method-specific colors
  - Mean as a horizontal bar with ±1 std whiskers
  - Significance bracket for TracIn vs Metadata (p=0.021*)
  - CV annotated below each method name
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
FIG_DIR = Path(__file__).resolve().parent
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Raw per-seed data (best epoch per seed by min val-planningADE) ─────
seeds = [3407, 42, 2024]

methods_data = {
    'TracIn':   [1.6868, 1.6804, 1.7457],
    'Hybrid':   [1.7716, 1.8478, 1.6795],
    'Baseline': [1.9166, 1.5931, 1.8065],
    'Metadata': [1.8317, 1.8026, 1.8307],
    'Loss SPL': [1.7275, 1.7256, 2.5550],
}

# Sort by mean ADE (best first)
method_order = sorted(methods_data.keys(),
                      key=lambda m: np.mean(methods_data[m]))

# ── Colors ────────────────────────────────────────────────────────────
palette = {
    'TracIn':    '#1b7837',
    'Baseline':  '#4575b4',
    'Metadata':  '#e08214',
    'Loss SPL':  '#c51b7d',
    'Hybrid':    '#7570b3',
}

seed_markers = {3407: 'o', 42: 's', 2024: '^'}

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# ── Broken y-axis: two vertically stacked subplots ───────────────────
# Upper panel (small): outlier region 2.50–2.62
# Lower panel (large): main cluster 1.50–1.97
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(4.5, 3.2), sharex=True,
    gridspec_kw={'height_ratios': [1, 5], 'hspace': 0.06}
)

# Y-limits
y_bot_lo, y_bot_hi = 1.50, 1.97
y_top_lo, y_top_hi = 2.50, 2.62
ax_bot.set_ylim(y_bot_lo, y_bot_hi)
ax_top.set_ylim(y_top_lo, y_top_hi)

# Hide spines at the break
ax_top.spines['bottom'].set_visible(False)
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_bot.spines['right'].set_visible(False)

ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax_top.tick_params(axis='y', labelsize=7.5)
ax_top.set_yticks([2.55])
ax_bot.tick_params(axis='y', labelsize=8)

# No break marks — the gap between panels suffices to indicate the axis break

# Shared y-label
fig.text(0.01, 0.5, 'Planning ADE (m)', va='center', rotation='vertical',
         fontsize=10)

# ── Helper: draw data on both axes ────────────────────────────────────
x_positions = {method: i for i, method in enumerate(method_order)}

for ax in [ax_top, ax_bot]:
    ax.grid(axis='y', alpha=0.15, linewidth=0.4)
    ax.set_axisbelow(True)

    for i, method in enumerate(method_order):
        x = x_positions[method]
        vals = np.array(methods_data[method])
        mean_val = vals.mean()
        std_val = vals.std()
        color = palette[method]

        # Mean horizontal bar
        bar_hw = 0.25
        ax.plot([x - bar_hw, x + bar_hw], [mean_val, mean_val],
                color=color, linewidth=2.8, solid_capstyle='round', zorder=5)

        # Std whiskers
        ax.plot([x, x], [mean_val - std_val, mean_val + std_val],
                color=color, linewidth=1.3, zorder=4, alpha=0.5)
        cap_hw = 0.1
        for y_cap in [mean_val - std_val, mean_val + std_val]:
            ax.plot([x - cap_hw, x + cap_hw], [y_cap, y_cap],
                    color=color, linewidth=1.3, zorder=4, alpha=0.5)

        # Individual seed points (jittered)
        for s_idx, seed in enumerate(seeds):
            val = methods_data[method][s_idx]
            jitter = (s_idx - 1) * 0.07
            ax.scatter(x + jitter, val, marker=seed_markers[seed], s=32,
                       color=color, edgecolors='white', linewidths=0.5,
                       zorder=6, alpha=0.9)

# ── Mean annotations on the lower axis ────────────────────────────────
for i, method in enumerate(method_order):
    if method == 'Loss SPL':
        continue  # handled separately below
    x = x_positions[method]
    vals = np.array(methods_data[method])
    mean_val = vals.mean()
    std_val = vals.std()
    color = palette[method]

    # Annotation position: below whisker, but stay within y limits
    annot_y = mean_val - std_val
    if annot_y < y_bot_lo + 0.03:
        # Too close to bottom, place to the side
        ax_bot.annotate(f'{mean_val:.3f}', xy=(x + 0.28, mean_val),
                        fontsize=7, color=color, ha='left', va='center',
                        fontweight='bold')
    else:
        ax_bot.annotate(f'{mean_val:.3f}', xy=(x, annot_y),
                        xytext=(0, -8), textcoords='offset points',
                        fontsize=7, color=color, ha='center', va='top',
                        fontweight='bold')

# Special case: Loss SPL mean (2.003) falls in the break gap.
# Show annotation on upper panel
x_loss = x_positions['Loss SPL']
vals_loss = np.array(methods_data['Loss SPL'])
mean_loss = vals_loss.mean()
ax_top.annotate(f'{mean_loss:.3f}', xy=(x_loss, y_top_lo),
                xytext=(8, 2), textcoords='offset points',
                fontsize=7, color=palette['Loss SPL'], ha='left', va='bottom',
                fontweight='bold', clip_on=False)

# ── Significance bracket on lower axis: TracIn vs Metadata ────────────
x_tracin = x_positions['TracIn']
x_meta = x_positions['Metadata']

bracket_y = 1.935
bracket_h = 0.012

ax_bot.plot([x_tracin, x_tracin, x_meta, x_meta],
            [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
            color='#333333', linewidth=0.8, zorder=7, clip_on=False)
ax_bot.text((x_tracin + x_meta) / 2, bracket_y + bracket_h + 0.005,
            'p = 0.021*', ha='center', va='bottom', fontsize=7.5,
            fontweight='bold', color='#333333', zorder=8)

# ── X-axis labels with CV ─────────────────────────────────────────────
ax_bot.set_xticks(range(len(method_order)))
xlabels = []
for method in method_order:
    vals = np.array(methods_data[method])
    cv = vals.std() / vals.mean() * 100
    xlabels.append(f'{method}\n(CV={cv:.1f}%)')
ax_bot.set_xticklabels(xlabels, fontsize=8, fontweight='medium',
                       linespacing=1.3)

# ── Seed legend on lower axis ─────────────────────────────────────────
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='#555', markerfacecolor='#555',
               markersize=4, linestyle='None', label='Seed 3407'),
    plt.Line2D([0], [0], marker='s', color='#555', markerfacecolor='#555',
               markersize=4, linestyle='None', label='Seed 42'),
    plt.Line2D([0], [0], marker='^', color='#555', markerfacecolor='#555',
               markersize=4, linestyle='None', label='Seed 2024'),
    plt.Line2D([0], [0], color='#555', linewidth=2.5,
               solid_capstyle='round', label='Mean ± 1 s.d.'),
]
# Horizontal legend below the figure
fig.legend(handles=legend_elements, loc='lower center', fontsize=7,
           ncol=4, frameon=False, handletextpad=0.3,
           columnspacing=1.2, bbox_to_anchor=(0.53, -0.02))

# Adjust margins
fig.subplots_adjust(left=0.14, bottom=0.22)

# ── Save ──────────────────────────────────────────────────────────────
for ext in ['pdf', 'png']:
    fig.savefig(FIG_DIR / f'fig1_multiseed_valADE_v2.{ext}')
plt.close()

print("Generated fig1_multiseed_valADE_v2.pdf and fig1_multiseed_valADE_v2.png")
print(f"Output directory: {FIG_DIR}")
