"""Bar plot for convergence speed / specialization across algorithm pairs.

Phase 1: simple bar plot since only one pair (NL+NL).
Phase 2+: full heatmap across multiple algorithm pairs.

Usage:
    python overcooked_speed/analysis/plot_heatmap.py \
        --summary_csv logs/sweep/summary_all.csv --save_dir logs/figures
"""
import argparse
import os
import sys
import numpy as np
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def plot_bar_metrics(summary_file, save_dir):
    """Bar plot showing convergence and specialization metrics across pairs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = []
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("No data in summary CSV")
        return

    pairs = [f"{r['agent0_type']}+{r['agent1_type']}" for r in rows]
    metric_groups = [
        ('final_reward_mean', 'Final Reward', 'reward'),
        ('T_reward_mean', 'Convergence Episode (T_reward)', 'conv'),
        ('final_specialization_mean', 'Final Specialization', 'spec'),
        ('T_specialization_mean', 'Specialization Episode (T_spec)', 'conv_spec'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (key, ylabel, tag) in enumerate(metric_groups):
        ax = axes[idx]
        values = [float(r.get(key, 0)) for r in rows]
        stds = [float(r.get(key.replace('_mean', '_std'), 0)) for r in rows]

        x = np.arange(len(pairs))
        bars = ax.bar(x, values, yerr=stds, capsize=5, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(pairs)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Convergence & Specialization Metrics', fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'metrics_bar.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_csv', default='logs/sweep/summary_all.csv')
    parser.add_argument('--save_dir', default='logs/figures')
    args = parser.parse_args()

    plot_bar_metrics(args.summary_csv, args.save_dir)


if __name__ == '__main__':
    main()
