#!/usr/bin/env python
"""Summarize LTS-PPO / RNN-IPPO / NL-IPPO experiment results from CSV.

Usage:
    python scripts/summarize_lts_results.py --log_dir logs/exp5_lts_selfplay
    python scripts/summarize_lts_results.py --log_dirs logs/exp4 logs/exp5  # compare
"""
import argparse
import csv
import os
import sys
import json
import numpy as np
from collections import defaultdict


# ── Field definitions ──
REWARD_FIELDS = ['reward']
LTS_FIELDS = [
    'loss_dy', 'dy_acc', 'loss_df', 'loss_df_action', 'loss_df_event',
    'loss_dc', 'loss_bel_cons',
    'ratio_mean', 'ratio_std', 'ratio_max', 'clip_fraction',
    'logprob_delta_sq', 'approx_kl_ppo',
    'belief_norm', 'belief_delta_mean', 'belief_encoder_grad_norm',
    'inter_memory_norm', 'inter_memory_filled',
]
PPO_FIELDS = ['loss', 'entropy', 'grad_norm', 'approx_kl', 'value_loss']


def load_csv(csv_path):
    """Load CSV, return list of dicts with numeric values."""
    if not os.path.exists(csv_path):
        return None
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            numeric_row = {}
            for k, v in row.items():
                if v is None or v == '' or v == 'None':
                    numeric_row[k] = None
                else:
                    try:
                        numeric_row[k] = float(v)
                    except (ValueError, TypeError):
                        numeric_row[k] = v
            rows.append(numeric_row)
    return rows


def safe_mean(vals):
    """Mean of non-None values, or NaN if empty."""
    clean = [v for v in vals if v is not None]
    return float(np.mean(clean)) if clean else float('nan')


def safe_sum(vals):
    """Sum of non-None values."""
    clean = [v for v in vals if v is not None]
    return float(np.sum(clean)) if clean else float('nan')


def compute_stats(rows, prefix=''):
    """Compute summary statistics from loaded CSV rows.

    Args:
        rows: list of dicts
        prefix: agent prefix filter ('a0_' or 'a1_') — if '', use raw field names.
                For RNN/NL agents that lack LTS fields, returns NaN.

    Returns:
        dict of stat_name → value
    """
    N = len(rows)
    if N == 0:
        return {}

    def col(name):
        """Get column values."""
        key = f'{prefix}{name}' if prefix else name
        return [r.get(key) for r in rows]

    def col_last(k, name):
        """Mean of column over last k rows."""
        key = f'{prefix}{name}' if prefix else name
        vals = [r.get(key) for r in rows[-k:]]
        return safe_mean(vals)

    def col_best(k, name, mode='max'):
        """Best moving average of column over window k."""
        key = f'{prefix}{name}' if prefix else name
        all_vals = [r.get(key) for r in rows if r.get(key) is not None]
        if len(all_vals) < k:
            return safe_mean(all_vals) if all_vals else float('nan')
        ma = np.convolve(all_vals, np.ones(k) / k, mode='valid')
        return float(np.max(ma)) if mode == 'max' else float(np.min(ma))

    stats = {}

    # ── Reward stats ──
    reward_vals = [r.get('reward') for r in rows if r.get('reward') is not None]
    stats['num_episodes'] = N
    stats['final10_return'] = col_last(10, 'reward')
    stats['final50_return'] = col_last(50, 'reward')
    stats['final100_return'] = col_last(100, 'reward')
    stats['best50_return'] = col_best(50, 'reward', 'max')
    stats['mean_return'] = safe_mean(reward_vals)
    stats['auc_return'] = safe_sum(reward_vals)
    stats['max_return'] = float(np.max(reward_vals)) if reward_vals else float('nan')

    # ── PPO diagnostics ──
    for f in PPO_FIELDS:
        if prefix:
            stats[f'mean_{f}'] = col_last(50, f) if N >= 10 else safe_mean(col(f))
        else:
            stats[f'mean_{f}'] = col_last(50, f) if N >= 10 else safe_mean(col(f))

    # ── LTS diagnostics (may be N/A for non-LTS agents) ──
    # Detect non-LTS agents: check if belief_norm or inter_memory_norm are present and non-zero
    # (RNN/NL agents have all LTS fields as 0.0 in CSV)
    is_lts = False
    if rows:
        # Check belief_norm: only LTS agents have non-zero belief_norm
        for check_field in ['a0_belief_norm', 'a0_inter_memory_filled']:
            ck = check_field if not prefix else f'{prefix}{check_field.replace("a0_", "")}'
            if ck in rows[0]:
                vals = [float(r.get(ck, 0) or 0) for r in rows[:10]]
                if any(v > 1e-6 for v in vals):
                    is_lts = True
                    break

    for f in LTS_FIELDS:
        if not is_lts:
            stats[f'final50_{f}'] = 'N/A'
            continue
        search_key = f'a0_{f}' if not prefix else f'{prefix}{f}'
        if rows and search_key not in rows[0]:
            search_key = f
        if rows and search_key in rows[0]:
            vals_last50 = [r.get(search_key) for r in rows[-50:]
                          if r.get(search_key) is not None]
            stats[f'final50_{f}'] = safe_mean(vals_last50) if vals_last50 else 'N/A'
        else:
            stats[f'final50_{f}'] = 'N/A'

    return stats


def format_val(v):
    """Format a value for markdown table."""
    if v == 'N/A' or v is None:
        return 'N/A'
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if np.isnan(v):
            return 'N/A'
        # Format based on magnitude
        if abs(v) < 0.001:
            return f'{v:.2e}'
        if abs(v) < 0.01:
            return f'{v:.5f}'
        if abs(v) < 1:
            return f'{v:.4f}'
        if abs(v) < 100:
            return f'{v:.2f}'
        return f'{v:.1f}'
    return str(v)


def print_markdown_table(all_stats, labels):
    """Print a markdown comparison table."""
    # Determine which columns to show
    key_fields = [
        'num_episodes', 'final10_return', 'final50_return', 'final100_return',
        'best50_return', 'mean_return', 'auc_return', 'max_return',
        'final50_loss_dy', 'final50_dy_acc', 'final50_loss_df_action',
        'final50_loss_dc', 'final50_belief_norm', 'final50_belief_delta_mean',
        'final50_ratio_max', 'final50_clip_fraction', 'final50_approx_kl_ppo',
        'final50_belief_encoder_grad_norm', 'final50_inter_memory_filled',
        'final50_inter_memory_norm',
    ]
    # Short names for table header
    short_names = {
        'num_episodes': 'Episodes',
        'final10_return': 'final10_r',
        'final50_return': 'final50_r',
        'final100_return': 'final100_r',
        'best50_return': 'best50_r',
        'mean_return': 'mean_r',
        'auc_return': 'auc_r',
        'max_return': 'max_r',
        'final50_loss_dy': 'loss_dy',
        'final50_dy_acc': 'dy_acc',
        'final50_loss_df_action': 'loss_df_act',
        'final50_loss_dc': 'loss_dc',
        'final50_belief_norm': 'bel_norm',
        'final50_belief_delta_mean': 'bel_delta',
        'final50_ratio_max': 'ratio_max',
        'final50_clip_fraction': 'clip_frac',
        'final50_approx_kl_ppo': 'approx_kl',
        'final50_belief_encoder_grad_norm': 'bel_grad',
        'final50_inter_memory_filled': 'inter_fill',
        'final50_inter_memory_norm': 'inter_norm',
    }

    # Build header
    header = '| Run | ' + ' | '.join(short_names[k] for k in key_fields) + ' |'
    sep = '|---' * (len(key_fields) + 1) + '|'

    print(header)
    print(sep)

    for label, stats in zip(labels, all_stats):
        vals = []
        for k in key_fields:
            v = stats.get(k, 'N/A')
            vals.append(format_val(v))
        row = '| ' + label + ' | ' + ' | '.join(vals) + ' |'
        print(row)


def plot_curves(rows, log_dir, label=''):
    """Generate diagnostic plots, saved to log_dir."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available, skipping plots")
        return

    N = len(rows)
    if N < 2:
        return

    episodes = np.arange(N)

    def has_col(name):
        vals = [r.get(name) for r in rows]
        return any(v is not None for v in vals)

    def get_col(name, prefix=''):
        key = f'{prefix}{name}' if prefix else name
        return np.array([r.get(key) if r.get(key) is not None else np.nan
                         for r in rows], dtype=np.float64)

    def smooth(y, w=20):
        """Simple moving average, handles NaN."""
        if len(y) < w:
            return y
        kernel = np.ones(w) / w
        valid = ~np.isnan(y)
        y_filled = y.copy()
        y_filled[~valid] = 0
        smoothed = np.convolve(y_filled, kernel, mode='same')
        # Edge correction
        for i in range(w):
            k = min(i + 1, w)
            smoothed[i] = np.nansum(y[max(0, i - k + 1):i + 1]) / k
            smoothed[-i - 1] = np.nansum(y[-i - 1:min(N, -i - 1 + k)]) / k
        return smoothed

    # 1. Return curve
    fig, ax = plt.subplots(figsize=(10, 4))
    reward = get_col('reward')
    ax.plot(episodes, reward, alpha=0.2, color='blue', linewidth=0.5)
    ax.plot(episodes, smooth(reward, 20), color='blue', linewidth=1.5, label='MA(20)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Return Curve {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, 'return_curve.png'), dpi=100)
    plt.close(fig)

    # 2. Aux losses curve (if LTS fields exist)
    if has_col('a0_loss_dy'):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            ls = '-'
            axes[0].plot(episodes, smooth(get_col('loss_dy', prefix), 20),
                        color=color, linestyle=ls, label=f'{prefix}D_y')
        axes[0].axhline(y=np.log(6), color='gray', linestyle='--', alpha=0.5, label='random CE')
        axes[0].set_ylabel('loss_dy')
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            axes[1].plot(episodes, smooth(get_col('loss_df_action', prefix), 20),
                        color=color, label=f'{prefix}D_f action')
        axes[1].set_ylabel('loss_df_action')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            axes[2].plot(episodes, smooth(get_col('loss_dc', prefix), 20),
                        color=color, label=f'{prefix}D_c')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('loss_dc')
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)
        fig.suptitle(f'Auxiliary Losses {label}')
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, 'aux_losses_curve.png'), dpi=100)
        plt.close(fig)

    # 3. D_y accuracy curve
    if has_col('a0_dy_acc'):
        fig, ax = plt.subplots(figsize=(10, 4))
        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            ax.plot(episodes, smooth(get_col('dy_acc', prefix), 20),
                   color=color, label=f'{prefix}')
        ax.axhline(y=1.0 / 6.0, color='gray', linestyle='--', alpha=0.5, label='random (16.7%)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('dy_acc')
        ax.set_title(f'D_y Accuracy {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, 'dy_acc_curve.png'), dpi=100)
        plt.close(fig)

    # 4. PPO diagnostics curve
    ppo_has = any(has_col(f'{prefix}{f}')
                  for f in ['ratio_max', 'clip_fraction', 'approx_kl_ppo']
                  for prefix in ['a0_', ''])
    if ppo_has:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            axes[0].plot(episodes, smooth(get_col('ratio_max', prefix), 20),
                        color=color, label=f'{prefix}')
        axes[0].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='warning')
        axes[0].set_ylabel('ratio_max')
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            axes[1].plot(episodes, smooth(get_col('clip_fraction', prefix), 20),
                        color=color, label=f'{prefix}')
        axes[1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='warning')
        axes[1].set_ylabel('clip_fraction')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            axes[2].plot(episodes, smooth(get_col('approx_kl_ppo', prefix), 20),
                        color=color, label=f'{prefix}')
        axes[2].axhline(y=0.02, color='red', linestyle='--', alpha=0.5, label='warning')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('approx_kl_ppo')
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)
        fig.suptitle(f'PPO Diagnostics {label}')
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, 'ppo_diagnostics_curve.png'), dpi=100)
        plt.close(fig)

    # 5. Belief curve
    if has_col('a0_belief_norm'):
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            axes[0].plot(episodes, smooth(get_col('belief_norm', prefix), 20),
                        color=color, label=f'{prefix}')
        axes[0].set_ylabel('belief_norm')
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        for prefix, color in [('a0_', 'blue'), ('a1_', 'orange')]:
            axes[1].plot(episodes, smooth(get_col('belief_delta_mean', prefix), 20),
                        color=color, label=f'{prefix}')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('belief_delta_mean')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
        fig.suptitle(f'Belief Diagnostics {label}')
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, 'belief_curve.png'), dpi=100)
        plt.close(fig)

    print(f"  Plots saved to {log_dir}/")


def summarize_one(log_dir):
    """Summarize a single experiment log directory."""
    csv_path = os.path.join(log_dir, 'episodes.csv')
    rows = load_csv(csv_path)
    if rows is None:
        print(f"ERROR: No episodes.csv found in {log_dir}")
        return None

    # Detect agent types from first row
    has_lts0 = any(k.startswith('a0_loss_dy') for k in rows[0].keys())
    has_lts1 = any(k.startswith('a1_loss_dy') for k in rows[0].keys())

    # Overall stats (no prefix)
    stats = compute_stats(rows)
    # Per-agent stats
    if has_lts0:
        a0_stats = compute_stats(rows, prefix='a0_')
        for k, v in a0_stats.items():
            stats[f'a0_{k}'] = v
    if has_lts1:
        a1_stats = compute_stats(rows, prefix='a1_')
        for k, v in a1_stats.items():
            stats[f'a1_{k}'] = v

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default=None, help='Single experiment log dir')
    parser.add_argument('--log_dirs', nargs='*', default=None, help='Multiple log dirs for comparison')
    parser.add_argument('--plot', action='store_true', default=True, help='Generate plots (default: True)')
    parser.add_argument('--no_plot', action='store_false', dest='plot', help='Skip plot generation')
    args = parser.parse_args()

    if args.log_dir:
        log_dirs = [args.log_dir]
        labels = [os.path.basename(args.log_dir.rstrip('/'))]
    elif args.log_dirs:
        log_dirs = args.log_dirs
        labels = [os.path.basename(d.rstrip('/')) for d in log_dirs]
    else:
        print("ERROR: specify --log_dir or --log_dirs")
        sys.exit(1)

    all_stats = []
    for log_dir, label in zip(log_dirs, labels):
        print(f"\n{'='*60}")
        print(f"Summarizing: {label}")
        print(f"  dir: {log_dir}")
        stats = summarize_one(log_dir)
        if stats is None:
            continue
        all_stats.append(stats)

        # Print per-experiment summary
        print(f"  Episodes: {stats.get('num_episodes', '?')}")
        print(f"  final10_return:  {format_val(stats.get('final10_return'))}")
        print(f"  final50_return:  {format_val(stats.get('final50_return'))}")
        print(f"  final100_return: {format_val(stats.get('final100_return'))}")
        print(f"  best50_return:   {format_val(stats.get('best50_return'))}")
        print(f"  mean_return:     {format_val(stats.get('mean_return'))}")
        print(f"  auc_return:      {format_val(stats.get('auc_return'))}")
        print(f"  max_return:      {format_val(stats.get('max_return'))}")
        print(f"  --- LTS diagnostics (agent 0, final50) ---")
        for f in LTS_FIELDS:
            v = stats.get(f'final50_{f}', 'N/A')
            if v != 'N/A':
                print(f"  {f}: {format_val(v)}")

        # Generate plots
        if args.plot:
            rows = load_csv(os.path.join(log_dir, 'episodes.csv'))
            if rows:
                plot_curves(rows, log_dir, label)

    # Comparison table (if multiple runs)
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print("Comparison Table:")
        print_markdown_table(all_stats, labels[:len(all_stats)])

    # Save summary JSON
    for log_dir, stats in zip(log_dirs, all_stats):
        summary_path = os.path.join(log_dir, 'summary_lts.json')
        # Convert numpy values
        clean_stats = {}
        for k, v in stats.items():
            if isinstance(v, (np.integer,)):
                clean_stats[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean_stats[k] = float(v)
            elif isinstance(v, str) or v is None:
                clean_stats[k] = v
            else:
                try:
                    clean_stats[k] = float(v)
                except (ValueError, TypeError):
                    clean_stats[k] = str(v)
        with open(summary_path, 'w') as f:
            json.dump(clean_stats, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
