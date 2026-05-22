"""Aggregate baseline sweep results and generate comparison plots.

Usage:
    python overcooked_speed/analysis/baseline_results.py \
        --sweep_dir logs/baseline_sweep --save_dir logs/figures
"""
import argparse
import json
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_summary(sweep_dir, config_name):
    """Load summary_all.json from a sweep subdirectory."""
    path = os.path.join(sweep_dir, config_name, 'summary_all.json')
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def load_all_seed_episodes(sweep_dir, config_name):
    """Load per-seed episode CSVs, return concatenated DataFrame-friendly dict."""
    base = os.path.join(sweep_dir, config_name, 'nl_nl')
    if not os.path.exists(base):
        return None
    all_rows = []
    for seed_dir in sorted(os.listdir(base)):
        csv_path = os.path.join(base, seed_dir, 'episodes.csv')
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['seed'] = seed_dir
                    all_rows.append(row)
    return all_rows


def compute_episode_stats(rows, key, window=50):
    """Compute mean and std of `key` across seeds, per episode.

    Returns:
        episodes: np.array of episode numbers
        mean: np.array of mean values
        std: np.array of std values
    """
    seeds = sorted(set(r['seed'] for r in rows))
    max_ep = max(int(r['episode']) for r in rows) + 1
    curves = {}
    for seed in seeds:
        seed_rows = [r for r in rows if r['seed'] == seed]
        seed_rows.sort(key=lambda r: int(r['episode']))
        values = np.array([float(r[key]) for r in seed_rows])
        curves[seed] = values

    min_len = min(len(v) for v in curves.values())
    episodes = np.arange(min_len)

    stacked = np.array([v[:min_len] for v in curves.values()])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return episodes, mean, std


def print_results_table(summaries):
    """Print a formatted results table."""
    keys = [
        'final_reward_mean', 'final_reward_std',
        'reward_auc_mean', 'reward_auc_std',
        'T_reward_mean', 'T_reward_std',
        'final_specialization_mean', 'final_specialization_std',
        'final_specialization_gated_mean', 'final_specialization_gated_std',
        'T_specialization_mean', 'T_specialization_std',
        'T_reward_success_rate', 'T_specialization_success_rate',
    ]

    headers = ['Metric'] + list(summaries.keys())
    print("\n" + "=" * 120)
    print("BASELINE SANITY SWEEP RESULTS")
    print("=" * 120)

    metrics_map = {
        'final_reward_mean': 'Final Reward (last 50)',
        'final_reward_std': '  ± std',
        'reward_auc_mean': 'Reward AUC',
        'reward_auc_std': '  ± std',
        'T_reward_mean': 'T_reward (convergence ep)',
        'T_reward_std': '  ± std',
        'final_specialization_mean': 'Final Specialization (raw)',
        'final_specialization_std': '  ± std',
        'final_specialization_gated_mean': 'Final Specialization (gated)',
        'final_specialization_gated_std': '  ± std',
        'T_specialization_mean': 'T_specialization',
        'T_specialization_std': '  ± std',
        'T_reward_success_rate': 'T_reward success rate',
        'T_specialization_success_rate': 'T_spec success rate',
    }

    rows_data = {}
    for metric_key in keys:
        human_name = metrics_map.get(metric_key, metric_key)
        row = [human_name]
        for config_name, s in summaries.items():
            if s is None:
                row.append('N/A')
            else:
                v = s.get(metric_key, None)
                if v is None:
                    row.append('N/A')
                elif isinstance(v, float):
                    row.append(f'{v:.4f}')
                else:
                    row.append(str(v))
        rows_data[metric_key] = row

    # Print table
    col_widths = [max(30, max(len(str(rows_data[k][i])) for k in rows_data))
                  for i in range(len(headers))]
    fmt = '  '.join(f'{{:<{w}}}' for w in col_widths)
    print(fmt.format(*headers))
    print('-' * sum(col_widths))
    for metric_key in keys:
        print(fmt.format(*rows_data[metric_key]))
    print("=" * 120)


def plot_learning_curves(sweep_dir, summaries, save_dir):
    """Plot reward and specialization learning curves for all 4 configs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    configs = list(summaries.keys())
    colors = ['#2196F3', '#FF5722', '#90CAF9', '#FFAB91']  # dark, dark, light, light
    linestyles = ['-', '-', '--', '--']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Reward curve ──
    ax = axes[0]
    for idx, config_name in enumerate(configs):
        rows = load_all_seed_episodes(sweep_dir, config_name)
        if rows is None:
            continue
        eps, mean, std = compute_episode_stats(rows, 'reward')
        window = 10
        if len(mean) > window:
            smoothed = np.convolve(mean, np.ones(window)/window, mode='valid')
            eps_smoothed = eps[window-1:]
            smoothed_std = np.convolve(std, np.ones(window)/window, mode='valid')
        else:
            smoothed = mean
            eps_smoothed = eps
            smoothed_std = std
        label = config_name.replace('_', ' ')
        ax.plot(eps_smoothed, smoothed, color=colors[idx], linestyle=linestyles[idx],
                linewidth=2, label=label)
        ax.fill_between(eps_smoothed,
                        smoothed - smoothed_std/np.sqrt(5),
                        smoothed + smoothed_std/np.sqrt(5),
                        color=colors[idx], alpha=0.12)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (MA-10, ±SE)')
    ax.set_title('Reward Learning Curves (5 seeds)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Specialization curve using s_overall (raw, always valid) ──
    ax = axes[1]
    for idx, config_name in enumerate(configs):
        rows = load_all_seed_episodes(sweep_dir, config_name)
        if rows is None:
            continue
        eps, mean, std = compute_episode_stats(rows, 's_overall')
        window = 10
        if len(mean) > window:
            smoothed = np.convolve(mean, np.ones(window)/window, mode='valid')
            eps_smoothed = eps[window-1:]
            smoothed_std = np.convolve(std, np.ones(window)/window, mode='valid')
        else:
            smoothed = mean
            eps_smoothed = eps
            smoothed_std = std

        label = config_name.replace('_', ' ')
        ax.plot(eps_smoothed, smoothed, color=colors[idx], linestyle=linestyles[idx],
                linewidth=2, label=label)
        ax.fill_between(eps_smoothed,
                        np.maximum(0, smoothed - smoothed_std/np.sqrt(5)),
                        np.minimum(1, smoothed + smoothed_std/np.sqrt(5)),
                        color=colors[idx], alpha=0.12)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Specialization (gated)')
    ax.set_title('Specialization Curves (5 seeds)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('IPPO vs MAPPO — Baseline Sanity Sweep', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'baseline_learning_curves.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def plot_bar_comparison(summaries, save_dir):
    """Bar plot comparing key metrics across 4 configs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    configs = list(summaries.keys())
    config_labels = [c.replace('_', '\n') for c in configs]
    colors = ['#2196F3', '#FF5722', '#90CAF9', '#FFAB91']

    metric_pairs = [
        ('final_reward_mean', 'final_reward_std', 'Final Reward'),
        ('T_reward_mean', 'T_reward_std', 'T_reward (convergence ep)'),
        ('final_specialization_mean', 'final_specialization_std', 'Final Specialization'),
        ('reward_auc_mean', 'reward_auc_std', 'Reward AUC'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for idx, (mean_key, std_key, title) in enumerate(metric_pairs):
        ax = axes[idx // 2, idx % 2]
        means = [summaries[c].get(mean_key, 0) or 0 for c in configs]
        stds = [summaries[c].get(std_key, 0) or 0 for c in configs]
        x = np.arange(len(configs))
        bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, edgecolor='white',
                      linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Baseline Comparison: IPPO vs MAPPO × egocentric vs global_concat',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'baseline_comparison_bars.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', default='logs/baseline_sweep')
    parser.add_argument('--save_dir', default='logs/figures')
    args = parser.parse_args()

    configs = {
        'ippo_egocentric': None,
        'mappo_egocentric': None,
        'ippo_global_concat': None,
        'mappo_global_concat': None,
    }

    for config_name in configs:
        configs[config_name] = load_summary(args.sweep_dir, config_name)

    print_results_table(configs)
    plot_learning_curves(args.sweep_dir, configs, args.save_dir)
    plot_bar_comparison(configs, args.save_dir)


if __name__ == '__main__':
    main()
