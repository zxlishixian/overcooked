"""Plot learning curves from sweep experiments.

Usage:
    python overcooked_speed/analysis/plot_learning_curves.py \
        --log_dir logs/long_nl_nl --save_dir logs/figures
"""
import argparse
import os
import sys
import json
import numpy as np
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def load_episode_csv(csv_path):
    """Load episode CSV into dict of arrays."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v != '' else np.nan for k, v in row.items()})
    return {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}


def plot_reward_curves(log_dir, pair_name, seeds, save_dir):
    """Plot reward curves for one algorithm pair across seeds."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_rewards = []
    for seed in seeds:
        csv_path = os.path.join(log_dir, pair_name, f'seed_{seed}', 'episodes.csv')
        if os.path.exists(csv_path):
            data = load_episode_csv(csv_path)
            all_rewards.append(data['reward'])

    if not all_rewards:
        print(f"  No data for {pair_name}")
        return

    min_len = min(len(r) for r in all_rewards)
    rewards = np.array([r[:min_len] for r in all_rewards])

    mean_r = np.mean(rewards, axis=0)
    std_r = np.std(rewards, axis=0)

    window = max(5, len(mean_r) // 20)
    kernel = np.ones(window) / window
    mean_smooth = np.convolve(mean_r, kernel, mode='valid')

    plt.figure(figsize=(8, 5))
    x = np.arange(len(mean_smooth))
    plt.plot(x, mean_smooth, label=f'{pair_name}')
    plt.fill_between(x, mean_smooth - std_r[:len(mean_smooth)],
                     mean_smooth + std_r[:len(mean_smooth)],
                     alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Reward (smoothed)')
    plt.title(f'Learning Curve: {pair_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'reward_{pair_name}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_specialization_curves(log_dir, pair_name, seeds, save_dir):
    """Plot raw + gated specialization curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_spec_raw = []
    all_spec_gated = []
    for seed in seeds:
        csv_path = os.path.join(log_dir, pair_name, f'seed_{seed}', 'episodes.csv')
        if os.path.exists(csv_path):
            data = load_episode_csv(csv_path)
            all_spec_raw.append(data['s_overall'])
            all_spec_gated.append(data['s_overall_gated'])

    if not all_spec_raw:
        return

    min_len = min(len(s) for s in all_spec_raw)
    spec_raw = np.array([s[:min_len] for s in all_spec_raw])
    spec_gated = np.array([s[:min_len] for s in all_spec_gated])

    mean_raw = np.mean(spec_raw, axis=0)
    mean_gated = np.nanmean(spec_gated, axis=0)

    window = max(5, len(mean_raw) // 20)
    kernel = np.ones(window) / window
    raw_smooth = np.convolve(mean_raw, kernel, mode='valid')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(np.arange(len(raw_smooth)), raw_smooth, label=f'{pair_name}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Overall Specialization (raw)')
    ax1.set_title(f'Raw Specialization: {pair_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    valid_mask = ~np.isnan(mean_gated)
    if valid_mask.sum() > 0:
        ax2.plot(np.arange(len(mean_gated))[valid_mask],
                 mean_gated[valid_mask], '.', markersize=2, label=f'{pair_name}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Overall Specialization (gated)')
    ax2.set_title(f'Gated Specialization: {pair_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'spec_{pair_name}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_action_distribution(log_dir, pair_name, seeds, save_dir):
    """Plot action distribution over training."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    action_names = ['up', 'down', 'right', 'left', 'stay', 'interact']

    all_a0 = {a: [] for a in action_names}
    all_a1 = {a: [] for a in action_names}

    for seed in seeds:
        csv_path = os.path.join(log_dir, pair_name, f'seed_{seed}', 'episodes.csv')
        if not os.path.exists(csv_path):
            continue
        data = load_episode_csv(csv_path)
        for a in action_names:
            all_a0[a].append(data[f'a0_action_{a}'])
            all_a1[a].append(data[f'a1_action_{a}'])

    if not all_a0['up']:
        return

    min_len = min(len(arr) for arr in all_a0['up'])
    a0_frac = {}
    a1_frac = {}
    for a in action_names:
        stacked = np.array([arr[:min_len] for arr in all_a0[a]])
        a0_frac[a] = np.mean(stacked, axis=0) / 400  # fraction of steps
        stacked = np.array([arr[:min_len] for arr in all_a1[a]])
        a1_frac[a] = np.mean(stacked, axis=0) / 400

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    x = np.arange(min_len)

    for i, a in enumerate(action_names):
        ax1.plot(x, a0_frac[a], color=colors[i], label=a, alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Fraction of actions')
    ax1.set_title(f'Agent 0 Action Distribution')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    for i, a in enumerate(action_names):
        ax2.plot(x, a1_frac[a], color=colors[i], label=a, alpha=0.8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Fraction of actions')
    ax2.set_title(f'Agent 1 Action Distribution')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Action Distribution: {pair_name}', fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'actions_{pair_name}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def plot_learning_metrics(log_dir, pair_name, seeds, save_dir):
    """Plot loss, entropy, and gradient norm curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metric_keys = [
        ('a0_loss', 'a1_loss', 'Policy Loss'),
        ('a0_entropy', 'a1_entropy', 'Entropy'),
        ('a0_grad_norm', 'a1_grad_norm', 'Gradient Norm'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (k0, k1, title) in enumerate(metric_keys):
        ax = axes[idx]
        all_a0 = []
        all_a1 = []
        for seed in seeds:
            csv_path = os.path.join(log_dir, pair_name, f'seed_{seed}', 'episodes.csv')
            if not os.path.exists(csv_path):
                continue
            data = load_episode_csv(csv_path)
            all_a0.append(data[k0])
            all_a1.append(data[k1])

        if not all_a0:
            continue

        min_len = min(len(arr) for arr in all_a0)
        a0 = np.array([arr[:min_len] for arr in all_a0])
        a1 = np.array([arr[:min_len] for arr in all_a1])

        window = max(3, min_len // 20)
        kernel = np.ones(window) / window
        ax.plot(np.arange(min_len - window + 1),
                np.convolve(np.mean(a0, axis=0), kernel, mode='valid'),
                label='Agent 0')
        ax.plot(np.arange(min_len - window + 1),
                np.convolve(np.mean(a1, axis=0), kernel, mode='valid'),
                label='Agent 1')
        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Learning Metrics: {pair_name}', fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'metrics_{pair_name}.png')
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='logs/long_nl_nl')
    parser.add_argument('--save_dir', default='logs/figures')
    parser.add_argument('--pairs', default=None,
                        help='Comma-separated pair names, e.g. nl_nl')
    args = parser.parse_args()

    if args.pairs is None:
        pair_dirs = [d for d in os.listdir(args.log_dir)
                     if os.path.isdir(os.path.join(args.log_dir, d))]
    else:
        pair_dirs = [args.pairs.replace(',', '_')]

    for pair_name in pair_dirs:
        pair_path = os.path.join(args.log_dir, pair_name)
        if not os.path.isdir(pair_path):
            continue
        seeds = [d.split('_')[1] for d in os.listdir(pair_path)
                 if d.startswith('seed_')]
        seeds = sorted([int(s) for s in seeds])

        plot_reward_curves(args.log_dir, pair_name, seeds, args.save_dir)
        plot_specialization_curves(args.log_dir, pair_name, seeds, args.save_dir)
        plot_action_distribution(args.log_dir, pair_name, seeds, args.save_dir)
        plot_learning_metrics(args.log_dir, pair_name, seeds, args.save_dir)


if __name__ == '__main__':
    main()
