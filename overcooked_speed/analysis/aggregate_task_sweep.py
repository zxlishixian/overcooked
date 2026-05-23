"""Aggregate task shaping sweep results into a summary table."""
import json
import os
import sys
import glob
import numpy as np
import csv

SWEEP_DIR = 'logs/task_shaping_sweep'

def load_all_summaries(sweep_dir):
    """Load all summary.json files from sweep directory."""
    summaries = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, '*/*/summary.json'))):
        # Path: logs/task_shaping_sweep/{type}_l{lam}_c{clip}/seed_{seed}/summary.json
        with open(path) as f:
            d = json.load(f)
        # Extract config from directory names
        parts = path.split('/')
        config_dir = parts[-2]  # {type}_l{lam}_c{clip}
        seed_dir = parts[-3]    # seed_{seed}
        d['_config'] = config_dir
        d['_seed_dir'] = seed_dir
        summaries.append(d)
    return summaries


def aggregate(summaries):
    """Group by (shaping_type, lambda_role, bonus_clip) and compute stats."""
    groups = {}
    for s in summaries:
        st = s.get('shaping_type', '?')
        lam = s.get('lambda_role', 0)
        clip = s.get('bonus_clip', 0)
        key = (st, lam, clip)
        if key not in groups:
            groups[key] = []
        groups[key].append(s)

    rows = []
    for (st, lam, clip), group in sorted(groups.items()):
        n = len(group)
        metrics = {}
        for field in ['final_reward', 'reward_auc', 'final_specialization',
                       'mean_shaping_applied', 'shaping_clip_rate',
                       'mean_total_potting', 'mean_total_soup_pickup',
                       'mean_total_soup_delivery', 'mean_total_task_events']:
            vals = [s.get(field, float('nan')) for s in group]
            # Handle None
            vals = [v if v is not None else float('nan') for v in vals]
            metrics[f'{field}_mean'] = np.nanmean(vals)
            metrics[f'{field}_std'] = np.nanstd(vals)

        # T_reward: handle None (never converged) as 300 (max episodes)
        t_vals = []
        for s in group:
            t = s.get('T_reward')
            if t is None or (isinstance(t, float) and np.isnan(t)):
                t_vals.append(300)
            else:
                t_vals.append(float(t))
        metrics['T_reward_mean'] = np.mean(t_vals)
        metrics['T_reward_std'] = np.std(t_vals)

        # T_specialization
        ts_vals = []
        for s in group:
            t = s.get('T_specialization')
            if t is None or (isinstance(t, float) and np.isnan(t)):
                ts_vals.append(300)
            else:
                ts_vals.append(float(t))
        metrics['T_specialization_mean'] = np.mean(ts_vals)
        metrics['T_specialization_std'] = np.std(ts_vals)

        rows.append({
            'shaping_type': st,
            'lambda_role': lam,
            'bonus_clip': clip,
            'n_seeds': n,
            **metrics,
        })
    return rows


def print_table(rows):
    """Print formatted results table."""
    print(f"\n{'='*140}")
    print("Task-Level Shaping Sweep Results (300ep × 3 seeds)")
    print(f"{'='*140}")
    header = (f"{'shaping_type':30s} {'lambda':>6s} {'clip':>5s} "
              f"{'final_r':>8s} {'r_auc':>8s} {'T_rew':>7s} {'T_spec':>7s} "
              f"{'final_spec':>9s} {'mean_pot':>8s} {'mean_del':>8s} "
              f"{'mean_shaping':>12s} {'clip_rate':>9s}")
    print(header)
    print("-" * 140)

    for r in sorted(rows, key=lambda x: -x['final_reward_mean']):
        print(f"{r['shaping_type']:30s} {r['lambda_role']:6.3f} {r['bonus_clip']:5.1f} "
              f"{r['final_reward_mean']:7.1f}±{r['final_reward_std']:.1f} "
              f"{r['reward_auc_mean']:7.0f} "
              f"{r['T_reward_mean']:6.0f} "
              f"{r['T_specialization_mean']:6.0f} "
              f"{r['final_specialization_mean']:8.3f} "
              f"{r['mean_total_potting_mean']:7.1f} "
              f"{r['mean_total_soup_delivery_mean']:7.1f} "
              f"{r['mean_shaping_applied_mean']:11.3f} "
              f"{r['shaping_clip_rate_mean']:8.2f}")

    # Print baselines for reference
    print(f"\n{'─'*140}")
    print("Reference baselines (500ep × 5 seeds):")
    print(f"  {'IPPO-ego':30s}          final_reward=29.6±6.8  auc=8699  T_rew=300  T_spec=117")
    print(f"  {'raw_clipped_role':30s}          final_reward=45.3±18.9 auc=10096 T_rew=365  T_spec=78")
    print(f"  {'MAPPO-ego':30s}          final_reward=80.0±21.4 auc=18068 T_rew=246  T_spec=50")


def write_csv(rows, output_path):
    """Write results to CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == '__main__':
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else SWEEP_DIR
    summaries = load_all_summaries(sweep_dir)
    print(f"Loaded {len(summaries)} summaries from {sweep_dir}")

    if not summaries:
        print("No results found!")
        sys.exit(1)

    rows = aggregate(summaries)
    print_table(rows)

    output_csv = os.path.join(sweep_dir, 'task_sweep_summary.csv')
    write_csv(rows, output_csv)
