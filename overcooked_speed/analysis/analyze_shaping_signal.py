"""Analyze correlation between shaping signals and reward.

Reads episode CSVs from one or more experiment log directories and computes
Pearson (and Spearman if scipy available) correlation between shaping/task
signals and episode reward.

Usage:
    python overcooked_speed/analysis/analyze_shaping_signal.py \
        --log_dirs logs/ablation/normalized_l003 \
        --output_csv logs/correlation_summary.csv
"""
import argparse
import csv
import os
import sys
import glob
import numpy as np

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_episodes(log_dir):
    """Load all episode CSVs from a log directory (recursively)."""
    rows = []
    run_id = os.path.basename(log_dir.rstrip('/'))

    for csv_path in sorted(glob.glob(os.path.join(log_dir, '**', 'episodes.csv'),
                                      recursive=True)):
        seed_dir = os.path.basename(os.path.dirname(csv_path))
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['_run_id'] = run_id
                row['_seed'] = seed_dir
                row['_csv_path'] = csv_path
                rows.append(row)
    return rows


def safe_float(val, default=np.nan):
    if val is None or val == '' or val == '?':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def compute_derived_signals(rows):
    """Compute derived aggregate signals from episode data."""
    n = len(rows)
    signals = {}

    # Direct CSV fields
    direct_fields = [
        'reward',
        'a0_role_bonus_raw', 'a1_role_bonus_raw',
        'a0_role_bonus_applied', 'a1_role_bonus_applied',
        'agent0_shaping_raw', 'agent1_shaping_raw',
        'agent0_shaping_applied', 'agent1_shaping_applied',
        'mean_shaping_applied',
        'complementarity_current', 'complementarity_delta',
        's_overall', 's_delivery', 's_cooking',
        'total_task_events', 'total_potting',
        'total_soup_pickup', 'total_soup_delivery',
    ]

    for field in direct_fields:
        vals = []
        for r in rows:
            v = safe_float(r.get(field, np.nan))
            vals.append(v)
        arr = np.array(vals, dtype=np.float64)
        if not np.all(np.isnan(arr)):
            signals[field] = arr

    # Derived: mean raw role bonus
    if ('a0_role_bonus_raw' in signals and 'a1_role_bonus_raw' in signals):
        signals['mean_role_bonus_raw'] = np.nanmean(
            np.stack([signals['a0_role_bonus_raw'],
                      signals['a1_role_bonus_raw']]), axis=0)
    elif ('agent0_shaping_raw' in signals and 'agent1_shaping_raw' in signals):
        signals['mean_shaping_raw'] = np.nanmean(
            np.stack([signals['agent0_shaping_raw'],
                      signals['agent1_shaping_raw']]), axis=0)

    # Derived: mean applied role bonus
    if ('a0_role_bonus_applied' in signals and 'a1_role_bonus_applied' in signals):
        signals['mean_role_bonus_applied'] = np.nanmean(
            np.stack([signals['a0_role_bonus_applied'],
                      signals['a1_role_bonus_applied']]), axis=0)
    elif ('agent0_shaping_applied' in signals and 'agent1_shaping_applied' in signals):
        signals['mean_shaping_applied'] = np.nanmean(
            np.stack([signals['agent0_shaping_applied'],
                      signals['agent1_shaping_applied']]), axis=0)

    # Derived: event aggregates (from per-agent fields)
    event_pairs = [
        ('total_cooking',
         ['a0_pickup_onion', 'a1_pickup_onion',
          'a0_place_onion_in_pot', 'a1_place_onion_in_pot']),
        ('total_delivery',
         ['a0_pickup_dish', 'a1_pickup_dish',
          'a0_pickup_soup', 'a1_pickup_soup',
          'a0_deliver_soup', 'a1_deliver_soup']),
    ]
    for name, fields in event_pairs:
        vals = np.zeros(n, dtype=np.float64)
        for f in fields:
            col = [safe_float(r.get(f, 0), 0) for r in rows]
            vals += np.array(col, dtype=np.float64)
        signals[name] = vals

    # total_task_events from events
    if 'total_task_events' not in signals and 'total_cooking' in signals:
        signals['total_task_events'] = (signals['total_cooking'] +
                                         signals['total_delivery'])

    # total_potting from events
    if 'total_potting' not in signals:
        vals = np.zeros(n, dtype=np.float64)
        for f in ['a0_place_onion_in_pot', 'a1_place_onion_in_pot']:
            col = [safe_float(r.get(f, 0), 0) for r in rows]
            vals += np.array(col, dtype=np.float64)
        signals['total_potting'] = vals

    # total_soup_delivery from events
    if 'total_soup_delivery' not in signals:
        vals = np.zeros(n, dtype=np.float64)
        for f in ['a0_deliver_soup', 'a1_deliver_soup']:
            col = [safe_float(r.get(f, 0), 0) for r in rows]
            vals += np.array(col, dtype=np.float64)
        signals['total_soup_delivery'] = vals

    # total_soup_pickup from events
    if 'total_soup_pickup' not in signals:
        vals = np.zeros(n, dtype=np.float64)
        for f in ['a0_pickup_soup', 'a1_pickup_soup']:
            col = [safe_float(r.get(f, 0), 0) for r in rows]
            vals += np.array(col, dtype=np.float64)
        signals['total_soup_pickup'] = vals

    return signals


def compute_correlations(signals, signal_names, target_name='reward'):
    """Compute Pearson (and Spearman) correlation for each signal vs target."""
    results = []
    target = signals.get(target_name)
    if target is None or len(target) < 3:
        return results

    for name in signal_names:
        sig = signals.get(name)
        if sig is None:
            continue

        # Remove NaN pairs
        mask = ~(np.isnan(sig) | np.isnan(target))
        x = sig[mask]
        y = target[mask]
        if len(x) < 3:
            continue

        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = None
        if HAS_SCIPY:
            spearman, _ = spearmanr(x, y)
            spearman = float(spearman)

        results.append({
            'signal': name,
            'target': target_name,
            'pearson': pearson,
            'spearman': spearman if spearman is not None else '',
            'n': len(x),
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze correlation between shaping signals and reward')
    parser.add_argument('--log_dirs', nargs='+', required=True,
                        help='One or more log directories containing episode CSVs')
    parser.add_argument('--output_csv', help='Path to write correlation_summary.csv')
    args = parser.parse_args()

    if not HAS_SCIPY:
        print("Note: scipy not available — reporting Pearson only.\n")

    all_signal_names = [
        'mean_role_bonus_raw', 'mean_role_bonus_applied',
        'mean_shaping_raw', 'mean_shaping_applied',
        'complementarity_current', 'complementarity_delta',
        's_overall', 's_delivery', 's_cooking',
        'total_cooking', 'total_delivery', 'total_task_events',
        'total_potting', 'total_soup_pickup', 'total_soup_delivery',
    ]

    # Also try reward_t (current) and reward_{t+1} (next episode)
    all_results = []

    for log_dir in args.log_dirs:
        if not os.path.isdir(log_dir):
            print(f"WARNING: {log_dir} not found, skipping")
            continue

        rows = load_episodes(log_dir)
        if not rows:
            print(f"WARNING: no episode CSV found in {log_dir}")
            continue

        signals = compute_derived_signals(rows)
        run_id = os.path.basename(log_dir.rstrip('/'))

        # Determine which signals are present
        available = [s for s in all_signal_names if s in signals]
        if not available:
            print(f"WARNING: no known signals in {log_dir}")
            continue

        # 1. corr(signal_t, reward_t)
        results = compute_correlations(signals, available, 'reward')
        for r in results:
            r['log_dir'] = run_id
            r['target_type'] = 'same_episode'
        all_results.extend(results)

        # 2. corr(signal_t, reward_{t+1}) — next episode
        if 'reward' in signals:
            reward = signals['reward']
            reward_next = np.roll(reward, -1)
            reward_next[-1] = np.nan  # last episode has no next
            signals_next = dict(signals)
            signals_next['reward'] = reward_next
            results = compute_correlations(signals_next, available, 'reward')
            for r in results:
                r['log_dir'] = run_id
                r['target_type'] = 'next_episode'
            all_results.extend(results)

        # 3. corr(signal_t, moving_avg_reward_{t:t+10})
        if 'reward' in signals and len(signals['reward']) >= 10:
            reward_ma = np.convolve(signals['reward'],
                                     np.ones(10)/10, mode='same')
            # Edge: use shorter windows
            for i in range(min(5, len(reward_ma))):
                reward_ma[i] = np.mean(signals['reward'][:i+1])
            for i in range(len(reward_ma)-5, len(reward_ma)):
                reward_ma[i] = np.mean(signals['reward'][i:])
            signals_ma = dict(signals)
            signals_ma['reward'] = reward_ma
            results = compute_correlations(signals_ma, available, 'reward')
            for r in results:
                r['log_dir'] = run_id
                r['target_type'] = 'moving_avg_10'
            all_results.extend(results)

    if not all_results:
        print("No correlation results computed.")
        return

    # ── Print summary ──
    same_ep = [r for r in all_results if r['target_type'] == 'same_episode'
               and r['pearson'] is not None]
    if same_ep:
        sorted_pos = sorted(same_ep, key=lambda r: -abs(r['pearson']))

        print("=" * 70)
        print("Same-episode correlations with reward (across all runs)")
        print("=" * 70)
        for r in sorted_pos[:20]:
            sp = f"  spearman={r['spearman']:.3f}" if r['spearman'] != '' else ''
            print(f"  {r['log_dir']:30s} {r['signal']:30s} "
                  f"pearson={r['pearson']:+.3f}{sp}  (n={r['n']})")

        # Aggregate by signal across runs
        print("\n--- Aggregated by signal (mean |pearson| across runs) ---")
        from collections import defaultdict
        by_signal = defaultdict(list)
        for r in same_ep:
            by_signal[r['signal']].append(r['pearson'])

        for name, vals in sorted(by_signal.items(),
                                  key=lambda kv: -abs(np.mean(kv[1]))):
            print(f"  {name:35s}  mean_pearson={np.mean(vals):+.3f}  "
                  f"mean_abs={np.mean(np.abs(vals)):.3f}  (n_runs={len(vals)})")

    # ── Write CSV ──
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)
        fieldnames = ['log_dir', 'signal', 'target', 'target_type',
                      'pearson', 'spearman', 'n']
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    'log_dir': r['log_dir'],
                    'signal': r['signal'],
                    'target': r['target'],
                    'target_type': r['target_type'],
                    'pearson': f"{r['pearson']:.6f}" if r['pearson'] is not None else '',
                    'spearman': f"{r['spearman']:.6f}" if r.get('spearman') and r['spearman'] != '' else '',
                    'n': r['n'],
                })
        print(f"\nWrote {len(all_results)} correlation rows to {args.output_csv}")


if __name__ == '__main__':
    main()
