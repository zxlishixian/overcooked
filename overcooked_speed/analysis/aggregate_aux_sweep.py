"""Aggregate auxiliary task actor-loss sweep results."""
import csv
import glob
import json
import os
import sys

import numpy as np

SWEEP_DIR = 'logs/aux_task_sweep'

FIELDS = [
    'final_reward', 'reward_auc', 'T_reward',
    'final_specialization', 'T_specialization',
    'mean_total_potting', 'mean_total_soup_pickup',
    'mean_total_soup_delivery',
]


def load_summaries(sweep_dir):
    summaries = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, '*', 'seed_*', 'summary.json'))):
        with open(path) as f:
            item = json.load(f)
        item['_path'] = path
        summaries.append(item)
    return summaries


def finite_or_nan(value):
    if value is None:
        return float('nan')
    return float(value)


def aggregate(summaries):
    groups = {}
    for item in summaries:
        key = (item.get('aux_task_type', '?'), float(item.get('aux_coef', 0.0)))
        groups.setdefault(key, []).append(item)

    rows = []
    for (aux_type, coef), group in sorted(groups.items()):
        row = {
            'aux_task_type': aux_type,
            'aux_coef': coef,
            'n_seeds': len(group),
        }
        for field in FIELDS:
            vals = [finite_or_nan(g.get(field)) for g in group]
            row[f'{field}_mean'] = float(np.nanmean(vals))
            row[f'{field}_std'] = float(np.nanstd(vals))

        aux_scores = [
            0.5 * (finite_or_nan(g.get('mean_aux_score0')) +
                   finite_or_nan(g.get('mean_aux_score1')))
            for g in group
        ]
        aux_advs = [
            0.5 * (finite_or_nan(g.get('mean_aux_adv0')) +
                   finite_or_nan(g.get('mean_aux_adv1')))
            for g in group
        ]
        aux_losses = [
            0.5 * (finite_or_nan(g.get('mean_aux_loss0')) +
                   finite_or_nan(g.get('mean_aux_loss1')))
            for g in group
        ]
        row['mean_aux_score_mean'] = float(np.nanmean(aux_scores))
        row['mean_aux_score_std'] = float(np.nanstd(aux_scores))
        row['mean_aux_adv_mean'] = float(np.nanmean(aux_advs))
        row['mean_aux_adv_std'] = float(np.nanstd(aux_advs))
        row['mean_aux_loss_mean'] = float(np.nanmean(aux_losses))
        row['mean_aux_loss_std'] = float(np.nanstd(aux_losses))
        rows.append(row)
    return rows


def print_table(rows):
    print('\nAux Task Actor-Loss Sweep Results (300ep x 3 seeds)')
    print('=' * 132)
    header = (
        f"{'aux_task_type':16s} {'coef':>7s} {'final_r':>11s} {'r_auc':>11s} "
        f"{'T_rew':>9s} {'final_spec':>12s} {'T_spec':>9s} "
        f"{'aux_score':>11s} {'aux_adv':>10s} {'aux_loss':>10s} "
        f"{'pot':>8s} {'soup_pk':>8s} {'soup_del':>9s}"
    )
    print(header)
    print('-' * 132)
    for r in sorted(rows, key=lambda x: -x['final_reward_mean']):
        print(
            f"{r['aux_task_type']:16s} {r['aux_coef']:7.3f} "
            f"{r['final_reward_mean']:6.1f}+/-{r['final_reward_std']:<4.1f} "
            f"{r['reward_auc_mean']:7.0f}+/-{r['reward_auc_std']:<5.0f} "
            f"{r['T_reward_mean']:6.0f}+/-{r['T_reward_std']:<3.0f} "
            f"{r['final_specialization_mean']:7.3f}+/-{r['final_specialization_std']:<5.3f} "
            f"{r['T_specialization_mean']:6.0f}+/-{r['T_specialization_std']:<3.0f} "
            f"{r['mean_aux_score_mean']:10.3f} "
            f"{r['mean_aux_adv_mean']:9.3f} "
            f"{r['mean_aux_loss_mean']:9.3f} "
            f"{r['mean_total_potting_mean']:8.2f} "
            f"{r['mean_total_soup_pickup_mean']:8.2f} "
            f"{r['mean_total_soup_delivery_mean']:9.2f}"
        )
    print('\nReference: IPPO-ego final_reward=29.6+/-6.8 auc=8699 T_rew=300 T_spec=117')
    print('Reference: best task reward-shaping final_reward=25.4 T_spec=34')
    print('Reference: raw_clipped role_shaping final_reward=45.3 auc=10096 T_spec=78')
    print('Reference: MAPPO-ego final_reward=80.0+/-21.4 auc=18068 T_rew=246 T_spec=50')


def write_csv(rows, output_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} rows to {output_path}')


if __name__ == '__main__':
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else SWEEP_DIR
    summaries = load_summaries(sweep_dir)
    print(f'Loaded {len(summaries)} summaries from {sweep_dir}')
    if not summaries:
        raise SystemExit(1)
    rows = aggregate(summaries)
    print_table(rows)
    write_csv(rows, os.path.join(sweep_dir, 'aux_sweep_summary.csv'))
