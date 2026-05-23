"""Multi-seed sweep of algorithm pairs.

Usage:
    python overcooked_speed/experiments/sweep_pairs.py \
        --layout cramped_room --pairs nl,nl \
        --num_episodes 50 --seeds 0 1 --log_dir logs/sweep_smoke
"""
import argparse
import json
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from overcooked_speed.experiments.run_pair import run_pair


def compute_summary_stats(summaries):
    """Aggregate multiple seed summaries for one pair."""
    keys = ['final_reward', 'reward_auc', 'T_reward', 'T_specialization',
            'final_specialization', 'final_sdelivery', 'final_scooking',
            'final_specialization_gated', 'final_sdelivery_gated',
            'final_scooking_gated',
            'T_reward_ok', 'T_specialization_ok',
            'T_specialization_gated_ok',
            'mean_shaping_applied', 'shaping_clip_rate',
            'mean_total_task_events', 'mean_total_soup_delivery',
            'mean_total_potting', 'mean_total_soup_pickup']
    agg = {}
    for k in keys:
        if k in summaries[0]:
            vals = [s[k] for s in summaries]
            if isinstance(summaries[0][k], bool):
                agg[k.replace('_ok', '_success_rate')] = np.mean([1 if v else 0 for v in vals])
            elif summaries[0][k] is None or (isinstance(summaries[0][k], float) and np.isnan(summaries[0][k])):
                # Handle None/NaN — count valid
                valid = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if valid:
                    agg[f'{k}_mean'] = np.mean(valid)
                    agg[f'{k}_std'] = np.std(valid)
                    agg[f'{k}_valid_count'] = len(valid)
                else:
                    agg[f'{k}_mean'] = None
                    agg[f'{k}_std'] = None
                    agg[f'{k}_valid_count'] = 0
            else:
                agg[f'{k}_mean'] = np.mean(vals)
                agg[f'{k}_std'] = np.std(vals)
    agg['num_seeds'] = len(summaries)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--pairs', default='nl,nl',
                        help='Comma-separated agent0,agent1.')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--log_dir', default='logs/sweep')
    parser.add_argument('--min_delivery_events', type=int, default=1)
    parser.add_argument('--min_cooking_events', type=int, default=3)
    parser.add_argument('--device', default='auto',
                        help="'auto', 'cpu', or GPU index like '0'")
    parser.add_argument('--reward_shaping', action='store_true', default=True,
                        help='Enable reward shaping')
    parser.add_argument('--no_reward_shaping', action='store_false',
                        dest='reward_shaping')
    parser.add_argument('--ent_coef', type=float, default=0.05)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--algo', default='ippo', choices=['ippo', 'mappo'],
                        help='Algorithm: ippo (independent PPO) or mappo (centralized-critic MAPPO)')
    parser.add_argument('--obs_mode', default='egocentric',
                        choices=['egocentric', 'global_concat', 'local'],
                        help='Observation mode')
    parser.add_argument('--shaping_type', default='none',
                        choices=['none', 'raw_clipped', 'constant_bonus',
                                 'event_density_bonus', 'event_binary_bonus',
                                 'delivery_chain_bonus', 'delivery_chain_raw_clipped',
                                 'normalized', 'weighted_normalized',
                                 'delta_complementarity',
                                 'self_task_progress', 'team_task_progress',
                                 'teammate_task_progress', 'task_lookahead_rule',
                                 'task_lola_rule', 'team_bottleneck_progress'],
                        help='Shaping bonus type (default none)')
    parser.add_argument('--role_shaping', action='store_true', default=False,
                        help='[DEPRECATED] Use --shaping_type instead')
    parser.add_argument('--role_window', type=int, default=20,
                        help='Past episodes for teammate role tendency (default 20)')
    parser.add_argument('--lambda_role', type=float, default=0.1,
                        help='Shaping bonus weight in training reward (default 0.1)')
    parser.add_argument('--bonus_clip', type=float, default=1.0,
                        help='Max absolute applied bonus per episode (default 1.0)')
    parser.add_argument('--role_bonus_clip', type=float, default=None,
                        help='[DEPRECATED] Use --bonus_clip')
    parser.add_argument('--role_bonus_type', default=None,
                        choices=['raw_clipped', 'normalized', 'weighted_normalized',
                                 'delta_complementarity'],
                        help='[DEPRECATED] Use --shaping_type')
    parser.add_argument('--w_onion_pickup', type=float, default=0.1)
    parser.add_argument('--w_potting', type=float, default=1.0)
    parser.add_argument('--w_dish_pickup', type=float, default=0.2)
    parser.add_argument('--w_soup_pickup', type=float, default=0.7)
    parser.add_argument('--w_delivery', type=float, default=1.0)
    args = parser.parse_args()

    # ── Resolve shaping_type, bonus_clip (backward compat) ──
    shaping_type = args.shaping_type
    bonus_clip = args.bonus_clip
    if args.role_shaping and shaping_type == 'none':
        shaping_type = 'raw_clipped'
    if args.role_bonus_type is not None:
        shaping_type = args.role_bonus_type
    if args.role_bonus_clip is not None:
        bonus_clip = args.role_bonus_clip

    parts = args.pairs.split(',')
    agent0_type = parts[0].strip()
    agent1_type = parts[1].strip() if len(parts) > 1 else agent0_type

    gpu_id = None
    if args.device == 'cpu':
        gpu_id = -1
    elif args.device != 'auto':
        gpu_id = int(args.device)

    all_summaries = []

    for seed in args.seeds:
        pair_log_dir = os.path.join(
            args.log_dir, f'{agent0_type}_{agent1_type}', f'seed_{seed}')
        summary, _, _ = run_pair(
            layout=args.layout,
            agent0_type=agent0_type,
            agent1_type=agent1_type,
            num_episodes=args.num_episodes,
            seed=seed,
            log_dir=pair_log_dir,
            horizon=args.horizon,
            lr=args.lr,
            gamma=args.gamma,
            min_delivery_events=args.min_delivery_events,
            min_cooking_events=args.min_cooking_events,
            reward_shaping=args.reward_shaping,
            ent_coef=args.ent_coef,
            ppo_epochs=args.ppo_epochs,
            device=gpu_id if gpu_id is not None else 'auto',
            algo=args.algo,
            obs_mode=args.obs_mode,
            role_shaping=args.role_shaping,
            role_window=args.role_window,
            lambda_role=args.lambda_role,
            role_bonus_clip=args.role_bonus_clip or bonus_clip,
            role_bonus_type=args.role_bonus_type or 'raw_clipped',
            shaping_type=shaping_type,
            bonus_clip=bonus_clip,
            w_onion_pickup=args.w_onion_pickup,
            w_potting=args.w_potting,
            w_dish_pickup=args.w_dish_pickup,
            w_soup_pickup=args.w_soup_pickup,
            w_delivery=args.w_delivery,
        )
        summary['agent0_type'] = agent0_type
        summary['agent1_type'] = agent1_type
        all_summaries.append(summary)

        with open(os.path.join(pair_log_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    agg = compute_summary_stats(all_summaries)
    agg['agent0_type'] = agent0_type
    agg['agent1_type'] = agent1_type
    agg['layout'] = args.layout
    agg['seeds'] = args.seeds
    if shaping_type != 'none':
        agg['shaping_type'] = shaping_type
        agg['lambda_role'] = args.lambda_role
        agg['role_window'] = args.role_window
        agg['bonus_clip'] = bonus_clip
        agg['w_onion_pickup'] = args.w_onion_pickup
        agg['w_potting'] = args.w_potting
        agg['w_dish_pickup'] = args.w_dish_pickup
        agg['w_soup_pickup'] = args.w_soup_pickup
        agg['w_delivery'] = args.w_delivery

    os.makedirs(args.log_dir, exist_ok=True)
    csv_path = os.path.join(args.log_dir, 'summary_all.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(agg.keys()))
        writer.writeheader()
        writer.writerow(agg)

    with open(os.path.join(args.log_dir, 'summary_all.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY: {agent0_type}+{agent1_type} on {args.layout}")
    print(f"{'='*60}")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
