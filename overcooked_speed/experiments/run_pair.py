"""Train one algorithm pair and record per-episode metrics.

Usage:
    python overcooked_speed/experiments/run_pair.py \
        --layout cramped_room --agent0 nl --agent1 nl \
        --num_episodes 50 --seed 0 --log_dir logs/smoke
"""
import argparse
import json
import os
import sys
import time
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from overcooked_speed.envs.overcooked_wrapper import OvercookedWrapper
from overcooked_speed.envs.event_tracker import EventTracker
from overcooked_speed.envs import get_free_gpus
from overcooked_speed.analysis.metrics import (
    compute_all_metrics,
    delivery_specialization,
    cooking_specialization,
    gated_delivery_specialization,
    gated_cooking_specialization,
    gated_overall_specialization,
)
from overcooked_speed.agents import create_agent, MAPPOManager


def select_device(gpu_id=None):
    """Select torch device with GPU occupancy check."""
    import torch

    if gpu_id == -1:
        return torch.device('cpu'), -1

    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            return torch.device('cpu'), -1
        return device, gpu_id

    if not torch.cuda.is_available():
        return torch.device('cpu'), -1

    free = get_free_gpus(max_gpus=2)
    if free:
        device = torch.device(f'cuda:{free[0]}')
        print(f"Auto-selected GPU {free[0]} (free: {free})")
        return device, free[0]
    else:
        print("No free GPU found, using CPU")
        return torch.device('cpu'), -1


def run_pair(layout, agent0_type, agent1_type, num_episodes, seed, log_dir,
             horizon=400, lr=1e-3, gamma=0.99, hidden_dim=256,
             reward_threshold=20.0, spec_threshold=0.3,
             min_delivery_events=1, min_cooking_events=3,
             reward_shaping=True, ent_coef=0.05, ppo_epochs=4,
             device='auto', algo='ippo'):
    """Run one agent pair for num_episodes and log results.

    Args:
        algo: 'ippo' (independent PPO) or 'mappo' (centralized-critic MAPPO).
    """

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    if isinstance(device, str):
        gpu_id = None if device == 'auto' else int(device)
    else:
        gpu_id = device
    torch_device, used_gpu = select_device(gpu_id)
    if torch_device.type == 'cuda':
        torch.cuda.manual_seed(seed)

    print(f"Using device: {torch_device} (GPU {used_gpu})" if used_gpu >= 0
          else f"Using device: {torch_device}")
    print(f"Algorithm: {algo.upper()}")

    env = OvercookedWrapper(layout_name=layout, horizon=horizon,
                            reward_shaping=reward_shaping)
    obs_dim = env.obs_dim
    n_actions = env.NUM_ACTIONS

    if algo == 'mappo':
        mappo = MAPPOManager(obs_dim, n_actions, lr=lr, gamma=gamma,
                             hidden_dim=hidden_dim, device=torch_device,
                             ppo_epochs=ppo_epochs, ent_coef=ent_coef)
    else:
        agent0 = create_agent(agent0_type, 0, obs_dim, n_actions, lr, gamma, hidden_dim,
                              device=torch_device, ent_coef=ent_coef,
                              ppo_epochs=ppo_epochs)
        agent1 = create_agent(agent1_type, 1, obs_dim, n_actions, lr, gamma, hidden_dim,
                              device=torch_device, ent_coef=ent_coef,
                              ppo_epochs=ppo_epochs)

    tracker = EventTracker()
    episode_rewards = []
    episode_event_counts = []

    # CSV: episode-level fields
    action_names = ['up', 'down', 'right', 'left', 'stay', 'interact']
    base_fields = ['episode', 'reward']
    a0_event_fields = [f'a0_pickup_onion', f'a0_pickup_dish', f'a0_pickup_soup',
                       f'a0_place_onion_in_pot', f'a0_deliver_soup',
                       f'a0_stay', f'a0_blocked', f'a0_invalid', f'a0_total']
    a1_event_fields = [f'a1_pickup_onion', f'a1_pickup_dish', f'a1_pickup_soup',
                       f'a1_place_onion_in_pot', f'a1_deliver_soup',
                       f'a1_stay', f'a1_blocked', f'a1_invalid', f'a1_total']
    a0_action_fields = [f'a0_action_{n}' for n in action_names]
    a1_action_fields = [f'a1_action_{n}' for n in action_names]
    learn_fields = ['a0_loss', 'a1_loss', 'a0_entropy', 'a1_entropy',
                    'a0_grad_norm', 'a1_grad_norm',
                    'a0_approx_kl', 'a1_approx_kl',
                    'a0_value_loss', 'a1_value_loss']
    mappo_extra = ['critic_loss', 'value_mean', 'advantage_mean']
    spec_fields = ['s_delivery', 's_cooking', 's_overall',
                   's_delivery_gated', 's_cooking_gated', 's_overall_gated']

    fieldnames = (base_fields + a0_event_fields + a1_event_fields +
                  a0_action_fields + a1_action_fields +
                  learn_fields + (mappo_extra if algo == 'mappo' else []) +
                  spec_fields)

    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'episodes.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    t_start = time.time()

    for ep in range(num_episodes):
        obs0, obs1 = env.reset()
        tracker.reset()

        if algo == 'mappo':
            mappo.train()
        else:
            agent0.train()
            agent1.train()

        done = False
        ep_reward = 0.0

        while not done:
            if algo == 'mappo':
                a0, a1 = mappo.act(obs0, obs1)
            else:
                a0 = agent0.act(obs0)
                a1 = agent1.act(obs1)

            (next_obs0, next_obs1), reward, done, info = env.step((a0, a1))
            tracker.step((a0, a1), info)

            if algo == 'mappo':
                mappo.store_reward(reward)
            else:
                agent0.store_reward(reward)
                agent1.store_reward(reward)

            obs0 = next_obs0
            obs1 = next_obs1
            ep_reward += reward

        game_stats = env.get_game_stats()
        tracker.post_process(game_stats, ep_reward)

        episode_rewards.append(ep_reward)
        counts = tracker.get_all_counts()
        episode_event_counts.append(counts)

        # End-of-episode training
        if algo == 'mappo':
            log = mappo.end_episode()
            a0_log = {
                'loss': log['actor0_loss'],
                'entropy': log['entropy0'],
                'grad_norm': log['a0_grad_norm'],
                'approx_kl': log['a0_approx_kl'],
                'value_loss': log['value_loss'],
            }
            a1_log = {
                'loss': log['actor1_loss'],
                'entropy': log['entropy1'],
                'grad_norm': log['a1_grad_norm'],
                'approx_kl': log['a1_approx_kl'],
                'value_loss': log['value_loss'],
            }
        else:
            log = {}
            a0_log = agent0.end_episode()
            a1_log = agent1.end_episode()

        # ── Specialization ──
        c0 = counts[0]
        c1 = counts[1]
        del0, del1 = c0['deliver_soup'], c1['deliver_soup']
        cook0 = c0['pickup_onion'] + c0['place_onion_in_pot']
        cook1 = c1['pickup_onion'] + c1['place_onion_in_pot']

        s_del_raw = delivery_specialization([del0, del1])
        s_cook_raw = cooking_specialization([cook0, cook1])
        s_overall_raw = 0.5 * (s_del_raw + s_cook_raw)

        s_del_gated = gated_delivery_specialization([del0, del1], min_delivery_events)
        s_cook_gated = gated_cooking_specialization([cook0, cook1], min_cooking_events)
        s_overall_gated = gated_overall_specialization(
            [del0, del1], [cook0, cook1], min_delivery_events, min_cooking_events)

        row = {
            'episode': ep,
            'reward': ep_reward,
            # Event counts agent 0
            'a0_pickup_onion': c0['pickup_onion'],
            'a0_pickup_dish': c0['pickup_dish'],
            'a0_pickup_soup': c0['pickup_soup'],
            'a0_place_onion_in_pot': c0['place_onion_in_pot'],
            'a0_deliver_soup': c0['deliver_soup'],
            'a0_stay': c0['stay'], 'a0_blocked': c0['blocked'],
            'a0_invalid': c0['invalid_action'], 'a0_total': c0['total_actions'],
            # Event counts agent 1
            'a1_pickup_onion': c1['pickup_onion'],
            'a1_pickup_dish': c1['pickup_dish'],
            'a1_pickup_soup': c1['pickup_soup'],
            'a1_place_onion_in_pot': c1['place_onion_in_pot'],
            'a1_deliver_soup': c1['deliver_soup'],
            'a1_stay': c1['stay'], 'a1_blocked': c1['blocked'],
            'a1_invalid': c1['invalid_action'], 'a1_total': c1['total_actions'],
            # Action distribution agent 0
            'a0_action_up': c0['action_up'],
            'a0_action_down': c0['action_down'],
            'a0_action_right': c0['action_right'],
            'a0_action_left': c0['action_left'],
            'a0_action_stay': c0['action_stay'],
            'a0_action_interact': c0['action_interact'],
            # Action distribution agent 1
            'a1_action_up': c1['action_up'],
            'a1_action_down': c1['action_down'],
            'a1_action_right': c1['action_right'],
            'a1_action_left': c1['action_left'],
            'a1_action_stay': c1['action_stay'],
            'a1_action_interact': c1['action_interact'],
            # Learning metrics
            'a0_loss': a0_log['loss'],
            'a1_loss': a1_log['loss'],
            'a0_entropy': a0_log['entropy'],
            'a1_entropy': a1_log['entropy'],
            'a0_grad_norm': a0_log['grad_norm'],
            'a1_grad_norm': a1_log['grad_norm'],
            'a0_approx_kl': a0_log.get('approx_kl', 0),
            'a1_approx_kl': a1_log.get('approx_kl', 0),
            'a0_value_loss': a0_log.get('value_loss', 0),
            'a1_value_loss': a1_log.get('value_loss', 0),
            # Specialization (raw)
            's_delivery': s_del_raw,
            's_cooking': s_cook_raw,
            's_overall': s_overall_raw,
            # Specialization (gated)
            's_delivery_gated': '' if np.isnan(s_del_gated) else s_del_gated,
            's_cooking_gated': '' if np.isnan(s_cook_gated) else s_cook_gated,
            's_overall_gated': '' if np.isnan(s_overall_gated) else s_overall_gated,
        }
        if algo == 'mappo':
            row['critic_loss'] = log.get('critic_loss', 0)
            row['value_mean'] = log.get('value_mean', 0)
            row['advantage_mean'] = log.get('advantage_mean', 0)
        writer.writerow(row)

    csv_file.close()

    summary = compute_all_metrics(episode_rewards, episode_event_counts,
                                   min_delivery_events, min_cooking_events)
    summary.update({
        'seed': seed,
        'layout': layout,
        'agent0_type': agent0_type,
        'agent1_type': agent1_type,
        'num_episodes': num_episodes,
        'horizon': horizon,
        'time_elapsed': time.time() - t_start,
        'device': str(torch_device),
        'min_delivery_events': min_delivery_events,
        'min_cooking_events': min_cooking_events,
    })

    with open(os.path.join(log_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Done. seed={seed} layout={layout} "
          f"a0={agent0_type} a1={agent1_type} "
          f"final_reward={summary['final_reward']:.2f} "
          f"T_reward={summary['T_reward']} "
          f"final_spec={summary['final_specialization']:.3f} "
          f"spec_gated={summary['final_specialization_gated']} "
          f"time={summary['time_elapsed']:.0f}s")

    return summary, episode_rewards, episode_event_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--agent0', default='nl',
                        choices=['nl'],
                        help='Agent 0 type (future: lola, lookahead, ideal_jpi)')
    parser.add_argument('--agent1', default='nl',
                        choices=['nl'],
                        help='Agent 1 type (future: lola, lookahead, ideal_jpi)')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--log_dir', default='logs/smoke')
    parser.add_argument('--reward_threshold', type=float, default=20.0)
    parser.add_argument('--spec_threshold', type=float, default=0.3)
    parser.add_argument('--min_delivery_events', type=int, default=1)
    parser.add_argument('--min_cooking_events', type=int, default=3)
    parser.add_argument('--device', default='auto',
                        help="'auto', 'cpu', or GPU index like '0'")
    parser.add_argument('--reward_shaping', action='store_true', default=True,
                        help='Enable reward shaping for intermediate milestones')
    parser.add_argument('--no_reward_shaping', action='store_false',
                        dest='reward_shaping',
                        help='Disable reward shaping (use sparse reward only)')
    parser.add_argument('--ent_coef', type=float, default=0.05,
                        help='Entropy bonus coefficient (default 0.05)')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='PPO update epochs per episode (default 4)')
    parser.add_argument('--algo', default='ippo', choices=['ippo', 'mappo'],
                        help='Algorithm: ippo (independent PPO) or mappo (centralized-critic MAPPO)')
    args = parser.parse_args()

    gpu_id = None
    if args.device == 'cpu':
        gpu_id = -1
    elif args.device != 'auto':
        gpu_id = int(args.device)

    run_pair(
        layout=args.layout,
        agent0_type=args.agent0,
        agent1_type=args.agent1,
        num_episodes=args.num_episodes,
        seed=args.seed,
        log_dir=args.log_dir,
        horizon=args.horizon,
        lr=args.lr,
        gamma=args.gamma,
        reward_threshold=args.reward_threshold,
        spec_threshold=args.spec_threshold,
        min_delivery_events=args.min_delivery_events,
        min_cooking_events=args.min_cooking_events,
        reward_shaping=args.reward_shaping,
        ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs,
        device=gpu_id if gpu_id is not None else 'auto',
        algo=args.algo,
    )


if __name__ == '__main__':
    main()
