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
from collections import deque
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
from overcooked_speed.agents import create_agent, MAPPOManager, RoleShapingManager
from overcooked_speed.agents.lts_agent import LTSAgent
from overcooked_speed.agents.rnn_agent import RNNAgent
from overcooked_speed.utils.teammate_features import extract_teammate_y



def resolve_lts_preset(name):
    """Resolve named LTS-PPO preset to a dict of overrides.

    Args:
        name: preset name — 'small', 'base', or 'no_inter'

    Returns:
        dict of (key, value) overrides for LTS defaults.
    """
    presets = {
        'small': dict(
            L=5, M=3, K=5, belief_dim=32, hidden_dim=64,
            enc_out_dim=32, obs_enc_hidden=64,
            intra_hidden=32, inter_hidden=32,
            alpha=0.05, beta=0.02, eta=0.02,
            belief_cons_coef=0.0,
        ),
        'base': dict(
            L=20, M=10, K=10, belief_dim=64, hidden_dim=256,
            enc_out_dim=64, obs_enc_hidden=128,
            intra_hidden=64, inter_hidden=64,
            alpha=0.1, beta=0.05, eta=0.05,
            belief_cons_coef=0.0,
        ),
        'no_inter': dict(
            L=20, M=0, K=10, belief_dim=64, hidden_dim=256,
            enc_out_dim=64, obs_enc_hidden=128,
            intra_hidden=64, inter_hidden=64,
            alpha=0.1, beta=0.05, eta=0.0,
            belief_cons_coef=0.0,
        ),
    }
    if name is None:
        return {}
    name = name.lower()
    if name not in presets:
        raise ValueError(f"Unknown LTS preset '{name}'. "
                         f"Available: {list(presets.keys())}")
    return presets[name]


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



class RunningWindowNormalizer:
    """Running window normalizer for scalar episode-level aux scores."""

    def __init__(self, window=100, eps=1e-8, clip=5.0):
        self.values = deque(maxlen=window)
        self.eps = eps
        self.clip = clip

    def normalize(self, value):
        if len(self.values) < 2:
            return 0.0
        arr = np.asarray(self.values, dtype=np.float32)
        std = float(arr.std())
        if std < self.eps:
            return 0.0
        adv = (float(value) - float(arr.mean())) / (std + self.eps)
        return float(np.clip(adv, -self.clip, self.clip))

    def record(self, value):
        self.values.append(float(value))


def task_score(counts, w_onion_pickup=0.1, w_potting=1.0,
               w_dish_pickup=0.2, w_soup_pickup=0.7, w_delivery=1.0):
    return (
        w_onion_pickup * counts.get('pickup_onion', 0) +
        w_potting * counts.get('place_onion_in_pot', 0) +
        w_dish_pickup * counts.get('pickup_dish', 0) +
        w_soup_pickup * counts.get('pickup_soup', 0) +
        w_delivery * counts.get('deliver_soup', 0)
    )


def compute_aux_scores(counts, aux_task_type, w_onion_pickup=0.1,
                       w_potting=1.0, w_dish_pickup=0.2,
                       w_soup_pickup=0.7, w_delivery=1.0):
    score0 = task_score(counts[0], w_onion_pickup, w_potting,
                        w_dish_pickup, w_soup_pickup, w_delivery)
    score1 = task_score(counts[1], w_onion_pickup, w_potting,
                        w_dish_pickup, w_soup_pickup, w_delivery)
    if aux_task_type == 'self_task':
        return float(score0), float(score1)
    if aux_task_type == 'team_task':
        team_score = float(score0 + score1)
        return team_score, team_score
    if aux_task_type == 'teammate_task':
        return float(score1), float(score0)
    raise ValueError(f"Unknown aux_task_type: {aux_task_type}")


def compute_task_totals(counts):
    total_onion_pickup = (counts[0].get('pickup_onion', 0) +
                          counts[1].get('pickup_onion', 0))
    total_potting = (counts[0].get('place_onion_in_pot', 0) +
                     counts[1].get('place_onion_in_pot', 0))
    total_dish_pickup = (counts[0].get('pickup_dish', 0) +
                         counts[1].get('pickup_dish', 0))
    total_soup_pickup = (counts[0].get('pickup_soup', 0) +
                         counts[1].get('pickup_soup', 0))
    total_soup_delivery = (counts[0].get('deliver_soup', 0) +
                           counts[1].get('deliver_soup', 0))
    total_task_events = (total_onion_pickup + total_potting +
                         total_dish_pickup + total_soup_pickup +
                         total_soup_delivery)
    return {
        'total_task_events': float(total_task_events),
        'total_potting': float(total_potting),
        'total_soup_pickup': float(total_soup_pickup),
        'total_soup_delivery': float(total_soup_delivery),
        'total_onion_pickup': float(total_onion_pickup),
        'total_dish_pickup': float(total_dish_pickup),
    }


def mean_task_totals(episode_event_counts):
    if not episode_event_counts:
        return {
            'mean_total_task_events': 0.0,
            'mean_total_potting': 0.0,
            'mean_total_soup_pickup': 0.0,
            'mean_total_soup_delivery': 0.0,
        }
    totals = [compute_task_totals(c) for c in episode_event_counts]
    return {
        'mean_total_task_events': float(np.mean([t['total_task_events'] for t in totals])),
        'mean_total_potting': float(np.mean([t['total_potting'] for t in totals])),
        'mean_total_soup_pickup': float(np.mean([t['total_soup_pickup'] for t in totals])),
        'mean_total_soup_delivery': float(np.mean([t['total_soup_delivery'] for t in totals])),
    }


EVENT_KEYS = ['onion_pickup', 'potting_onion', 'dish_pickup',
               'soup_pickup', 'soup_delivery']


def held_to_int(held_obj):
    """Map held object to int: 0=none, 1=onion/ingredient, 2=dish, 3=soup."""
    if held_obj is None:
        return 0
    name = getattr(held_obj, 'name', '')
    if name in ('onion', 'tomato'):
        return 1
    if name == 'dish':
        return 2
    if name == 'soup':
        return 3
    return 0


def run_pair(layout, agent0_type, agent1_type, num_episodes, seed, log_dir,
             horizon=400, lr=1e-3, gamma=0.99, hidden_dim=256,
             nl_hidden_dim=256,
             reward_threshold=20.0, spec_threshold=0.3,
             min_delivery_events=1, min_cooking_events=3,
             reward_shaping=True, ent_coef=0.05, ppo_epochs=4,
             device='auto', algo='ippo', obs_mode='egocentric',
             role_shaping=False, role_window=20, lambda_role=0.1,
             role_bonus_clip=1.0, role_bonus_type='raw_clipped',
             shaping_type='none', bonus_clip=1.0,
             w_onion_pickup=0.1, w_potting=1.0, w_dish_pickup=0.2,
             w_soup_pickup=0.7, w_delivery=1.0,
             aux_task_loss=False, aux_task_type='self_task', aux_coef=0.0,
             aux_norm_window=100, aux_adv_clip=5.0,
             critic_mode='normal', teammate_probs_mode='true',
             save_trajectories=False,
             # LTS-PPO args
             L=20, M=10, K=10, belief_dim=64,
             intra_feat_dim=23, y_dim=16, future_dim=11, c_dim=17,
             obs_enc_hidden=128, intra_hidden=64, inter_hidden=64,
             enc_out_dim=64, alpha=0.1, beta=0.05, eta=0.05,
             belief_cons_coef=0.0, belief_lr_scale=0.5, df_event=False,
             # RNN-IPPO args
             rnn_K=10, rnn_hidden_dim=64,
             # Fixed teammate
             fixed_teammate=None, warmup_inter_memory_episodes=0,
             save_model_dir=None):
    """Run one agent pair for num_episodes and log results.

    Args:
        algo: 'ippo' (independent PPO) or 'mappo' (centralized-critic MAPPO).
        obs_mode: 'egocentric' (agent-specific ~520-dim) or 'global_concat'
                  (both agents see same ~1040-dim).
        shaping_type: shaping bonus type (see VALID_SHAPING_TYPES).
        lambda_role: weight of shaping bonus in training reward.
        bonus_clip: max absolute applied bonus per episode.
        w_onion_pickup, w_potting, w_dish_pickup, w_soup_pickup, w_delivery:
            task-progress weights (for task-level shaping and aux-task scores).
        aux_task_loss: Add an episode-level auxiliary actor loss for IPPO only.
        role_shaping, role_bonus_type, role_bonus_clip: deprecated aliases.
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
    print(f"Algorithm: {algo.upper()}  obs_mode: {obs_mode}")

    env = OvercookedWrapper(layout_name=layout, horizon=horizon,
                            reward_shaping=reward_shaping, obs_mode=obs_mode)
    # Auto-detect dimensions from env (no hardcoding)
    obs0, obs1 = env.reset()
    obs_dim = obs0.shape[0]
    global_obs_dim = env.global_obs_dim
    n_actions = env.NUM_ACTIONS

    print(f"actor_obs_dim={obs_dim}  global_obs_dim={global_obs_dim}")

    # Resolve effective shaping_type (backward compat with role_shaping / role_bonus_type)
    effective_type = shaping_type
    if role_shaping and effective_type == 'none':
        effective_type = role_bonus_type if role_bonus_type else 'raw_clipped'

    if effective_type != 'none' and algo == 'mappo':
        raise NotImplementedError(
            "Shaping is currently only supported for IPPO (--algo ippo). "
            "MAPPO already uses a centralized critic which captures role dynamics.")

    if critic_mode == 'policy_conditioned' and algo == 'mappo':
        raise NotImplementedError(
            "Policy-conditioned critic is not supported for MAPPO. "
            "MAPPO already uses a centralized critic with global state.")

    critic_extra_dim = 6 if critic_mode == 'policy_conditioned' else 0
    if critic_mode == 'policy_conditioned':
        print(f"Critic mode: policy_conditioned "
              f"(critic_extra_dim={critic_extra_dim}, "
              f"critic_extra_hidden_dim=32)")

    if aux_task_type not in {'self_task', 'team_task', 'teammate_task'}:
        raise ValueError(
            "aux_task_type must be one of: self_task, team_task, teammate_task")

    if aux_task_loss and algo != 'ippo':
        raise NotImplementedError(
            "Auxiliary task actor loss is currently supported only for IPPO. "
            "MAPPO is intentionally left unchanged for this experiment.")

    if aux_task_loss:
        print(f"Aux task loss enabled: type={aux_task_type} "
              f"coef={aux_coef} norm_window={aux_norm_window} "
              f"adv_clip={aux_adv_clip}")
        aux_normalizers = [
            RunningWindowNormalizer(window=aux_norm_window, clip=aux_adv_clip),
            RunningWindowNormalizer(window=aux_norm_window, clip=aux_adv_clip),
        ]
    else:
        aux_normalizers = None

    if effective_type != 'none':
        print(f"Shaping enabled: type={effective_type} "
              f"window={role_window} lambda={lambda_role} clip={bonus_clip}")
        role_mgr = RoleShapingManager(window=role_window, bonus_clip=bonus_clip,
                                       shaping_type=effective_type,
                                       lambda_role=lambda_role,
                                       w_onion_pickup=w_onion_pickup,
                                       w_potting=w_potting,
                                       w_dish_pickup=w_dish_pickup,
                                       w_soup_pickup=w_soup_pickup,
                                       w_delivery=w_delivery)

    if algo == 'mappo':
        mappo = MAPPOManager(obs_dim, n_actions, lr=lr, gamma=gamma,
                             hidden_dim=hidden_dim, device=torch_device,
                             ppo_epochs=ppo_epochs, ent_coef=ent_coef,
                             global_obs_dim=global_obs_dim)
        print(f"MAPPO: actor input={obs_dim}, critic input={global_obs_dim}")
    else:
        lts_kwargs = dict(
            L=L, M=M, K=K, belief_dim=belief_dim,
            intra_feat_dim=intra_feat_dim, y_dim=y_dim,
            future_dim=future_dim, c_dim=c_dim,
            obs_enc_hidden=obs_enc_hidden,
            intra_hidden=intra_hidden, inter_hidden=inter_hidden,
            enc_out_dim=enc_out_dim,
            alpha=alpha, beta=beta, eta=eta,
            belief_cons_coef=belief_cons_coef,
            belief_lr_scale=belief_lr_scale,
            df_event=df_event,
        )
        rnn_kwargs = dict(
            K=rnn_K, rnn_hidden_dim=rnn_hidden_dim,
            intra_feat_dim=intra_feat_dim, y_dim=y_dim,
        )
        agent_kwargs = {**lts_kwargs, **rnn_kwargs}

        # hidden_dim from preset (lts_ac_hidden_dim) only applies to LTS agents
        _hd0 = hidden_dim if agent0_type == 'lts_ppo' else nl_hidden_dim
        _hd1 = hidden_dim if agent1_type == 'lts_ppo' else nl_hidden_dim
        agent0 = create_agent(agent0_type, 0, obs_dim, n_actions, lr, gamma, _hd0,
                              device=torch_device, ent_coef=ent_coef,
                              ppo_epochs=ppo_epochs,
                              critic_extra_dim=critic_extra_dim,
                              **agent_kwargs)
        agent1 = create_agent(agent1_type, 1, obs_dim, n_actions, lr, gamma, _hd1,
                              device=torch_device, ent_coef=ent_coef,
                              ppo_epochs=ppo_epochs,
                              critic_extra_dim=critic_extra_dim,
                              **agent_kwargs)

        # ── Fixed teammate mode ──
        if fixed_teammate is not None:
            ckpt = torch.load(fixed_teammate, map_location=torch_device)
            # Handle both raw state_dict and dict-wrapped checkpoints
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
            elif isinstance(ckpt, dict) and 'policy' in ckpt:
                sd = ckpt['policy']
            else:
                sd = ckpt
            agent1.policy.load_state_dict(sd, strict=False)
            if isinstance(agent1, LTSAgent) and isinstance(ckpt, dict):
                if 'belief_encoder' in ckpt:
                    agent1.belief_encoder.load_state_dict(ckpt['belief_encoder'], strict=False)
            agent1.eval()
            for p in agent1.policy.parameters():
                p.requires_grad = False
            if isinstance(agent1, LTSAgent):
                for p in agent1.belief_encoder.parameters():
                    p.requires_grad = False
            print(f"Fixed teammate loaded from {fixed_teammate} "
                  f"(frozen, eval mode)")
            if warmup_inter_memory_episodes > 0 and isinstance(agent0, LTSAgent):
                print(f"Warming up inter-memory for "
                      f"{warmup_inter_memory_episodes} episodes...")

    # Agent type checks for LTS/RNN-specific handling
    is_lts0 = isinstance(agent0, LTSAgent) if algo != 'mappo' else False
    is_lts1 = isinstance(agent1, LTSAgent) if algo != 'mappo' else False
    is_rnn0 = isinstance(agent0, RNNAgent) if algo != 'mappo' else False
    is_rnn1 = isinstance(agent1, RNNAgent) if algo != 'mappo' else False
    _y_dim = y_dim  # used for zero-initialization

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
    critic_diag_fields = [
        'critic_mode', 'teammate_probs_mode',
        'a0_return_mean', 'a1_return_mean',
        'a0_value_mean', 'a1_value_mean',
        'a0_explained_variance', 'a1_explained_variance',
        'a0_critic_extra_entropy', 'a1_critic_extra_entropy',
        'a0_value_extra_sensitivity', 'a1_value_extra_sensitivity',
        'a0_value_mean_shuffled', 'a1_value_mean_shuffled',
        'a0_explained_variance_shuffled', 'a1_explained_variance_shuffled',
        'a0_shuffled_diag_note', 'a1_shuffled_diag_note',
    ]
    mappo_extra = ['critic_loss', 'value_mean', 'advantage_mean']
    aux_fields = [
        'aux_task_loss_enabled', 'aux_task_type', 'aux_coef',
        'aux_norm_window', 'aux_score0', 'aux_score1',
        'aux_adv0', 'aux_adv1', 'aux_loss0', 'aux_loss1',
    ]
    task_event_fields = [
        'total_task_events', 'total_potting', 'total_soup_pickup',
        'total_soup_delivery', 'total_onion_pickup', 'total_dish_pickup',
    ]
    shaping_fields = [
        'shaping_type', 'lambda_role', 'bonus_clip',
        'agent0_shaping_raw', 'agent1_shaping_raw',
        'agent0_shaping_applied', 'agent1_shaping_applied',
        'mean_shaping_applied', 'shaping_clip_rate',
        'a0_shaped_reward', 'a1_shaped_reward',
        *task_event_fields,
        # Role-specific (populated only for role types)
        'teammate0_p_cook', 'teammate0_p_deliver',
        'teammate1_p_cook', 'teammate1_p_deliver',
        'complementarity_current', 'complementarity_delta',
        # Task-lookahead-specific
        'teammate0_p_potting', 'teammate0_p_delivery',
        'teammate1_p_potting', 'teammate1_p_delivery',
    ]
    spec_fields = ['s_delivery', 's_cooking', 's_overall',
                   's_delivery_gated', 's_cooking_gated', 's_overall_gated']
    lts_diag_fields = [
        'a0_loss_dy', 'a1_loss_dy',
        'a0_dy_acc', 'a1_dy_acc',
        'a0_loss_df', 'a1_loss_df',
        'a0_loss_df_action', 'a1_loss_df_action',
        'a0_loss_df_event', 'a1_loss_df_event',
        'a0_loss_dc', 'a1_loss_dc',
        'a0_loss_bel_cons', 'a1_loss_bel_cons',
        'a0_ratio_mean', 'a1_ratio_mean',
        'a0_ratio_std', 'a1_ratio_std',
        'a0_ratio_max', 'a1_ratio_max',
        'a0_clip_fraction', 'a1_clip_fraction',
        'a0_logprob_delta_sq', 'a1_logprob_delta_sq',
        'a0_approx_kl_ppo', 'a1_approx_kl_ppo',
        'a0_belief_norm', 'a1_belief_norm',
        'a0_belief_delta_mean', 'a1_belief_delta_mean',
        'a0_belief_encoder_grad_norm', 'a1_belief_encoder_grad_norm',
        'a0_inter_memory_norm', 'a1_inter_memory_norm',
        'a0_inter_memory_filled', 'a1_inter_memory_filled',
        'a0_skipped_dy', 'a1_skipped_dy',
        'a0_skipped_df', 'a1_skipped_df',
        'a0_param_total', 'a1_param_total',
    ]

    extra = []
    if algo == 'mappo':
        extra += mappo_extra
    if aux_task_loss:
        extra += aux_fields
    if effective_type != 'none':
        extra += shaping_fields
    extra += lts_diag_fields

    fieldnames = (base_fields + a0_event_fields + a1_event_fields +
                  a0_action_fields + a1_action_fields +
                  learn_fields + critic_diag_fields + extra + spec_fields)

    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'episodes.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    aux_episode_logs = []

    # ── Trajectory saving buffers ──
    traj_data = None
    if save_trajectories:
        traj_data = {
            'obs_0': [], 'actions_0': [], 'probs_0': [], 'held_0': [], 'rewards_0': [],
            'obs_1': [], 'actions_1': [], 'probs_1': [], 'held_1': [], 'rewards_1': [],
            'events_0': [], 'events_1': [],
            'episode_lengths': [],
        }

    t_start = time.time()

    for ep in range(num_episodes):
        obs0, obs1 = env.reset()
        tracker.reset()

        # Skip warmup episodes for LTS agent during fixed-teammate warmup
        is_warmup = (fixed_teammate is not None and
                     warmup_inter_memory_episodes > 0 and
                     ep < warmup_inter_memory_episodes and
                     is_lts0)

        if algo == 'mappo':
            mappo.train()
        else:
            agent0.train()
            agent1.train()

        done = False
        ep_reward = 0.0

        # LTS/RNN: init prev-step tracking vars (t-1 data, zero at step 0)
        if is_lts0 or is_rnn0:
            prev_ty0 = np.zeros(_y_dim, dtype=np.float32)
            prev_a0 = 0
            prev_r0 = 0.0
        if is_lts1 or is_rnn1:
            prev_ty1 = np.zeros(_y_dim, dtype=np.float32)
            prev_a1 = 0
            prev_r1 = 0.0

        # Per-episode trajectory buffers
        ep_traj = None
        if save_trajectories:
            ep_traj = {
                'obs_0': [], 'actions_0': [], 'probs_0': [], 'held_0': [], 'rewards_0': [],
                'obs_1': [], 'actions_1': [], 'probs_1': [], 'held_1': [], 'rewards_1': [],
            }

        while not done:
            # Build intra features BEFORE acting (uses t-1 data for causality)
            if is_lts0:
                agent0.build_next_intra_feature(prev_ty0, prev_a0, prev_r0)
            if is_rnn0:
                agent0.build_next_intra_feature(prev_ty0, prev_a0, prev_r0)
            if is_lts1:
                agent1.build_next_intra_feature(prev_ty1, prev_a1, prev_r1)
            if is_rnn1:
                agent1.build_next_intra_feature(prev_ty1, prev_a1, prev_r1)

            if algo == 'mappo':
                global_obs = env.get_global_obs()
                a0, a1 = mappo.act(obs0, obs1, global_obs)
                probs0 = probs1 = np.ones(n_actions, dtype=np.float32) / n_actions
            elif critic_mode == 'policy_conditioned':
                # Always collect probs when saving trajectories (for any teammate_probs_mode)
                if teammate_probs_mode in ('true', 'shuffled'):
                    probs0 = agent0.get_action_probs(obs0)
                    probs1 = agent1.get_action_probs(obs1)
                    a0 = agent0.act(obs0, critic_extra=probs1)
                    a1 = agent1.act(obs1, critic_extra=probs0)
                else:
                    # uniform mode
                    extra0 = np.ones(6, dtype=np.float32) / 6.0
                    extra1 = np.ones(6, dtype=np.float32) / 6.0
                    a0 = agent0.act(obs0, critic_extra=extra0)
                    a1 = agent1.act(obs1, critic_extra=extra1)
                    probs0 = agent0.get_action_probs(obs0) if save_trajectories else extra0
                    probs1 = agent1.get_action_probs(obs1) if save_trajectories else extra1
            else:
                a0 = agent0.act(obs0)
                a1 = agent1.act(obs1)
                probs0 = agent0.get_action_probs(obs0) if save_trajectories else np.ones(n_actions, dtype=np.float32) / n_actions
                probs1 = agent1.get_action_probs(obs1) if save_trajectories else np.ones(n_actions, dtype=np.float32) / n_actions

            (next_obs0, next_obs1), reward, done, info = env.step((a0, a1))
            tracker.step((a0, a1), info)
            state_info = env.get_state_info()

            if algo == 'mappo':
                mappo.store_reward(reward)
            else:
                agent0.store_reward(reward)
                agent1.store_reward(reward)

            # ── LTS/RNN: store teammate info & update prev-step vars ──
            if is_lts0:
                agent0.store_teammate_info(state_info, a1)
                prev_ty0 = extract_teammate_y(0, state_info, a1, y_dim=_y_dim)
                prev_a0 = a0
                prev_r0 = reward
            elif is_rnn0:
                prev_ty0 = extract_teammate_y(0, state_info, a1, y_dim=_y_dim)
                agent0.store_teammate_y(prev_ty0)
                prev_a0 = a0
                prev_r0 = reward
            if is_lts1:
                agent1.store_teammate_info(state_info, a0)
                prev_ty1 = extract_teammate_y(1, state_info, a0, y_dim=_y_dim)
                prev_a1 = a1
                prev_r1 = reward
            elif is_rnn1:
                prev_ty1 = extract_teammate_y(1, state_info, a0, y_dim=_y_dim)
                agent1.store_teammate_y(prev_ty1)
                prev_a1 = a1
                prev_r1 = reward

            # ── Trajectory data collection ──
            if ep_traj is not None:
                ep_traj['obs_0'].append(obs0.copy())
                ep_traj['actions_0'].append(a0)
                ep_traj['probs_0'].append(probs0.copy())
                ep_traj['held_0'].append(held_to_int(state_info['player_0_held']))
                ep_traj['rewards_0'].append(reward)
                ep_traj['obs_1'].append(obs1.copy())
                ep_traj['actions_1'].append(a1)
                ep_traj['probs_1'].append(probs1.copy())
                ep_traj['held_1'].append(held_to_int(state_info['player_1_held']))
                ep_traj['rewards_1'].append(reward)

            obs0 = next_obs0
            obs1 = next_obs1
            ep_reward += reward

        game_stats = env.get_game_stats()
        tracker.post_process(game_stats, ep_reward)

        episode_rewards.append(ep_reward)
        counts = tracker.get_all_counts()
        episode_event_counts.append(counts)

        # ── Trajectory: extract per-step events ──
        if ep_traj is not None:
            T = len(ep_traj['obs_0'])
            events_0 = np.zeros((T, 5), dtype=np.uint8)
            events_1 = np.zeros((T, 5), dtype=np.uint8)
            for ei, ek in enumerate(EVENT_KEYS):
                for step_idx in game_stats[ek][0]:
                    if step_idx < T:
                        events_0[step_idx, ei] = 1
                for step_idx in game_stats[ek][1]:
                    if step_idx < T:
                        events_1[step_idx, ei] = 1
            traj_data['events_0'].append(events_0)
            traj_data['events_1'].append(events_1)
            traj_data['obs_0'].append(np.array(ep_traj['obs_0'], dtype=np.float32))
            traj_data['actions_0'].append(np.array(ep_traj['actions_0'], dtype=np.int8))
            traj_data['probs_0'].append(np.array(ep_traj['probs_0'], dtype=np.float32))
            traj_data['held_0'].append(np.array(ep_traj['held_0'], dtype=np.int8))
            traj_data['rewards_0'].append(np.array(ep_traj['rewards_0'], dtype=np.float32)[:, None])
            traj_data['obs_1'].append(np.array(ep_traj['obs_1'], dtype=np.float32))
            traj_data['actions_1'].append(np.array(ep_traj['actions_1'], dtype=np.int8))
            traj_data['probs_1'].append(np.array(ep_traj['probs_1'], dtype=np.float32))
            traj_data['held_1'].append(np.array(ep_traj['held_1'], dtype=np.int8))
            traj_data['rewards_1'].append(np.array(ep_traj['rewards_1'], dtype=np.float32)[:, None])
            traj_data['episode_lengths'].append(T)

        # ── Shaping (IPPO only, episode-level, prior to training) ──
        s_raw0 = s_raw1 = 0.0
        s_app0 = s_app1 = 0.0
        s_clipped = False
        p_cook_0 = p_deliver_0 = 0.5
        p_cook_1 = p_deliver_1 = 0.5
        p_pot_0 = p_del_0 = 0.5
        p_pot_1 = p_del_1 = 0.5
        comp_current = 0.5
        comp_delta = 0.0
        totals = compute_task_totals(counts)
        total_task_events = totals['total_task_events']
        total_potting = totals['total_potting']
        total_soup_pickup = totals['total_soup_pickup']
        total_soup_delivery = totals['total_soup_delivery']
        total_onion_pickup = totals['total_onion_pickup']
        total_dish_pickup = totals['total_dish_pickup']

        aux_score0 = aux_score1 = 0.0
        aux_adv0 = aux_adv1 = 0.0
        aux_loss0 = aux_loss1 = 0.0
        if aux_task_loss:
            aux_score0, aux_score1 = compute_aux_scores(
                counts, aux_task_type, w_onion_pickup, w_potting,
                w_dish_pickup, w_soup_pickup, w_delivery)
            aux_adv0 = aux_normalizers[0].normalize(aux_score0)
            aux_adv1 = aux_normalizers[1].normalize(aux_score1)
            aux_normalizers[0].record(aux_score0)
            aux_normalizers[1].record(aux_score1)

        if effective_type != 'none':
            # For task-level types that need both agents' counts
            role_mgr.set_episode_counts(counts[0], counts[1])

            # For delta type: set both agents' counts before computing bonus
            if effective_type == 'delta_complementarity':
                role_mgr.set_delta_counts(counts[0], counts[1])

            s_raw0, s_app0 = role_mgr.compute_bonus(0, counts[0])
            s_raw1, s_app1 = role_mgr.compute_bonus(1, counts[1])
            s_clipped = (abs(s_raw0 - s_app0) > 1e-8 or
                        abs(s_raw1 - s_app1) > 1e-8)

            # For role types: get teammate tendencies and complementarity
            if role_mgr.is_role_type:
                p_cook_0, p_deliver_0 = role_mgr.get_teammate_tendency(0)
                p_cook_1, p_deliver_1 = role_mgr.get_teammate_tendency(1)
                if effective_type == 'delta_complementarity':
                    comp_current = role_mgr.get_complementarity(counts[0], counts[1])
                    comp_delta = s_app0  # same for both agents

            # For task-lookahead types: get teammate task tendencies
            if role_mgr.is_task_lookahead_type:
                p_pot_0, p_del_0 = role_mgr.get_teammate_task_tendency(0)
                p_pot_1, p_del_1 = role_mgr.get_teammate_task_tendency(1)

            shaped0 = lambda_role * s_app0
            shaped1 = lambda_role * s_app1
            agent0.add_terminal_bonus(float(shaped0))
            agent1.add_terminal_bonus(float(shaped1))
            ep_reward_shaped0 = ep_reward + float(shaped0)
            ep_reward_shaped1 = ep_reward + float(shaped1)

            role_mgr.record_episode(counts[0], counts[1])


        # End-of-episode training
        # Compute LTS episode characteristics (before end_episode call)
        c0_vec = None
        c1_vec = None
        if is_lts0:
            c0_vec = agent0.compute_current_characteristic(game_stats)
        if is_lts1:
            c1_vec = agent1.compute_current_characteristic(game_stats)

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
            shuffle_extra = (critic_mode == 'policy_conditioned' and
                            teammate_probs_mode == 'shuffled')

            # During warmup, skip training but still collect inter-memory for LTS agent
            if is_warmup:
                if is_lts0 and c0_vec is not None and M > 0:
                    agent0._inter_memory.push(c0_vec)
                agent0._clear_all_buffers()
                agent1._clear_buffer()
                a0_log = agent0._empty_lts_log()
                a1_log = {
                    'loss': 0.0, 'mean_return': 0.0, 'entropy': 0.0,
                    'grad_norm': 0.0, 'approx_kl': 0.0, 'value_loss': 0.0,
                    'aux_loss': 0.0,
                    'value_mean': 0.0, 'explained_variance': 0.0,
                    'critic_extra_entropy': 0.0, 'value_extra_sensitivity': 0.0,
                    'sensitivity_note': None,
                    'value_mean_shuffled': None,
                    'explained_variance_shuffled': None,
                    'shuffled_diag_note': None,
                }
            elif aux_task_loss:
                a0_log = agent0.end_episode(
                    aux_adv=aux_adv0, aux_coef=aux_coef,
                    shuffle_critic_extra=shuffle_extra,
                    teammate_characteristic=c0_vec)
                if fixed_teammate is not None:
                    agent1._clear_buffer()
                    a1_log = {
                        'loss': 0.0, 'mean_return': 0.0, 'entropy': 0.0,
                        'grad_norm': 0.0, 'approx_kl': 0.0, 'value_loss': 0.0,
                        'aux_loss': 0.0,
                        'value_mean': 0.0, 'explained_variance': 0.0,
                        'critic_extra_entropy': 0.0, 'value_extra_sensitivity': 0.0,
                        'sensitivity_note': None,
                        'value_mean_shuffled': None,
                        'explained_variance_shuffled': None,
                        'shuffled_diag_note': None,
                    }
                else:
                    a1_log = agent1.end_episode(
                        aux_adv=aux_adv1, aux_coef=aux_coef,
                        shuffle_critic_extra=shuffle_extra,
                        teammate_characteristic=c1_vec)
                aux_loss0 = a0_log.get('aux_loss', 0)
                aux_loss1 = a1_log.get('aux_loss', 0)
                aux_episode_logs.append({
                    'aux_score0': aux_score0,
                    'aux_score1': aux_score1,
                    'aux_adv0': aux_adv0,
                    'aux_adv1': aux_adv1,
                    'aux_loss0': aux_loss0,
                    'aux_loss1': aux_loss1,
                })
            else:
                a0_log = agent0.end_episode(
                    shuffle_critic_extra=shuffle_extra,
                    teammate_characteristic=c0_vec)
                if fixed_teammate is not None:
                    # Fixed teammate: frozen, skip training
                    agent1._clear_buffer()
                    a1_log = {
                        'loss': 0.0, 'mean_return': 0.0, 'entropy': 0.0,
                        'grad_norm': 0.0, 'approx_kl': 0.0, 'value_loss': 0.0,
                        'aux_loss': 0.0,
                        'value_mean': 0.0, 'explained_variance': 0.0,
                        'critic_extra_entropy': 0.0, 'value_extra_sensitivity': 0.0,
                        'sensitivity_note': None,
                        'value_mean_shuffled': None,
                        'explained_variance_shuffled': None,
                        'shuffled_diag_note': None,
                    }
                else:
                    a1_log = agent1.end_episode(
                        shuffle_critic_extra=shuffle_extra,
                        teammate_characteristic=c1_vec)

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
            # Critic-conditioning diagnostics (always present)
            'critic_mode': critic_mode,
            'teammate_probs_mode': teammate_probs_mode,
            'a0_return_mean': a0_log.get('mean_return', 0),
            'a1_return_mean': a1_log.get('mean_return', 0),
            'a0_value_mean': a0_log.get('value_mean', 0),
            'a1_value_mean': a1_log.get('value_mean', 0),
            'a0_explained_variance': a0_log.get('explained_variance', 0),
            'a1_explained_variance': a1_log.get('explained_variance', 0),
            'a0_critic_extra_entropy': a0_log.get('critic_extra_entropy', 0),
            'a1_critic_extra_entropy': a1_log.get('critic_extra_entropy', 0),
            'a0_value_extra_sensitivity': a0_log.get('value_extra_sensitivity', 0),
            'a1_value_extra_sensitivity': a1_log.get('value_extra_sensitivity', 0),
            # Shuffled-mode diagnostics (None in non-shuffled modes)
            'a0_value_mean_shuffled': a0_log.get('value_mean_shuffled') or '',
            'a1_value_mean_shuffled': a1_log.get('value_mean_shuffled') or '',
            'a0_explained_variance_shuffled': a0_log.get('explained_variance_shuffled') or '',
            'a1_explained_variance_shuffled': a1_log.get('explained_variance_shuffled') or '',
            'a0_shuffled_diag_note': a0_log.get('shuffled_diag_note') or '',
            'a1_shuffled_diag_note': a1_log.get('shuffled_diag_note') or '',
            # Specialization (raw)
            's_delivery': s_del_raw,
            's_cooking': s_cook_raw,
            's_overall': s_overall_raw,
            # Specialization (gated)
            's_delivery_gated': '' if np.isnan(s_del_gated) else s_del_gated,
            's_cooking_gated': '' if np.isnan(s_cook_gated) else s_cook_gated,
            's_overall_gated': '' if np.isnan(s_overall_gated) else s_overall_gated,
            # LTS/RNN diagnostics
            'a0_loss_dy': a0_log.get('loss_dy', 0),
            'a1_loss_dy': a1_log.get('loss_dy', 0),
            'a0_dy_acc': a0_log.get('dy_acc', 0),
            'a1_dy_acc': a1_log.get('dy_acc', 0),
            'a0_loss_df': a0_log.get('loss_df', 0),
            'a1_loss_df': a1_log.get('loss_df', 0),
            'a0_loss_df_action': a0_log.get('loss_df_action', 0),
            'a1_loss_df_action': a1_log.get('loss_df_action', 0),
            'a0_loss_df_event': a0_log.get('loss_df_event', 0),
            'a1_loss_df_event': a1_log.get('loss_df_event', 0),
            'a0_loss_dc': a0_log.get('loss_dc', 0),
            'a1_loss_dc': a1_log.get('loss_dc', 0),
            'a0_loss_bel_cons': a0_log.get('loss_bel_cons', 0),
            'a1_loss_bel_cons': a1_log.get('loss_bel_cons', 0),
            'a0_ratio_mean': a0_log.get('ratio_mean', 0),
            'a1_ratio_mean': a1_log.get('ratio_mean', 0),
            'a0_ratio_std': a0_log.get('ratio_std', 0),
            'a1_ratio_std': a1_log.get('ratio_std', 0),
            'a0_ratio_max': a0_log.get('ratio_max', 0),
            'a1_ratio_max': a1_log.get('ratio_max', 0),
            'a0_clip_fraction': a0_log.get('clip_fraction', 0),
            'a1_clip_fraction': a1_log.get('clip_fraction', 0),
            'a0_logprob_delta_sq': a0_log.get('logprob_delta_sq', 0),
            'a1_logprob_delta_sq': a1_log.get('logprob_delta_sq', 0),
            'a0_approx_kl_ppo': a0_log.get('approx_kl_ppo', 0),
            'a1_approx_kl_ppo': a1_log.get('approx_kl_ppo', 0),
            'a0_belief_norm': a0_log.get('belief_norm', 0),
            'a1_belief_norm': a1_log.get('belief_norm', 0),
            'a0_belief_delta_mean': a0_log.get('belief_delta_mean', 0),
            'a1_belief_delta_mean': a1_log.get('belief_delta_mean', 0),
            'a0_belief_encoder_grad_norm': a0_log.get('belief_encoder_grad_norm', 0),
            'a1_belief_encoder_grad_norm': a1_log.get('belief_encoder_grad_norm', 0),
            'a0_inter_memory_norm': a0_log.get('inter_memory_norm', 0),
            'a1_inter_memory_norm': a1_log.get('inter_memory_norm', 0),
            'a0_inter_memory_filled': a0_log.get('inter_memory_filled', 0),
            'a1_inter_memory_filled': a1_log.get('inter_memory_filled', 0),
            'a0_skipped_dy': a0_log.get('skipped_dy', 0),
            'a1_skipped_dy': a1_log.get('skipped_dy', 0),
            'a0_skipped_df': a0_log.get('skipped_df', 0),
            'a1_skipped_df': a1_log.get('skipped_df', 0),
            'a0_param_total': (isinstance(agent0, (LTSAgent, RNNAgent)) and
                              agent0.param_counts.get('total', 0) or 0),
            'a1_param_total': (isinstance(agent1, (LTSAgent, RNNAgent)) and
                              agent1.param_counts.get('total', 0) or 0),
        }
        if algo == 'mappo':
            row['critic_loss'] = log.get('critic_loss', 0)
            row['value_mean'] = log.get('value_mean', 0)
            row['advantage_mean'] = log.get('advantage_mean', 0)
        if aux_task_loss:
            row['aux_task_loss_enabled'] = True
            row['aux_task_type'] = aux_task_type
            row['aux_coef'] = aux_coef
            row['aux_norm_window'] = aux_norm_window
            row['aux_score0'] = aux_score0
            row['aux_score1'] = aux_score1
            row['aux_adv0'] = aux_adv0
            row['aux_adv1'] = aux_adv1
            row['aux_loss0'] = aux_loss0
            row['aux_loss1'] = aux_loss1
        if effective_type != 'none':
            row['shaping_type'] = effective_type
            row['lambda_role'] = lambda_role
            row['bonus_clip'] = bonus_clip
            row['agent0_shaping_raw'] = s_raw0
            row['agent1_shaping_raw'] = s_raw1
            row['agent0_shaping_applied'] = s_app0
            row['agent1_shaping_applied'] = s_app1
            row['mean_shaping_applied'] = (s_app0 + s_app1) / 2.0
            row['shaping_clip_rate'] = 1.0 if s_clipped else 0.0
            row['a0_shaped_reward'] = ep_reward_shaped0
            row['a1_shaped_reward'] = ep_reward_shaped1
            row['total_task_events'] = total_task_events
            row['total_potting'] = total_potting
            row['total_soup_pickup'] = total_soup_pickup
            row['total_soup_delivery'] = total_soup_delivery
            row['total_onion_pickup'] = total_onion_pickup
            row['total_dish_pickup'] = total_dish_pickup
            # Role-specific fields (only for role types)
            row['teammate0_p_cook'] = p_cook_0
            row['teammate0_p_deliver'] = p_deliver_0
            row['teammate1_p_cook'] = p_cook_1
            row['teammate1_p_deliver'] = p_deliver_1
            row['complementarity_current'] = comp_current
            row['complementarity_delta'] = comp_delta
            # Task-lookahead-specific fields
            row['teammate0_p_potting'] = p_pot_0
            row['teammate0_p_delivery'] = p_del_0
            row['teammate1_p_potting'] = p_pot_1
            row['teammate1_p_delivery'] = p_del_1
        writer.writerow(row)

    csv_file.close()

    # ── Save trajectory npz ──
    if save_trajectories:
        lengths = traj_data['episode_lengths']
        E = len(lengths)
        max_T = max(lengths)

        def pad2d(arr_list, max_T, dtype=np.float32):
            """Pad list of 2d arrays (T, D) to (E, max_T, D)."""
            D = arr_list[0].shape[-1]
            out = np.zeros((E, max_T, D), dtype=dtype)
            for i, arr in enumerate(arr_list):
                out[i, :len(arr)] = arr
            return out

        def pad1d(arr_list, max_T, dtype=np.int8):
            """Pad list of 1d arrays (T,) to (E, max_T)."""
            out = np.full((E, max_T), -1, dtype=dtype)
            for i, arr in enumerate(arr_list):
                out[i, :len(arr)] = arr
            return out

        obs_dim = traj_data['obs_0'][0].shape[-1]
        np.savez(
            os.path.join(log_dir, 'trajectories.npz'),
            obs_0=pad2d(traj_data['obs_0'], max_T, np.float32),
            actions_0=pad1d(traj_data['actions_0'], max_T, np.int8),
            probs_0=pad2d(traj_data['probs_0'], max_T, np.float32),
            events_0=pad2d(traj_data['events_0'], max_T, np.uint8),
            held_0=pad1d(traj_data['held_0'], max_T, np.int8),
            rewards_0=pad2d(traj_data['rewards_0'], max_T, np.float32),
            obs_1=pad2d(traj_data['obs_1'], max_T, np.float32),
            actions_1=pad1d(traj_data['actions_1'], max_T, np.int8),
            probs_1=pad2d(traj_data['probs_1'], max_T, np.float32),
            events_1=pad2d(traj_data['events_1'], max_T, np.uint8),
            held_1=pad1d(traj_data['held_1'], max_T, np.int8),
            rewards_1=pad2d(traj_data['rewards_1'], max_T, np.float32),
            episode_lengths=np.array(lengths, dtype=np.int32),
            episode_ids=np.arange(E, dtype=np.int32),
            metadata_json=json.dumps({
                'event_source': 'agent_specific',
                'obs_dim': obs_dim,
                'layout': layout, 'seed': seed,
                'algo': algo, 'critic_mode': critic_mode,
                'teammate_probs_mode': teammate_probs_mode,
            }),
        )
        print(f"Saved trajectory npz: {E} episodes, max_T={max_T}, obs_dim={obs_dim}")

    summary = compute_all_metrics(episode_rewards, episode_event_counts,
                                   min_delivery_events, min_cooking_events)
    summary.update(mean_task_totals(episode_event_counts))

    # Compute mean critic-conditioning diagnostic fields over all episodes from CSV
    critic_diag_means = {}
    import csv as csv_module
    csv_path = os.path.join(log_dir, 'episodes.csv')
    critic_keys = ['a0_return_mean', 'a1_return_mean',
                   'a0_value_mean', 'a1_value_mean',
                   'a0_explained_variance', 'a1_explained_variance',
                   'a0_critic_extra_entropy', 'a1_critic_extra_entropy',
                   'a0_value_extra_sensitivity', 'a1_value_extra_sensitivity',
                   'a0_value_mean_shuffled', 'a1_value_mean_shuffled',
                   'a0_explained_variance_shuffled', 'a1_explained_variance_shuffled']
    try:
        with open(csv_path, 'r') as f:
            reader = csv_module.DictReader(f)
            critic_vals = {k: [] for k in critic_keys}
            for row in reader:
                for k in critic_keys:
                    v = row.get(k, '')
                    if v != '' and v is not None:
                        try:
                            critic_vals[k].append(float(v))
                        except (ValueError, TypeError):
                            pass
        for k in critic_keys:
            vals = critic_vals[k]
            critic_diag_means[f'mean_{k}'] = float(np.mean(vals)) if vals else 0.0
    except Exception:
        for k in critic_keys:
            critic_diag_means[f'mean_{k}'] = 0.0

    summary.update(critic_diag_means)

    summary.update({
        'seed': seed,
        'layout': layout,
        'agent0_type': agent0_type,
        'agent1_type': agent1_type,
        'algo': algo,
        'obs_mode': obs_mode,
        'actor_obs_dim': obs_dim,
        'global_obs_dim': global_obs_dim,
        'num_episodes': num_episodes,
        'horizon': horizon,
        'time_elapsed': time.time() - t_start,
        'device': str(torch_device),
        'min_delivery_events': min_delivery_events,
        'min_cooking_events': min_cooking_events,
        'role_shaping': role_shaping,
        'critic_mode': critic_mode,
        'teammate_probs_mode': teammate_probs_mode,
        'policy_conditioned_critic': critic_mode == 'policy_conditioned',
        'critic_input_dim': obs_dim + (6 if critic_mode == 'policy_conditioned' else 0),
        'teammate_prob_dim': 6 if critic_mode == 'policy_conditioned' else 0,
        'critic_extra_dim': critic_extra_dim,
        'critic_extra_hidden_dim': 32 if critic_mode == 'policy_conditioned' else 0,
    })
    if aux_task_loss:
        summary.update({
            'aux_task_loss': True,
            'aux_task_type': aux_task_type,
            'aux_coef': aux_coef,
            'aux_norm_window': aux_norm_window,
            'aux_adv_clip': aux_adv_clip,
            'mean_aux_score0': float(np.mean([x['aux_score0'] for x in aux_episode_logs])) if aux_episode_logs else 0.0,
            'mean_aux_score1': float(np.mean([x['aux_score1'] for x in aux_episode_logs])) if aux_episode_logs else 0.0,
            'mean_aux_adv0': float(np.mean([x['aux_adv0'] for x in aux_episode_logs])) if aux_episode_logs else 0.0,
            'mean_aux_adv1': float(np.mean([x['aux_adv1'] for x in aux_episode_logs])) if aux_episode_logs else 0.0,
            'mean_aux_loss0': float(np.mean([x['aux_loss0'] for x in aux_episode_logs])) if aux_episode_logs else 0.0,
            'mean_aux_loss1': float(np.mean([x['aux_loss1'] for x in aux_episode_logs])) if aux_episode_logs else 0.0,
        })
    else:
        summary.update({'aux_task_loss': False})
    if effective_type != 'none':
        # Read back CSV to compute aggregate shaping stats
        import csv as csv_module
        csv_path = os.path.join(log_dir, 'episodes.csv')
        all_applied = []
        all_clip_rate_vals = []
        all_task_events = []
        all_soup_del = []
        all_potting = []
        all_soup_pickup = []
        with open(csv_path, 'r') as f:
            reader = csv_module.DictReader(f)
            for row in reader:
                if row.get('mean_shaping_applied'):
                    all_applied.append(abs(float(row['mean_shaping_applied'])))
                if row.get('shaping_clip_rate'):
                    all_clip_rate_vals.append(float(row['shaping_clip_rate']))
                if row.get('total_task_events'):
                    all_task_events.append(float(row['total_task_events']))
                if row.get('total_soup_delivery'):
                    all_soup_del.append(float(row['total_soup_delivery']))
                if row.get('total_potting'):
                    all_potting.append(float(row['total_potting']))
                if row.get('total_soup_pickup'):
                    all_soup_pickup.append(float(row['total_soup_pickup']))
        summary.update({
            'shaping_type': effective_type,
            'lambda_role': lambda_role,
            'bonus_clip': bonus_clip,
            'role_window': role_window,
            'mean_shaping_applied': float(np.mean(all_applied)) if all_applied else 0.0,
            'shaping_clip_rate': float(np.mean(all_clip_rate_vals)) if all_clip_rate_vals else 0.0,
            'mean_total_task_events': summary['mean_total_task_events'],
            'mean_total_soup_delivery': summary['mean_total_soup_delivery'],
            'mean_total_potting': summary['mean_total_potting'],
            'mean_total_soup_pickup': summary['mean_total_soup_pickup'],
        })
        # Include task weight params in summary
        summary.update({
            'w_onion_pickup': w_onion_pickup,
            'w_potting': w_potting,
            'w_dish_pickup': w_dish_pickup,
            'w_soup_pickup': w_soup_pickup,
            'w_delivery': w_delivery,
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

    # ── Save model checkpoints ──
    if save_model_dir is not None and algo != 'mappo':
        import torch as _torch_save
        os.makedirs(save_model_dir, exist_ok=True)
        for i, agent in enumerate([agent0, agent1]):
            path = os.path.join(save_model_dir, f'model_agent{i}.pt')
            if isinstance(agent, LTSAgent):
                _torch_save.save({
                    'policy': agent.ac_network.state_dict(),
                    'belief_encoder': agent.belief_encoder.state_dict(),
                }, path)
            else:
                _torch_save.save(agent.policy.state_dict(), path)
            print(f"Saved model to {path}")
        # Also save LTS-specific config
        if isinstance(agent0, LTSAgent) or isinstance(agent1, LTSAgent):
            import json as _json_save
            cfg_path = os.path.join(save_model_dir, 'lts_config.json')
            cfg = {
                'L': L, 'M': M, 'K': K, 'belief_dim': belief_dim,
                'y_dim': y_dim, 'intra_feat_dim': intra_feat_dim,
                'future_dim': (agent0.future_dim if isinstance(agent0, LTSAgent)
                              else agent1.future_dim if isinstance(agent1, LTSAgent)
                              else None),
                'c_dim': c_dim, 'df_event': df_event,
            }
            with open(cfg_path, 'w') as f:
                _json_save.dump(cfg, f, indent=2)
            print(f"Saved LTS config to {cfg_path}")

    return summary, episode_rewards, episode_event_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', default='cramped_room')
    parser.add_argument('--agent0', default='nl',
                        choices=['nl', 'lts_ppo', 'rnn_ppo'],
                        help='Agent 0 type')
    parser.add_argument('--agent1', default='nl',
                        choices=['nl', 'lts_ppo', 'rnn_ppo'],
                        help='Agent 1 type')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for actor/critic MLP (default 256)')
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
    parser.add_argument('--critic_mode', default='normal',
                        choices=['normal', 'policy_conditioned'],
                        help='Critic mode: normal (V(obs)) or '
                             'policy_conditioned (V(obs, teammate_probs))')
    parser.add_argument('--teammate_probs_mode', default='true',
                        choices=['true', 'uniform', 'shuffled'],
                        help='Teammate probs mode for policy_conditioned critic: '
                             'true=real probs, uniform=constant [1/6]*6, '
                             'shuffled=real probs permuted before PPO update')
    parser.add_argument('--obs_mode', default='egocentric',
                        choices=['egocentric', 'global_concat', 'local'],
                        help='Observation mode: egocentric (~520-dim per agent), '
                             'global_concat (~1040-dim both agents), '
                             'local (reserved for future)')
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
    parser.add_argument('--w_onion_pickup', type=float, default=0.1,
                        help='Task weight: onion pickup (default 0.1)')
    parser.add_argument('--w_potting', type=float, default=1.0,
                        help='Task weight: potting onion (default 1.0)')
    parser.add_argument('--w_dish_pickup', type=float, default=0.2,
                        help='Task weight: dish pickup (default 0.2)')
    parser.add_argument('--w_soup_pickup', type=float, default=0.7,
                        help='Task weight: soup pickup (default 0.7)')
    parser.add_argument('--w_delivery', type=float, default=1.0,
                        help='Task weight: soup delivery (default 1.0)')
    parser.add_argument('--aux_task_loss', action='store_true', default=False,
                        help='Enable episode-level task auxiliary actor loss for IPPO')
    parser.add_argument('--aux_task_type', default='self_task',
                        choices=['self_task', 'team_task', 'teammate_task'],
                        help='Aux task score type (default self_task)')
    parser.add_argument('--aux_coef', type=float, default=0.0,
                        help='Auxiliary actor loss coefficient (default 0.0)')
    parser.add_argument('--aux_norm_window', type=int, default=100,
                        help='Running normalization window for aux scores (default 100)')
    parser.add_argument('--aux_adv_clip', type=float, default=5.0,
                        help='Max absolute normalized aux advantage (default 5.0)')
    parser.add_argument('--save_trajectories', action='store_true', default=False,
                        help='Save per-timestep trajectory data as npz for '
                             'Phase B future-predictor training')

    # LTS-PPO arguments
    lts_group = parser.add_argument_group('LTS-PPO')
    lts_group.add_argument('--lts_preset', default=None,
                           choices=['small', 'base', 'no_inter'],
                           help='LTS-PPO preset: small (debug), base, no_inter (ablation)')
    lts_group.add_argument('--lts_L', type=int, default=20,
                           help='Intra-history length (default 20)')
    lts_group.add_argument('--lts_M', type=int, default=10,
                           help='Inter-memory episode count (default 10)')
    lts_group.add_argument('--lts_K', type=int, default=10,
                           help='D_f lookahead steps (default 10)')
    lts_group.add_argument('--lts_belief_dim', type=int, default=64,
                           help='Belief embedding dimension (default 64)')
    lts_group.add_argument('--lts_y_dim', type=int, default=16,
                           help='Teammate observable state dimension (default 16)')
    lts_group.add_argument('--lts_intra_feat_dim', type=int, default=23,
                           help='Intra-history feature dimension (default 23)')
    lts_group.add_argument('--lts_future_dim', type=int, default=None,
                           help='D_f output dimension (auto: 6 if no df_event, 11 with df_event)')
    lts_group.add_argument('--lts_c_dim', type=int, default=17,
                           help='Episode characteristic dimension (default 17)')
    lts_group.add_argument('--lts_enc_out_dim', type=int, default=64,
                           help='Encoder output dimension (default 64)')
    lts_group.add_argument('--lts_obs_enc_hidden', type=int, default=128,
                           help='ObsEncoder hidden dim (default 128)')
    lts_group.add_argument('--lts_intra_hidden', type=int, default=64,
                           help='IntraEncoder GRU hidden dim (default 64)')
    lts_group.add_argument('--lts_inter_hidden', type=int, default=64,
                           help='InterEncoder MLP hidden dim (default 64)')
    lts_group.add_argument('--lts_alpha', type=float, default=0.1,
                           help='D_y loss weight (default 0.1)')
    lts_group.add_argument('--lts_beta', type=float, default=0.05,
                           help='D_f loss weight (default 0.05)')
    lts_group.add_argument('--lts_eta', type=float, default=0.05,
                           help='D_c loss weight (default 0.05)')
    lts_group.add_argument('--lts_belief_cons_coef', type=float, default=0.0,
                           help='Belief consistency reg coefficient (default 0.0)')
    lts_group.add_argument('--lts_belief_lr_scale', type=float, default=0.5,
                           help='Belief encoder LR scale relative to AC (default 0.5)')
    lts_group.add_argument('--lts_df_event', action='store_true', default=False,
                           help='Enable event prediction in D_f head')

    # RNN-IPPO arguments
    rnn_group = parser.add_argument_group('RNN-IPPO')
    rnn_group.add_argument('--rnn_K', type=int, default=10,
                           help='RNN-IPPO history length (default 10)')
    rnn_group.add_argument('--rnn_hidden_dim', type=int, default=64,
                           help='RNN-IPPO GRU hidden dim (default 64)')

    # Fixed teammate
    parser.add_argument('--fixed_teammate', default=None,
                        help='Path to fixed teammate checkpoint for agent1')
    parser.add_argument('--warmup_inter_memory_episodes', type=int, default=0,
                        help='Warmup episodes for inter-memory before training '
                             '(fixed-teammate mode, default 0)')
    parser.add_argument('--save_model_dir', default=None,
                        help='Directory to save model checkpoints after training')

    args = parser.parse_args()

    # ── Resolve LTS preset ──
    preset = resolve_lts_preset(args.lts_preset)
    lts_kwargs = {
        'L': preset.get('L', args.lts_L),
        'M': preset.get('M', args.lts_M),
        'K': preset.get('K', args.lts_K),
        'belief_dim': preset.get('belief_dim', args.lts_belief_dim),
        'y_dim': args.lts_y_dim,
        'intra_feat_dim': args.lts_intra_feat_dim,
        'future_dim': args.lts_future_dim,
        'c_dim': args.lts_c_dim,
        'enc_out_dim': preset.get('enc_out_dim', args.lts_enc_out_dim),
        'obs_enc_hidden': preset.get('obs_enc_hidden', args.lts_obs_enc_hidden),
        'intra_hidden': preset.get('intra_hidden', args.lts_intra_hidden),
        'inter_hidden': preset.get('inter_hidden', args.lts_inter_hidden),
        'alpha': preset.get('alpha', args.lts_alpha),
        'beta': preset.get('beta', args.lts_beta),
        'eta': preset.get('eta', args.lts_eta),
        'belief_cons_coef': preset.get('belief_cons_coef', args.lts_belief_cons_coef),
        'belief_lr_scale': args.lts_belief_lr_scale,
        'df_event': args.lts_df_event,
    }
    # Preset hidden_dim only applies to LTS agents; NL/RNN agents use args.hidden_dim
    lts_ac_hidden_dim = preset.get('hidden_dim', args.hidden_dim)
    nl_hidden_dim = args.hidden_dim

    # ── Resolve shaping_type, bonus_clip (backward compat) ──
    shaping_type = args.shaping_type
    bonus_clip = args.bonus_clip

    # --role_shaping flag implies raw_clipped if no explicit shaping_type
    if args.role_shaping and shaping_type == 'none':
        shaping_type = 'raw_clipped'

    # --role_bonus_type overrides shaping_type (deprecated)
    if args.role_bonus_type is not None:
        shaping_type = args.role_bonus_type

    # --role_bonus_clip overrides --bonus_clip (deprecated)
    if args.role_bonus_clip is not None:
        bonus_clip = args.role_bonus_clip

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
        hidden_dim=lts_ac_hidden_dim,
        nl_hidden_dim=nl_hidden_dim,
        reward_threshold=args.reward_threshold,
        spec_threshold=args.spec_threshold,
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
        aux_task_loss=args.aux_task_loss,
        aux_task_type=args.aux_task_type,
        aux_coef=args.aux_coef,
        aux_norm_window=args.aux_norm_window,
        aux_adv_clip=args.aux_adv_clip,
        critic_mode=args.critic_mode,
        teammate_probs_mode=args.teammate_probs_mode,
        save_trajectories=args.save_trajectories,
        # LTS-PPO
        **lts_kwargs,
        # RNN-IPPO
        rnn_K=args.rnn_K,
        rnn_hidden_dim=args.rnn_hidden_dim,
        # Fixed teammate
        fixed_teammate=args.fixed_teammate,
        warmup_inter_memory_episodes=args.warmup_inter_memory_episodes,
        save_model_dir=args.save_model_dir,
    )


if __name__ == '__main__':
    main()
