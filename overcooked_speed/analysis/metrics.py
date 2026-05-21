"""Convergence speed and role specialization metrics.

All metrics operate on per-episode data arrays (reward, event counts).
"""
import numpy as np


def moving_average(x, window=10):
    """Simple moving average of 1D array x."""
    if len(x) < window:
        return np.array([np.mean(x)] * len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def final_reward(rewards, last_n=50):
    """Mean reward over the last N episodes."""
    n = min(last_n, len(rewards))
    return float(np.mean(rewards[-n:]))


def reward_auc(rewards):
    """Area under the reward curve: sum of per-episode rewards."""
    return float(np.sum(rewards))


def moving_avg_reward(rewards, window=10):
    """Return moving-average-smoothed reward array."""
    return moving_average(rewards, window)


def T_reward(rewards, threshold, window=10, stable_K=20):
    """First episode where moving-average reward reaches threshold and stays.

    Returns (convergence_episode, converged_bool).
    If never reached, returns (total_episodes, False).
    """
    ma = moving_average(rewards, window)
    if len(ma) < stable_K:
        return len(rewards), False
    for i in range(len(ma) - stable_K):
        if np.all(ma[i:i + stable_K] >= threshold):
            return i + window, True
    return len(rewards), False


def delivery_specialization(delivery_counts):
    """S_delivery = |c0 - c1| / (c0 + c1 + eps)."""
    c0, c1 = delivery_counts[0], delivery_counts[1]
    denom = c0 + c1 + 1e-8
    return abs(c0 - c1) / denom


def cooking_specialization(cooking_counts):
    """S_cooking = |c0 - c1| / (c0 + c1 + eps)."""
    c0, c1 = cooking_counts[0], cooking_counts[1]
    denom = c0 + c1 + 1e-8
    return abs(c0 - c1) / denom


def overall_specialization(delivery_counts, cooking_counts):
    """S_overall = 0.5 * (S_delivery + S_cooking)."""
    return 0.5 * (delivery_specialization(delivery_counts) +
                  cooking_specialization(cooking_counts))


# ── Gated specialization (with minimum event thresholds) ──────────────────

def gated_delivery_specialization(delivery_counts, min_events=1):
    """S_delivery, or NaN if total deliveries < min_events."""
    c0, c1 = delivery_counts[0], delivery_counts[1]
    if c0 + c1 < min_events:
        return float('nan')
    return delivery_specialization(delivery_counts)


def gated_cooking_specialization(cooking_counts, min_events=3):
    """S_cooking, or NaN if total cooking events < min_events."""
    c0, c1 = cooking_counts[0], cooking_counts[1]
    if c0 + c1 < min_events:
        return float('nan')
    return cooking_specialization(cooking_counts)


def gated_overall_specialization(delivery_counts, cooking_counts,
                                  min_delivery=1, min_cooking=3):
    """S_overall if BOTH individual specializations are valid, else NaN."""
    s_del = gated_delivery_specialization(delivery_counts, min_delivery)
    s_cook = gated_cooking_specialization(cooking_counts, min_cooking)
    if np.isnan(s_del) or np.isnan(s_cook):
        return float('nan')
    return 0.5 * (s_del + s_cook)


def invalid_action_rate(event_counts, agent_id):
    """invalid_count / total_actions."""
    c = event_counts[agent_id]
    return c['invalid_action'] / max(c['total_actions'], 1)


def blocking_rate(event_counts, agent_id):
    """blocked / total_actions."""
    c = event_counts[agent_id]
    return c['blocked'] / max(c['total_actions'], 1)


def idle_rate(event_counts, agent_id):
    """stay / total_actions."""
    c = event_counts[agent_id]
    return c['stay'] / max(c['total_actions'], 1)


def T_specialization(spec_values, threshold, window=10, stable_K=20):
    """First episode where specialization >= threshold and stays.

    Returns (convergence_episode, converged_bool).
    """
    ma = moving_average(spec_values, window)
    if len(ma) < stable_K:
        return len(spec_values), False
    for i in range(len(ma) - stable_K):
        if np.all(ma[i:i + stable_K] >= threshold):
            return i + window, True
    return len(spec_values), False


def compute_all_metrics(episode_rewards, episode_event_counts,
                         min_delivery_events=1, min_cooking_events=3):
    """Compute full metrics dict from episode data.

    Args:
        episode_rewards: list/array of per-episode total reward (scalar per ep)
        episode_event_counts: list of {0: counts, 1: counts} per episode
        min_delivery_events: gating threshold for delivery specialization
        min_cooking_events: gating threshold for cooking specialization

    Returns dict with all standard metrics.
    """
    rewards = np.array(episode_rewards)
    n_eps = len(rewards)

    # Delivery and cooking counts summed across all episodes
    delivery_0 = sum(e[0].get('deliver_soup', 0) for e in episode_event_counts)
    delivery_1 = sum(e[1].get('deliver_soup', 0) for e in episode_event_counts)
    cooking_0 = sum(e[0].get('pickup_onion', 0) + e[0].get('place_onion_in_pot', 0) for e in episode_event_counts)
    cooking_1 = sum(e[1].get('pickup_onion', 0) + e[1].get('place_onion_in_pot', 0) for e in episode_event_counts)

    # Raw specialization (always computed)
    s_del_raw = delivery_specialization([delivery_0, delivery_1])
    s_cook_raw = cooking_specialization([cooking_0, cooking_1])
    s_overall_raw = 0.5 * (s_del_raw + s_cook_raw)

    # Gated specialization
    s_del_gated = gated_delivery_specialization([delivery_0, delivery_1], min_delivery_events)
    s_cook_gated = gated_cooking_specialization([cooking_0, cooking_1], min_cooking_events)
    s_overall_gated = gated_overall_specialization(
        [delivery_0, delivery_1], [cooking_0, cooking_1],
        min_delivery_events, min_cooking_events)

    # Per-episode specialization trajectory (raw + gated)
    raw_traj = np.zeros(n_eps)
    gated_traj = np.full(n_eps, np.nan)
    for i in range(n_eps):
        evt = episode_event_counts[i]
        d0 = evt[0].get('deliver_soup', 0)
        d1 = evt[1].get('deliver_soup', 0)
        c0 = evt[0].get('pickup_onion', 0) + evt[0].get('place_onion_in_pot', 0)
        c1 = evt[1].get('pickup_onion', 0) + evt[1].get('place_onion_in_pot', 0)
        raw_traj[i] = overall_specialization([d0, d1], [c0, c1])
        gated_traj[i] = gated_overall_specialization(
            [d0, d1], [c0, c1], min_delivery_events, min_cooking_events)

    t_reward_ep, t_reward_ok = T_reward(rewards, threshold=20.0)
    t_spec_ep, t_spec_ok = T_specialization(raw_traj, threshold=0.3)

    # Use gated trajectory for T_specialization if enough valid points
    valid_gated = gated_traj[~np.isnan(gated_traj)]
    if len(valid_gated) >= 10:
        t_spec_gated_ep, t_spec_gated_ok = T_specialization(valid_gated, threshold=0.3)
    else:
        t_spec_gated_ep, t_spec_gated_ok = len(rewards), False

    final_r = final_reward(rewards)
    r_auc = reward_auc(rewards)

    return {
        'final_reward': final_r,
        'reward_auc': r_auc,
        'T_reward': int(t_reward_ep),
        'T_reward_ok': t_reward_ok,
        # Raw specialization (always computed, fragile at low event counts)
        'T_specialization': int(t_spec_ep),
        'T_specialization_ok': t_spec_ok,
        'final_specialization': float(s_overall_raw),
        'final_sdelivery': float(s_del_raw),
        'final_scooking': float(s_cook_raw),
        # Gated specialization
        'final_specialization_gated': (float(s_overall_gated)
                                        if not np.isnan(s_overall_gated) else None),
        'final_sdelivery_gated': (float(s_del_gated)
                                   if not np.isnan(s_del_gated) else None),
        'final_scooking_gated': (float(s_cook_gated)
                                  if not np.isnan(s_cook_gated) else None),
        'T_specialization_gated': int(t_spec_gated_ep),
        'T_specialization_gated_ok': t_spec_gated_ok,
        # Event totals at final episode
        'total_deliveries_0': int(delivery_0),
        'total_deliveries_1': int(delivery_1),
        'total_cooking_0': int(cooking_0),
        'total_cooking_1': int(cooking_1),
        # Trajectories
        'spec_trajectory': raw_traj.tolist(),
        'spec_trajectory_gated': [float(x) if not np.isnan(x) else None
                                   for x in gated_traj],
    }
