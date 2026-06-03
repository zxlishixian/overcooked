"""Teammate observable feature extraction for LTS-PPO.

y_{-i}: observable teammate behavior state (y_dim dims, default 16)
intra_feat: per-step input for intra-history encoder (intra_feat_dim dims, default 23)
c_{-i}^e: episode-level teammate characteristic (c_dim dims, default 17)
"""
import numpy as np

ORIENTATION_TO_INDEX = {
    (0, -1): 0,  # up (north in grid coordinates)
    (0, 1):  1,  # down
    (1, 0):  2,  # right
    (-1, 0): 3,  # left
}


def _orientation_to_onehot(orientation, n_orient=4):
    idx = ORIENTATION_TO_INDEX.get(orientation, 0)
    vec = np.zeros(n_orient, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def _held_to_onehot(held_obj, n_held=4):
    """Map held object to one-hot: 0=none, 1=onion, 2=dish, 3=soup."""
    idx = 0
    if held_obj is not None:
        name = getattr(held_obj, 'name', '')
        if name in ('onion', 'tomato'):
            idx = 1
        elif name == 'dish':
            idx = 2
        elif name == 'soup':
            idx = 3
    vec = np.zeros(n_held, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def extract_teammate_y(agent_id, state_info, teammate_action, y_dim=16):
    """Extract teammate-observable features y_{-i}^t.

    Args:
        agent_id: 0 or 1 (ego agent index; teammate is 1 - agent_id)
        state_info: dict from env.get_state_info() with keys:
            player_{tid}_pos, player_{tid}_orient, player_{tid}_held
        teammate_action: teammate's action at current step (int 0-5)
        y_dim: total y dimension (default 16)

    Returns:
        np.ndarray of shape (y_dim,) float32:
            [0:2]   position (x, y) as raw ints
            [2:6]   orientation one-hot (4)
            [6:10]  held object one-hot (4)
            [10:16] teammate action one-hot (6)
    """
    tid = 1 - agent_id
    pos = state_info.get(f'player_{tid}_pos', (0, 0))
    orient = state_info.get(f'player_{tid}_orient', (0, -1))
    held = state_info.get(f'player_{tid}_held', None)

    pos_arr = np.array(pos, dtype=np.float32)
    orient_oh = _orientation_to_onehot(orient)
    held_oh = _held_to_onehot(held)
    action_oh = np.zeros(6, dtype=np.float32)
    if 0 <= teammate_action < 6:
        action_oh[teammate_action] = 1.0

    y = np.concatenate([pos_arr, orient_oh, held_oh, action_oh]).astype(np.float32)
    if len(y) != y_dim:
        raise ValueError(f"teammate_y dim mismatch: expected {y_dim}, got {len(y)}")
    return y


def build_intra_feature(teammate_y, own_action, reward, intra_feat_dim=23):
    """Build per-step intra-history feature vector.

    Args:
        teammate_y: (y_dim,) teammate observable state from previous step
        own_action: ego agent's action at previous step (int 0-5)
        reward: reward at previous step (float)
        intra_feat_dim: total intra feature dimension (default 23)

    Returns:
        np.ndarray of shape (intra_feat_dim,) float32:
            [0:y_dim]         teammate_y
            [y_dim:y_dim+6]   own_action one-hot (6)
            [-1]              reward (1)
    """
    y_dim = len(teammate_y)
    own_action_oh = np.zeros(6, dtype=np.float32)
    if 0 <= own_action < 6:
        own_action_oh[own_action] = 1.0

    feat = np.concatenate([
        teammate_y.astype(np.float32),
        own_action_oh,
        np.array([float(reward)], dtype=np.float32),
    ]).astype(np.float32)

    if len(feat) != intra_feat_dim:
        raise ValueError(f"intra_feat dim mismatch: expected {intra_feat_dim}, got {len(feat)}")
    return feat


def compute_episode_characteristic(teammate_actions, teammate_helds,
                                    teammate_positions, game_stats,
                                    total_actions, c_dim=17,
                                    n_actions=6, n_held=4):
    """Compute episode-level teammate characteristic c_{-i}^e.

    Uses only ego-observable data: tracked teammate actions, held objects,
    positions, and post-episode game_stats (public, not privileged).

    Args:
        teammate_actions: list of ints, teammate actions over episode
        teammate_helds: list of ints, teammate held-object over episode
            (0=none, 1=onion, 2=dish, 3=soup)
        teammate_positions: list of (x, y) tuples over episode
        game_stats: dict from env.get_game_stats() — post-episode public stats
        total_actions: int, total actions in episode (for normalization)
        c_dim: total c dimension (default 17)
        n_actions: number of action types (default 6)
        n_held: number of held-object types (default 4)

    Returns:
        np.ndarray of shape (c_dim,) float32:
            [0:6]    action distribution histogram (normalized)
            [6:11]   event counts normalized by total_actions (5 events)
            [11]     avg distance traveled per step
            [12:16]  avg held-object distribution (4)
            [16]     delivery count normalized
    """
    T = len(teammate_actions)
    if T == 0:
        return np.zeros(c_dim, dtype=np.float32)

    # Action distribution (6 dims)
    action_hist = np.zeros(n_actions, dtype=np.float32)
    for a in teammate_actions:
        if 0 <= a < n_actions:
            action_hist[a] += 1.0
    action_hist /= T

    # Event counts from game_stats (5 dims) — post-episode public stats
    event_keys = ['onion_pickup', 'potting_onion', 'dish_pickup',
                  'soup_pickup', 'soup_delivery']
    event_counts = np.zeros(5, dtype=np.float32)
    for ei, ek in enumerate(event_keys):
        if ek in game_stats:
            # game_stats[ek] is a list of per-agent timestep lists
            # For c_{-i}^e we care about the TEAMMATE's events
            event_counts[ei] = float(len(game_stats[ek][1]) if len(game_stats[ek]) > 1
                                     else len(game_stats[ek][0]))
    if total_actions > 0:
        event_counts /= total_actions

    # Avg distance traveled (1 dim)
    dist = 0.0
    if T > 1 and len(teammate_positions) > 1:
        for t in range(1, min(T, len(teammate_positions))):
            p_prev = teammate_positions[t - 1]
            p_cur = teammate_positions[t]
            dx = p_cur[0] - p_prev[0]
            dy = p_cur[1] - p_prev[1]
            dist += float(abs(dx) + abs(dy))
        dist /= (T - 1)

    # Avg held-object distribution (4 dims)
    held_hist = np.zeros(n_held, dtype=np.float32)
    for h in teammate_helds:
        if 0 <= h < n_held:
            held_hist[h] += 1.0
    held_hist /= max(T, 1)

    # Delivery count normalized (1 dim) — already in event_counts[4]
    delivery_norm = event_counts[4]

    c = np.concatenate([
        action_hist,
        event_counts,
        np.array([dist], dtype=np.float32),
        held_hist,
        np.array([delivery_norm], dtype=np.float32),
    ]).astype(np.float32)

    if len(c) != c_dim:
        raise ValueError(f"c dim mismatch: expected {c_dim}, got {len(c)}")
    return c
