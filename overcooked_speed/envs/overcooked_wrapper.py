"""Overcooked environment wrapper with unified interface.

Supports standard overcooked_ai layouts, custom layouts, and optional
reward shaping for intermediate milestones.
"""
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from overcooked_speed.envs import get_layout_spec


class OvercookedWrapper:
    """Wraps OvercookedEnv with agent-centric observation API."""

    ACTION_SPACE = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), 'interact']
    NUM_ACTIONS = len(ACTION_SPACE)
    ACTION_NAMES = ['up', 'down', 'right', 'left', 'stay', 'interact']

    def __init__(self, layout_name='cramped_room', horizon=400,
                 reward_shaping=False, obs_mode='egocentric'):
        if obs_mode not in ('egocentric', 'global_concat', 'local'):
            raise ValueError(
                f"Unknown obs_mode '{obs_mode}'. "
                f"Supported: egocentric, global_concat, local")
        is_custom, spec = get_layout_spec(layout_name)
        if is_custom:
            mdp = OvercookedGridworld.from_grid(spec['grid'], base_layout_params={
                'start_all_orders': spec.get('start_all_orders', []),
                'start_bonus_orders': spec.get('start_bonus_orders', []),
                'rew_shaping_params': spec.get('rew_shaping_params', None),
            })
        else:
            mdp = OvercookedGridworld.from_layout_name(layout_name)

        self._env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
        self.layout_name = layout_name
        self.horizon = horizon
        self.num_players = 2
        self.reward_shaping = reward_shaping
        self.obs_mode = obs_mode

        # Track held objects + game_stats deltas for reward shaping
        self._prev_held = {0: None, 1: None}
        self._prev_pot_counts = {0: 0, 1: 0}

    def reset(self):
        self._env.reset()
        self._prev_held = {0: None, 1: None}
        self._prev_pot_counts = {0: 0, 1: 0}
        return self.get_obs(0), self.get_obs(1)

    def step(self, joint_action):
        act0 = self.ACTION_SPACE[int(joint_action[0])]
        act1 = self.ACTION_SPACE[int(joint_action[1])]

        next_state, sparse_r, done, _ = self._env.step((act0, act1))

        shaped_r = sparse_r
        if self.reward_shaping:
            shaped_r = sparse_r + self._compute_shaped_reward()

        obs0 = self.get_obs(0)
        obs1 = self.get_obs(1)

        info = self.get_state_info()
        info['reward'] = float(shaped_r)
        info['sparse_reward'] = float(sparse_r)

        return (obs0, obs1), shaped_r, done, info

    def _compute_shaped_reward(self):
        """Compute shaping reward from held-object transitions + game_stats.

        +2  place onion in pot (verified via game_stats['potting_onion'] delta)
        +3  pickup soup (from pot)
        Total shaped per soup cycle = 3×2 + 3 = 9 vs 20 for delivery.
        """
        shaping = 0.0
        s = self._env.state
        gs = self._env.game_stats

        for agent_id in (0, 1):
            held = s.players[agent_id].held_object
            prev = self._prev_held[agent_id]

            # Soup pickup: None → soup (held-object transition)
            if prev is None and held is not None:
                obj_name = str(held).lower()
                if 'soup' in obj_name:
                    shaping += 3.0

            # Pot placement: verified via game_stats delta (not held-object,
            # because dropping onion on counter looks the same)
            pot_list = gs.get('potting_onion', [[], []])[agent_id]
            new_pots = len(pot_list) - self._prev_pot_counts[agent_id]
            shaping += 2.0 * new_pots
            self._prev_pot_counts[agent_id] = len(pot_list)

            self._prev_held[agent_id] = held

        return shaping

    def get_obs(self, agent_id):
        """Return agent-centric observation based on obs_mode.

        egocentric:   enc[agent_id].flatten()  — agent-specific ~520-dim
        global_concat: np.array(enc).flatten() — both agents see same ~1040-dim
        local:        reserved for future partial-observation work
        """
        enc = self._env.lossless_state_encoding_mdp(self._env.state)
        if self.obs_mode == "egocentric":
            return np.array(enc[agent_id], dtype=np.float32).flatten()
        elif self.obs_mode == "global_concat":
            return np.array(enc, dtype=np.float32).flatten()
        elif self.obs_mode == "local":
            raise NotImplementedError(
                "local observation mode is reserved for future work")
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")

    def get_global_obs(self):
        """Full global state encoding for centralized critic (MAPPO).

        Always returns the concatenated dual-perspective encoding regardless of
        obs_mode — this ensures the centralized critic sees the full state even
        when actors are limited to egocentric views.
        """
        enc = self._env.lossless_state_encoding_mdp(self._env.state)
        return np.array(enc, dtype=np.float32).flatten()

    def get_state_info(self):
        s = self._env.state
        p0 = s.players[0]
        p1 = s.players[1]
        return {
            'timestep': s.timestep,
            'player_0_pos': p0.position,
            'player_0_orient': p0.orientation,
            'player_0_held': p0.held_object,
            'player_1_pos': p1.position,
            'player_1_orient': p1.orientation,
            'player_1_held': p1.held_object,
        }

    def get_game_stats(self):
        return self._env.game_stats

    @property
    def obs_dim(self):
        """Actor observation dimension for the current obs_mode."""
        s = self._env.state
        if s is None:
            self._env.reset()
            s = self._env.state
        return self.get_obs(0).shape[0]

    @property
    def global_obs_dim(self):
        """Full global state dimension (always concatenated, for MAPPO critic)."""
        s = self._env.state
        if s is None:
            self._env.reset()
            s = self._env.state
        enc = self._env.lossless_state_encoding_mdp(s)
        return len(np.array(enc, dtype=np.float32).flatten())
