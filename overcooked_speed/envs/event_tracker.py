"""Event tracker for Overcooked episodes.

Tracks per-agent behavioral events within each episode via state diffing
and accumulated game statistics from OvercookedEnv.

Detected events:
    pickup_onion / pickup_dish / pickup_soup
    place_onion_in_pot (potting)
    deliver_soup
    stay / invalid_action / blocked
    action distribution (up/down/right/left/stay/interact counts)
    total_actions
"""
import numpy as np


class EventTracker:
    """Tracks per-agent events across an episode using state diffs."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_events = {
            0: self._empty_counts(),
            1: self._empty_counts(),
        }
        self._prev_held = {0: None, 1: None}
        self._prev_pos = {0: None, 1: None}

    @staticmethod
    def _empty_counts():
        return {
            'pickup_onion': 0,
            'pickup_dish': 0,
            'pickup_soup': 0,
            'place_onion_in_pot': 0,
            'deliver_soup': 0,
            'stay': 0,
            'invalid_action': 0,
            'blocked': 0,
            'total_actions': 0,
            # Action distribution (index→count)
            'action_up': 0,
            'action_down': 0,
            'action_right': 0,
            'action_left': 0,
            'action_stay': 0,
            'action_interact': 0,
        }

    def step(self, joint_action, state_info):
        """Record events for one step."""
        for agent_id in (0, 1):
            counts = self.episode_events[agent_id]
            counts['total_actions'] += 1

            action_idx = joint_action[agent_id]
            held = state_info[f'player_{agent_id}_held']
            pos = state_info[f'player_{agent_id}_pos']

            # Action distribution
            action_names = ['action_up', 'action_down', 'action_right',
                            'action_left', 'action_stay', 'action_interact']
            if 0 <= action_idx < len(action_names):
                counts[action_names[action_idx]] += 1

            prev_held = self._prev_held[agent_id]
            prev_pos = self._prev_pos[agent_id]

            # Object pickup: None → something
            if prev_held is None and held is not None:
                obj_name = str(held).lower()
                if 'onion' in obj_name:
                    counts['pickup_onion'] += 1
                elif 'dish' in obj_name:
                    counts['pickup_dish'] += 1
                elif 'soup' in obj_name:
                    counts['pickup_soup'] += 1

            # Stay detection
            if action_idx == 4:
                counts['stay'] += 1

            # Blocked detection: agent tried to move but position didn't change
            if action_idx < 4:
                if prev_pos is not None and pos == prev_pos:
                    counts['blocked'] += 1

            self._prev_held[agent_id] = held
            self._prev_pos[agent_id] = pos

    def post_process(self, game_stats, total_reward):
        """Finalize event counts from game_stats."""
        for agent_id in (0, 1):
            counts = self.episode_events[agent_id]

            n_potting = len(game_stats.get('potting_onion', [[], []])[agent_id])
            counts['place_onion_in_pot'] += n_potting

            n_delivery = len(game_stats.get('soup_delivery', [[], []])[agent_id])
            counts['deliver_soup'] += n_delivery

    def get_counts(self, agent_id):
        return self.episode_events[agent_id].copy()

    def get_all_counts(self):
        return {
            0: self.episode_events[0].copy(),
            1: self.episode_events[1].copy(),
        }
