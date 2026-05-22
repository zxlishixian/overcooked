"""Unified shaping manager for mechanism diagnosis.

Shaping types:
  none                        - No shaping (IPPO baseline)
  raw_clipped                 - Original role complementarity (kept for ablation)
  normalized / weighted_normalized / delta_complementarity — role-based variants
  constant_bonus              - Fixed 1.0 bonus every episode
  event_density_bonus         - Useful events / total_actions
  event_binary_bonus          - 1.0 if any useful event occurred
  delivery_chain_bonus        - Weighted task-progress, normalized
  delivery_chain_raw_clipped  - Weighted task-progress, raw clipped

No second-order gradients. No neural teammate models.
"""
import numpy as np

# Types that use teammate role history
_ROLE_TYPES = {'raw_clipped', 'normalized', 'weighted_normalized',
               'delta_complementarity'}

# All valid shaping types
VALID_SHAPING_TYPES = _ROLE_TYPES | {
    'none', 'constant_bonus', 'event_density_bonus', 'event_binary_bonus',
    'delivery_chain_bonus', 'delivery_chain_raw_clipped',
}


class RoleShapingManager:
    """Unified shaping manager supporting role-based and diagnostic types."""

    def __init__(self, window=20, bonus_clip=1.0, shaping_type='none',
                 lambda_role=0.1):
        self.window = window
        self.bonus_clip = bonus_clip
        self.shaping_type = shaping_type
        self.lambda_role = lambda_role

        # Role history (only populated for _ROLE_TYPES)
        self._history = []
        self._complementarity_history = []

        # Delta complementarity cache
        self._cached_delta_bonus = None
        self._delta_counts0 = None
        self._delta_counts1 = None

    @property
    def is_role_type(self):
        return self.shaping_type in _ROLE_TYPES

    # ── Public API ──

    def compute_bonus(self, agent_id, counts):
        """Return (raw_bonus, applied_bonus) for one agent."""
        if self.shaping_type == 'none':
            return 0.0, 0.0
        raw = self._compute_raw(agent_id, counts)
        applied = float(np.clip(raw, -self.bonus_clip, self.bonus_clip))
        return float(raw), applied

    def record_episode(self, counts0, counts1):
        """Store episode counts for teammate-tendency history."""
        if not self.is_role_type:
            return

        cook0 = (counts0.get('pickup_onion', 0) +
                 counts0.get('place_onion_in_pot', 0))
        deliver0 = (counts0.get('pickup_dish', 0) +
                    counts0.get('pickup_soup', 0) +
                    counts0.get('deliver_soup', 0))
        cook1 = (counts1.get('pickup_onion', 0) +
                 counts1.get('place_onion_in_pot', 0))
        deliver1 = (counts1.get('pickup_dish', 0) +
                    counts1.get('pickup_soup', 0) +
                    counts1.get('deliver_soup', 0))

        self._history.append((cook0, deliver0, cook1, deliver1))
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]

        eps = 1e-8
        total0 = cook0 + deliver0 + eps
        total1 = cook1 + deliver1 + eps
        C = ((cook0 / total0) * (deliver1 / total1) +
             (deliver0 / total0) * (cook1 / total1))
        self._complementarity_history.append(C)
        if len(self._complementarity_history) > self.window:
            self._complementarity_history = self._complementarity_history[-self.window:]

    def get_teammate_tendency(self, agent_id):
        """Return (p_cook, p_deliver) for the teammate from previous episodes."""
        if not self.is_role_type or len(self._history) == 0:
            return 0.5, 0.5
        teammate = 1 - agent_id
        total_cook = sum(h[2 * teammate] for h in self._history)
        total_deliver = sum(h[2 * teammate + 1] for h in self._history)
        total = total_cook + total_deliver
        if total == 0:
            return 0.5, 0.5
        return total_cook / total, total_deliver / total

    def get_complementarity(self, counts0, counts1):
        """Compute current complementarity from two agents' counts."""
        return _compute_complementarity(counts0, counts1)

    def set_delta_counts(self, counts0, counts1):
        """Store both agents' counts before computing delta bonus."""
        self._delta_counts0 = counts0
        self._delta_counts1 = counts1

    # ── Backward-compatible wrappers ──

    def compute_role_bonus(self, agent_id, counts):
        """Return applied bonus (after clip). Legacy API."""
        _, applied = self.compute_bonus(agent_id, counts)
        return applied

    def get_raw_bonus(self, agent_id, counts):
        """Return raw bonus (before clip)."""
        raw, _ = self.compute_bonus(agent_id, counts)
        return raw

    # ── Internal dispatch ──

    def _compute_raw(self, agent_id, counts):
        dispatch = {
            'raw_clipped': self._raw_clipped,
            'normalized': self._normalized,
            'weighted_normalized': self._weighted_normalized,
            'delta_complementarity': self._delta_complementarity,
            'constant_bonus': lambda a, c: 1.0,
            'event_density_bonus': self._event_density,
            'event_binary_bonus': self._event_binary,
            'delivery_chain_bonus': self._delivery_chain_normalized,
            'delivery_chain_raw_clipped': self._delivery_chain_raw,
        }
        return dispatch[self.shaping_type](agent_id, counts)

    # ── Role-based raw computations ──

    def _raw_clipped(self, agent_id, counts):
        p_cook, p_deliver = self.get_teammate_tendency(agent_id)
        cooking = counts.get('pickup_onion', 0) + counts.get('place_onion_in_pot', 0)
        delivery = (counts.get('pickup_dish', 0) + counts.get('pickup_soup', 0) +
                    counts.get('deliver_soup', 0))
        return p_cook * delivery + p_deliver * cooking

    def _normalized(self, agent_id, counts):
        p_cook, p_deliver = self.get_teammate_tendency(agent_id)
        cooking = counts.get('pickup_onion', 0) + counts.get('place_onion_in_pot', 0)
        delivery = (counts.get('pickup_dish', 0) + counts.get('pickup_soup', 0) +
                    counts.get('deliver_soup', 0))
        denom = cooking + delivery + 1e-8
        return (p_cook * delivery + p_deliver * cooking) / denom

    def _weighted_normalized(self, agent_id, counts):
        p_cook, p_deliver = self.get_teammate_tendency(agent_id)
        w_cook = (0.2 * counts.get('pickup_onion', 0) +
                  1.0 * counts.get('place_onion_in_pot', 0))
        w_deliver = (0.2 * counts.get('pickup_dish', 0) +
                     0.5 * counts.get('pickup_soup', 0) +
                     1.0 * counts.get('deliver_soup', 0))
        denom = w_cook + w_deliver + 1e-8
        return (p_cook * w_deliver + p_deliver * w_cook) / denom

    def _delta_complementarity(self, agent_id, counts):
        if agent_id == 1 and self._cached_delta_bonus is not None:
            bonus = self._cached_delta_bonus
            self._cached_delta_bonus = None
            return bonus

        if (agent_id == 0 and self._delta_counts0 is not None and
                self._delta_counts1 is not None):
            C_current = _compute_complementarity(
                self._delta_counts0, self._delta_counts1)
            if len(self._complementarity_history) == 0:
                C_window = 0.5
            else:
                C_window = np.mean(self._complementarity_history)
            bonus = C_current - C_window
            self._cached_delta_bonus = bonus
            return bonus
        return 0.0

    # ── Diagnostic raw computations ──

    @staticmethod
    def _useful_events(counts):
        return (counts.get('pickup_onion', 0) +
                counts.get('place_onion_in_pot', 0) +
                counts.get('pickup_dish', 0) +
                counts.get('pickup_soup', 0) +
                counts.get('deliver_soup', 0))

    @staticmethod
    def _weighted_chain(counts):
        return (0.1 * counts.get('pickup_onion', 0) +
                0.5 * counts.get('place_onion_in_pot', 0) +
                0.2 * counts.get('pickup_dish', 0) +
                0.7 * counts.get('pickup_soup', 0) +
                1.0 * counts.get('deliver_soup', 0))

    def _event_density(self, agent_id, counts):
        useful = self._useful_events(counts)
        total = counts.get('total_actions', 400)
        return useful / (total + 1e-8)

    def _event_binary(self, agent_id, counts):
        return 1.0 if self._useful_events(counts) > 0 else 0.0

    def _delivery_chain_normalized(self, agent_id, counts):
        raw = self._weighted_chain(counts)
        total = counts.get('total_actions', 400)
        return raw / (total + 1e-8)

    def _delivery_chain_raw(self, agent_id, counts):
        return self._weighted_chain(counts)


def _compute_complementarity(counts0, counts1):
    """Compute role complementarity C from two agents' counts."""
    eps = 1e-8
    cook0 = counts0.get('pickup_onion', 0) + counts0.get('place_onion_in_pot', 0)
    deliver0 = (counts0.get('pickup_dish', 0) + counts0.get('pickup_soup', 0) +
                counts0.get('deliver_soup', 0))
    cook1 = counts1.get('pickup_onion', 0) + counts1.get('place_onion_in_pot', 0)
    deliver1 = (counts1.get('pickup_dish', 0) + counts1.get('pickup_soup', 0) +
                counts1.get('deliver_soup', 0))
    total0 = cook0 + deliver0 + eps
    total1 = cook1 + deliver1 + eps
    p0_cook = cook0 / total0
    p0_deliver = deliver0 / total0
    p1_cook = cook1 / total1
    p1_deliver = deliver1 / total1
    return p0_cook * p1_deliver + p0_deliver * p1_cook
