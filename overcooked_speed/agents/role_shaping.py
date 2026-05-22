"""Role-level LOLA-like teammate-aware reward shaping.

Instead of second-order gradients or theta-space LOLA, we use a simple,
interpretable role bonus: estimate teammate's role tendency from recent
history, then reward complementary behavior.

For agent i, given teammate j's recent event history:
    p_j_cook = cook_j / (cook_j + deliver_j + eps)
    p_j_deliver = deliver_j / (cook_j + deliver_j + eps)

    role_bonus_i = p_j_cook * delivery_event_i + p_j_deliver * cooking_event_i

where:
    cooking_event = onion_pickup + place_onion_in_pot
    delivery_event = dish_pickup + soup_pickup + soup_delivery

Early episodes use neutral prior (0.5, 0.5) to avoid spurious shaping.
"""
import numpy as np


class RoleShapingManager:
    """Tracks per-agent role history and computes complementary role bonuses.

    Uses only information from previous episodes — no future leakage.
    """

    def __init__(self, window=20, bonus_clip=1.0):
        self.window = window
        self.bonus_clip = bonus_clip

        # Rolling history: list of (cook_0, deliver_0, cook_1, deliver_1) tuples
        self._history = []

    def record_episode(self, counts0, counts1):
        """Record per-agent event counts from a completed episode."""
        cook0 = counts0.get('pickup_onion', 0) + counts0.get('place_onion_in_pot', 0)
        deliver0 = (counts0.get('pickup_dish', 0) +
                    counts0.get('pickup_soup', 0) +
                    counts0.get('deliver_soup', 0))
        cook1 = counts1.get('pickup_onion', 0) + counts1.get('place_onion_in_pot', 0)
        deliver1 = (counts1.get('pickup_dish', 0) +
                    counts1.get('pickup_soup', 0) +
                    counts1.get('deliver_soup', 0))

        self._history.append((cook0, deliver0, cook1, deliver1))
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]

    def get_teammate_tendency(self, agent_id):
        """Get teammate's role probabilities from recent history.

        Returns (p_cook, p_deliver) for the teammate of agent_id.
        Uses only previous episodes, not the current one.
        Early episodes return neutral prior (0.5, 0.5).
        """
        if len(self._history) == 0:
            return 0.5, 0.5

        # Aggregate over history for the teammate
        teammate = 1 - agent_id
        total_cook = sum(h[2 * teammate] for h in self._history)
        total_deliver = sum(h[2 * teammate + 1] for h in self._history)
        total = total_cook + total_deliver

        if total == 0:
            return 0.5, 0.5

        p_cook = total_cook / total
        p_deliver = total_deliver / total
        return p_cook, p_deliver

    def compute_role_bonus(self, agent_id, counts):
        """Compute role bonus for agent_id from this episode's event counts.

        Uses teammate tendency from *previous* episodes only.
        """
        p_cook, p_deliver = self.get_teammate_tendency(agent_id)

        cooking_event = counts.get('pickup_onion', 0) + counts.get('place_onion_in_pot', 0)
        delivery_event = (counts.get('pickup_dish', 0) +
                          counts.get('pickup_soup', 0) +
                          counts.get('deliver_soup', 0))

        bonus = p_cook * delivery_event + p_deliver * cooking_event
        return np.clip(bonus, -self.bonus_clip, self.bonus_clip)
