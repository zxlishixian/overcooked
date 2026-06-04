from .pg_agent import PGAgent
from .policy import ActorCritic
from .mappo_agent import MAPPOManager, CentralizedCritic
from .role_shaping import RoleShapingManager
from .belief_ppo_agent import BeliefPPOAgent
from .rnn_agent import RNNAgent

AGENT_REGISTRY = {
    'nl': PGAgent,
    'belief_ppo': BeliefPPOAgent,
    'rnn_ppo': RNNAgent,
}


def create_agent(agent_type, agent_id, *args, **kwargs):
    """Factory: create an agent by type string.

    Args:
        agent_type: 'nl', 'belief_ppo', 'rnn_ppo'
        agent_id: 0 or 1
        *args, **kwargs: passed to agent constructor

    Returns PGAgent (or subclass) instance.
    """
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type '{agent_type}'. "
                         f"Available: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[agent_type](agent_id, *args, **kwargs)
