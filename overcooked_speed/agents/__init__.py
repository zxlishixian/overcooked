from .pg_agent import PGAgent
from .policy import ActorCritic
from .mappo_agent import MAPPOManager, CentralizedCritic
from .role_shaping import RoleShapingManager

# Future: from .lola_agent import LOLAAgent
# Future: from .lookahead_agent import LookaheadAgent
# Future: from .ideal_jpi_agent import IdealJpiAgent
# Future: from .non_agent import NonAgent

AGENT_REGISTRY = {
    'nl': PGAgent,
    # 'non': NonAgent,        # TODO: fixed-policy baseline
    # 'lola': LOLAAgent,       # TODO: second-order LOLA
    # 'lookahead': LookaheadAgent,  # TODO: second-order Lookahead
    # 'ideal_jpi': IdealJpiAgent,   # TODO: exact analytical Jπ
}


def create_agent(agent_type, agent_id, *args, **kwargs):
    """Factory: create an agent by type string.

    Args:
        agent_type: 'nl' (currently only NL supported)
        agent_id: 0 or 1
        *args, **kwargs: passed to agent constructor

    Returns PGAgent (or future agent) instance.
    """
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type '{agent_type}'. "
                         f"Available: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[agent_type](agent_id, *args, **kwargs)
