# Overcooked: Multi-Agent Convergence Speed Research

Cooperative multi-agent reinforcement learning framework based on [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai), with a research module for studying training convergence speed and role specialization emergence.

## Structure

```
overcooked/
├── overcooked_ai_py/       # Environment: Overcooked MDP, agents, planning, visualization
├── overcooked_speed/       # Research framework
│   ├── agents/             # MARL agents (PPO, future: LOLA, Lookahead)
│   ├── envs/               # Environment wrapper, event tracker
│   ├── analysis/           # Metrics and plotting
│   ├── experiments/        # Training scripts (run_pair, sweep_pairs)
│   ├── algos/              # Algorithm implementations (future)
│   └── configs/            # Experiment configurations (future)
├── setup.py
└── pyproject.toml
```

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.8. Core dependency: `torch`, `numpy<2.0.0`, `pygame`, `gymnasium`.

## Quick Start

```bash
# Single training run (PPO agents, 100 episodes)
python overcooked_speed/experiments/run_pair.py \
    --layout cramped_room --agent0 nl --agent1 nl \
    --num_episodes 100 --seed 0 --log_dir logs/demo

# Multi-seed sweep
python overcooked_speed/experiments/sweep_pairs.py \
    --layout cramped_room --pairs nl,nl \
    --num_episodes 500 --seeds 0 1 2 --log_dir logs/sweep

# GPU training
python overcooked_speed/experiments/sweep_pairs.py \
    --layout cramped_room --pairs nl,nl \
    --num_episodes 500 --seeds 0 1 2 --device auto
```

## Agent Types

| Type | Description |
|------|-------------|
| `nl` | PPO naive learner (independent Actor-Critic with GAE) |

## Key Findings

- PPO + reward shaping (pot placement +2, soup pickup +3) enables consistent learning
- NL+NL pairs spontaneously develop role specialization (delivery: 0.90, cooking: 0.72)
- Specialization emerges before reward convergence (T_spec ≈ 47 vs T_reward ≈ 222)
- Role assignment is emergent, not fixed by agent index

## License

MIT — see [LICENSE](LICENSE)
