# Overcooked-Speed: Multi-Agent Convergence & Role Specialization

A research framework for studying **how different learning algorithm pairs** affect training convergence speed and spontaneous role specialization in cooperative multi-agent reinforcement learning (MARL), built on top of [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai).

## Motivation

In cooperative MARL, agents must learn not just to act, but to **coordinate**. A central open question is: *how does the choice of learning algorithm — for each agent independently — shape convergence speed and the emergence of role specialization?*

For example, with two PPO agents (NL+NL), we observe:
- **Role specialization emerges spontaneously** — one agent becomes the "cook" (fetching onions, filling pots), the other becomes the "deliverer" (picking up soup, serving)
- **Specialization emerges before reward convergence** — T_spec ≈ 47 vs T_reward ≈ 222 (episodes)
- **Role assignment is emergent, not pre-assigned** — which agent takes which role depends on random initialization, not agent index

This framework is designed to systematically compare different algorithm pairings — currently NL+NL (PPO independent learners), with second-order methods (LOLA, Lookahead) planned for future work.

## Key Concepts

### Convergence Speed (T_reward)
The first episode where the moving-average reward reaches a threshold (default: 20) and stays above it for K consecutive episodes. Measured in episodes — **lower is faster**.

### Role Specialization (S)
Based on per-agent event counts. For delivery:

$$S_{delivery} = \frac{|c_0 - c_1|}{c_0 + c_1 + \epsilon}$$

where $c_i$ is agent $i$'s delivery count. $S = 0$ means equal sharing; $S = 1$ means one agent does everything. Cooking specialization $S_{cooking}$ is defined analogously using onion-pickup + pot-placement counts. Overall specialization is the mean: $S_{overall} = (S_{delivery} + S_{cooking}) / 2$.

### Gated Specialization
Raw specialization is fragile when event counts are low (e.g., 0 deliveries → $S = 0/0 \approx 0$, implying "equal" which is misleading). Gated metrics require minimum event thresholds (default: 1 delivery, 3 cooking events) and return NaN if unmet.

### Reward Shaping
The Overcooked sparse reward (+20 per soup delivery) is too sparse for PPO to learn from scratch — the probability of accidentally completing the full pipeline is near zero. We add intermediate shaping rewards verified via game state diffs:

| Event | Reward | Detection |
|-------|--------|-----------|
| Place onion in pot | +2 | `game_stats['potting_onion']` delta (MDP-verified) |
| Pick up soup from pot | +3 | held-object transition (`None → soup`) |
| **Total shaped per soup** | **9** | vs. **20** sparse delivery reward |

The shaping uses MDP-verified game stats to prevent reward exploitation — early versions that rewarded onion-pickup transitions directly were gamed by agents spamming interact at the dispenser.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Experiments                          │
│              run_pair.py  /  sweep_pairs.py              │
├─────────────────────────────────────────────────────────┤
│  Agent 0 (PGAgent/PPO)      Agent 1 (PGAgent/PPO)       │
│  ┌─────────────────────┐   ┌─────────────────────┐      │
│  │ ActorCritic (1040→  │   │ ActorCritic (1040→  │      │
│  │  256→6/1)           │   │  256→6/1)           │      │
│  │ GAE + PPO Clip +    │   │ GAE + PPO Clip +    │      │
│  │ Multi-epoch replay  │   │ Multi-epoch replay  │      │
│  └─────────────────────┘   └─────────────────────┘      │
├─────────────────────────────────────────────────────────┤
│                  OvercookedWrapper                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │  • lossless_state_encoding (~1040-dim obs)       │   │
│  │  • Reward shaping (game_stats-verified)          │   │
│  │  • Custom layout support (from_grid)             │   │
│  │  • GPU auto-detection with occupancy check       │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│                  Event Tracker (per-step)                │
│  Pickups │ Pot placements │ Deliveries │ Actions │ Stay │
├─────────────────────────────────────────────────────────┤
│              Overcooked-AI (OvercookedGridworld)         │
│               MDP dynamics, state transitions            │
└─────────────────────────────────────────────────────────┘
```

### PPO Agent (PGAgent)
- **Network**: Shared FC(obs_dim→256) → ReLU, separate actor(256→6) and critic(256→1) heads, orthogonal initialization
- **Advantage**: GAE (λ=0.95, γ=0.99)
- **Policy update**: PPO clipped surrogate (ε=0.2), 4 epochs per episode, minibatch size 64
- **Value loss**: MSE, coefficient 0.5
- **Entropy bonus**: coefficient 0.05 (prevents premature convergence)
- **Gradient clipping**: max norm 0.5

### Observation Space
The `lossless_state_encoding` from Overcooked-AI provides a full grid state representation:
- ~1040 dimensions for cramped_room (5×5 grid)
- ~2080 dimensions for larger custom layouts

## Installation

```bash
git clone https://github.com/zxlishixian/overcooked.git
cd overcooked
pip install -e .
```

**Requirements**: Python ≥ 3.8, `torch ≥ 1.10`, `numpy < 2.0`, `pygame`, `gymnasium`, `matplotlib`.

## Quick Start

```bash
# Single training run (PPO agents, 100 episodes, CPU)
python overcooked_speed/experiments/run_pair.py \
    --layout cramped_room --agent0 nl --agent1 nl \
    --num_episodes 100 --seed 0 --log_dir logs/demo

# Multi-seed sweep (500 episodes × 3 seeds, GPU auto-detect)
python overcooked_speed/experiments/sweep_pairs.py \
    --layout cramped_room --pairs nl,nl \
    --num_episodes 500 --seeds 0 1 2 \
    --device auto --log_dir logs/sweep

# Plot learning curves from sweep results
python overcooked_speed/analysis/plot_learning_curves.py \
    --log_dir logs/sweep --save_dir logs/figures
```

### CLI Reference

`run_pair.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--layout` | `cramped_room` | Layout name or custom layout key |
| `--agent0/--agent1` | `nl` | Agent type (`nl` = PPO) |
| `--num_episodes` | 100 | Training episodes |
| `--horizon` | 400 | Max steps per episode |
| `--lr` | 1e-3 | Learning rate (Adam) |
| `--gamma` | 0.99 | Discount factor |
| `--device` | `auto` | `auto`/`cpu`/GPU index (`0`, `1`, etc.) |
| `--reward_shaping` | True | Enable intermediate milestone rewards |
| `--ent_coef` | 0.05 | Entropy bonus coefficient |
| `--ppo_epochs` | 4 | PPO update epochs per episode |
| `--log_dir` | `logs/smoke` | Output directory for CSVs and summary JSON |

`sweep_pairs.py` adds:

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs` | `nl,nl` | Comma-separated agent0,agent1 types |
| `--seeds` | `0 1 2` | Random seeds for multi-seed averaging |

## Key Findings (v5: game_stats-verified reward shaping)

**500 episodes × 3 seeds on cramped_room, NL+NL (PPO)**

| Metric | Mean ± Std |
|--------|-----------|
| Final Reward (last 50 ep) | 33.7 ± 9.2 |
| T_reward (convergence ep) | 222 ± 60 |
| **Final Specialization** | **0.81 ± 0.04** |
| Delivery Specialization | 0.90 ± 0.07 |
| Cooking Specialization | 0.72 ± 0.03 |
| T_specialization | 47 ± 16 |
| Total Deliveries (per seed) | 653–768 |

### Observations
1. **Role specialization is robust**: across all 3 seeds, one agent consistently becomes the cook (7,000+ cooking events) and the other the deliverer (600+ deliveries)
2. **Specialization precedes convergence**: T_spec ≈ 47 vs T_reward ≈ 222 — agents learn *who does what* long before they learn *how to do it well*
3. **Delivery specialization (0.90) > Cooking specialization (0.72)**: the delivery role is more sharply divided because only one agent can deliver at the serving station at a time
4. **Reward shaping anti-exploitation**: using `game_stats['potting_onion']` delta (MDP-verified) prevents the reward hacking seen in earlier versions where agents spammed onion pickups without progressing the task

### Reward Shaping Evolution

| Version | Design | Outcome |
|---------|--------|---------|
| v1–v2 | +1 onion pickup, +5 soup pickup | Exploited: 46,679 pickups vs 11 deliveries |
| v3 | +5 soup pickup only | Dead: 0 deliveries, no exploration signal |
| v4 | +2 onion drop, +3 soup pickup | Exploited: drop-on-counter looks like pot placement |
| **v5** | **+2 pot placement (game_stats Δ), +3 soup pickup** | ✅ Clean learning, strong specialization |

## Project Structure

```
overcooked/
├── overcooked_ai_py/                 # Environment (from Overcooked-AI)
│   ├── mdp/                          # OvercookedGridworld, OvercookedEnv, dynamics
│   ├── agents/                       # Scripted agents (rule-based, benchmarking)
│   ├── planning/                     # Planners for action simulation
│   ├── visualization/                # Pygame-based state renderer
│   └── data/                         # Layouts, graphics, test fixtures
│
├── overcooked_speed/                 # Research framework (this project)
│   ├── agents/
│   │   ├── pg_agent.py               # PPO agent with GAE + clip
│   │   └── policy.py                 # Actor-Critic network (shared MLP)
│   ├── envs/
│   │   ├── overcooked_wrapper.py     # Unified env API + reward shaping
│   │   ├── event_tracker.py          # Per-step event counting (state diffs)
│   │   └── __init__.py               # GPU detection, custom layout registry
│   ├── analysis/
│   │   ├── metrics.py                # Convergence & specialization metrics
│   │   ├── plot_learning_curves.py   # Reward/spec/action/learning curves
│   │   └── plot_heatmap.py           # Bar/heatmap across algorithm pairs
│   ├── experiments/
│   │   ├── run_pair.py               # Single pair training + CSV logging
│   │   └── sweep_pairs.py            # Multi-seed sweep + aggregation
│   ├── algos/                        # (future) Algorithm implementations
│   └── configs/                      # (future) Experiment config files
│
├── setup.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Future Research Directions

### Phase 2: Second-Order Learning Algorithms
Implement and benchmark agents that account for *other agents' learning*:

| Algorithm | Key Idea |
|-----------|----------|
| **LOLA** (Learning with Opponent-Learning Awareness) | Each agent differentiates through the *other agent's* policy update when computing its own gradient — anticipating how the co-player will change |
| **Lookahead** | Simulate k steps of joint learning then take the first gradient step — a form of model-based multi-agent planning |
| **Ideal Jπ** | Exact analytical joint policy gradient — serves as an upper-bound oracle for cooperative settings |

**Hypothesis**: LOLA and Lookahead should accelerate T_reward (faster convergence) and produce more stable specialization compared to naive learners, because they account for co-adaptation effects that NL agents treat as environmental noise.

### Phase 3: Algorithm Pair Analysis
Cross-comparing mixed pairs (e.g., LOLA+NL, LOLA+Lookahead) to study:
- Does having even one "aware" agent improve convergence for both?
- Is there an optimal pair for convergence speed vs final performance trade-off?
- How does algorithm choice affect the *type* of role division that emerges?

### Phase 4: Custom Layout Studies
The framework supports custom layouts via the `CUSTOM_LAYOUTS` registry. An asymmetric layout (`custom_asymmetric_roles`, 8×5 grid) separates cooking and delivery zones spatially, designed to encourage stronger role specialization. Future work:
- Measure specialization strength as a function of spatial separation
- Study whether certain algorithm pairs are more robust to layout changes

### Phase 5: Hyperparameter Sensitivity
- Entropy coefficient sweep (current: 0.05)
- PPO clip range and epoch count
- Discount factor (γ) effects on multi-agent credit assignment

## GPU Support

The framework auto-detects free GPUs via `nvidia-smi` with occupancy checks:

- GPU utilization ≤ 5% AND free memory ≥ 500 MB → eligible
- Excludes GPUs already in `CUDA_VISIBLE_DEVICES`
- Capped at 2 GPUs maximum (safety constraint)

```bash
# Auto-select free GPU
python overcooked_speed/experiments/sweep_pairs.py --device auto ...

# Force CPU
python overcooked_speed/experiments/sweep_pairs.py --device cpu ...

# Use specific GPU
python overcooked_speed/experiments/sweep_pairs.py --device 0 ...
```

## Contributing

This is a research project in active development. Contributions, questions, and collaborations are welcome — please open an issue or pull request.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on [HumanCompatibleAI/overcooked_ai](https://github.com/HumanCompatibleAI/overcooked_ai), a cooperative multi-agent environment introduced in:

> Carroll, M., Shah, R., Ho, M. K., Griffiths, T., Seshia, S., Abbeel, P., & Dragan, A. (2019). *On the Utility of Learning about Humans for Human-AI Coordination*. NeurIPS 2019.
