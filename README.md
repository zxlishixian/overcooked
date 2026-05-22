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
┌──────────────────────────────────────────────────────────────┐
│                       Experiments                            │
│                run_pair.py  /  sweep_pairs.py                │
├──────────────────────────────────────────────────────────────┤
│                          --algo                              │
│          ippo                         mappo                  │
│  ┌───────────────────┐    ┌─────────────────────────────┐   │
│  │ Agent 0  Agent 1  │    │   CentralizedCritic V(s)    │   │
│  │ (PGAgent) (PGAgent)│    │   ┌──────────┐              │   │
│  │ ┌──────┐ ┌──────┐ │    │   │ Actor 0  │  Actor 1     │   │
│  │ │Actor │ │Actor │ │    │   │ (policy) │  (policy)    │   │
│  │ │Critic│ │Critic│ │    │   └──────────┘              │   │
│  │ └──────┘ └──────┘ │    │   Shared GAE advantage       │   │
│  │ Separate GAE + PPO │    │   CTDE: centralized training │   │
│  │                    │    └─────────────────────────────┘   │
│  │ + RoleShapingMgr   │                                      │
│  │   teammate tendency│                                      │
│  │   complementary    │                                      │
│  │   role bonus       │                                      │
│  └───────────────────┘                                      │
├──────────────────────────────────────────────────────────────┤
│                     OvercookedWrapper                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  • obs_mode: egocentric (~520-dim) / global_concat  │  │
│  │    (~1040-dim), get_global_obs() for MAPPO critic    │  │
│  │  • Reward shaping (game_stats-verified)               │  │
│  │  • Custom layout support (from_grid)                  │  │
│  │  • GPU auto-detection with occupancy check            │  │
│  └───────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│                   Event Tracker (per-step)                   │
│   Pickups │ Pot placements │ Deliveries │ Actions │ Stay    │
├──────────────────────────────────────────────────────────────┤
│               Overcooked-AI (OvercookedGridworld)            │
│                 MDP dynamics, state transitions              │
└──────────────────────────────────────────────────────────────┘
```

### PPO Agent (PGAgent)
- **Network**: Shared FC(obs_dim→256) → ReLU, separate actor(256→6) and critic(256→1) heads, orthogonal initialization
- **Advantage**: GAE (λ=0.95, γ=0.99)
- **Policy update**: PPO clipped surrogate (ε=0.2), 4 epochs per episode, minibatch size 64
- **Value loss**: MSE, coefficient 0.5
- **Entropy bonus**: coefficient 0.05 (prevents premature convergence)
- **Gradient clipping**: max norm 0.5

### MAPPO (Multi-Agent PPO) — `--algo mappo`
- **Paradigm**: CTDE (Centralized Training, Decentralized Execution)
- **Actors**: Two decentralized ActorCritic networks, each outputting independent action policies
- **Critic**: One centralized 3-layer MLP V(s_global) — same architecture as PGAgent critic but sees full global state
- **Advantage**: Both actors share the same team advantage computed by the centralized critic via GAE
- **Actor update**: PPO clipped surrogate (same hyperparams as IPPO), no per-actor value loss term
- **Critic update**: MSE regression on returns, independent optimizer
- **Key difference from IPPO**: Rather than each agent learning its own value function from its own perspective, MAPPO uses a single centralized critic that sees the full global state — this stabilizes value estimation in cooperative tasks where individual observations are partial or noisy

### Unified Shaping API — `--shaping_type`

A general-purpose mechanism diagnosis API that supports 10 shaping types for understanding *why* role shaping works (or doesn't). All types inject a per-episode bonus via `add_terminal_bonus()`, propagated backward through the trajectory by GAE.

**Architecture**: A single `RoleShapingManager` class in [role_shaping.py](overcooked_speed/agents/role_shaping.py) dispatches via `_compute_raw()` based on `shaping_type`. Methods that use teammate role history (`_ROLE_TYPES`) call `record_episode()` to maintain a sliding window of past episode counts. Diagnostic methods compute the bonus purely from the current episode's event counts.

**Shaping types:**

| `--shaping_type` | Mechanism tested | Uses teammate history | Bonus formula |
|------------------|------------------|----------------------|---------------|
| `none` | No shaping (baseline) | — | 0 |
| `raw_clipped` | Role complementarity (original) | Yes (K=20) | `p_cook×delivery + p_deliver×cooking` |
| `normalized` | Event-normalized complementarity | Yes (K=20) | `(p_cook×delivery + p_deliver×cooking) / total_events` |
| `weighted_normalized` | Weighted complementarity | Yes (K=20) | Weighted by event importance (delivery=1.0, potting=1.0, pickup=0.2) |
| `delta_complementarity` | Complementarity improvement over baseline | Yes (K=20) | `C_current − mean(C_history)` |
| `constant_bonus` | Constant reward shift | No | 1.0 |
| `event_density_bonus` | Task-progress density | No | `useful_events / total_actions` |
| `event_binary_bonus` | Simple event-occurrence signal | No | 1.0 if any useful event occurred |
| `delivery_chain_bonus` | Weighted task-progress (normalized) | No | `weighted_chain / total_actions` |
| `delivery_chain_raw_clipped` | Weighted task-progress (raw, clipped) | No | `weighted_chain` (clipped to ±bonus_clip) |

**Key design decisions:**
| Choice | Rationale |
|--------|-----------|
| Window K=20 | Smooths noise, adapts to role drift |
| Bonus clip ±1.0 (default) | Prevents extreme bonuses from destabilizing PPO |
| λ_role scales bonus before GAE injection | Allows tuning of shaping strength vs environment reward |
| IPPO only | MAPPO's centralized critic already captures role dynamics |
| Diagnostic types don't use teammate history | Isolates the mechanism — any improvement comes from the signal itself, not from tracking teammate behavior |

**Backward compatibility**: `--role_shaping`, `--role_bonus_type`, and `--role_bonus_clip` are preserved as deprecated aliases that map to `--shaping_type` and `--bonus_clip`.

### Observation Space

The `lossless_state_encoding_mdp` from Overcooked-AI returns a 2-agent dual-perspective encoding `enc[0], enc[1]` where each perspective is ~520-dim for cramped_room (5×5 grid, 26 channels). The wrapper supports two observation modes via `--obs_mode`:

| Mode | Dim per agent | Agent 0 obs | Agent 1 obs | Use case |
|------|--------------|-------------|-------------|----------|
| `egocentric` (default) | ~520 | `enc[0].flatten()` | `enc[1].flatten()` | Fair comparison: IPPO vs MAPPO with same actor input |
| `global_concat` | ~1040 | `np.array(enc).flatten()` | `np.array(enc).flatten()` (identical) | Idealized upper-bound ablation |
| `local` | — | — | — | Reserved for future partial-observation work |

**Key design**: In `egocentric` mode, agent 0 and agent 1 receive **different** 520-dim vectors (agent-specific channel ordering), even though the underlying state information is the same. This matters for fair comparison — IPPO agents see the same information as MAPPO actors, and MAPPO's advantage comes only from its centralized critic, which always uses `get_global_obs()` (~1040-dim, regardless of `obs_mode`).

**Fair comparison matrix**:

| Config | Actor obs | Critic obs | What it isolates |
|--------|-----------|------------|-----------------|
| IPPO-egocentric | 520-dim ego | 520-dim ego (own critic) | IPPO limited baseline |
| **MAPPO-egocentric** | **520-dim ego** | **1040-dim global** | **Centralized critic benefit** |
| IPPO-global_concat | 1040-dim both | 1040-dim both (own critic) | Upper-bound ablation |
| MAPPO-global_concat | 1040-dim both | 1040-dim global | Upper-bound ablation |

## Installation

```bash
git clone https://github.com/zxlishixian/overcooked.git
cd overcooked
pip install -e .
```

**Requirements**: Python ≥ 3.8, `torch ≥ 1.10`, `numpy < 2.0`, `pygame`, `gymnasium`, `matplotlib`.

## Quick Start

```bash
# IPPO: Independent PPO learners (default) — egocentric obs
python overcooked_speed/experiments/run_pair.py \
    --layout cramped_room --agent0 nl --agent1 nl \
    --num_episodes 100 --seed 0 --log_dir logs/demo

# IPPO with global_concat obs (ablation)
python overcooked_speed/experiments/run_pair.py \
    --layout cramped_room --algo ippo --obs_mode global_concat \
    --agent0 nl --agent1 nl \
    --num_episodes 100 --seed 0 --log_dir logs/demo_ippo_global

# MAPPO: egocentric actors + centralized global critic (key comparison)
python overcooked_speed/experiments/run_pair.py \
    --layout cramped_room --algo mappo --obs_mode egocentric \
    --agent0 nl --agent1 nl \
    --num_episodes 100 --seed 0 --log_dir logs/demo_mappo

# IPPO + Role Shaping (teammate-aware complementary reward)
python overcooked_speed/experiments/run_pair.py \
    --layout cramped_room --algo ippo --obs_mode egocentric \
    --shaping_type raw_clipped --lambda_role 0.1 --role_window 20 \
    --agent0 nl --agent1 nl \
    --num_episodes 100 --seed 0 --log_dir logs/demo_role

# IPPO + Diagnostic shaping (e.g., event density bonus)
python overcooked_speed/experiments/run_pair.py \
    --layout cramped_room --algo ippo --obs_mode egocentric \
    --shaping_type event_density_bonus --lambda_role 0.03 \
    --agent0 nl --agent1 nl \
    --num_episodes 100 --seed 0 --log_dir logs/demo_density

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
| `--algo` | `ippo` | Algorithm: `ippo` (independent PPO) or `mappo` (centralized-critic MAPPO) |
| `--obs_mode` | `egocentric` | Observation mode: `egocentric` (~520-dim), `global_concat` (~1040-dim), `local` (reserved) |
| `--shaping_type` | `none` | Shaping bonus type: `none`, `raw_clipped`, `normalized`, `weighted_normalized`, `delta_complementarity`, `constant_bonus`, `event_density_bonus`, `event_binary_bonus`, `delivery_chain_bonus`, `delivery_chain_raw_clipped` |
| `--bonus_clip` | 1.0 | Max absolute bonus per episode (applied after bonus_type raw compute) |
| `--lambda_role` | 0.1 | Shaping bonus weight in training reward |
| `--role_window` | 20 | Past episodes for teammate role tendency (role-based types only) |
| `--role_shaping` | `False` | ⚠️ Deprecated — use `--shaping_type raw_clipped` |
| `--role_bonus_type` | — | ⚠️ Deprecated — use `--shaping_type` |
| `--role_bonus_clip` | — | ⚠️ Deprecated — use `--bonus_clip` |
| `--log_dir` | `logs/smoke` | Output directory for CSVs and summary JSON |

`sweep_pairs.py` adds:

| Flag | Default | Description |
|------|---------|-------------|
| `--pairs` | `nl,nl` | Comma-separated agent0,agent1 types |
| `--seeds` | `0 1 2` | Random seeds for multi-seed averaging |
| `--algo` | `ippo` | Algorithm: `ippo` or `mappo` |
| `--obs_mode` | `egocentric` | Observation mode: `egocentric`, `global_concat`, `local` |
| `--shaping_type` | `none` | Shaping bonus type (same 10 options as run_pair) |
| `--bonus_clip` | 1.0 | Max absolute bonus per episode |
| `--lambda_role` | 0.1 | Shaping bonus weight |
| `--role_window` | 20 | Past episodes for teammate tendency |
| `--role_shaping` | `False` | ⚠️ Deprecated — use `--shaping_type raw_clipped` |

## Key Findings

### Baseline Sweep: IPPO vs MAPPO (500ep × 5 seeds)

**cramped_room, egocentric obs (520-dim actor), shaped reward**

| Config | final_reward | reward_auc | T_reward | final_spec | T_spec |
|--------|-------------|------------|----------|------------|--------|
| IPPO-egocentric | 29.6 ± 6.8 | 8699 ± 989 | 299.8 | 0.768 ± 0.039 | 116.6 |
| IPPO-global_concat | 34.5 ± 7.7 | 9952 ± 884 | 237.8 | 0.804 ± 0.042 | 53.0 |
| MAPPO-egocentric | **80.0 ± 21.4** | **18068 ± 4446** | **246.0** | **0.867 ± 0.038** | **49.6** |
| MAPPO-global_concat | 78.1 ± 18.1 | 16404 ± 2983 | 195.0 | 0.811 ± 0.088 | 40.0 |

### IPPO + Role Shaping (500ep × 5 seeds)

| Config | final_reward | reward_auc | T_reward | final_spec | T_spec |
|--------|-------------|------------|----------|------------|--------|
| IPPO-ego (baseline) | 29.6 ± 6.8 | 8699 ± 989 | 299.8 | 0.768 ± 0.039 | 116.6 |
| **IPPO-role-shaping** | **45.3 ± 18.9** | **10096 ± 2740** | **365.0** | **0.733 ± 0.039** | **78.0** |
| MAPPO-ego (upper bound) | 80.0 ± 21.4 | 18068 ± 4446 | 246.0 | 0.867 ± 0.038 | 49.6 |

**Per-seed breakdown (IPPO-role-shaping):**

| Seed | final_reward | T_reward | spec | Agent 0 | Agent 1 |
|------|-------------|----------|------|---------|---------|
| 0 | 23.3 | 500 ✗ | 0.669 | 318 del / 1757 cook | 30 del / 5412 cook |
| 1 | 22.2 | 500 ✗ | 0.740 | 349 del / 1514 cook | 23 del / 6142 cook |
| 2 | 52.6 | 303 ✓ | 0.715 | 23 del / 1931 cook | 448 del / 6261 cook |
| 3 | 66.2 | 225 ✓ | 0.780 | 792 del / 1689 cook | 32 del / 7626 cook |
| 4 | 62.0 | 297 ✓ | 0.761 | 18 del / 6735 cook | 582 del / 1778 cook |

### Observations

1. **MAPPO dominates IPPO**: With identical 520-dim egocentric actor input, MAPPO's centralized critic achieves 2.7× higher reward (80.0 vs 29.6) — value estimation, not policy optimization, is the bottleneck in this cooperative task
2. **Observation mode matters less for MAPPO**: MAPPO-egocentric (80.0) ≈ MAPPO-global_concat (78.1) — the centralized critic already sees full global state regardless of actor obs mode
3. **Global observation helps IPPO modestly**: IPPO-global (34.5) beats IPPO-ego (29.6) by +17% — seeing both perspectives helps even without centralized training
4. **Role shaping improves IPPO by +53%** (45.3 vs 29.6) but with high variance — 3/5 seeds converge well (52-66 reward), 2/5 fail (22-23). When it works, specialization emerges faster (T_spec=78 vs 117) and reward approaches the IPPO-global baseline. When it doesn't, the constant bonus signal (saturated at clip=1.0) fails to provide useful gradient information
5. **Role shaping does not approach MAPPO**: MAPPO-ego still leads by 1.8× in reward (80.0 vs 45.3). The centralized critic captures richer coordination signals than the simple role-count bonus
6. **Current limitation — bonus saturation**: With `bonus_clip=1.0`, the raw role bonus (p_cook × deliveries + p_deliver × cooking) routinely exceeds 1.0 in a 400-step episode (15-20+ events), so the bonus clips to 1.0 almost every episode. This turns `lambda_role * bonus = 0.1` into a constant shift rather than a behavior-sensitive gradient. Future work: increase clip, normalize by episode steps, or use ranking-based bonus
7. **Reward shaping anti-exploitation**: using `game_stats['potting_onion']` delta (MDP-verified) prevents the reward hacking seen in earlier versions where agents spammed onion pickups without progressing the task

### Mechanism Diagnosis: Why Does Role Shaping Help? (300ep × 3 seeds)

To disentangle *which* component of `raw_clipped` role shaping drives the improvement, we ran a controlled ablation comparing 5 diagnostic shaping types, each at 2 bonus weights (λ=0.03, 0.1), all with the same PPO hyperparameters:

| Config | λ=0.03 | λ=0.1 | Clip rate |
|--------|--------|-------|-----------|
| **IPPO-ego baseline** (500ep) | — | **29.6 ± 6.8** | — |
| **raw_clipped** role shaping (500ep) | — | **45.3 ± 18.9** | ~100% |
| `constant_bonus` | 18.7 ± 1.8 | 22.9 ± 5.0 | 0% |
| `event_binary_bonus` | 18.7 ± 1.8 | 22.9 ± 5.0 | 0% |
| `event_density_bonus` | **27.1 ± 6.2** | 20.5 ± 4.7 | 0% |
| `delivery_chain_bonus` | 14.9 ± 4.1 | 20.6 ± 0.6 | 0% |
| `delivery_chain_raw_clipped` | 22.7 ± 5.9 | 18.2 ± 4.2 | ~97% |

**Five diagnostic questions answered:**

1. **Does constant_bonus reproduce raw_clipped improvement?** No. At λ=0.1, constant_bonus achieves 22.9 vs raw_clipped 45.3. A constant +0.1 reward shift per episode does not explain the 53% gain — the signal must carry behavior-relevant information.

2. **Is event_density better than constant?** Yes, at low λ (27.1 vs 18.7). Density bonus = `useful_events / total_actions` carries information about action efficiency, which constant_bonus cannot. However, at higher λ (0.1), density drops to 20.5, suggesting the signal becomes too noisy when amplified.

3. **Does delivery_chain beat role complementarity?** No. Neither chain variant (normalized or raw_clipped) reaches the IPPO-ego baseline (29.6), let alone raw_clipped role shaping (45.3). The weighted task-progress signal alone is insufficient.

4. **What signals correlate with reward?** Correlation analysis across all existing logs (10,000+ episodes) reveals that **task-progress event counts**, not role complementarity, are the strongest reward predictors:

| Signal | Pearson r | Spearman ρ |
|--------|-----------|-------------|
| `total_potting` (onions placed in pots) | **+0.813** | **+0.963** |
| `total_task_events` | +0.728 | +0.755 |
| `total_soup_delivery` | +0.698 | +0.711 |
| `mean_role_bonus_raw` | +0.334 | +0.504 |
| `complementarity_current` | +0.356 | +0.512 |
| `complementarity_delta` | +0.160 | +0.242 |

The Spearman correlation is even stronger for total_potting (ρ=0.963) — the *rank ordering* of episodes by potting events almost perfectly predicts reward ranking. This is expected in Overcooked where soup delivery requires 3 onions per pot.

5. **What does this mean?** The evidence points to **task-progress density during early training** as the mechanism, not pure constant shift or role complementarity. However, `event_density_bonus` alone (27.1) does not match `raw_clipped` (45.3), suggesting that **combining** task-progress signals with teammate-history-aware role complementarity produces a synergistic effect that neither component achieves alone.

**Caveats**: Diagnostic sweep was 300ep × 3 seeds; baseline comparisons are against 500ep × 5 seeds. Results may not be directly comparable at the same episode count.

### Correlation Analysis Tool

```bash
# Compute Pearson & Spearman correlation between shaping signals and reward
python overcooked_speed/analysis/analyze_shaping_signal.py \
    --log_dirs logs/ippo_role_shaping_ego_500 logs/ablation/normalized_l003 \
    --output_csv logs/correlation_summary.csv
```

The script reads all episode CSVs recursively, computes derived signals (total_cooking, total_delivery, total_task_events, etc.), and reports three correlation types:
- **same_episode**: corr(signal_t, reward_t)
- **next_episode**: corr(signal_t, reward_{t+1}) — predictive power
- **moving_avg_10**: corr(signal_t, MA_reward_{t:t+10}) — smoothed reward

**MAPPO CSV fields** (beyond IPPO):

| Field | Description |
|-------|-------------|
| `critic_loss` | Centralized critic MSE loss |
| `value_mean` | Mean V(s) across episode timesteps |
| `advantage_mean` | Mean GAE advantage (before normalization) |

**Shaping CSV fields** (beyond IPPO):

| Field | Description |
|-------|-------------|
| `shaping_type` | Active shaping type string |
| `lambda_role` | Shaping bonus weight |
| `bonus_clip` | Max absolute applied bonus |
| `agent0_shaping_raw` / `agent1_shaping_raw` | Raw bonus per agent (before clip) |
| `agent0_shaping_applied` / `agent1_shaping_applied` | Applied bonus per agent (after clip) |
| `mean_shaping_applied` | Mean applied bonus across both agents |
| `shaping_clip_rate` | Fraction of episodes where raw ≠ applied |
| `a0_shaped_reward` / `a1_shaped_reward` | Episode reward + λ_role × applied_bonus |
| `teammate0_p_cook` / `teammate0_p_deliver` | Agent 0's teammate (agent 1) role tendency |
| `teammate1_p_cook` / `teammate1_p_deliver` | Agent 1's teammate (agent 0) role tendency |
| `complementarity_current` | Current episode complementarity C |
| `complementarity_delta` | C_current − mean(C_history) |
| `total_task_events` | Sum of cooking + delivery events |
| `total_potting` | Total onions placed in pots |
| `total_soup_pickup` | Total soups picked up |
| `total_soup_delivery` | Total soups delivered |

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
│   │   ├── pg_agent.py               # IPPO agent with GAE + clip
│   │   ├── mappo_agent.py            # MAPPO: centralized critic + two actors
│   │   ├── role_shaping.py           # Role-level LOLA-like teammate-aware bonus
│   │   └── policy.py                 # Actor-Critic network (shared MLP)
│   ├── envs/
│   │   ├── overcooked_wrapper.py     # Unified env API + reward shaping
│   │   ├── event_tracker.py          # Per-step event counting (state diffs)
│   │   └── __init__.py               # GPU detection, custom layout registry
│   ├── analysis/
│   │   ├── metrics.py                # Convergence & specialization metrics
│   │   ├── analyze_shaping_signal.py # Correlation: shaping signals vs reward
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

**Current baseline**: IPPO (independent PPO), MAPPO (centralized-critic CTDE), and **IPPO+role-shaping** (teammate-aware complementary bonus) are implemented. MAPPO provides the centralized-training reference point — LOLA/Lookahead should be compared against all three to isolate the benefit of second-order gradient information from simpler teammate-aware heuristics.

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
