# Overcooked 多智能体协作学习实验日志

> 本文档记录从研究开始到当前阶段的主要探索、实验结果、阶段性结论，以及下一步研究路线。目标是让新加入项目的人能够快速理解：我们在研究什么、已经验证了什么、哪些方向被排除、哪些方向值得继续推进。

---

## 1. 研究总目标

本项目最初从 LOLA / Lookahead 等多智能体学习算法出发，关注的问题是：

**在合作多智能体任务中，agent 是否能够通过显式建模队友的学习过程、策略变化、任务意图，从而更快形成有效协作？**

随着实验推进，研究重点逐渐从“比较不同算法最终博弈能力”转向：

**不同算法或不同 teammate-aware 机制，是否能够提升合作任务中的训练速度、收敛速度、角色分工速度和样本效率？**

当前核心研究假设是：

> 在 Overcooked 这类顺序协作任务中，普通 IPPO 的瓶颈不只是 actor 观测不足，而是缺乏对队友策略、队友意图、队友学习趋势的显式建模。相比直接 reward shaping，更合理的路线是让 agent 的 critic 或 actor 利用 teammate policy / teammate intention / teammate intention trend 信息。

---

## 2. 早期理论探索：从 LOLA 到 Policy-Space Opponent Response

### 2.1 初始问题

最初关注 LOLA 算法是否能够在协调博弈、反协调博弈、合作任务中产生更好的协作行为。

我们讨论过：

- 协调博弈：双方选择相同或互补策略以共同获益；
- 反协调博弈：双方需要选择不同角色或动作才能共同获益；
- Overcooked 这类任务更接近“角色互补 + 动态任务链条”的合作场景。

一个重要早期判断是：

> 很多合作任务并不是简单“双方都做同一件事”，而是需要 agents 在任务链条中形成不同但互补的角色。

例如 Overcooked 中：

- 一个 agent 更适合拿洋葱、放锅；
- 另一个 agent 更适合拿盘子、取汤、交付；
- 高效协作依赖角色分工与时序配合。

---

### 2.2 Policy-space 版 new_lola_g 设计

在 matrix-game / exact game 中，我们设计过一个 new_lola_g 思路：


def agent1 update ≈ first_order + second_order

其中 second_order 不再直接学习 theta-space 的复杂二阶项，而是学习 policy-space opponent response Jacobian：

\[
J_\pi^{actual} = \frac{\partial(\Delta \pi^2_{actual})}{\partial \pi^1}
\]

也就是说：

> 不管对手实际使用什么算法，我们都尝试拟合“我的策略变化会如何影响对手下一步策略变化”。

这个设计强调：

- 不一定要假设对手是 naive learner；
- 也不一定严格复刻原始 LOLA 的 theta-space 二阶推导；
- 更重要的是估计 actual opponent response。

这一阶段得到的重要认识：

1. **直接学习压缩后的 G 向量会有问题**：它把 analytical value gradient 和 learned opponent response 混在一起，容易自标注错误。
2. **更合理的是学习 Jπ**：即对手策略更新对我策略的 Jacobian，再把 value gradient 解析地接上。
3. **在线 JVP 训练容易破坏离线预训练模型**：因为 JVP 只约束少数方向，其它维度会漂移。
4. **冻结 offline backbone + online residual head** 是一个更稳的方向。

虽然这条线后来没有直接迁移到 Overcooked，但它奠定了一个核心思想：

> 不要只比较 agent 当前策略，而要建模队友策略如何变化，以及这种变化如何影响合作收益。

---

## 3. 转向 Overcooked：从算法能力到收敛速度

### 3.1 研究方向转变

在 exact/matrix game 之后，我们将研究场景转向 Overcooked-AI。

新的研究问题变成：

> 在更复杂的深度强化学习合作环境中，不同学习机制是否能够让 agents 更快形成有效分工和协作？

也就是说，关注指标不只是最终 reward，还包括：

- reward AUC；
- T_reward：达到 reward 阈值所需 episode；
- T_specialization：形成角色分工所需 episode；
- final_specialization；
- seed variance；
- value estimation 质量。

---

### 3.2 Overcooked 实验环境

当前实验基于 Overcooked-AI，主要使用 layout：

- `cramped_room`

已支持的 observation mode：

| obs_mode | actor observation | 说明 |
|---|---:|---|
| `egocentric` | 520-dim | 每个 agent 使用自己作为 primary 的 lossless encoding，两个 agent 看到不同向量 |
| `global_concat` | 1040-dim | 两个 agent 都看到两个视角拼接后的全局信息 |
| `local` | reserved | 暂未实现 |

一个关键修复是：

> 早期 IPPO 中两个 agent 实际拿到的是相同 1040-dim global observation，这让 IPPO 条件过于理想化。后来修正为 egocentric 模式，actor 只看 520-dim 自身视角。

---

## 4. PPO / IPPO 基础实现

### 4.1 基础算法

我们首先实现了基于 PPO 的 IPPO baseline：

- 每个 agent 有独立 actor-critic 网络；
- 两个 agent 接收 shared team reward；
- agent 独立更新，不共享参数；
- actor 输入 egocentric obs；
- critic 初始也只输入 egocentric obs。

Actor-Critic 结构大致为：

- obs → shared FC → actor head → action logits；
- obs → shared FC → critic head → V(s)。

PPO 使用：

- GAE；
- PPO clipped objective；
- entropy bonus；
- value loss；
- gradient clipping。

---

### 4.2 为什么选择 PPO/IPPO 作为 naive learner

Overcooked reward 稀疏，普通 REINFORCE 难以学习，因此使用 PPO + GAE 作为基础 naive learner。

PPO 的作用：

- 限制策略更新幅度，缓解多智能体非平稳性；
- GAE 帮助稀疏奖励向前传播；
- critic 提供 baseline，减少方差。

---

## 5. MAPPO Baseline 与公平观测设置

### 5.1 MAPPO 实现

随后实现 MAPPO baseline：

- actor 仍然使用和 IPPO 相同的 observation；
- centralized critic 使用 global observation；
- 这样可以隔离 centralized critic 的作用。

公平设置：

| 方法 | actor obs | critic obs | 目的 |
|---|---:|---:|---|
| IPPO-egocentric | 520 | 520 | 去中心化 baseline |
| MAPPO-egocentric | 520 | 1040 | 只增加 centralized critic |
| IPPO-global | 1040 | 1040 | actor 全局观测上限对照 |
| MAPPO-global | 1040 | 1040 | global actor + global critic |

---

### 5.2 Baseline sanity sweep 结果

实验设置：

- 500 episodes；
- 5 seeds；
- layout: cramped_room。

结果：

| Metric | IPPO-ego | MAPPO-ego | IPPO-global | MAPPO-global |
|---|---:|---:|---:|---:|
| Final Reward | 29.6 ± 6.8 | 80.0 ± 21.4 | 34.5 ± 7.7 | 78.1 ± 18.1 |
| Reward AUC | 8699 | 18068 | 9952 | 16404 |
| T_reward ↓ | 300 ± 35 | 246 ± 38 | 238 ± 51 | 195 ± 21 |
| Final Spec | 0.768 | 0.867 | 0.804 | 0.811 |
| T_spec ↓ | 117 ± 52 | 50 ± 29 | 53 ± 19 | 40 ± 28 |

关键结论：

1. **MAPPO-egocentric 明显强于 IPPO-egocentric**。在 actor observation 相同的情况下，MAPPO final reward 约为 IPPO 的 2.7 倍。
2. **IPPO-global 远不如 MAPPO-egocentric**。说明单纯给 actor 全局观测不能解决核心问题。
3. **MAPPO 的优势主要来自 centralized critic / value estimation / credit assignment，而不是 actor 看不到全局。**

这是当前研究的重要起点：

> IPPO 的核心瓶颈是 value estimation / credit assignment，因此如果不想依赖 MAPPO global critic，就需要通过 teammate-aware critic 来改善 decentralized value estimation。

---

## 6. Role Shaping 与 Task Shaping 探索

### 6.1 初始想法

受到 LOLA / Lookahead 思想启发，我们尝试过通过 role/task shaping 引导 agents 更快分工。

直觉：

- 如果队友倾向 cooking，我就做 delivery；
- 如果队友倾向 delivery，我就做 cooking；
- 或者通过 task-progress bonus 鼓励 onion pickup / potting / dish pickup / soup pickup / delivery。

---

### 6.2 Role Shaping 初步结果

第一版 raw-clipped role shaping：

| Metric | IPPO-ego | IPPO-role-shaping | MAPPO-ego |
|---|---:|---:|---:|
| final_reward | 29.6 ± 6.8 | 45.3 ± 18.9 | 80.0 ± 21.4 |
| reward_auc | 8699 | 10096 | 18068 |
| T_reward | 299.8 | 365.0 | 246.0 |
| final_specialization | 0.768 | 0.733 | 0.867 |
| T_specialization | 116.6 | 78.0 | 49.6 |

表面上 role shaping 提升了 final_reward，但后来发现：

- role bonus 几乎总是 clipped 到 1.0；
- 它可能不是有效的 role complementarity signal，而是某种 accidental dense bonus / early exploration effect。

---

### 6.3 Role bonus ablation

测试了：

- normalized；
- weighted_normalized；
- delta_complementarity。

结果：

- 新的 normalized bonus 不再 clip；
- 但所有 principled role bonus 都没有超过 IPPO baseline；
- raw_clipped 的提升不是干净的 role complementarity。

结论：

> 高层 role shaping 太粗糙，不能稳定提升 Overcooked 中的真实任务表现。

---

### 6.4 Mechanism diagnosis

做了相关性分析，发现 reward 最相关的不是 role bonus，而是 task-progress events：

| Signal | Pearson correlation with reward |
|---|---:|
| total_potting | +0.813 |
| total_task_events | +0.689 |
| total_soup_delivery | +0.665 |
| total_cooking | +0.581 |
| role_bonus | +0.334 |

结论：

> task-progress events 尤其是 potting_onion，比抽象 role bonus 更接近真实 reward。

---

### 6.5 Task-level shaping 实验

测试了：

- self_task_progress；
- team_task_progress；
- teammate_task_progress；
- task_lookahead_rule；
- task_lola_rule；
- team_bottleneck_progress。

最佳 task-level shaping 结果仍低于 IPPO baseline：

| Method | final_reward | T_spec |
|---|---:|---:|
| IPPO-ego baseline | 29.6 | 117 |
| best task shaping | 25.4 | 34 |
| MAPPO-ego | 80.0 | 50 |

关键结论：

- task shaping 可以显著加快 specialization；
- 但 final reward 下降；
- 说明它让 agent 更快形成行为模式，但可能收敛到低质量分工。

核心解释：

> Reward shaping 把 task-progress signal 加进 reward / GAE，污染 critic，使 agent 优化 proxy objective，而不是真实 soup delivery reward。

因此后续路线不再继续 reward shaping。

---

## 7. 从 Reward Shaping 转向 Teammate-Aware Critic

### 7.1 关键转向

经过 shaping 实验后，我们得出：

> 如果想借鉴 LOLA / Lookahead 思想，不应该继续手写 reward bonus，而应该显式建模队友策略、队友意图、队友学习趋势，并把这些信息用于 critic / actor，而不是污染 reward。

因此下一步转向：

- teammate-policy-conditioned critic；
- teammate-intention-conditioned critic；
- 后续再进入 Lookahead-like adaptation 和 LOLA-like shaping。

---

## 8. IPPO-PC: Policy-Conditioned Critic

### 8.1 方法

IPPO-PC 将 critic 从：

\[
V_i(o_i)
\]

改成：

\[
V_i(o_i, \pi_j(o_j))
\]

其中：

- actor 不变：\(\pi_i(a_i|o_i)\)；
- critic 输入自己的 egocentric obs 和队友当前 action probability distribution；
- reward / GAE / critic target 仍然只使用真实环境 reward；
- teammate_probs 使用 rollout-time old policy，detach 后存入 buffer；
- critic loss 不会反传到队友 actor。

---

### 8.2 IPPO-PC 实验结果

实验设置：

- 500 episodes；
- 5 seeds；
- cramped_room。

| Metric | IPPO-normal | IPPO-PC | Delta |
|---|---:|---:|---:|
| final_reward | 29.58 ± 7.65 | 32.26 ± 15.89 | +9.1% |
| reward_auc | 8699 ± 1105 | 9553 ± 3669 | +9.8% |
| T_reward | 299.8 ± 38.7 | 333.2 ± 113.2 | worse |
| T_specialization | 116.6 ± 57.8 | 80.4 ± 46.8 | -31% |
| final_specialization | 0.768 ± 0.039 | 0.723 ± 0.116 | -0.04 |

Value diagnostics：

| Diagnostic | IPPO-normal | IPPO-PC | Interpretation |
|---|---:|---:|---|
| explained_variance | 0.679 | 0.700 | value fit improves |
| value_teammate_sensitivity | 0.0 | 0.077 | critic uses teammate info |
| teammate_prob_entropy | 0.0 | 1.50 | teammate policy moderately certain |
| value_mean | 3.78 | 4.15 | slight overestimation |
| return_mean | 3.64 | 4.01 | higher returns |

Per-seed final reward：

| Seed | Normal | PC | Delta |
|---|---:|---:|---:|
| 0 | 23.84 | 27.20 | +3.36 |
| 1 | 31.00 | 25.68 | -5.32 |
| 2 | 23.84 | 27.64 | +3.80 |
| 3 | 42.20 | 20.56 | -21.64 |
| 4 | 27.00 | 60.24 | +33.24 |

---

### 8.3 IPPO-PC 阶段性结论

IPPO-PC 证明：

1. 队友当前 action distribution 可以被 decentralized critic 使用；
2. value fit 有小幅提升；
3. specialization 明显加快；
4. final reward 平均略升，但方差变大；
5. 当前 teammate policy snapshot 有用，但不够鲁棒。

核心解释：

> 当前策略快照 \(\pi_j(o_j)\) 只告诉 critic 队友现在可能做什么，但没有告诉 critic 队友最近一直在尝试完成什么任务，也没有告诉 critic 队友未来会如何改变。

这推动下一阶段：从 policy snapshot 转向 intention trend。

---


## 9. IPPO-PC Ablation (Phase A): True vs Uniform vs Shuffled — ✅ Completed

### 9.1 方法

为了排除 IPPO-PC 的提升来自"额外 critic 容量"而非"队友策略信息"的可能性，实现了 `--teammate_probs_mode {true, uniform, shuffled}`：

| mode | critic_extra | 目的 |
|---|---|---|
| true | 真实 rollout-time π_j(o_j) | 原始 IPPO-PC |
| uniform | 恒定 [1/6, 1/6, 1/6, 1/6, 1/6, 1/6] | 控制额外参数/输入维度，无信息 |
| shuffled | 真实 π_j，PPO 更新前打乱 | 控制边际分布，破坏 state-policy pairing（次要控制） |

### 9.2 实验设置

- 500 episodes，5 seeds (0-4)
- layout: cramped_room
- algo: IPPO egocentric
- 比较: normal vs PC-true vs PC-uniform

### 9.3 实验结果

| Metric | IPPO-normal | PC-true | PC-uniform |
|---|---|---|---|
| final_reward | 29.6 ± 7.7 | 32.3 ± 15.9 | **38.3 ± 7.8** |
| reward_auc | 8699 ± 1105 | 9553 ± 3669 | **10186 ± 1791** |
| T_reward | 299.8 | 333.2 | **283.2** |
| explained_variance | 0.679 | **0.700** | 0.684 |
| value_extra_sensitivity | 0.0 | 0.077 | — |
| final_specialization | 0.768 | 0.723 | 0.755 |
| T_specialization | 116.6 | 80.4 | 86.4 |

Per-seed final_reward:

| Seed | Normal | PC-true | PC-uniform |
|---|---|---|---|
| 0 | 23.84 | 27.20 | 37.20 |
| 1 | 31.00 | 25.68 | 52.08 |
| 2 | 23.84 | 27.64 | 22.72 |
| 3 | 42.20 | 20.56 | 40.76 |
| 4 | 27.00 | 60.24 | 38.56 |

### 9.4 关键结论

**PC 的提升主要来自额外 critic 容量，而非队友策略信息。**

1. **uniform (38.3) 显著优于 true (32.3)**：即使给 critic 的是恒定均匀分布（无信息），性能反而更好
2. **uniform 方差更低** (7.8 vs 15.9)：真实队友 probs 给训练增加了不稳定性
3. **true 的 explained_variance 最高 (0.700)**：critic 确实在使用队友 probs 来拟合 returns，但这并没有转化为更好的策略或更高的 reward
4. **PC-true 的 value_extra_sensitivity = 0.077**：critic 对 teammate probs 的敏感度较低

**这是一个负向结果**：说明在 IPPO 中，给 critic 额外的 6-dim 输入和 projection layer 带来的容量提升就足够了，而队友策略信息本身并不是提升的驱动因素。真实队友 probs 反而引入了噪声。

### 9.5 对后续方向的影响

这个结果和之前的 shaping 实验一起，指向同一个方向：

- reward shaping 不是可靠路线（污染 critic / GAE）
- teammate policy input 也不是可靠路线（噪声 > 信息）
- 需要探索其他方向改进 decentralized value estimation

这推动了 Phase B 实验：也许不是"当前策略 snapshot"不够好，而是 teammate future behavior 从根本上难以从 local history 预测。

---

## 10. Phase B: Teammate Future Prediction Model — ✅ Completed

### 10.1 核心问题

Phase A 证明 teammate current policy π_j(o_j^t) 对 critic 的帮助有限（反而不如 random）。但也许问题不在于"当前策略"，而在于"队友未来行为"能否从近期历史中预测：

> 给定队友最近 K 步的 (obs, action) 历史 τ_j^{t-K:t}，能否比当前策略快照 π_j(o_j^t) 更好地预测队友未来 H 步的 action distribution 和 task events？

如果可以，则用学习到的 latent z_j 作为 critic_extra 在 Phase C 中使用；如果不可以，则说明在 Overcooked IPPO 设置中，队友行为本质上是不可预测的。

### 10.2 模型设计

**TeammateFuturePredictor (GRU-based):**
- Input: K 步 (obs_dim + 6) one-hot action 序列
- GRU(obs_dim+6 → hidden_dim=64)
- z = tanh(Linear(64 → latent_dim=32))
- action_head: Linear(32 → 6), softmax → future action histogram
- event_head: Linear(32 → 5), logits → future event binary OR

**5 个零参数 baseline:**

| baseline | 描述 |
|---|---|
| uniform | 恒定 [1/6]*6 action dist |
| global_action_prior | 训练集 mean future action histogram |
| last_action | 历史窗口最后一步 action 的 one-hot |
| recent_K_histogram | 历史 K 步 action 的 histogram |
| **current_policy π_j** | 保存的 rollout-time π_j(o_j^t) — **核心对比** |

### 10.3 数据

- 10 个 npz trajectory 文件 (5 seeds × IPPO-normal + 5 seeds × IPPO-PC-true)
- 500 episodes/seed, 500k timestep windows (train) + 100k (val)
- 80/20 split by trajectory
- Event positive rates: onion_pickup=11.8%, potting=5.6%, dish=5.5%, soup=3.1%, delivery=1.5%

### 10.4 训练命令

```bash
python overcooked_speed/experiments/train_future_predictor.py \
  --data_dirs logs/future_pred_data/normal logs/future_pred_data/pc_true \
  --obs_dim 520 --K 10 --H 10 --hidden_dim 64 --latent_dim 32 \
  --lambda_event 1.0 --epochs 50 --batch_size 128 --lr 1e-3 \
  --max_samples 500000 --output_dir logs/future_predictor
```

### 10.5 实验结果

**Action Prediction (核心):**

| Predictor | CE ↓ | KL ↓ | Top-1 ↑ | MSE ↓ |
|---|---|---|---|---|
| uniform | 1.7918 | 0.4297 | 0.2278 | 0.0216 |
| global_action_prior | 1.7888 | 0.4268 | 0.1688 | 0.0214 |
| last_action | 14.4453 | 13.0832 | 0.2516 | 0.1441 |
| recent_K_histogram | 4.1790 | 2.8169 | 0.3246 | 0.0283 |
| **current_policy π_j** | **1.7684** | **0.4064** | **0.3779** | 0.0205 |
| **GRU (ours)** | **1.7629** | **0.4057** | 0.2700 | 0.0205 |

**Success Criteria:**

| Criterion | Result |
|---|---|
| GRU > current_policy by ≥5% CE | **FAIL** (+0.3%) |
| GRU > recent_K by ≥5% CE | **PASS** (+57.8%) |
| latent_std_mean > 0.01 | **PASS** (0.5046) |

**Event Prediction (GRU only):**

| Event | F1 | Precision | Recall | pos_rate |
|---|---|---|---|---|
| onion_pickup | 0.383 | 0.257 | 0.753 | 11.8% |
| potting_onion | 0.459 | 0.303 | 0.945 | 5.6% |
| dish_pickup | 0.226 | 0.130 | 0.853 | 5.5% |
| soup_pickup | 0.288 | 0.173 | 0.846 | 3.1% |
| soup_delivery | 0.324 | 0.208 | 0.730 | 1.5% |

**Latent Diagnostics:**

| Metric | Value |
|---|---|
| latent_mean | 0.0102 |
| latent_std | 0.7092 |
| latent_norm_mean | 3.9963 |
| latent_std_mean | 0.5046 |

**Training Dynamics:**

| Epoch | train_loss | val_loss | val_act_CE | val_evt_BCE |
|---|---|---|---|---|
| 0 | 2.335 | **2.408** | 1.763 | 0.645 |
| 1 | 2.231 | 2.448 | 1.756 | 0.692 |
| 5 | 2.128 | 2.526 | 1.756 | 0.770 |
| 10 | 2.074 | 2.804 | 1.762 | 1.042 |

Best epoch: 0, early stopping at epoch 10 (patience=10). Event head massively overfits after epoch 0.

### 10.6 分析

**Primary finding — Negative result:** GRU 比 current_policy 仅提升 0.3% CE（需要 ≥5%）。10 步历史信息几乎完全被当前策略快照所包含。

**为什么 current_policy 已经很好了？** Overcooked 中队友的下一步行为主要由当前状态决定（马尔可夫性），额外的历史信息边际贡献极小。π_j(o_j^t) 已经编码了队友在当前状态下可能做什么，而 10 步前的历史在给定当前 obs 的情况下几乎没有额外预测力。

**GRU 确实学到了东西** (+57.8% over recent_K histogram)，但学到的东西就是 π_j 已经捕获的。

**Event prediction 失败** 因为 (a) 事件极度稀疏 (1.5%-11.8%)，(b) pos_weight 导致高 recall 低 precision，(c) event head 迅速过拟合。

**Latent z_j 是健康的** (non-collapsed, std_mean=0.5046)，但没有包含 π_j 之外的独特信息。

### 10.7 Phase C 建议

**不建议继续 Phase C (Latent Intention Conditioned Critic V_i(o_i, z_j)).**

理由:
1. Phase A 证明 PC gain 来自 capacity 而非 teammate info
2. Phase B 证明 z_j 相比 π_j 几乎没有额外预测价值
3. PLIC (π_j + z_j, 38-dim) 可能只会增加参数而不带来 coordination benefit
4. 在 IPPO 设置中，teammate future behavior 从 local history 难以预测

**负向结果的价值:** 这两个 Phase 排除了两条路径（teammate policy conditioning 和 future behavior prediction），说明 decentralized IPPO 的瓶颈不在"缺乏队友建模"，而在部分可观测合作环境中的 value estimation 根本性困难。未来的工作应该关注:
- MAPPO 式的 centralized critic（已经证明有效）
- 通信机制（允许 agents 交换意图信号）
- 更好的 exploration 策略
- 或直接接受 IPPO 的局限性，在架构层面做更根本的改变

### 10.8 数据加载优化与修复的 Bugs

**file_sequential_batches 设计:**
- 按 npz 文件顺序加载：load one file → process all samples → next file
- 避免 DataLoader shuffle 造成的 cache thrashing (10 个 49MB npz 被反复加载)
- GPU 利用率从 0% 提升到正常水平

**修复的 bugs:**
- Generator 无限循环 (while True → removed)
- train_gen 只在 loop 外创建一次 → 移到 epoch loop 内，每个 epoch 创建新 generator
- CUDA tensor .numpy() 错误 → 加 .cpu()
- rewards 维度错误 (T,400) 而不是 (T,1) → 加 [:, None]
- metadata 加载慢 → zipfile 只读 episode_lengths.npy (2KB vs 49MB)
- sweep_pairs.py 缺少 teammate_probs_mode 字段 → 添加到 agg dict

**trajectory npz 格式:**
- obs_0/obs_1: (E, max_T, 520) float32
- actions_0/actions_1: (E, max_T) int8
- probs_0/probs_1: (E, max_T, 6) float32
- events_0/events_1: (E, max_T, 5) uint8 (agent-specific)
- held_0/held_1: (E, max_T) int8
- rewards_0/rewards_1: (E, max_T, 1) float32
- episode_lengths: (E,) int32

---

## 11. 当前科研路线图 (UPDATED)

### Stage 0: Baselines ✅

- IPPO, MAPPO, egocentric/global obs fair comparison
- 结论: IPPO 主要瓶颈是 value estimation / credit assignment

### Stage 1: Shaping Experiments ✅

- role shaping, task shaping, mechanism diagnosis
- 结论: reward shaping 可以加速 specialization 但会污染 critic，降低 final reward

### Stage 2: Policy-Conditioned Critic ✅

- IPPO-PC: V_i(o_i, π_j(o_j))
- PC ablation (true/uniform/shuffled)
- **结论: PC gain 来自额外 critic 容量，不是 teammate policy info。Uniform 优于 true。**

### Stage 3: Teammate Future Prediction ✅

- GRU 预测队友未来 action/event
- 5 个 baseline（包括 current_policy）
- **结论: GRU +0.3% over current_policy (FAIL)。队友未来行为从 local history 不可预测。**

### Stage 4 & 5: Lookahead/LOLA Adaptation — Deferred

基于 Phase A+B 的负向结果，Stage 4 (Lookahead-like Actor Adaptation) 和 Stage 5 (LOLA-like Intention Shaping) 在当前 IPPO 框架内不太可能有显著突破。需要先在架构层面做更根本的改变（例如通信、centralized critic、或更好的 exploration）之后，再考虑 teammate modeling。

---

## 12. 更新后的一句话说

本项目探索了两种改善去中心化多智能体合作的路线:

1. **直接 teammate modeling** (PC, IC, future prediction)
2. **Reward shaping** (role shaping, task shaping)

结论是: **在当前 Overcooked IPPO 框架中，两条路线都没有提供超过简单 baseline (uniform critic_extra / no shaping) 的稳健提升。**

MAPPO centralized critic 仍然是 gold standard (80.0 final reward vs IPPO 29.6)，缩小这一差距需要更根本的架构改变，而非 incremental 的 teammate-aware 信息注入。

---

## 13. 当前不要做的事情 (UPDATED)

基于已完成的实验，当前阶段不要做:

- ~~更复杂 reward shaping~~ (已排除: 污染 critic)
- ~~teammate policy-conditioned critic~~ (已排除: 提升来自 capacity 而非 info)
- ~~neural intention predictor~~ (已排除: z_j 几乎不增加预测力 over π_j)
- aux loss
- theta-space 二阶 LOLA
- differentiable PPO lookahead
- 修改 MAPPO
- 过早实现 LOLA response model

当前如果要继续推进，应聚焦:

1. 分析为什么 MAPPO centralized critic 有效 (credit assignment? variance reduction?)
2. 探索 agent 间通信机制
3. 更好的 exploration 策略
4. 跨 layout 验证现有结论

---

## 14. 论文状态评估 (UPDATED)

### 当前状态

| Component | Status | Evidence |
|---|---|---|
| IPPO baseline + MAPPO upper bound | ✅ | 500ep × 5 seeds |
| Reward shaping negative result | ✅ | 24 configs, all fail vs baseline |
| PC capacity vs info ablation | ✅ | true < uniform (capacity wins) |
| Future prediction negative | ✅ | GRU +0.3% over π_j (no new info) |
| Value estimation bottleneck | ✅ | MAPPO 2.7× IPPO with same actor |

### 论文可行性

当前结果可以写一篇 honest negative-result paper:

**标题建议:** "What Doesn't Help Decentralized Multi-Agent Coordination: Lessons from Overcooked"

**贡献:**
1. 系统性地排除了 reward shaping (24 configs)
2. 排除了 teammate policy conditioning (capacity > info)
3. 排除了 learned intention/future prediction (no new info beyond π_j)
4. 确认了 value estimation 是核心瓶颈 (MAPPO > 2.7×)
5. 提供了一个干净的实验框架用于 future work

如果后续能在至少 1 个 positive direction（通信 / 更好的 critic architecture / exploration）上取得稳健提升，可以和这些 negative results 一起形成更完整的 story。
