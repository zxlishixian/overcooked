"""BeliefPPOAgent: Variational Belief PPO agent (POMDP belief-MDP route).

q_φ(b_t | o_t, h_t [, m_t]):
  - b_t ~ N(mu, sigma) — variational posterior over hidden state
  - π(a_t | o_t, b_t) — belief-conditioned policy
  - V(o_t, b_t) — belief-conditioned critic (decentralized)

Auxiliary losses:
  - Reward prediction: D_r(b_t, a_t) → r_hat, MSE loss
  - KL regularization: KL(N(mu, sigma) || N(0, I)) with free-bits + annealing
  - Optional obs prediction: D_o(b_t, a_t) → e_obs_next_hat (default off)

PPO gradients flow through the belief encoder (per-minibatch recompute).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pg_agent import PGAgent
from ..models.belief_encoder import (
    VariationalBeliefEncoder, RewardPredictor, ObsPredictor, QueryOutcomePredictor,
)
from ..models.belief_actor_critic import BeliefActorCritic
from ..models.memory_bank import MemoryBank
from ..models.historical_context import HistoricalContextMemory


class BeliefPPOAgent(PGAgent):
    """Belief-PPO agent with variational belief inference.

    Extends PGAgent. Key additions:
    - VariationalBeliefEncoder (mu, logvar → b_t via reparameterization)
    - Local history ring buffer (L steps)
    - Optional MemoryBank (cross-episode retrieval, rank-based top-percent)
    - Reward prediction auxiliary head
    - KL regularization with free-bits + annealing
    """

    def __init__(self, agent_id, obs_dim, n_actions=6, lr=1e-3, gamma=0.99,
                 hidden_dim=256, device='cpu',
                 ppo_epochs=10, ppo_clip=0.2, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 critic_extra_dim=0,
                 # Belief-specific
                 belief_dim=32,
                 history_len=10,
                 obs_out=64,
                 hist_out=64,
                 gru_hidden=64,
                 query_dim=64,
                 belief_lr_scale=0.5,
                 # Aux loss weights
                 rew_coef=0.05,
                 obs_pred_coef=0.0,
                 # KL
                 kl_coef=1e-4,
                 kl_warmup_episodes=100,
                 free_nats=1.0,
                 # Memory (MVP-A: disabled)
                 belief_use_memory=False,
                 memory_size=2000,
                 memory_top_percent=0.05,
                 memory_topk_max=20,
                 memory_min_entries=10,
                 memory_include_teammate_action=False,
                 # New: deterministic belief, nonzero reward weighting
                 belief_deterministic=False,
                 belief_nonzero_reward_weight=1.0,
                 # MVP-B2: query outcome, temperature, query_hidden
                 belief_query_hidden=128,
                 belief_query_outcome_coef=0.01,
                 belief_memory_temperature=0.1,
                 # MVP-C: Historical Context
                 belief_use_historical_context=False,
                 historical_top_percent=0.05,
                 historical_topk_max=40,
                 historical_min_entries=10,
                 historical_attn_dim=64,
                 **_kwargs):
        super().__init__(agent_id, obs_dim, n_actions, lr, gamma, hidden_dim,
                         device, ppo_epochs, ppo_clip, gae_lambda,
                         vf_coef, ent_coef, max_grad_norm, critic_extra_dim=0)

        self.obs_dim = obs_dim

        # Belief hyperparams
        self.belief_dim = belief_dim
        self.history_len = history_len
        self.obs_out = obs_out
        self.hist_out = hist_out
        self.gru_hidden = gru_hidden
        self.query_dim = query_dim
        self.belief_lr_scale = belief_lr_scale
        self.rew_coef = rew_coef
        self.obs_pred_coef = obs_pred_coef
        self.kl_coef = kl_coef
        self.kl_warmup_episodes = kl_warmup_episodes
        self.free_nats = free_nats
        self.use_memory = belief_use_memory
        self.memory_include_teammate_action = memory_include_teammate_action
        self.deterministic = belief_deterministic
        self.nonzero_reward_weight = belief_nonzero_reward_weight
        self.query_hidden = belief_query_hidden
        self.query_outcome_coef = belief_query_outcome_coef
        self.memory_temperature = belief_memory_temperature
        self.use_historical_context = belief_use_historical_context
        self.historical_top_percent = historical_top_percent
        self.historical_topk_max = historical_topk_max
        self.historical_min_entries = historical_min_entries
        self.historical_attn_dim = historical_attn_dim

        # v_dim for memory values
        if memory_include_teammate_action:
            self.v_dim = obs_out + 1 + n_actions
        else:
            self.v_dim = obs_out + 1

        # Belief encoder
        self.belief_encoder = VariationalBeliefEncoder(
            obs_dim=obs_dim, n_actions=n_actions,
            belief_dim=belief_dim, history_len=history_len,
            obs_out=obs_out, hist_out=hist_out,
            gru_hidden=gru_hidden, query_dim=query_dim, query_hidden=belief_query_hidden,
            use_memory=belief_use_memory, v_dim=self.v_dim if belief_use_memory else None,
            deterministic=belief_deterministic,
            use_historical_context=belief_use_historical_context,
            historical_attn_dim=historical_attn_dim,
        ).to(device)

        # Belief-conditioned actor-critic
        self.ac_network = BeliefActorCritic(
            obs_dim=obs_dim, belief_dim=belief_dim,
            hidden_dim=hidden_dim, n_actions=n_actions, device=device,
        )

        # Auxiliary predictors
        self.reward_predictor = RewardPredictor(
            belief_dim=belief_dim, n_actions=n_actions, hidden=64,
        ).to(device)

        # Query outcome predictor (MVP-B2)
        self.query_outcome_predictor = None
        if belief_query_outcome_coef > 0 and belief_use_memory:
            self.query_outcome_predictor = QueryOutcomePredictor(
                query_dim=query_dim, obs_out=obs_out, hidden=128,
            ).to(device)

        self.obs_predictor = None
        if obs_pred_coef > 0:
            self.obs_predictor = ObsPredictor(
                belief_dim=belief_dim, n_actions=n_actions,
                obs_out=obs_out, hidden=64,
            ).to(device)

        # Historical context memory (MVP-C)
        self.historical_memory = None
        if belief_use_historical_context:
            self.historical_memory = HistoricalContextMemory(
                memory_size=memory_size, obs_dim=obs_dim, device=device,
            )

        # Memory bank (optional, legacy)
        self.memory_bank = None
        if belief_use_memory and not belief_use_historical_context:
            self.memory_bank = MemoryBank(
                memory_size=memory_size, v_dim=self.v_dim,
                top_percent=memory_top_percent, topk_max=memory_topk_max,
                min_entries=memory_min_entries,
                temperature=belief_memory_temperature, device=device,
            )

        # Single optimizer with param groups
        ac_params = list(self.ac_network.parameters())
        enc_params = list(self.belief_encoder.parameters())
        aux_params = list(self.reward_predictor.parameters())
        if self.obs_predictor is not None:
            aux_params += list(self.obs_predictor.parameters())
        if self.query_outcome_predictor is not None:
            aux_params += list(self.query_outcome_predictor.parameters())

        self.optimizer = torch.optim.Adam([
            {'params': ac_params, 'lr': lr},
            {'params': enc_params, 'lr': lr * belief_lr_scale},
            {'params': aux_params, 'lr': lr},
        ])

        self.policy = self.ac_network  # for backward compat
        self._log_param_counts()

        # Episode buffers
        self._beliefs = []
        self._next_obs = []  # o_{t+1}, for obs prediction target
        self._history_buffer = []  # deque of (obs, action, reward) tuples
        self._teammate_actions = []  # teammate action per step (MVP-C)

        # hist_input_dim for sequence building
        self.hist_input_dim = obs_dim + 1 + n_actions

    def _log_param_counts(self):
        def count(m):
            return sum(p.numel() for p in m.parameters())

        n_enc = count(self.belief_encoder)
        n_ac = count(self.ac_network)
        n_rew = count(self.reward_predictor)
        n_obs = count(self.obs_predictor) if self.obs_predictor else 0
        n_mem = 0
        if self.memory_bank is not None:
            n_mem = sum(k.numel() for k in [self.memory_bank._keys, self.memory_bank._values]
                       if k.numel() > 0)
        n_total = n_enc + n_ac + n_rew + n_obs

        print(f"[BeliefPPO-{self.agent_id}] Parameter counts:")
        print(f"  belief_encoder:     {n_enc:,}")
        print(f"  belief_ac:          {n_ac:,}")
        print(f"  reward_predictor:   {n_rew:,}")
        if n_obs > 0:
            print(f"  obs_predictor:      {n_obs:,}")
        if n_mem > 0:
            print(f"  memory_bank (keys+vals): {n_mem:,}")
        print(f"  total:              {n_total:,}")

        self._param_counts = {
            'belief_encoder': n_enc, 'belief_ac': n_ac,
            'reward_predictor': n_rew, 'obs_predictor': n_obs,
            'memory_bank': n_mem, 'total': n_total,
        }

    @property
    def param_counts(self):
        return getattr(self, '_param_counts', {})

    # ── History management ──

    def push_history(self, obs, action, reward):
        """Push a past transition to the history ring buffer.

        Called in run_pair episode loop BEFORE act().
        Uses (t-1) data: obs_{t-1}, action_{t-1}, reward_{t-1}.
        For step 0, called with zeros.
        """
        self._history_buffer.append((obs.copy(), action, float(reward)))
        if len(self._history_buffer) > self.history_len:
            self._history_buffer = self._history_buffer[-self.history_len:]

    # ── Override PGAgent.act ──

    def act(self, obs, deterministic=False, critic_extra=None):
        """Build belief from obs + history [+ historical context], then sample action."""
        # Build history_seq from ring buffer
        history_seq = self._build_history_seq()

        # Historical context (MVP-C)
        hist_mem_obs, hist_mem_actions = None, None
        if self.use_historical_context and self.historical_memory is not None:
            hist_mem_obs, hist_mem_actions = self.historical_memory.get_all()

        # Legacy: Retrieve memory
        retrieved_mem = None
        if self.use_memory and self.memory_bank is not None and len(self.memory_bank) > 0:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            hist_t = torch.as_tensor(history_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, _, _, query, _ = self.belief_encoder.encode(obs_t, hist_t)
                retrieved_mem = self.memory_bank.retrieve(query)
            retrieved_mem = retrieved_mem.squeeze(0).cpu().numpy()
        elif self.use_memory and not self.use_historical_context:
            retrieved_mem = np.zeros(self.v_dim, dtype=np.float32)

        # Encode belief (no grad during rollout)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            hist_t = torch.as_tensor(history_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            mem_t = torch.as_tensor(retrieved_mem, dtype=torch.float32, device=self.device).unsqueeze(0) if retrieved_mem is not None else None
            _, _, b_t, _, _ = self.belief_encoder.encode(
                obs_t, hist_t,
                retrieved_memory=mem_t if self.use_memory else None,
                hist_mem_obs=hist_mem_obs, hist_mem_actions=hist_mem_actions)
            belief_np = b_t.squeeze(0).cpu().numpy()

        # Sample action from belief-conditioned AC
        action, log_prob, value = self.ac_network.act(obs, belief_np, deterministic)

        # Store to buffers
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._beliefs.append(belief_np)

        return action

    def _build_history_seq(self):
        """Build (L, hist_input_dim) numpy array from ring buffer.

        Left-padded with zeros for t < L.
        """
        L = self.history_len
        seq = np.zeros((L, self.hist_input_dim), dtype=np.float32)
        start_idx = L - len(self._history_buffer)
        for i, (obs, action, reward) in enumerate(self._history_buffer):
            idx = start_idx + i
            if 0 <= idx < L:
                action_oh = np.zeros(self.n_actions, dtype=np.float32)
                if 0 <= action < self.n_actions:
                    action_oh[action] = 1.0
                seq[idx, :self.obs_dim] = obs
                seq[idx, self.obs_dim] = reward
                seq[idx, self.obs_dim + 1:] = action_oh
        return seq

    def store_next_obs(self, next_obs):
        """Store o_{t+1} for obs prediction target."""
        self._next_obs.append(next_obs.copy())

    def store_teammate_action(self, action):
        """Store teammate action for historical context memory (MVP-C)."""
        self._teammate_actions.append(action)

    # ── Override PGAgent.end_episode ──

    def end_episode(self, episode_num=0, aux_adv=None, aux_coef=0.0,
                    shuffle_critic_extra=False, teammate_characteristic=None,
                    **_kwargs):
        """Compute GAE, run PPO + aux losses, optionally store to memory.

        Each PPO minibatch recomputes beliefs with current encoder (WITH grad).
        Rollout-time old_log_probs are fixed.
        """
        if len(self._rewards) == 0:
            self._clear_all_buffers()
            return self._empty_belief_log()

        self._dones[-1] = True

        # ── GAE ──
        advantages, returns = self._compute_gae(
            self._rewards, self._values, self._dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Tensors ──
        obs_t = torch.as_tensor(np.stack(self._obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(self._actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(self._log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(self._rewards, dtype=torch.float32, device=self.device)

        # Action onehots for reward/obs predictors
        actions_oh = F.one_hot(actions_t, num_classes=self.n_actions).float()

        # Next obs targets for obs predictor
        next_obs_t = None
        if self.obs_pred_coef > 0 and len(self._next_obs) > 0:
            next_obs_t = torch.as_tensor(np.stack(self._next_obs), dtype=torch.float32,
                                         device=self.device)

        # History sequences
        history_seqs_t = self._build_history_sequences_tensor()

        # Historical context (MVP-C): get memory for PPO update and store episode
        hist_mem_obs_t, hist_mem_act_t = None, None
        hist_diag = {}
        if self.use_historical_context and self.historical_memory is not None:
            hist_mem_obs_t, hist_mem_act_t = self.historical_memory.get_all()

        # Memory retrieval (if enabled, legacy)
        memory_t = None
        memory_diag = {}
        if self.use_memory and self.memory_bank is not None:
            with torch.no_grad():
                _, _, _, queries = self.belief_encoder.encode(obs_t, history_seqs_t)
                retrieved, mem_diag = self.memory_bank.retrieve(queries, return_diagnostics=True)
            memory_t = retrieved  # (T, v_dim)
            memory_diag = mem_diag

        T = len(obs_t)

        # ── KL annealing ──
        effective_kl_coef = self.kl_coef * min(episode_num / max(self.kl_warmup_episodes, 1), 1.0)

        # ── Accumulators ──
        total_loss = 0.0; total_entropy = 0.0; total_value_loss = 0.0
        total_rew_loss = 0.0; total_rew_loss_nz = 0.0; total_obs_loss = 0.0
        total_q_out_loss = 0.0
        total_kl_loss = 0.0; total_kl_raw = 0.0
        total_grad_norm = 0.0; total_belief_grad_norm = 0.0
        clip_count = 0; n_updates = 0
        ratio_sum = 0.0; ratio_sum_sq = 0.0; ratio_max_val = 0.0; ratio_count = 0
        nz_rew_count = 0; total_n = 0

        # ── PPO update loop ──
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, 64):
                end = min(start + 64, T)
                mb_idx = perm[start:end]

                mb_obs = obs_t[mb_idx]
                mb_act = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]
                mb_rew = rewards_t[mb_idx]
                mb_act_oh = actions_oh[mb_idx]
                mb_hist = history_seqs_t[mb_idx]

                mb_mem = None
                if self.use_memory and memory_t is not None:
                    mb_mem = memory_t[mb_idx]

                # Recompute beliefs WITH grad
                mu, logvar, mb_beliefs = self.belief_encoder(
                    mb_obs, mb_hist, retrieved_memory=mb_mem,
                    hist_mem_obs=hist_mem_obs_t, hist_mem_actions=hist_mem_act_t)

                # ── PPO loss ──
                new_lp, values, entropy = self.ac_network.evaluate(
                    mb_obs, mb_beliefs, mb_act)

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()

                # ── Reward prediction loss ──
                rew_hat = self.reward_predictor(mb_beliefs, mb_act_oh).squeeze(-1)
                rew_loss_all = F.mse_loss(rew_hat, mb_rew, reduction='none')

                # Nonzero reward weighting
                nz_mask = mb_rew.abs() > 1e-8
                weights = torch.ones_like(mb_rew)
                weights[nz_mask] = self.nonzero_reward_weight
                rew_loss = (weights * rew_loss_all).mean()

                # Nonzero reward loss (unweighted, for diagnostics)
                rew_loss_nz = rew_loss_all[nz_mask].mean() if nz_mask.any() else torch.zeros(
                    (), dtype=torch.float32, device=self.device)
                nz_rew_count += int(nz_mask.sum().item())
                total_n += len(mb_idx)

                # ── Obs prediction loss (optional) ──
                obs_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if self.obs_pred_coef > 0 and self.obs_predictor is not None and next_obs_t is not None:
                    mb_next_obs = next_obs_t[mb_idx]
                    with torch.no_grad():
                        next_obs_target = self.belief_encoder.obs_encoder(mb_next_obs)
                    next_obs_hat = self.obs_predictor(mb_beliefs, mb_act_oh)
                    obs_loss = F.mse_loss(next_obs_hat, next_obs_target)

                # ── Query outcome loss (MVP-B2) ──
                q_out_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if self.query_outcome_predictor is not None and self.use_memory:
                    # Get query vectors from the encoder for this minibatch
                    _, _, _, q_vecs = self.belief_encoder.encode(mb_obs, mb_hist)
                    # Target: [obs_emb_next, reward]
                    with torch.no_grad():
                        if len(self._next_obs) > 0:
                            next_obs_all = np.stack(self._next_obs)
                            mb_next_obs = torch.as_tensor(
                                next_obs_all[mb_idx.cpu().numpy()],
                                dtype=torch.float32, device=self.device)
                        else:
                            mb_next_obs = mb_obs
                        next_obs_emb = self.belief_encoder.obs_encoder(mb_next_obs)
                        outcome_target = torch.cat([next_obs_emb, mb_rew.unsqueeze(-1)], dim=-1)
                    outcome_hat = self.query_outcome_predictor(q_vecs)
                    q_out_loss = F.mse_loss(outcome_hat, outcome_target)

                # ── KL loss (free-bits + annealing) ──
                kl_loss, kl_raw = VariationalBeliefEncoder.compute_kl(
                    mu, logvar, free_nats=self.free_nats)

                # ── Total loss ──
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * entropy.mean()
                        + self.rew_coef * rew_loss
                        + self.obs_pred_coef * obs_loss
                        + self.query_outcome_coef * q_out_loss
                        + effective_kl_coef * kl_loss)

                self.optimizer.zero_grad()
                loss.backward()

                # Belief encoder grad norm
                bel_norm_sq = 0.0
                for p in self.belief_encoder.parameters():
                    if p.grad is not None:
                        bel_norm_sq += p.grad.data.norm(2).item() ** 2
                bel_grad_norm = bel_norm_sq ** 0.5

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.ac_network.parameters()) +
                    list(self.belief_encoder.parameters()) +
                    list(self.reward_predictor.parameters()) +
                    (list(self.obs_predictor.parameters()) if self.obs_predictor else []) +
                    (list(self.query_outcome_predictor.parameters()) if self.query_outcome_predictor else []),
                    self.max_grad_norm)
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()

                self.optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.mean().item()
                total_value_loss += value_loss.item()
                total_rew_loss += rew_loss.item()
                total_rew_loss_nz += rew_loss_nz.item()
                total_obs_loss += obs_loss.item()
                total_q_out_loss += q_out_loss.item()
                total_kl_loss += kl_loss.item()
                total_kl_raw += kl_raw.item()
                total_grad_norm += grad_norm
                total_belief_grad_norm += bel_grad_norm

                with torch.no_grad():
                    r_vals = ratio.detach()
                    ratio_sum += r_vals.sum().item()
                    ratio_sum_sq += (r_vals ** 2).sum().item()
                    ratio_max_val = max(ratio_max_val, float(r_vals.max().item()))
                    ratio_count += r_vals.numel()
                    cfrac = ((ratio - 1).abs() > self.ppo_clip).float().mean().item()
                clip_count += cfrac
                n_updates += 1

        n_updates = max(1, n_updates)

        # ── Store episode to historical memory (MVP-C) ──
        if self.use_historical_context and self.historical_memory is not None and T > 0:
            self.historical_memory.add(self._obs, self._teammate_actions)

        # ── Store episode to memory bank (legacy) ──
        if self.use_memory and self.memory_bank is not None and T > 0:
            self.memory_bank.store_episode(
                history_seqs=history_seqs_t,
                next_obs_embs=self._build_all_obs_raw(obs_t),
                rewards=rewards_t,
                query_mlp=self.belief_encoder.query_mlp,
                obs_encoder=self.belief_encoder.obs_encoder,
                history_encoder=self.belief_encoder.history_encoder,
                teammate_actions=(actions_t if self.memory_include_teammate_action else None),
            )

        # ── Diagnostics ──
        ratio_mean_val = ratio_sum / max(ratio_count, 1)
        ratio_std_val = (max(ratio_sum_sq / max(ratio_count, 1) - ratio_mean_val ** 2, 0.0)) ** 0.5

        with torch.no_grad():
            # Final beliefs (no memory retrieval, for diagnostics)
            _, _, _, _, hist_diag_final = self.belief_encoder.encode(
                obs_t, history_seqs_t,
                hist_mem_obs=hist_mem_obs_t, hist_mem_actions=hist_mem_act_t)
            if hist_diag_final:
                hist_diag = hist_diag_final
            mu_final, logvar_final, b_final = self.belief_encoder(
                obs_t, history_seqs_t,
                hist_mem_obs=hist_mem_obs_t, hist_mem_actions=hist_mem_act_t)
            belief_norm_val = float(b_final.norm(dim=-1).mean().item())
            belief_mu_norm_val = float(mu_final.norm(dim=-1).mean().item())
            # Correct belief_std: use logvar, not mu
            belief_std_per_dim = torch.exp(0.5 * logvar_final)  # (batch, belief_dim)
            belief_std_val = float(belief_std_per_dim.mean().item())
            belief_std_min_val = float(belief_std_per_dim.min().item())
            belief_std_max_val = float(belief_std_per_dim.max().item())
            belief_logvar_mean_val = float(logvar_final.mean().item())
            belief_logvar_min_val = float(logvar_final.min().item())
            belief_logvar_max_val = float(logvar_final.max().item())

        # Memory diagnostics
        mem_size = len(self.memory_bank) if self.memory_bank else 0
        zero_frac = 1.0 if mem_size == 0 else 0.0

        self._clear_all_buffers()

        return {
            'loss': total_loss / n_updates,
            'mean_return': float(returns_t.mean().item()) if T > 0 else 0.0,
            'entropy': total_entropy / n_updates,
            'grad_norm': total_grad_norm / n_updates,
            'approx_kl': 0.0,  # legacy
            'value_loss': total_value_loss / n_updates,
            'aux_loss': total_rew_loss / n_updates,
            'value_mean': 0.0,
            'explained_variance': 0.0,
            'critic_extra_entropy': 0.0,
            'value_extra_sensitivity': 0.0,
            'sensitivity_note': None,
            'value_mean_shuffled': None,
            'explained_variance_shuffled': None,
            'shuffled_diag_note': None,
            # Belief diagnostics
            'loss_rew': total_rew_loss / n_updates,
            'loss_rew_nonzero': total_rew_loss_nz / n_updates,
            'nonzero_reward_frac': nz_rew_count / max(total_n, 1),
            'loss_obs': total_obs_loss / n_updates,
            'loss_query_outcome': total_q_out_loss / n_updates,
            'loss_kl': total_kl_loss / n_updates,
            'kl_raw': total_kl_raw / n_updates,
            'effective_kl_coef': effective_kl_coef,
            'belief_norm': belief_norm_val,
            'belief_mu_norm': belief_mu_norm_val,
            'belief_std_mean': belief_std_val,
            'belief_std_min': belief_std_min_val,
            'belief_std_max': belief_std_max_val,
            'belief_logvar_mean': belief_logvar_mean_val,
            'belief_logvar_min': belief_logvar_min_val,
            'belief_logvar_max': belief_logvar_max_val,
            'belief_encoder_grad_norm': total_belief_grad_norm / n_updates,
            'ratio_mean': ratio_mean_val,
            'ratio_std': ratio_std_val,
            'ratio_max': ratio_max_val,
            'clip_fraction': clip_count / n_updates,
            'approx_kl_ppo': 0.0,
            'logprob_delta_sq': 0.0,
            # Memory diagnostics
            'memory_size': mem_size,
            'retrieval_k_selected': memory_diag.get('k_selected', 0),
            'retrieval_top_percent': self.memory_bank.top_percent if self.memory_bank else 0.0,
            'retrieval_sim_mean_selected': memory_diag.get('sim_mean', 0.0),
            'retrieval_sim_max': memory_diag.get('sim_max', 0.0),
            'retrieval_sim_min_selected': memory_diag.get('sim_min', 0.0),
            'retrieval_sim_gap': memory_diag.get('sim_gap', 0.0),
            'retrieval_weight_max': memory_diag.get('weight_max', 0.0),
            'retrieval_weight_min': memory_diag.get('weight_min', 0.0),
            'retrieval_weight_std': memory_diag.get('weight_std', 0.0),
            'retrieval_entropy': memory_diag.get('entropy', 0.0),
            'retrieval_zero_fraction': zero_frac,
            # Historical context diagnostics (MVP-C)
            'hist_memory_size': len(self.historical_memory) if self.historical_memory else 0,
            'hist_k_selected': hist_diag.get('k_selected', 0),
            'hist_sim_mean_selected': hist_diag.get('sim_mean', 0.0),
            'hist_sim_max': hist_diag.get('sim_max', 0.0),
            'hist_sim_min_selected': hist_diag.get('sim_min', 0.0),
            'hist_sim_gap': hist_diag.get('sim_gap', 0.0),
            'hist_attention_entropy': hist_diag.get('entropy', 0.0),
            'hist_weight_max': hist_diag.get('weight_max', 0.0),
            'hist_weight_std': hist_diag.get('weight_std', 0.0),
        }

    # ── Internal helpers ──

    def _build_history_sequences_tensor(self):
        """Build (T, L, hist_input_dim) tensor from stored episode data.

        For each timestep t, history_seq[t] contains:
        [(o_{t-L}, a_{t-L}, r_{t-L}), ..., (o_{t-1}, a_{t-1}, r_{t-1})]
        Left-padded with zeros for t < L.
        """
        T = len(self._obs)
        L = self.history_len
        seqs = np.zeros((T, L, self.hist_input_dim), dtype=np.float32)

        for t in range(T):
            for offset in range(L):
                src = t - (L - 1) + offset
                if 0 <= src < T:
                    action_oh = np.zeros(self.n_actions, dtype=np.float32)
                    if 0 <= self._actions[src] < self.n_actions:
                        action_oh[self._actions[src]] = 1.0
                    seqs[t, offset, :self.obs_dim] = self._obs[src]
                    seqs[t, offset, self.obs_dim] = self._rewards[src]
                    seqs[t, offset, self.obs_dim + 1:] = action_oh

        return torch.as_tensor(seqs, dtype=torch.float32, device=self.device)

    def _build_all_obs_raw(self, obs_t):
        """Build raw observation tensor for all steps + next_obs.

        Returns: (T+1, obs_dim) — raw o_0 ... o_T
        where o_T is the last next_obs.
        MemoryBank.store_episode will encode these via ObsEncoder internally.
        """
        if len(self._next_obs) > 0:
            next_t = np.stack(self._next_obs)
            all_obs_np = np.concatenate([self._obs, next_t[-1:]], axis=0)
        else:
            all_obs_np = np.array(self._obs)
        return torch.as_tensor(all_obs_np, dtype=torch.float32, device=self.device)

    # ── Buffer management ──

    def _clear_all_buffers(self):
        super()._clear_buffer()
        self._beliefs = []
        self._next_obs = []
        self._history_buffer = []
        self._teammate_actions = []

    def _empty_belief_log(self):
        return {
            'loss': 0.0, 'mean_return': 0.0, 'entropy': 0.0,
            'grad_norm': 0.0, 'approx_kl': 0.0, 'value_loss': 0.0,
            'aux_loss': 0.0,
            'value_mean': 0.0, 'explained_variance': 0.0,
            'critic_extra_entropy': 0.0, 'value_extra_sensitivity': 0.0,
            'sensitivity_note': None,
            'value_mean_shuffled': None,
            'explained_variance_shuffled': None,
            'shuffled_diag_note': None,
            'loss_rew': 0.0, 'loss_rew_nonzero': 0.0, 'nonzero_reward_frac': 0.0,
            'loss_obs': 0.0, 'loss_query_outcome': 0.0, 'loss_kl': 0.0, 'kl_raw': 0.0,
            'effective_kl_coef': 0.0,
            'belief_norm': 0.0, 'belief_mu_norm': 0.0, 'belief_std_mean': 0.0,
            'belief_std_min': 0.0, 'belief_std_max': 0.0,
            'belief_logvar_mean': 0.0, 'belief_logvar_min': 0.0, 'belief_logvar_max': 0.0,
            'belief_encoder_grad_norm': 0.0,
            'ratio_mean': 0.0, 'ratio_std': 0.0, 'ratio_max': 0.0,
            'clip_fraction': 0.0, 'approx_kl_ppo': 0.0, 'logprob_delta_sq': 0.0,
            'memory_size': 0, 'retrieval_k_selected': 0, 'retrieval_top_percent': 0.0,
            'retrieval_sim_mean_selected': 0.0, 'retrieval_sim_max': 0.0,
            'retrieval_sim_min_selected': 0.0, 'retrieval_entropy': 0.0,
            'retrieval_zero_fraction': 0.0,
            'hist_memory_size': 0, 'hist_k_selected': 0,
            'hist_sim_mean_selected': 0.0, 'hist_sim_max': 0.0,
            'hist_sim_min_selected': 0.0, 'hist_sim_gap': 0.0,
            'hist_attention_entropy': 0.0, 'hist_weight_max': 0.0,
            'hist_weight_std': 0.0,
        }

    # ── Mode switching ──

    def train(self):
        self.ac_network.train()
        self.belief_encoder.train()
        self.reward_predictor.train()
        if self.obs_predictor is not None:
            self.obs_predictor.train()

    def eval(self):
        self.ac_network.eval()
        self.belief_encoder.eval()
        self.reward_predictor.eval()
        if self.obs_predictor is not None:
            self.obs_predictor.eval()

    def get_parameters(self):
        params = list(self.ac_network.parameters()) + list(self.belief_encoder.parameters())
        params += list(self.reward_predictor.parameters())
        if self.obs_predictor is not None:
            params += list(self.obs_predictor.parameters())
        return np.concatenate([p.data.cpu().numpy().ravel() for p in params])
