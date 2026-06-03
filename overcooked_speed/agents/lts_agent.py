"""LTSAgent: Dynamic-Belief-style PPO agent for Overcooked.

Each agent maintains a belief embedding b_t^i that approximates the teammate's
latent behavioral state. The belief is built from:
  1. Current local observation o_t^i (ObsEncoder)
  2. Intra-episode history of teammate observable features (IntraEncoder/GRU)
  3. Cross-episode memory of episode-level teammate characteristics (InterEncoder)

Actor and Critic condition on [obs || belief] (decentralized critic).
Auxiliary heads D_y, D_f, D_c provide additional representation learning signal.
PPO gradients flow through the belief encoder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pg_agent import PGAgent
from ..models.lts_belief import LTSBeliefNetwork
from ..models.lts_actor_critic import LTSActorCritic
from ..utils.teammate_features import (
    extract_teammate_y, build_intra_feature, compute_episode_characteristic
)
from ..utils.inter_episode_memory import InterEpisodeMemory


class LTSAgent(PGAgent):
    """LTS-PPO agent with Dynamic Belief over teammate latent state.

    Extends PGAgent. Key additions:
    - Belief encoder network (LTSBeliefNetwork)
    - Belief-conditioned Actor-Critic (LTSActorCritic)
    - Intra-episode history ring buffer
    - Inter-episode memory of teammate characteristics
    - Auxiliary prediction losses (D_y, D_f, D_c)

    Single optimizer with param groups: AC at lr, belief encoder at lr * belief_lr_scale.
    """

    def __init__(self, agent_id, obs_dim, n_actions=6, lr=1e-3, gamma=0.99,
                 hidden_dim=256, device='cpu',
                 ppo_epochs=10, ppo_clip=0.2, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 critic_extra_dim=0,
                 # LTS-specific
                 L=20, M=10, K=10,
                 belief_dim=64,
                 intra_feat_dim=23, y_dim=16,
                 future_dim=None, c_dim=17,
                 obs_enc_hidden=128,
                 intra_hidden=64, inter_hidden=64,
                 enc_out_dim=64,
                 alpha=0.1, beta=0.05, eta=0.05,
                 belief_cons_coef=0.0,
                 belief_lr_scale=0.5,
                 df_event=False,
                 **_kwargs):
        # Call PGAgent.__init__ with critic_extra_dim=0 (belief replaces it)
        super().__init__(agent_id, obs_dim, n_actions, lr, gamma, hidden_dim,
                         device, ppo_epochs, ppo_clip, gae_lambda,
                         vf_coef, ent_coef, max_grad_norm, critic_extra_dim=0)

        # LTS hyperparameters
        self.L = L
        self.M = M
        self.K = K
        self.belief_dim = belief_dim
        self.intra_feat_dim = intra_feat_dim
        self.y_dim = y_dim
        self.df_event = df_event
        # Auto-derive future_dim if not explicitly provided
        if future_dim is None:
            future_dim = n_actions if not df_event else n_actions + 5
        self.future_dim = future_dim
        self.c_dim = c_dim
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.belief_cons_coef = belief_cons_coef
        self.belief_lr_scale = belief_lr_scale

        # Replace PGAgent's policy with belief-conditioned actor-critic
        self.ac_network = LTSActorCritic(
            obs_dim=obs_dim, belief_dim=belief_dim,
            hidden_dim=hidden_dim, n_actions=n_actions, device=device
        )

        # Belief encoder network
        self.belief_encoder = LTSBeliefNetwork(
            obs_dim=obs_dim,
            intra_feat_dim=intra_feat_dim, L=L,
            M=M, c_dim=c_dim,
            y_dim=y_dim, future_dim=future_dim,
            obs_hidden=obs_enc_hidden, obs_out=enc_out_dim,
            intra_hidden=intra_hidden, intra_out=enc_out_dim,
            inter_hidden=inter_hidden, inter_out=enc_out_dim,
            belief_dim=belief_dim, n_actions=n_actions,
        ).to(device)

        # Single optimizer with param groups
        self.optimizer = torch.optim.Adam([
            {'params': self.ac_network.parameters(), 'lr': lr},
            {'params': self.belief_encoder.parameters(), 'lr': lr * belief_lr_scale},
        ])

        # Override self.policy for backward compat (train/eval mode switching)
        self.policy = self.ac_network

        # Print parameter counts
        self._log_param_counts()

        # LTS episode buffers
        self._beliefs = []
        self._teammate_y = []
        self._intra_feats = []
        self._team_actions = []

        # Intra-episode history ring buffer
        self._intra_buffer = []

        # Inter-episode memory
        self._inter_memory = InterEpisodeMemory(M=M, c_dim=c_dim)

        # Episode-level accumulators for characteristic computation
        self._ep_team_actions = []
        self._ep_team_held = []
        self._ep_team_positions = []
        self._ep_rewards = []

        # Previous step tracking
        self._prev_team_action = 0
        self._prev_my_action = 0
        self._prev_reward = 0.0

    def _log_param_counts(self):
        """Print parameter counts by module to stdout."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        n_encoder = count(self.belief_encoder.obs_encoder) + \
                    count(self.belief_encoder.intra_encoder) + \
                    count(self.belief_encoder.inter_encoder) + \
                    count(self.belief_encoder.fusion)
        n_aux = count(self.belief_encoder.aux_heads)
        n_actor = count(self.ac_network.actor)
        n_critic = count(self.ac_network.critic)
        n_total = count(self.belief_encoder) + count(self.ac_network)

        self._param_counts = {
            'belief_encoder': n_encoder,
            'aux_heads': n_aux,
            'actor': n_actor,
            'critic': n_critic,
            'total': n_total,
        }
        print(f"[LTSAgent-{self.agent_id}] Parameter counts:")
        print(f"  belief_encoder: {n_encoder:,}")
        print(f"  aux_heads:      {n_aux:,}")
        print(f"  actor:          {n_actor:,}")
        print(f"  critic:         {n_critic:,}")
        print(f"  total:          {n_total:,}")

    @property
    def param_counts(self):
        return getattr(self, '_param_counts', {})

    # ── Override PGAgent.act ──

    def act(self, obs, deterministic=False, critic_extra=None):
        """Build belief from obs + intra_history + inter_memory, then sample action.

        critic_extra is ignored (belief replaces this role).
        """
        # Build intra_seq from ring buffer
        if len(self._intra_buffer) > 0:
            intra_seq = np.stack(self._intra_buffer, axis=0)
        else:
            intra_seq = np.zeros((0, self.intra_feat_dim), dtype=np.float32)

        # Pad to length L with zeros at beginning
        if intra_seq.shape[0] < self.L:
            pad = np.zeros((self.L - intra_seq.shape[0], self.intra_feat_dim),
                          dtype=np.float32)
            intra_seq = np.concatenate([pad, intra_seq], axis=0)
        elif intra_seq.shape[0] > self.L:
            intra_seq = intra_seq[-self.L:]

        # Get inter-episode memory
        inter_mem = self._inter_memory.get_memory()

        # Encode belief (no grad during rollout)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            intra_t = torch.as_tensor(intra_seq, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            inter_t = torch.as_tensor(inter_mem, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)

            belief, _, _, _ = self.belief_encoder(obs_t, intra_t, inter_t)
            belief_np = belief.squeeze(0).cpu().numpy()

        # Sample action from belief-conditioned actor-critic
        action, log_prob, value = self.ac_network.act(obs, belief_np, deterministic)

        # Store to PGAgent buffers
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)

        # Store to LTS buffers
        self._beliefs.append(belief_np)

        return action

    # ── Teammate feature tracking ──

    def store_teammate_info(self, state_info, teammate_action):
        """Extract and store teammate-observable features y_{-i}^t.

        Called AFTER env.step() when current state_info is available.
        """
        y_t = extract_teammate_y(self.agent_id, state_info, teammate_action,
                                  y_dim=self.y_dim)
        self._teammate_y.append(y_t)
        self._team_actions.append(teammate_action)

        # Episode accumulators for characteristic computation
        self._ep_team_actions.append(teammate_action)
        tid = 1 - self.agent_id
        held = state_info.get(f'player_{tid}_held', None)
        held_int = self._held_to_int(held)
        self._ep_team_held.append(held_int)
        pos = state_info.get(f'player_{tid}_pos', (0, 0))
        self._ep_team_positions.append(pos)

    def store_reward(self, reward):
        """Override to also track reward in episode accumulator."""
        super().store_reward(reward)
        self._ep_rewards.append(reward)

    def build_next_intra_feature(self, teammate_y_prev, my_action_prev, reward_prev):
        """Build intra-feat from (t-1) data and push to ring buffer.

        Called at the BEGINNING of each step (after previous step's data is available).
        For step 0, teammate_y_prev should be zeros.
        """
        feat = build_intra_feature(teammate_y_prev, my_action_prev, reward_prev,
                                    intra_feat_dim=self.intra_feat_dim)
        self._intra_buffer.append(feat)
        if len(self._intra_buffer) > self.L:
            self._intra_buffer = self._intra_buffer[-self.L:]

    # ── Episode characteristic ──

    def compute_current_characteristic(self, game_stats):
        """Compute episode characteristic c_{-i}^e from ego-observable data."""
        total_actions = len(self._ep_team_actions)
        return compute_episode_characteristic(
            teammate_actions=self._ep_team_actions,
            teammate_helds=self._ep_team_held,
            teammate_positions=self._ep_team_positions,
            game_stats=game_stats,
            total_actions=total_actions,
            c_dim=self.c_dim,
            n_actions=self.n_actions,
        )

    # ── Override PGAgent.end_episode ──

    def end_episode(self, aux_adv=None, aux_coef=0.0,
                    shuffle_critic_extra=False,
                    teammate_characteristic=None,
                    future_y_targets=None):
        """Compute GAE, run PPO updates + auxiliary belief losses.

        Each PPO epoch recomputes beliefs with the current encoder so PPO
        gradients flow into the encoder. Rollout-time old_log_probs are fixed.
        """
        if len(self._rewards) == 0:
            self._clear_all_buffers()
            return self._empty_lts_log()

        self._dones[-1] = True

        # ── Standard GAE computation (inherited from PGAgent) ──
        advantages, returns = self._compute_gae(
            self._rewards, self._values, self._dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Convert buffers to tensors ──
        obs_t = torch.as_tensor(np.stack(self._obs), dtype=torch.float32,
                                device=self.device)
        actions_t = torch.as_tensor(self._actions, dtype=torch.int64,
                                    device=self.device)
        old_log_probs_t = torch.as_tensor(self._log_probs, dtype=torch.float32,
                                          device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32,
                                       device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32,
                                    device=self.device)

        # Store old beliefs for diagnostics and optional belief consistency
        old_beliefs_t = torch.as_tensor(np.stack(self._beliefs), dtype=torch.float32,
                                        device=self.device)

        # ── Teammate y targets for D_y (teammate action at time t) ──
        dy_targets_t = torch.as_tensor(self._team_actions, dtype=torch.int64,
                                       device=self.device)

        # ── Future y targets for D_f ──
        if future_y_targets is not None:
            future_t = torch.as_tensor(future_y_targets, dtype=torch.float32,
                                       device=self.device)
        else:
            future_t = self._build_future_targets()

        # ── Episode characteristic target for D_c ──
        if teammate_characteristic is not None:
            c_t = torch.as_tensor(teammate_characteristic, dtype=torch.float32,
                                  device=self.device)
        else:
            c_t = torch.zeros(self.c_dim, dtype=torch.float32, device=self.device)

        # ── Update inter-episode memory ──
        if teammate_characteristic is not None and self.M > 0:
            self._inter_memory.push(teammate_characteristic)

        # ── Build intra_sequences for belief computation ──
        intra_seqs_t = self._build_intra_sequences_tensor()

        # ── Inter memory (same for all timesteps in this episode) ──
        inter_mem = self._inter_memory.get_memory()
        inter_t = torch.as_tensor(inter_mem, dtype=torch.float32,
                                  device=self.device).unsqueeze(0).expand(
            len(obs_t), -1)

        T = len(obs_t)
        use_aux = aux_adv is not None and aux_coef != 0.0

        total_loss = 0.0
        total_entropy = 0.0
        total_value_loss = 0.0
        total_aux_dy = 0.0
        total_aux_df = 0.0
        total_aux_df_action = 0.0
        total_aux_df_event = 0.0
        total_aux_dc = 0.0
        total_aux_bel_cons = 0.0
        total_logprob_delta_sq = 0.0
        total_approx_kl_ppo = 0.0
        total_grad_norm = 0.0
        total_belief_grad_norm = 0.0
        clip_count = 0
        n_updates = 0
        skipped_dy = 0
        skipped_df = 0
        # Tracking for dy_acc and ratio stats
        total_dy_correct = 0
        total_dy_count = 0
        ratio_sum = 0.0
        ratio_sum_sq = 0.0
        ratio_max_val = 0.0
        ratio_count = 0

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
                mb_intra = intra_seqs_t[mb_idx]
                mb_inter = inter_t[mb_idx]

                # Recompute beliefs for this minibatch (with grad)
                mb_beliefs, mb_dy, mb_df, _dc_ep = self.belief_encoder(
                    mb_obs, mb_intra, mb_inter)

                # Aux loss D_y: predict teammate current action (CE)
                loss_dy = F.cross_entropy(mb_dy, dy_targets_t[mb_idx])

                # dy_acc tracking
                with torch.no_grad():
                    dy_pred_label = mb_dy.argmax(dim=-1)
                    dy_correct = (dy_pred_label == dy_targets_t[mb_idx]).sum().item()
                total_dy_correct += dy_correct
                total_dy_count += len(mb_idx)

                # Aux loss D_f: soft CE on action histogram [+ optional event BCE]
                loss_df, loss_df_action, loss_df_event = self._compute_df_loss(
                    mb_df, future_t[mb_idx])

                # Aux loss D_c: predict episode characteristic from mean belief
                dc_pred = self.belief_encoder.aux_heads.dc_head(
                    mb_beliefs.mean(dim=0))
                loss_dc = F.mse_loss(dc_pred, c_t)

                # Belief consistency (optional)
                if self.belief_cons_coef > 0:
                    loss_bel_cons = F.mse_loss(mb_beliefs,
                                                old_beliefs_t[mb_idx].detach())
                else:
                    loss_bel_cons = torch.zeros((), dtype=torch.float32,
                                                 device=self.device)

                new_lp, values, entropy = self.ac_network.evaluate(
                    mb_obs, mb_beliefs, mb_act)

                # PPO clip loss
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip,
                                    1.0 + self.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()

                # Total loss
                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * entropy.mean()
                        + self.alpha * loss_dy
                        + self.beta * loss_df
                        + self.eta * loss_dc
                        + self.belief_cons_coef * loss_bel_cons)

                self.optimizer.zero_grad()
                loss.backward()

                # Compute belief encoder grad norm (before clipping)
                bel_norm_sq = 0.0
                for p in self.belief_encoder.parameters():
                    if p.grad is not None:
                        bel_norm_sq += p.grad.data.norm(2).item() ** 2
                bel_grad_norm = bel_norm_sq ** 0.5

                # Clip all parameters together
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.ac_network.parameters()) +
                    list(self.belief_encoder.parameters()),
                    self.max_grad_norm)

                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()

                self.optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.mean().item()
                total_value_loss += value_loss.item()
                total_aux_dy += loss_dy.item()
                total_aux_df += loss_df.item()
                total_aux_df_action += loss_df_action.item()
                total_aux_df_event += loss_df_event.item()
                total_aux_dc += loss_dc.item()
                total_aux_bel_cons += loss_bel_cons.item()
                total_grad_norm += grad_norm
                total_belief_grad_norm += bel_grad_norm

                with torch.no_grad():
                    logprob_delta_sq = 0.5 * ((new_lp - mb_old_lp) ** 2).mean().item()
                    kl_ppo = ((ratio - 1) - torch.log(ratio)).mean().item()
                    cfrac = ((ratio - 1).abs() > self.ppo_clip).float().mean().item()
                    # Ratio statistics
                    r_vals = ratio.detach()
                    ratio_sum += r_vals.sum().item()
                    ratio_sum_sq += (r_vals ** 2).sum().item()
                    ratio_max_val = max(ratio_max_val, float(r_vals.max().item()))
                    ratio_count += r_vals.numel()
                total_logprob_delta_sq += logprob_delta_sq
                total_approx_kl_ppo += kl_ppo
                clip_count += cfrac
                n_updates += 1

        n_updates = max(1, n_updates)

        # ── Ratio statistics ──
        ratio_mean_val = ratio_sum / max(ratio_count, 1)
        ratio_std_val = (max(ratio_sum_sq / max(ratio_count, 1) - ratio_mean_val ** 2, 0.0)) ** 0.5

        # ── dy_acc ──
        dy_acc_val = total_dy_correct / max(total_dy_count, 1)

        # ── Inter-memory diagnostics ──
        inter_memory_norm_val = float(np.linalg.norm(inter_mem))
        inter_memory_filled_val = self._inter_memory.get_count()

        # Belief diagnostics (recompute final beliefs without grad)
        with torch.no_grad():
            beliefs_final, _, _, _ = self.belief_encoder(obs_t, intra_seqs_t, inter_t)
            belief_norm_val = float(beliefs_final.norm(dim=-1).mean().item())
            belief_delta_val = float(
                (beliefs_final - old_beliefs_t).norm(dim=-1).mean().item())

        self._clear_all_buffers()

        return {
            # Standard PPO fields
            'loss': total_loss / n_updates,
            'mean_return': float(returns_t.mean().item()) if T > 0 else 0.0,
            'entropy': total_entropy / n_updates,
            'grad_norm': total_grad_norm / n_updates,
            'approx_kl': total_logprob_delta_sq / n_updates,
            'value_loss': total_value_loss / n_updates,
            'aux_loss': (total_aux_dy + total_aux_df + total_aux_dc) / n_updates,
            'value_mean': float(old_beliefs_t.mean().item()) if T > 0 else 0.0,
            'explained_variance': 0.0,
            'critic_extra_entropy': 0.0,
            'value_extra_sensitivity': 0.0,
            'sensitivity_note': None,
            'value_mean_shuffled': None,
            'explained_variance_shuffled': None,
            'shuffled_diag_note': None,
            # LTS-specific
            'loss_dy': total_aux_dy / n_updates,
            'loss_df': total_aux_df / n_updates,
            'loss_df_action': total_aux_df_action / n_updates,
            'loss_df_event': total_aux_df_event / n_updates,
            'loss_dc': total_aux_dc / n_updates,
            'loss_bel_cons': total_aux_bel_cons / n_updates,
            'dy_acc': dy_acc_val,
            'ratio_mean': ratio_mean_val,
            'ratio_std': ratio_std_val,
            'ratio_max': ratio_max_val,
            'clip_fraction': clip_count / n_updates,
            'logprob_delta_sq': total_logprob_delta_sq / n_updates,
            'approx_kl_ppo': total_approx_kl_ppo / n_updates,
            'belief_norm': belief_norm_val,
            'belief_delta_mean': belief_delta_val,
            'belief_encoder_grad_norm': total_belief_grad_norm / n_updates,
            'inter_memory_norm': inter_memory_norm_val,
            'inter_memory_filled': inter_memory_filled_val,
            'skipped_dy': skipped_dy,
            'skipped_df': skipped_df,
        }

    # ── Aux loss computation ──

    def _compute_df_loss(self, df_pred, future_target):
        """D_f loss: soft CE on action histogram [+ optional event BCE].

        df_pred: (T, future_dim) raw logits
            [0:n_actions] action logits → softmax → CE with soft target
            [n_actions:] event logits → BCE with binary target (if df_event enabled)
        future_target: (T, future_dim)
            [0:n_actions] soft action histogram, [n_actions:] binary events

        Returns:
            (total_loss, action_loss, event_loss_or_zero)
        """
        n_act = self.n_actions
        # Action histogram: manual soft cross-entropy
        action_logits = df_pred[:, :n_act]
        action_target = future_target[:, :n_act]
        action_log_probs = torch.log_softmax(action_logits, dim=-1)
        loss_action = -(action_target * action_log_probs).sum(-1).mean()

        # Event BCE (optional)
        loss_event = torch.zeros((), dtype=torch.float32, device=self.device)
        if self.df_event and df_pred.shape[-1] > n_act:
            n_evt = df_pred.shape[-1] - n_act
            if n_evt > 0:
                loss_event = F.binary_cross_entropy_with_logits(
                    df_pred[:, n_act:n_act + n_evt],
                    future_target[:, n_act:n_act + n_evt])
        return loss_action + loss_event, loss_action, loss_event

    def _build_future_targets(self):
        """Build D_f targets: K-step lookahead teammate action histogram + events.

        Returns:
            (T, future_dim) tensor on device.
        """
        T = len(self._teammate_y)
        targets = np.zeros((T, self.future_dim), dtype=np.float32)

        for t in range(T):
            fut_end = min(t + self.K, T)
            if fut_end > t:
                # Future teammate actions (from _team_actions)
                fut_actions = self._team_actions[t + 1:fut_end + 1]
                n_fut = len(fut_actions)
                if n_fut > 0:
                    for a in fut_actions:
                        if 0 <= a < 6:
                            targets[t, a] += 1.0
                    targets[t, :6] /= n_fut

        return torch.as_tensor(targets, dtype=torch.float32, device=self.device)

    def _build_intra_sequences_tensor(self):
        """Build (T, L, intra_feat_dim) tensor for belief recomputation.

        For each timestep t, intra_seq[t] contains features from steps
        (t-L+1) through t. Each feature is built from (t-1)-data, maintaining
        simultaneous-action causality.
        """
        T = len(self._obs)
        all_feats = np.zeros((T, self.intra_feat_dim), dtype=np.float32)

        zero_y = np.zeros(self.y_dim, dtype=np.float32)

        for i in range(T):
            if i > 0:
                y_prev = self._teammate_y[i - 1]
                a_prev = self._actions[i - 1]
                r_prev = self._rewards[i - 1]
            else:
                y_prev = zero_y
                a_prev = 0
                r_prev = 0.0
            all_feats[i] = build_intra_feature(y_prev, a_prev, r_prev,
                                                intra_feat_dim=self.intra_feat_dim)

        seqs = np.zeros((T, self.L, self.intra_feat_dim), dtype=np.float32)
        for t in range(T):
            for offset in range(self.L):
                src_idx = t - (self.L - 1) + offset
                if 0 <= src_idx < T:
                    seqs[t, offset] = all_feats[src_idx]

        return torch.as_tensor(seqs, dtype=torch.float32, device=self.device)

    # ── Buffer management ──

    def _clear_all_buffers(self):
        """Clear episode buffers (PGAgent's + LTS-specific)."""
        super()._clear_buffer()
        self._beliefs = []
        self._teammate_y = []
        self._intra_feats = []
        self._team_actions = []
        self._intra_buffer = []
        self._ep_team_actions = []
        self._ep_team_held = []
        self._ep_team_positions = []
        self._ep_rewards = []

    def _empty_lts_log(self):
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
            'loss_dy': 0.0, 'loss_df': 0.0, 'loss_df_action': 0.0,
            'loss_df_event': 0.0, 'loss_dc': 0.0,
            'loss_bel_cons': 0.0,
            'dy_acc': 0.0,
            'ratio_mean': 0.0, 'ratio_std': 0.0, 'ratio_max': 0.0,
            'clip_fraction': 0.0,
            'logprob_delta_sq': 0.0, 'approx_kl_ppo': 0.0,
            'belief_norm': 0.0, 'belief_delta_mean': 0.0,
            'belief_encoder_grad_norm': 0.0,
            'inter_memory_norm': 0.0,
            'inter_memory_filled': 0,
            'skipped_dy': 0, 'skipped_df': 0,
        }

    # ── Mode switching ──

    def train(self):
        self.ac_network.train()
        self.belief_encoder.train()

    def eval(self):
        self.ac_network.eval()
        self.belief_encoder.eval()

    def get_parameters(self):
        ac_params = np.concatenate([p.data.cpu().numpy().ravel()
                                    for p in self.ac_network.parameters()])
        bel_params = np.concatenate([p.data.cpu().numpy().ravel()
                                      for p in self.belief_encoder.parameters()])
        return np.concatenate([ac_params, bel_params])

    @staticmethod
    def _held_to_int(held_obj):
        if held_obj is None:
            return 0
        name = getattr(held_obj, 'name', '')
        if name in ('onion', 'tomato'):
            return 1
        if name == 'dish':
            return 2
        if name == 'soup':
            return 3
        return 0
