"""RNNAgent: RNN-IPPO baseline for LTS-PPO ablation.

Uses the same intra_feat input as LTS-PPO's IntraEncoder branch:
  intra_feat = [teammate_y (16) || own_action_onehot (6) || reward (1)]

K-step intra_feat sequence → GRU(rnn_hidden_dim) → last hidden → concat to obs
→ separate actor/critic MLP trunks.

No belief decomposition, no inter-memory, no auxiliary losses.
PPO gradients flow through the GRU.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pg_agent import PGAgent
from ..models.rnn_actor_critic import RNNActorCritic
from ..utils.teammate_features import build_intra_feature


class RNNAgent(PGAgent):
    """RNN-IPPO agent: GRU over intra-episode history, no belief structure.

    Input-aligned with LTS-PPO's intra branch. Used to test whether
    a generic GRU history encoder (without belief decomposition or
    inter-episode memory) matches LTS-PPO performance.
    """

    def __init__(self, agent_id, obs_dim, n_actions=6, lr=1e-3, gamma=0.99,
                 hidden_dim=256, device='cpu',
                 ppo_epochs=10, ppo_clip=0.2, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 critic_extra_dim=0,
                 # RNN-specific
                 K=10, rnn_hidden_dim=64,
                 intra_feat_dim=23, y_dim=16,
                 **_kwargs):
        super().__init__(agent_id, obs_dim, n_actions, lr, gamma, hidden_dim,
                         device, ppo_epochs, ppo_clip, gae_lambda,
                         vf_coef, ent_coef, max_grad_norm, critic_extra_dim=0)

        self.K = K
        self.rnn_hidden_dim = rnn_hidden_dim
        self.intra_feat_dim = intra_feat_dim
        self.y_dim = y_dim

        # Replace PGAgent's policy with RNN-conditioned actor-critic
        self.ac_network = RNNActorCritic(
            obs_dim=obs_dim, intra_feat_dim=intra_feat_dim,
            rnn_hidden_dim=rnn_hidden_dim, hidden_dim=hidden_dim,
            n_actions=n_actions, device=device,
        )

        # Single optimizer (PPO backprops through GRU)
        self.optimizer = torch.optim.Adam(self.ac_network.parameters(), lr=lr)
        self.policy = self.ac_network

        self._log_param_counts()

        # Intra-episode history ring buffer
        self._intra_buffer = []

        # Episode buffers for end_episode recomputation
        self._teammate_y = []

        # Previous step tracking
        self._prev_teammate_y = np.zeros(y_dim, dtype=np.float32)

    def _log_param_counts(self):
        def count(module):
            return sum(p.numel() for p in module.parameters())

        n_gru = count(self.ac_network.gru)
        n_actor = count(self.ac_network.actor)
        n_critic = count(self.ac_network.critic)
        n_total = count(self.ac_network)

        self._param_counts = {
            'gru': n_gru,
            'actor': n_actor,
            'critic': n_critic,
            'total': n_total,
        }
        print(f"[RNNAgent-{self.agent_id}] Parameter counts:")
        print(f"  gru:     {n_gru:,}")
        print(f"  actor:   {n_actor:,}")
        print(f"  critic:  {n_critic:,}")
        print(f"  total:   {n_total:,}")

    @property
    def param_counts(self):
        return getattr(self, '_param_counts', {})

    # ── Override PGAgent.act ──

    def act(self, obs, deterministic=False, critic_extra=None):
        """Build intra_seq from ring buffer, then sample action."""
        if len(self._intra_buffer) > 0:
            intra_seq = np.stack(self._intra_buffer, axis=0)
        else:
            intra_seq = np.zeros((0, self.intra_feat_dim), dtype=np.float32)

        if intra_seq.shape[0] < self.K:
            pad = np.zeros((self.K - intra_seq.shape[0], self.intra_feat_dim),
                          dtype=np.float32)
            intra_seq = np.concatenate([pad, intra_seq], axis=0)
        elif intra_seq.shape[0] > self.K:
            intra_seq = intra_seq[-self.K:]

        action, log_prob, value = self.ac_network.act(obs, intra_seq, deterministic)

        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)

        return action

    # ── Teammate feature tracking ──

    def build_next_intra_feature(self, teammate_y_prev, my_action_prev, reward_prev):
        """Build intra-feat from (t-1) data and push to ring buffer."""
        feat = build_intra_feature(teammate_y_prev, my_action_prev, reward_prev,
                                    intra_feat_dim=self.intra_feat_dim)
        self._intra_buffer.append(feat)
        if len(self._intra_buffer) > self.K:
            self._intra_buffer = self._intra_buffer[-self.K:]

    def store_teammate_y(self, teammate_y):
        """Store teammate y for end_episode intra_seq rebuild."""
        self._teammate_y.append(teammate_y)

    # ── Override PGAgent.end_episode ──

    def end_episode(self, aux_adv=None, aux_coef=0.0,
                    shuffle_critic_extra=False, **_kwargs):
        """Compute GAE, run PPO updates with GRU history.

        Each PPO epoch rebuilds intra_sequences from stored buffers so
        PPO gradients flow through the GRU.
        """
        if len(self._rewards) == 0:
            self._clear_all_buffers()
            return self._empty_rnn_log()

        self._dones[-1] = True

        advantages, returns = self._compute_gae(
            self._rewards, self._values, self._dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

        # Build intra_sequences (K, intra_feat_dim) per timestep
        intra_seqs_t = self._build_intra_sequences_tensor()

        T = len(obs_t)
        total_loss = 0.0
        total_entropy = 0.0
        total_value_loss = 0.0
        total_approx_kl = 0.0
        total_grad_norm = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(T, device=self.device)
            for start in range(0, T, 64):
                end = min(start + 64, T)
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_act = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]
                mb_intra = intra_seqs_t[mb_idx]

                new_lp, values, entropy = self.ac_network.evaluate(
                    mb_obs, mb_intra, mb_act)

                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip,
                                    1.0 + self.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()

                loss = (policy_loss + self.vf_coef * value_loss
                        - self.ent_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.ac_network.parameters(), self.max_grad_norm)
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()
                self.optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.mean().item()
                total_value_loss += value_loss.item()
                total_approx_kl += 0.5 * ((new_lp - mb_old_lp) ** 2).mean().item()
                total_grad_norm += grad_norm
                n_updates += 1

        n_updates = max(1, n_updates)
        self._clear_all_buffers()

        return {
            'loss': total_loss / n_updates,
            'mean_return': float(returns_t.mean().item()) if T > 0 else 0.0,
            'entropy': total_entropy / n_updates,
            'grad_norm': total_grad_norm / n_updates,
            'approx_kl': total_approx_kl / n_updates,
            'value_loss': total_value_loss / n_updates,
            'aux_loss': 0.0,
            'value_mean': 0.0,
            'explained_variance': 0.0,
            'critic_extra_entropy': 0.0,
            'value_extra_sensitivity': 0.0,
            'sensitivity_note': None,
            'value_mean_shuffled': None,
            'explained_variance_shuffled': None,
            'shuffled_diag_note': None,
        }

    def _build_intra_sequences_tensor(self):
        """Build (T, K, intra_feat_dim) tensor for GRU forward.

        For each timestep t, intra_seq[t] contains features from steps
        (t-K+1) through t, each built from (t-1)-data.
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

        seqs = np.zeros((T, self.K, self.intra_feat_dim), dtype=np.float32)
        for t in range(T):
            for offset in range(self.K):
                src_idx = t - (self.K - 1) + offset
                if 0 <= src_idx < T:
                    seqs[t, offset] = all_feats[src_idx]

        return torch.as_tensor(seqs, dtype=torch.float32, device=self.device)

    # ── Buffer management ──

    def _clear_all_buffers(self):
        super()._clear_buffer()
        self._intra_buffer = []
        self._teammate_y = []

    def _empty_rnn_log(self):
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
        }

    # ── Mode switching ──

    def train(self):
        self.ac_network.train()

    def eval(self):
        self.ac_network.eval()

    def get_parameters(self):
        return np.concatenate([p.data.cpu().numpy().ravel()
                              for p in self.ac_network.parameters()])
