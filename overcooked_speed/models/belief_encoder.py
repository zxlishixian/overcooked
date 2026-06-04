"""Variational Belief Encoder for Belief-PPO (POMDP belief-MDP).

q_φ(b_t | o_t, h_t [, m_t]):
  - ObsEncoder: current observation → e_obs
  - HistoryEncoder: GRU over L-step past (obs, action, reward) → e_hist
  - Optional MemoryBank retrieval: m_t (only when belief_use_memory=True)
  - FusionMLP: [e_obs, e_hist, (m_t)] → mu, logvar
  - b_t = mu + eps * std  (reparameterization)

Auxiliary heads:
  - RewardPredictor: D_r(b_t, a_t) → r_hat
  - ObsPredictor:     D_o(b_t, a_t) → e_obs_next_hat  (default disabled)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Encoder sub-modules ──

class ObsEncoder(nn.Module):
    """Single-layer MLP encoder for current observation o_t."""

    def __init__(self, obs_dim, obs_out=64):
        super().__init__()
        self.net = nn.Linear(obs_dim, obs_out)
        nn.init.orthogonal_(self.net.weight, gain=np.sqrt(2))
        nn.init.constant_(self.net.bias, 0)

    def forward(self, obs):
        return self.net(obs)


class HistoryEncoder(nn.Module):
    """GRU encoder over L-step past history sequence.

    Input: (batch, L, obs_dim + 1 + n_actions)  — raw (obs, reward_scalar, action_onehot)
    Output: (batch, hist_out)
    """

    def __init__(self, hist_input_dim, gru_hidden=64, hist_out=64):
        super().__init__()
        self.gru = nn.GRU(hist_input_dim, gru_hidden, batch_first=True)
        self.proj = nn.Linear(gru_hidden, hist_out)
        nn.init.orthogonal_(self.proj.weight, gain=np.sqrt(2))
        nn.init.constant_(self.proj.bias, 0)
        # Orthogonal init for GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, history_seq):
        """Args:
            history_seq: (batch, L, hist_input_dim) — left-padded with zeros for t < L
        Returns:
            (batch, hist_out)
        """
        _, h_n = self.gru(history_seq)
        return self.proj(h_n.squeeze(0))


class QueryMLP(nn.Module):
    """2-layer MLP that maps [e_obs, e_hist] → query vector for memory retrieval."""

    def __init__(self, obs_out=64, hist_out=64, query_hidden=128, query_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_out + hist_out, query_hidden),
            nn.ReLU(),
            nn.Linear(query_hidden, query_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, e_obs, e_hist):
        x = torch.cat([e_obs, e_hist], dim=-1)
        return F.normalize(self.net(x), dim=-1)  # L2 normalize


class FusionMLP(nn.Module):
    """MLP that fuses [e_obs, e_hist, (m_t)] → mu, logvar.

    Input dim is dynamic: obs_out + hist_out [+ v_dim] depending on use_memory.
    """

    def __init__(self, fusion_input_dim, belief_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_input_dim, belief_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(belief_dim, belief_dim)
        self.logvar_head = nn.Linear(belief_dim, belief_dim)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0)
        nn.init.orthogonal_(self.logvar_head.weight, gain=0.01)
        nn.init.constant_(self.logvar_head.bias, 0)

    def forward(self, fused_input):
        h = self.net(fused_input)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar


# ── Auxiliary predictor heads ──

class RewardPredictor(nn.Module):
    """Predict r_t from (b_t, a_t)."""

    def __init__(self, belief_dim=32, n_actions=6, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, belief, action_onehot):
        """Args:
            belief: (batch, belief_dim)
            action_onehot: (batch, n_actions)
        Returns:
            r_hat: (batch, 1)
        """
        return self.net(torch.cat([belief, action_onehot], dim=-1))


class ObsPredictor(nn.Module):
    """Predict next obs embedding from (b_t, a_t).

    Target: stopgrad(ObsEncoder(o_{t+1})).
    Default disabled (obs_pred_coef=0.0).
    """

    def __init__(self, belief_dim=32, n_actions=6, obs_out=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(belief_dim + n_actions, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_out),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, belief, action_onehot):
        """Returns: e_obs_next_hat: (batch, obs_out)"""
        return self.net(torch.cat([belief, action_onehot], dim=-1))


class ActionEncoder(nn.Module):
    """Encode teammate action for historical state fusion.

    Linear(n_actions → obs_out), additive with ObsEncoder output.
    """

    def __init__(self, n_actions=6, obs_out=64):
        super().__init__()
        self.net = nn.Linear(n_actions, obs_out)
        nn.init.orthogonal_(self.net.weight, gain=np.sqrt(2))
        nn.init.constant_(self.net.bias, 0)

    def forward(self, action_onehot):
        return self.net(action_onehot)


class QueryOutcomePredictor(nn.Module):
    """Predict future outcome (next_obs_emb || reward) from query vector.

    Target: concat(stopgrad(ObsEncoder(o_{t+1})), r_t)
    This provides direct supervision for the query/key representation.
    """

    def __init__(self, query_dim=64, obs_out=64, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(query_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_out + 1),  # obs_out + reward
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, query):
        """Returns: outcome_hat: (batch, obs_out + 1)"""
        return self.net(query)


# ── Main Variational Belief Encoder ──

class VariationalBeliefEncoder(nn.Module):
    """Full variational belief encoder.

    q(b_t | o_t, h_t [, m_t]):
      e_obs = ObsEncoder(o_t)
      e_hist = HistoryEncoder(history_seq)
      query = QueryMLP(e_obs, e_hist)  # for external MemoryBank retrieval
      m_t = retrieved_memory (or zeros)
      mu, logvar = FusionMLP([e_obs, e_hist, m_t])
      b_t = mu + eps * std
    """

    def __init__(self, obs_dim, n_actions=6, belief_dim=32,
                 history_len=10, obs_out=64, hist_out=64,
                 gru_hidden=64, query_dim=64, query_hidden=128,
                 use_memory=False, v_dim=None, deterministic=False,
                 use_historical_context=False, historical_attn_dim=64):
        super().__init__()
        self.belief_dim = belief_dim
        self.n_actions = n_actions
        self.history_len = history_len
        self.obs_out = obs_out
        self.hist_out = hist_out
        self.query_dim = query_dim
        self.use_memory = use_memory
        self.v_dim = v_dim
        self.deterministic = deterministic
        self.use_historical_context = use_historical_context

        # hist_input_dim: obs_dim + 1 (reward) + n_actions (action onehot)
        self.hist_input_dim = obs_dim + 1 + n_actions

        self.obs_encoder = ObsEncoder(obs_dim, obs_out)
        self.history_encoder = HistoryEncoder(self.hist_input_dim, gru_hidden, hist_out)
        self.query_mlp = QueryMLP(obs_out, hist_out, query_hidden, query_dim)

        # Historical context module (MVP-C)
        self.historical_module = None
        if use_historical_context:
            from .historical_context import HistoricalContextModule
            self.historical_module = HistoricalContextModule(
                obs_out=obs_out, attn_dim=historical_attn_dim,
                n_actions=n_actions,
            )

        # Fusion input dim
        fusion_input_dim = obs_out + hist_out
        if use_memory and v_dim is not None:
            fusion_input_dim += v_dim
        self.fusion = FusionMLP(fusion_input_dim, belief_dim)

    def encode(self, obs, history_seq, retrieved_memory=None, hist_mem_obs=None, hist_mem_actions=None):
        """Encode belief: returns (mu, logvar, b_t, query_vec, hist_diag).

        Args:
            obs: (batch, obs_dim)
            history_seq: (batch, L, hist_input_dim)
            retrieved_memory: (batch, v_dim) or None (legacy)
            hist_mem_obs: (N, obs_out) or None — pre-encoded historical obs (MVP-C)
            hist_mem_actions: (N,) int64 or None — historical teammate actions (MVP-C)

        Returns:
            mu, logvar, b_t, query, hist_diag (dict or None)
        """
        e_obs = self.obs_encoder(obs)           # (batch, obs_out)
        e_hist = self.history_encoder(history_seq)  # (batch, hist_out)
        query = self.query_mlp(e_obs, e_hist)   # (batch, query_dim)

        hist_diag = None
        f_ctx = e_obs  # default: no context

        # MVP-C: Historical context (residual)
        if self.use_historical_context and self.historical_module is not None:
            if hist_mem_obs is not None and hist_mem_obs.shape[0] > 0:
                # Encode historical raw obs through current ObsEncoder
                hist_obs_emb = self.obs_encoder(hist_mem_obs)  # (N, obs_out)
                f_ctx, hist_diag = self.historical_module(
                    e_obs, hist_obs_emb, hist_mem_actions)
            else:
                f_ctx = e_obs
                hist_diag = {
                    'k_selected': 0, 'sim_mean': 0.0, 'sim_max': 0.0,
                    'sim_min': 0.0, 'sim_gap': 0.0,
                    'entropy': 0.0, 'weight_max': 0.0, 'weight_min': 0.0,
                    'weight_std': 0.0,
                }

        # Fusion: use f_ctx (residual context) instead of raw e_obs
        if self.use_memory:
            if retrieved_memory is None:
                batch_size = e_obs.shape[0]
                retrieved_memory = torch.zeros(batch_size, self.v_dim, device=e_obs.device)
            fused = torch.cat([f_ctx, e_hist, retrieved_memory], dim=-1)
        else:
            fused = torch.cat([f_ctx, e_hist], dim=-1)

        mu, logvar = self.fusion(fused)
        b_t = self._reparameterize(mu, logvar)
        return mu, logvar, b_t, query, hist_diag

    def _reparameterize(self, mu, logvar):
        if self.deterministic:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs, history_seq, retrieved_memory=None,
                hist_mem_obs=None, hist_mem_actions=None):
        """Main forward: returns (mu, logvar, b_t)."""
        mu, logvar, b_t, _query, _diag = self.encode(
            obs, history_seq, retrieved_memory,
            hist_mem_obs=hist_mem_obs, hist_mem_actions=hist_mem_actions)
        return mu, logvar, b_t

    @staticmethod
    def compute_kl(mu, logvar, free_nats=1.0):
        """KL(N(mu, sigma) || N(0, I)) with free-bits per sample.

        Args:
            mu: (batch, belief_dim)
            logvar: (batch, belief_dim)
            free_nats: total free nats per sample (not per dim)

        Returns:
            kl_loss: scalar (clamped per-sample, mean over batch)
            kl_raw: scalar (raw per-sample KL, mean over batch)
        """
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
        kl_raw_per_sample = kl_per_dim.sum(dim=-1)     # (batch,)
        kl_raw = kl_raw_per_sample.mean()              # scalar
        kl_clamped = torch.clamp(kl_raw_per_sample - free_nats, min=0.0)
        kl_loss = kl_clamped.mean()                    # scalar
        return kl_loss, kl_raw
