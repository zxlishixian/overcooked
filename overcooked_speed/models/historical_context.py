"""Historical Context Module for Belief-PPO (MVP-C).

Dynamic-Belief-style cross-episode historical context:
- Adaptive dropout via raw obs-cosine similarity (no learned query/key)
- Historical state = obs_emb + teammate_action_emb (additive fusion)
- Learnable soft attention over selected historical states
- Residual context: f_ctx = e_obs + memory_context
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoricalContextMemory:
    """FIFO memory of past-episode (obs, teammate_action) pairs.

    Entries added at episode end only — no within-episode retrieval.
    """

    def __init__(self, memory_size=2000, obs_dim=520, device='cpu'):
        self.memory_size = memory_size
        self.obs_dim = obs_dim
        self.device = device

        self._obs = torch.zeros(0, obs_dim, device=device)
        self._actions = torch.zeros(0, dtype=torch.int64, device=device)
        self._write_idx = 0
        self._count = 0

    def _init_buffers(self, N):
        if self._obs.numel() == 0:
            self._obs = torch.zeros(self.memory_size, self.obs_dim, device=self.device)
            self._actions = torch.zeros(self.memory_size, dtype=torch.int64, device=self.device)

    def add(self, obs_list, action_list):
        """Batch-add (obs, teammate_action) pairs at episode end.

        Args:
            obs_list: list of (obs_dim,) numpy arrays
            action_list: list of int teammate actions
        """
        N = min(len(obs_list), len(action_list))
        if N == 0:
            return
        self._init_buffers(N)
        for i in range(N):
            idx = self._write_idx
            self._obs[idx] = torch.as_tensor(obs_list[i], dtype=torch.float32, device=self.device)
            self._actions[idx] = int(action_list[i])
            self._write_idx = (self._write_idx + 1) % self.memory_size
            self._count = min(self._count + 1, self.memory_size)

    def get_all(self):
        """Return all stored (obs, actions).

        Returns:
            obs: (N, obs_dim) tensor
            actions: (N,) int64 tensor
        """
        if self._count == 0:
            return (
                torch.zeros(0, self.obs_dim, device=self.device),
                torch.zeros(0, dtype=torch.int64, device=self.device),
            )
        return self._obs[:self._count].clone(), self._actions[:self._count].clone()

    def reset(self):
        self._write_idx = 0
        self._count = 0

    def __len__(self):
        return self._count


class HistoricalContextModule(nn.Module):
    """Dynamic-Belief-style historical context with adaptive dropout + attention.

    Flow:
      1. e_t = ObsEncoder(o_t)              (current obs embedding)
      2. e_j = ObsEncoder(o_j)              (historical obs embeddings)
      3. sim_j = cosine(e_t, e_j)           (raw cosine, no learned query)
      4. top p% filtering (adaptive dropout)
      5. s_j = e_j + ActionEncoder(a_j)     (historical state = obs + action)
      6. q = W_q(e_t), k_j = W_k(s_j), v_j = W_v(s_j)
      7. attention weights via scaled dot-product
      8. memory_context = weighted sum of v_j
      9. f_ctx = e_t + memory_context       (residual context)
    """

    def __init__(self, obs_out=64, attn_dim=64, n_actions=6,
                 top_percent=0.05, topk_max=40, min_entries=10):
        super().__init__()
        self.obs_out = obs_out
        self.attn_dim = attn_dim
        self.top_percent = top_percent
        self.topk_max = topk_max
        self.min_entries = min_entries

        # Action encoder for historical teammate actions
        self.action_encoder = nn.Linear(n_actions, obs_out)

        # Learnable attention projections
        self.W_q = nn.Linear(obs_out, attn_dim)
        self.W_k = nn.Linear(obs_out, attn_dim)  # operates on s_j (obs_out dim)
        self.W_v = nn.Linear(obs_out, attn_dim)

        # Init
        for m in [self.action_encoder, self.W_q, self.W_k, self.W_v]:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, e_t_all, memory_obs, memory_actions):
        """Compute historical context.

        Args:
            e_t_all: (batch, obs_out) — current obs embeddings (pre-encoded)
            memory_obs: (N, obs_dim) — raw historical observations
            memory_actions: (N,) int64 — historical teammate actions

        Returns:
            f_ctx: (batch, obs_out) — residual context
            diagnostics: dict
        """
        batch_size = e_t_all.shape[0]
        device = e_t_all.device
        N = memory_obs.shape[0]
        e_t = e_t_all  # already encoded externally

        if N == 0:
            return e_t, {
                'k_selected': 0, 'sim_mean': 0.0, 'sim_max': 0.0,
                'sim_min': 0.0, 'sim_gap': 0.0,
                'entropy': 0.0, 'weight_max': 0.0, 'weight_min': 0.0,
                'weight_std': 0.0,
            }

        # Encode historical obs (will be done inside call via separate encoder)
        # This is handled differently — the caller passes pre-encoded obs
        # Here we receive raw obs and encode them
        # Actually, memory_obs should be pre-encoded by the caller (via ObsEncoder)
        # For now, assume memory_obs is already (N, obs_out) — the caller handles encoding

        # Step 3: cosine similarity between current and historical obs embeddings
        e_t_norm = F.normalize(e_t_all, dim=-1)                    # (batch, obs_out)
        mem_norm = F.normalize(memory_obs, dim=-1)                 # (N, obs_out)
        sim = torch.mm(e_t_norm, mem_norm.T)                       # (batch, N)

        # Step 4: top p% filtering
        if N < self.min_entries:
            k = max(1, N)
        else:
            k_percent = max(1, int(np.ceil(N * self.top_percent)))
            k = min(k_percent, self.topk_max)

        topk_sim, topk_idx = torch.topk(sim, min(k, N), dim=-1)   # (batch, k)
        actual_k = topk_sim.shape[-1]

        # Step 5: historical state = obs_emb + action_emb
        # memory_actions_onehot: (N, n_actions)
        mem_act_oh = F.one_hot(memory_actions.long(), num_classes=6).float().to(device)
        act_emb = self.action_encoder(mem_act_oh)                   # (N, obs_out)
        s_j = memory_obs + act_emb                                  # (N, obs_out)

        # Step 6: attention projections
        q = self.W_q(e_t_all)                                       # (batch, attn_dim)
        k_all = self.W_k(s_j)                                       # (N, attn_dim)
        v_all = self.W_v(s_j)                                       # (N, attn_dim)

        # Gather top-k
        memory_context = torch.zeros(batch_size, self.attn_dim, device=device)
        for b in range(batch_size):
            k_sel = k_all[topk_idx[b]]                              # (k, attn_dim)
            v_sel = v_all[topk_idx[b]]                              # (k, attn_dim)
            scores = torch.mm(q[b:b+1], k_sel.T) / np.sqrt(self.attn_dim)  # (1, k)
            weights = torch.softmax(scores, dim=-1)                 # (1, k)
            memory_context[b] = (weights.squeeze(0).unsqueeze(-1) * v_sel).sum(dim=0)

        # Step 9: residual context — project back to obs_out
        # memory_context is (batch, attn_dim), need to map to (batch, obs_out)
        # Use a simple linear projection for the residual
        if not hasattr(self, 'residual_proj'):
            self.residual_proj = nn.Linear(self.attn_dim, self.obs_out).to(device)
            nn.init.orthogonal_(self.residual_proj.weight, gain=0.01)
            nn.init.constant_(self.residual_proj.bias, 0)
        f_ctx = e_t_all + self.residual_proj(memory_context)

        # Diagnostics
        sim_mean = float(topk_sim.mean().item()) if actual_k > 0 else 0.0
        sim_max = float(topk_sim.max().item()) if actual_k > 0 else 0.0
        sim_min = float(topk_sim.min().item()) if actual_k > 0 else 0.0
        sim_gap = sim_max - sim_min if actual_k > 0 else 0.0

        if actual_k > 0:
            # Compute attention weights for the first batch element as representative
            k_sel_0 = k_all[topk_idx[0]]
            scores_0 = torch.mm(q[0:1], k_sel_0.T) / np.sqrt(self.attn_dim)
            w0 = torch.softmax(scores_0, dim=-1).squeeze(0)
            entropy = float(-(w0 * torch.log(w0 + 1e-8)).sum().item())
            weight_max = float(w0.max().item())
            weight_min = float(w0.min().item())
            weight_std = float(w0.std().item())
        else:
            entropy, weight_max, weight_min, weight_std = 0.0, 0.0, 0.0, 0.0

        return f_ctx, {
            'k_selected': actual_k,
            'sim_mean': sim_mean, 'sim_max': sim_max, 'sim_min': sim_min,
            'sim_gap': sim_gap,
            'entropy': entropy,
            'weight_max': weight_max, 'weight_min': weight_min,
            'weight_std': weight_std,
        }
