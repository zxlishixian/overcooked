"""MemoryBank for Belief-PPO — optional cross-episode retrieval.

FIFO queue of (key, value) pairs. Retrieval uses rank-based top-percent
cosine similarity with a top-k cap.

MVP-A: not used (belief_use_memory=False).
MVP-B: enabled via --belief_use_memory True.
"""
import numpy as np
import torch
import torch.nn.functional as F


class MemoryBank:
    """FIFO memory bank with rank-based top-percent retrieval.

    Key: QueryMLP([e_obs, e_hist]).detach()  — same space as query
    Value: concat(stopgrad(ObsEncoder(o_next)), reward)
           [+ teammate_action_onehot if include_teammate_action=True]
    """

    def __init__(self, memory_size=2000, v_dim=None, top_percent=0.05,
                 topk_max=20, min_entries=10, temperature=0.1, device='cpu'):
        self.memory_size = memory_size
        self.v_dim = v_dim if v_dim is not None else 65  # obs_out(64) + 1
        self.top_percent = top_percent
        self.topk_max = topk_max
        self.min_entries = min_entries
        self.temperature = temperature
        self.device = device

        # Buffers
        self._keys = torch.zeros(0, device=device)
        self._values = torch.zeros(0, self.v_dim, device=device)
        self._write_idx = 0
        self._count = 0

    def _init_buffers(self, key_dim):
        """Lazy-init buffers when first key is added (key_dim is determined at runtime)."""
        self._keys = torch.zeros(self.memory_size, key_dim, device=self.device)
        self._values = torch.zeros(self.memory_size, self.v_dim, device=self.device)

    def add(self, keys, values):
        """Batch-add (key, value) pairs to the bank.

        Args:
            keys: (N, query_dim) tensor — QueryMLP([e_obs, e_hist]).detach()
            values: (N, v_dim) tensor — [next_obs_emb, reward, ...]
        """
        if len(keys.shape) == 1:
            keys = keys.unsqueeze(0)
        if len(values.shape) == 1:
            values = values.unsqueeze(0)

        if self._keys.numel() == 0:
            self._init_buffers(keys.shape[-1])

        N = keys.shape[0]
        for i in range(N):
            idx = self._write_idx
            # Store L2-normalized keys
            self._keys[idx] = F.normalize(keys[i].detach().to(self.device).unsqueeze(0), dim=-1).squeeze(0)
            self._values[idx] = values[i].detach().to(self.device)
            self._write_idx = (self._write_idx + 1) % self.memory_size
            self._count = min(self._count + 1, self.memory_size)

    def retrieve(self, queries, return_diagnostics=False):
        """Retrieve memory for a batch of queries.

        Args:
            queries: (batch, query_dim)
            return_diagnostics: if True, also return diagnostic dict

        Returns:
            m_t: (batch, v_dim) — weighted sum of retrieved values
            [diagnostics]: dict with diagnostic fields
        """
        batch_size = queries.shape[0]
        N = self._count

        if N == 0:
            result = torch.zeros(batch_size, self.v_dim, device=queries.device)
            if return_diagnostics:
                return result, {
                    'k_selected': 0, 'sim_mean': 0.0, 'sim_max': 0.0,
                    'sim_min': 0.0, 'entropy': 0.0,
                }
            return result

        # Determine k
        if N < self.min_entries:
            k = N
        else:
            k_percent = max(1, int(np.ceil(N * self.top_percent)))
            k = min(k_percent, self.topk_max)

        # Cosine similarity with temperature
        keys_norm = self._keys[:N]                              # already normalized
        sim = torch.mm(queries, keys_norm.T)                    # (batch, N)

        # Top-k per query
        topk_sim, topk_idx = torch.topk(sim, k, dim=-1)        # (batch, k)
        topk_weights = torch.softmax(topk_sim / self.temperature, dim=-1)  # (batch, k)

        # Weighted sum of values
        m_t = torch.zeros(batch_size, self.v_dim, device=queries.device)
        for b in range(batch_size):
            selected_values = self._values[topk_idx[b]]         # (k, v_dim)
            m_t[b] = (topk_weights[b].unsqueeze(-1) * selected_values).sum(dim=0)

        if return_diagnostics:
            k_selected = k
            sim_mean = float(topk_sim.mean().item())
            sim_max = float(topk_sim.max().item())
            sim_min = float(topk_sim.min().item())
            sim_gap = sim_max - sim_min
            entropy = float(-(topk_weights * torch.log(topk_weights + 1e-8)).sum(-1).mean().item())
            weight_max = float(topk_weights.max(dim=-1).values.mean().item())
            weight_min = float(topk_weights.min(dim=-1).values.mean().item())
            weight_std = float(topk_weights.std(dim=-1).mean().item())

            return m_t, {
                'k_selected': k_selected,
                'sim_mean': sim_mean, 'sim_max': sim_max, 'sim_min': sim_min,
                'sim_gap': sim_gap,
                'entropy': entropy,
                'weight_max': weight_max, 'weight_min': weight_min,
                'weight_std': weight_std,
            }
        return m_t

    def store_episode(self, history_seqs, next_obs_embs, rewards,
                      query_mlp, obs_encoder, history_encoder,
                      teammate_actions=None):
        """Build and add (key, value) pairs for an entire episode.

        Called at episode end. Uses .detach() on all inputs.

        Args:
            history_seqs: (T, L, hist_input_dim)
            next_obs_embs: (T+1, obs_out) — ObsEncoder(o_t) for all steps
            rewards: (T,) float tensor
            query_mlp: QueryMLP module
            obs_encoder: ObsEncoder module
            history_encoder: HistoryEncoder module
            teammate_actions: (T,) int tensor (optional, for include_teammate_action)

        Returns:
            (keys, values) tensors added
        """
        T = len(rewards)
        with torch.no_grad():
            e_obs = obs_encoder(next_obs_embs[:-1])           # first T steps: (T, obs_out)
            e_hist = history_encoder(history_seqs)             # (T, hist_out)
            keys = query_mlp(e_obs, e_hist)                    # (T, query_dim)

            # Values: encode o_{t+1} through ObsEncoder, then concat with reward
            next_e_obs = obs_encoder(next_obs_embs[1:])       # (T, obs_out) — o_{t+1}
            r = rewards.unsqueeze(-1)                          # (T, 1)
            if teammate_actions is not None:
                ta_onehot = F.one_hot(teammate_actions.long(), num_classes=6).float()
                values = torch.cat([next_e_obs, r, ta_onehot], dim=-1)
            else:
                values = torch.cat([next_e_obs, r], dim=-1)

        self.add(keys, values)
        return keys, values

    def reset(self):
        """Clear all stored memories."""
        self._write_idx = 0
        self._count = 0

    def __len__(self):
        return self._count
