"""LTS-PPO Belief Encoder Network.

ObsEncoder: MLP over current observation.
IntraEncoder: GRU over intra-episode history sequence.
InterEncoder: MLP over flattened cross-episode memory.
AuxHeads: D_y (current teammate action), D_f (future behavior), D_c (episode characteristic).
LTSBeliefNetwork: composes all encoders + fusion + aux heads.
"""
import numpy as np
import torch
import torch.nn as nn


class ObsEncoder(nn.Module):
    """MLP encoder for current observation o_t^i."""

    def __init__(self, obs_dim, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, obs):
        return self.net(obs)


class IntraEncoder(nn.Module):
    """GRU encoder over L-step intra-episode history sequence."""

    def __init__(self, intra_feat_dim=23, hidden_dim=64, out_dim=64):
        super().__init__()
        self.gru = nn.GRU(intra_feat_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim)
        nn.init.orthogonal_(self.proj.weight, gain=np.sqrt(2))
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, intra_seq):
        """Args:
            intra_seq: (batch, L, intra_feat_dim) — padded with zeros for t < L.
        Returns:
            (batch, out_dim)
        """
        _, h_n = self.gru(intra_seq)
        return self.proj(h_n.squeeze(0))


class InterEncoder(nn.Module):
    """MLP encoder over M cross-episode characteristic vectors (flattened)."""

    def __init__(self, M=10, c_dim=17, hidden_dim=64, out_dim=64):
        super().__init__()
        input_dim = M * c_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)

    def forward(self, inter_memory):
        """Args:
            inter_memory: (batch, M * c_dim)
        Returns:
            (batch, out_dim)
        """
        return self.net(inter_memory)


class AuxHeads(nn.Module):
    """Auxiliary prediction heads from belief embedding.

    D_y: predict teammate current action a_{-i}^t (n_actions-dim, CE target)
    D_f: predict future teammate behavior summary (future_dim-dim)
    D_c: predict episode-level teammate characteristic (c_dim-dim)
    """

    def __init__(self, belief_dim=64, n_actions=6, future_dim=11, c_dim=17):
        super().__init__()
        self.dy_head = nn.Linear(belief_dim, n_actions)
        self.df_head = nn.Linear(belief_dim, future_dim)
        self.dc_head = nn.Linear(belief_dim, c_dim)

        for head in [self.dy_head, self.df_head, self.dc_head]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.constant_(head.bias, 0)

    def forward(self, belief):
        """Args:
            belief: (batch, belief_dim)
        Returns:
            dy_pred: (batch, n_actions) — raw logits for CE
            df_pred: (batch, future_dim) — raw logits (action part) + logits (event part)
            dc_pred: (batch, c_dim) — raw predictions
        """
        dy = self.dy_head(belief)
        df = self.df_head(belief)
        dc = self.dc_head(belief)
        return dy, df, dc


class LTSBeliefNetwork(nn.Module):
    """Full belief encoder: ObsEncoder + IntraEncoder + InterEncoder + Fusion + AuxHeads.

    Forward returns (belief, dy_pred, df_pred, dc_pred).
    """

    def __init__(self, obs_dim,
                 intra_feat_dim=23, L=20,
                 M=10, c_dim=17,
                 y_dim=16, future_dim=11,
                 obs_hidden=128, obs_out=64,
                 intra_hidden=64, intra_out=64,
                 inter_hidden=64, inter_out=64,
                 belief_dim=64, n_actions=6):
        super().__init__()
        self.L = L
        self.M = M
        self.belief_dim = belief_dim
        self.n_actions = n_actions

        self.obs_encoder = ObsEncoder(obs_dim, obs_hidden, obs_out)
        self.intra_encoder = IntraEncoder(intra_feat_dim, intra_hidden, intra_out)
        self.inter_encoder = InterEncoder(M, c_dim, inter_hidden, inter_out)

        fusion_input_dim = obs_out + intra_out + inter_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, belief_dim),
            nn.ReLU(),
        )
        nn.init.orthogonal_(self.fusion[0].weight, gain=np.sqrt(2))
        nn.init.constant_(self.fusion[0].bias, 0)

        self.aux_heads = AuxHeads(belief_dim, n_actions, future_dim, c_dim)

    def forward(self, obs, intra_seq, inter_memory):
        """Args:
            obs: (batch, obs_dim)
            intra_seq: (batch, L, intra_feat_dim)
            inter_memory: (batch, M * c_dim)
        Returns:
            belief: (batch, belief_dim)
            dy_pred: (batch, n_actions)
            df_pred: (batch, future_dim)
            dc_pred: (batch, c_dim)
        """
        obs_enc = self.obs_encoder(obs)
        intra_enc = self.intra_encoder(intra_seq)
        inter_enc = self.inter_encoder(inter_memory)

        fused = self.fusion(torch.cat([obs_enc, intra_enc, inter_enc], dim=-1))
        dy, df, dc = self.aux_heads(fused)

        return fused, dy, df, dc
