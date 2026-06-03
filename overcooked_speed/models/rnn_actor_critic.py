"""RNN-IPPO Actor-Critic with GRU history encoder.

Input: [obs || GRU(intra_seq)] where intra_seq uses the same features as
LTS-PPO's IntraEncoder (teammate_y + own_action + reward).
No belief decomposition, no inter-memory, no auxiliary heads.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNActorCritic(nn.Module):
    """GRU-conditioned Actor-Critic for RNN-IPPO baseline.

    intra_seq (K, intra_feat_dim) → GRU → last_hidden
    [obs || last_hidden] → separate actor/critic MLP trunks.
    """

    def __init__(self, obs_dim, intra_feat_dim=23, rnn_hidden_dim=64,
                 hidden_dim=256, n_actions=6, device='cpu'):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.gru = nn.GRU(intra_feat_dim, rnn_hidden_dim, batch_first=True)

        input_dim = obs_dim + rnn_hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Orthogonal init (GRU weights only, skip 1D biases)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        for net in [self.actor, self.critic]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)

        self.to(device)

    def _encode_gru(self, intra_seq):
        """Run GRU on intra_seq and return last hidden state.

        Args:
            intra_seq: (batch, K, intra_feat_dim)
        Returns:
            (batch, rnn_hidden_dim)
        """
        _, h_n = self.gru(intra_seq)
        return h_n.squeeze(0)

    def forward(self, obs, intra_seq):
        """Args:
            obs: (batch, obs_dim)
            intra_seq: (batch, K, intra_feat_dim)
        Returns:
            action_logits: (batch, n_actions)
            value: (batch,)
        """
        gru_feat = self._encode_gru(intra_seq)
        x = torch.cat([obs, gru_feat], dim=-1)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def act(self, obs, intra_seq, deterministic=False):
        """Single-step action sampling (numpy in/out).

        Args:
            obs: (obs_dim,) numpy array
            intra_seq: (K, intra_feat_dim) numpy array (padded with zeros)
            deterministic: if True, argmax
        Returns:
            action_idx: int, log_prob: float, value: float
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            intra_t = torch.as_tensor(intra_seq, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            logits, value = self.forward(obs_t, intra_t)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, 1)
            log_prob = log_probs.gather(1, action).squeeze(-1)
        return int(action.item()), log_prob.item(), value.item()

    def evaluate(self, obs, intra_seq, action):
        """Batch evaluate for PPO update.

        Args:
            obs: (batch, obs_dim)
            intra_seq: (batch, K, intra_feat_dim)
            action: (batch,) int64
        Returns:
            log_prob: (batch,), value: (batch,), entropy: (batch,)
        """
        logits, values = self.forward(obs, intra_seq)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return log_prob, values, entropy

    def get_probs(self, obs, intra_seq):
        """Read-only action probabilities.

        Returns:
            numpy (n_actions,) float32 probs.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            intra_t = torch.as_tensor(intra_seq, dtype=torch.float32,
                                      device=self.device).unsqueeze(0)
            logits, _ = self.forward(obs_t, intra_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.cpu().numpy()
