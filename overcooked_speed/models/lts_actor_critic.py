"""Belief-conditioned Actor-Critic network for LTS-PPO.

Input: [obs || belief]
Separate MLP trunks for actor and critic (no shared layers).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LTSActorCritic(nn.Module):
    """Belief-conditioned Actor-Critic with separate actor/critic MLP trunks.

    Input: concat(obs, belief) → (obs_dim + belief_dim,)
    Actor: Linear → ReLU → Linear → n_actions logits
    Critic: Linear → ReLU → Linear → 1 scalar value
    """

    def __init__(self, obs_dim, belief_dim=64, hidden_dim=256,
                 n_actions=6, device='cpu'):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        input_dim = obs_dim + belief_dim

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

        # Orthogonal init with proper gains
        for net in [self.actor, self.critic]:
            for m in net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0)

        # Actor last layer: small gain to prevent early overconfidence
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.constant_(self.actor[-1].bias, 0)
        # Critic last layer: gain=1.0
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.constant_(self.critic[-1].bias, 0)

        self.to(device)

    def forward(self, obs, belief):
        """Args:
            obs: (batch, obs_dim)
            belief: (batch, belief_dim)
        Returns:
            action_logits: (batch, n_actions)
            value: (batch,)
        """
        x = torch.cat([obs, belief], dim=-1)
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def act(self, obs, belief, deterministic=False):
        """Single-step action sampling (numpy in/out, used during rollout).

        Args:
            obs: (obs_dim,) numpy array
            belief: (belief_dim,) numpy array
            deterministic: if True, argmax instead of sample
        Returns:
            action_idx: int
            log_prob: float
            value: float
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            belief_t = torch.as_tensor(belief, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
            logits, value = self.forward(obs_t, belief_t)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, 1)
            log_prob = log_probs.gather(1, action).squeeze(-1)
        return int(action.item()), log_prob.item(), value.item()

    def evaluate(self, obs, belief, action):
        """Batch evaluate for PPO update.

        Args:
            obs: (batch, obs_dim)
            belief: (batch, belief_dim)
            action: (batch,) int64
        Returns:
            log_prob: (batch,)
            value: (batch,)
            entropy: (batch,)
        """
        logits, values = self.forward(obs, belief)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return log_prob, values, entropy

    def get_probs(self, obs, belief):
        """Read-only action probabilities (for teammate policy queries).

        Returns:
            numpy (n_actions,) float32 probs.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            belief_t = torch.as_tensor(belief, dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
            logits, _ = self.forward(obs_t, belief_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.cpu().numpy()
