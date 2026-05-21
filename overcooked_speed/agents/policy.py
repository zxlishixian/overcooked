"""Actor-Critic network for Overcooked PPO agent.

Shared feature extractor with separate actor (policy) and critic (value) heads.
"""
import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """2-layer shared MLP with actor and critic heads.

    Input: lossless_state_encoding flattened (~1040-dim for cramped_room)
    Shared: 256 ReLU
    Actor: 6 action logits
    Critic: 1 scalar state value
    """

    def __init__(self, obs_dim, hidden_dim=256, n_actions=6, device='cpu'):
        super().__init__()
        self.device = device

        self.shared = nn.Linear(obs_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal init with standard gains
        nn.init.orthogonal_(self.shared.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.shared.bias, 0)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.constant_(self.critic.bias, 0)

        self.to(device)

    def forward(self, obs):
        """Return (action_logits, value) for a batch of observations."""
        h = torch.relu(self.shared(obs))
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def act(self, obs, deterministic=False):
        """Return (action_idx, log_prob, value) from numpy observation."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, value = self.forward(obs_t)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, 1)
            log_prob = log_probs.gather(1, action).squeeze(-1)
        return int(action.item()), log_prob.item(), value.item()

    def evaluate(self, obs, action):
        """Return (log_prob, value, entropy) for batch (with grad)."""
        logits, values = self.forward(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return log_prob, values, entropy
