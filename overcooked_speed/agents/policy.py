"""Actor-Critic network for Overcooked PPO agent.

Shared feature extractor with separate actor (policy) and critic (value) heads.
Supports policy-conditioned critic via critic_extra_dim > 0.
"""
import torch
import torch.nn as nn
import numpy as np


class ActorCritic(nn.Module):
    """2-layer shared MLP with actor and critic heads.

    Input: lossless_state_encoding flattened (~1040-dim for cramped_room)
    Shared: 256 ReLU
    Actor: 6 action logits
    Critic: 1 scalar state value (optionally conditioned on teammate policy)
    """

    def __init__(self, obs_dim, hidden_dim=256, n_actions=6, device='cpu',
                 critic_extra_dim=0):
        super().__init__()
        self.device = device
        self.critic_extra_dim = critic_extra_dim

        self.shared = nn.Linear(obs_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, n_actions)

        if critic_extra_dim > 0:
            self.critic_extra_proj = nn.Linear(critic_extra_dim, 32)
            nn.init.orthogonal_(self.critic_extra_proj.weight, gain=np.sqrt(2))
            nn.init.constant_(self.critic_extra_proj.bias, 0)
            self.critic = nn.Linear(hidden_dim + 32, 1)
        else:
            self.critic_extra_proj = None
            self.critic = nn.Linear(hidden_dim, 1)

        # Orthogonal init with standard gains
        nn.init.orthogonal_(self.shared.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.shared.bias, 0)
        nn.init.constant_(self.actor.bias, 0)
        nn.init.constant_(self.critic.bias, 0)

        self.to(device)

    def forward_actor(self, obs):
        """Actor-only forward: obs → action logits. No critic computation."""
        h = torch.relu(self.shared(obs))
        return self.actor(h)

    def forward(self, obs, critic_extra=None):
        """Return (action_logits, value) for a batch of observations.

        When critic_extra is provided and critic_extra_proj exists,
        the critic head receives concat(shared_features, proj(critic_extra)).
        Otherwise critic sees shared_features only.
        """
        h = torch.relu(self.shared(obs))
        logits = self.actor(h)

        if critic_extra is not None and self.critic_extra_proj is not None:
            extra_feat = torch.relu(self.critic_extra_proj(critic_extra))
            value = self.critic(torch.cat([h, extra_feat], dim=-1)).squeeze(-1)
        else:
            value = self.critic(h).squeeze(-1)

        return logits, value

    def _require_critic_extra(self):
        if self.critic_extra_proj is not None:
            raise ValueError(
                "critic_extra required in PC mode "
                "(critic_extra_dim > 0). ActorCritic was constructed with "
                f"critic_extra_dim={self.critic_extra_dim}")

    def act(self, obs, deterministic=False, critic_extra=None):
        """Return (action_idx, log_prob, value) from numpy observation."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            extra_t = None
            if critic_extra is not None:
                extra_t = torch.as_tensor(critic_extra, dtype=torch.float32,
                                          device=self.device).unsqueeze(0)
            else:
                self._require_critic_extra()

            logits, value = self.forward(obs_t, critic_extra=extra_t)
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, 1)
            log_prob = log_probs.gather(1, action).squeeze(-1)
        return int(action.item()), log_prob.item(), value.item()

    def evaluate(self, obs, action, critic_extra=None):
        """Return (log_prob, value, entropy) for batch (with grad)."""
        if critic_extra is None:
            self._require_critic_extra()
        logits, values = self.forward(obs, critic_extra=critic_extra)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return log_prob, values, entropy

    def get_probs(self, obs):
        """Return action probs (6-dim) from numpy obs via forward_actor only.

        No critic computation — safe regardless of critic_extra_dim.
        Runs under torch.no_grad(), returns numpy array.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            logits = self.forward_actor(obs_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.cpu().numpy()
