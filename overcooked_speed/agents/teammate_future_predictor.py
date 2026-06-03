"""TeammateFuturePredictor: GRU model that predicts teammate future actions and events.

Input: K-step history of (obs, action_onehot) for a single agent
Output: future action histogram (H-step soft target) + future event vector (binary OR over H steps)
Latent z = tanh(Linear(GRU(hidden) → latent_dim)) — usable as critic_extra in Phase C.
"""
import numpy as np
import torch
import torch.nn as nn


class TeammateFuturePredictor(nn.Module):
    """GRU encoder → latent z → action_head + event_head.

    No PPO modification. Pure supervised learning on collected trajectories.
    """

    def __init__(self, obs_dim, n_actions=6, n_events=5,
                 hidden_dim=64, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.n_events = n_events
        self.input_dim = obs_dim + n_actions

        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.action_head = nn.Linear(latent_dim, n_actions)
        self.event_head = nn.Linear(latent_dim, n_events)

    def forward(self, history_seq):
        """Forward pass.

        Args:
            history_seq: (batch, K, obs_dim + n_actions) float32 tensor

        Returns:
            action_pred: (batch, n_actions) softmax over future actions
            event_logits: (batch, n_events) logits for future events
            z: (batch, latent_dim) tanh-bounded latent vector
        """
        _, h_n = self.gru(history_seq)
        z = torch.tanh(self.latent_proj(h_n.squeeze(0)))
        action_pred = torch.softmax(self.action_head(z), dim=-1)
        event_logits = self.event_head(z)
        return action_pred, event_logits, z

    def encode(self, history_seq):
        """Return latent z only (for Phase C critic_extra)."""
        _, h_n = self.gru(history_seq)
        z = torch.tanh(self.latent_proj(h_n.squeeze(0)))
        return z


def build_history_sequence(obs_seq, action_seq):
    """Build (K, obs_dim + n_actions) input from raw arrays.

    Args:
        obs_seq: (K, obs_dim) float32 numpy
        action_seq: (K,) int numpy (0..n_actions-1)

    Returns:
        (K, obs_dim + 6) float32 numpy
    """
    n_actions = 6
    K = len(action_seq)
    action_onehot = np.zeros((K, n_actions), dtype=np.float32)
    action_onehot[np.arange(K), action_seq] = 1.0
    return np.concatenate([obs_seq, action_onehot], axis=-1)


def make_action_target(future_actions, n_actions=6):
    """Build soft-target action histogram from future action sequence.

    Args:
        future_actions: (H,) int numpy — actions t+1 .. t+H
        n_actions: action space size (default 6)

    Returns:
        (n_actions,) float32 — normalized histogram
    """
    H = len(future_actions)
    if H == 0:
        return np.ones(n_actions, dtype=np.float32) / n_actions
    target = np.zeros(n_actions, dtype=np.float32)
    for a in future_actions:
        if 0 <= a < n_actions:
            target[a] += 1.0
    return target / H


def make_event_target(future_events, n_events=5):
    """Build binary event target (logical OR over future window).

    Args:
        future_events: (H, n_events) uint8 or bool numpy
        n_events: number of event types (default 5)

    Returns:
        (n_events,) float32 — 1.0 if event occurred in window, 0.0 otherwise
    """
    if len(future_events) == 0:
        return np.zeros(n_events, dtype=np.float32)
    return np.any(future_events, axis=0).astype(np.float32)
