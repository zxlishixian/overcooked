"""PGAgent: PPO (Proximal Policy Optimization) agent.

Independent learner — each agent has its own Actor-Critic network.
Uses GAE for advantage estimation, PPO clip for stable policy updates,
and multi-epoch replay of collected trajectories.
"""
import numpy as np
from .policy import ActorCritic


class PGAgent:
    """PPO agent with GAE and clipped surrogate objective."""

    def __init__(self, agent_id, obs_dim, n_actions=6, lr=1e-3, gamma=0.99,
                 hidden_dim=256, device='cpu',
                 ppo_epochs=10, ppo_clip=0.2, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device

        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.policy = ActorCritic(obs_dim, hidden_dim, n_actions, device=device)
        self.optimizer = self._make_optimizer(lr)

        # Episode buffer
        self._obs = []
        self._actions = []
        self._log_probs = []
        self._values = []
        self._rewards = []
        self._dones = []

    def _make_optimizer(self, lr):
        import torch
        return torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, obs, deterministic=False):
        """Choose action, return action index. Stores log_prob and value."""
        action, log_prob, value = self.policy.act(obs, deterministic)
        self._obs.append(obs)
        self._actions.append(action)
        self._log_probs.append(log_prob)
        self._values.append(value)
        return action

    def store_reward(self, reward):
        self._rewards.append(reward)
        self._dones.append(False)

    def add_terminal_bonus(self, bonus):
        """Add bonus to the last stored reward (for role shaping at episode end)."""
        if self._rewards:
            self._rewards[-1] += bonus

    def end_episode(self):
        """Compute GAE and run PPO updates."""
        import torch

        if len(self._rewards) == 0:
            self._clear_buffer()
            return {'loss': 0, 'mean_return': 0, 'entropy': 0, 'grad_norm': 0,
                    'approx_kl': 0, 'value_loss': 0}

        # Mark last step as terminal for GAE
        self._dones[-1] = True

        # ── Compute GAE advantages ──
        advantages, returns = self._compute_gae(
            self._rewards, self._values, self._dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Convert buffer to tensors ──
        obs_t = torch.as_tensor(np.stack(self._obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(self._actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(self._log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # ── PPO update loop ──
        total_loss = 0
        total_entropy = 0
        total_value_loss = 0
        total_approx_kl = 0
        total_grad_norm = 0

        for _ in range(self.ppo_epochs):
            # Shuffle
            indices = torch.randperm(len(obs_t), device=self.device)
            for start in range(0, len(obs_t), 64):  # minibatch size 64
                end = start + 64
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_act = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]

                new_lp, values, entropy = self.policy.evaluate(mb_obs, mb_act)

                # PPO clip loss
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm)

                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()

                self.optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.mean().item()
                total_value_loss += value_loss.item()

                with torch.no_grad():
                    approx_kl = 0.5 * ((new_lp - mb_old_lp) ** 2).mean().item()
                total_approx_kl += approx_kl
                total_grad_norm += grad_norm

        n_updates = max(1, self.ppo_epochs * max(1, (len(obs_t) // 64)))
        self._clear_buffer()

        return {
            'loss': total_loss / n_updates,
            'mean_return': returns_t.mean().item(),
            'entropy': total_entropy / n_updates,
            'grad_norm': total_grad_norm / n_updates,
            'approx_kl': total_approx_kl / n_updates,
            'value_loss': total_value_loss / n_updates,
        }

    def _compute_gae(self, rewards, values, dones):
        """Compute GAE advantages and returns.

        Args:
            rewards: list of scalar rewards per step
            values: list of scalar V(s) per step
            dones: list of bool (True at terminal step)

        Returns:
            advantages: numpy array, returns: numpy array
        """
        T = len(rewards)
        gae = 0
        advantages = np.zeros(T, dtype=np.float32)
        returns_val = np.zeros(T, dtype=np.float32)

        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1] if t + 1 < T else 0

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - int(dones[t]))
            advantages[t] = gae

        returns_val = advantages + np.array(values, dtype=np.float32)
        return advantages, returns_val

    def _clear_buffer(self):
        self._obs = []
        self._actions = []
        self._log_probs = []
        self._values = []
        self._rewards = []
        self._dones = []

    def get_parameters(self):
        return np.concatenate([p.data.cpu().numpy().ravel()
                              for p in self.policy.parameters()])

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()
