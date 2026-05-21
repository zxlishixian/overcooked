"""MAPPO: Multi-Agent PPO with centralized critic.

Two decentralized actors (one per agent) share a single centralized critic V(s_global).
The critic is only used during training (CTDE: Centralized Training, Decentralized Execution).
Both actors update using the same team advantage computed by the centralized critic.

Reference: Yu et al. (2022) "The Surprising Effectiveness of PPO in Cooperative
Multi-Agent Games" https://arxiv.org/abs/2103.01955
"""
import numpy as np
import torch
import torch.nn as nn

from .policy import ActorCritic


class CentralizedCritic(nn.Module):
    """3-layer MLP value function: V(s_global) → scalar."""

    def __init__(self, obs_dim, hidden_dim=256, device='cpu'):
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Orthogonal init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        self.to(device)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class MAPPOManager:
    """Manages MAPPO training: two actors + one centralized critic.

    Data collection uses the actors' own critic heads (not the centralized critic).
    Training uses the centralized critic for GAE advantage computation, then updates
    both actors (PPO clip) and the centralized critic (MSE regression).
    """

    def __init__(self, obs_dim, n_actions=6, lr=1e-3, gamma=0.99,
                 hidden_dim=256, device='cpu',
                 ppo_epochs=4, ppo_clip=0.2, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.05, max_grad_norm=0.5,
                 critic_lr=None, global_obs_dim=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        if global_obs_dim is None:
            global_obs_dim = obs_dim

        # Two decentralized actors — each sees only its own egocentric/local obs
        self.actor0 = ActorCritic(obs_dim, hidden_dim, n_actions, device=device)
        self.actor1 = ActorCritic(obs_dim, hidden_dim, n_actions, device=device)

        # One centralized critic — sees full global state
        self.critic = CentralizedCritic(global_obs_dim, hidden_dim, device=device)

        # Separate optimizers
        critic_lr = critic_lr if critic_lr is not None else lr
        self.actor0_opt = torch.optim.Adam(self.actor0.parameters(), lr=lr)
        self.actor1_opt = torch.optim.Adam(self.actor1.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Per-agent episode buffers (actions, log_probs, observations)
        self._obs0, self._obs1 = [], []
        self._global_obs = []
        self._actions0, self._actions1 = [], []
        self._log_probs0, self._log_probs1 = [], []
        self._rewards = []
        self._dones = []

    def train(self):
        self.actor0.train()
        self.actor1.train()
        self.critic.train()

    def eval(self):
        self.actor0.eval()
        self.actor1.eval()
        self.critic.eval()

    def act(self, obs0, obs1, global_obs, deterministic=False):
        """Both actors sample actions independently. Returns (a0, a1).

        Args:
            obs0, obs1: actor-specific observations (controlled by obs_mode)
            global_obs: full global state for centralized critic V(s)
        """
        a0, lp0, _ = self.actor0.act(obs0, deterministic)
        a1, lp1, _ = self.actor1.act(obs1, deterministic)

        self._obs0.append(obs0)
        self._obs1.append(obs1)
        self._global_obs.append(global_obs)
        self._actions0.append(a0)
        self._actions1.append(a1)
        self._log_probs0.append(lp0)
        self._log_probs1.append(lp1)

        return a0, a1

    def store_reward(self, reward):
        self._rewards.append(reward)
        self._dones.append(False)

    def end_episode(self):
        """Compute centralized GAE, then update both actors + critic."""
        if len(self._rewards) == 0:
            self._clear_buffers()
            return self._empty_log()

        T = len(self._rewards)
        self._dones[-1] = True

        # ── Compute centralized critic values for all states ──
        # Critic uses full global state, which may differ from actor obs
        global_t = torch.as_tensor(np.stack(self._global_obs), dtype=torch.float32,
                                   device=self.device)
        with torch.no_grad():
            critic_values = self.critic(global_t).cpu().numpy()

        # ── Compute GAE advantages (shared by both actors) ──
        advantages, returns = self._compute_gae(
            self._rewards, critic_values, self._dones)

        value_mean = float(np.mean(critic_values))
        advantage_mean = float(np.mean(advantages))

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Convert buffers to tensors ──
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Each actor uses its own egocentric observations
        obs0_t = torch.as_tensor(np.stack(self._obs0), dtype=torch.float32, device=self.device)
        obs1_t = torch.as_tensor(np.stack(self._obs1), dtype=torch.float32, device=self.device)

        actions0_t = torch.as_tensor(self._actions0, dtype=torch.int64, device=self.device)
        actions1_t = torch.as_tensor(self._actions1, dtype=torch.int64, device=self.device)
        old_lp0_t = torch.as_tensor(self._log_probs0, dtype=torch.float32, device=self.device)
        old_lp1_t = torch.as_tensor(self._log_probs1, dtype=torch.float32, device=self.device)

        # ── Update both actors with PPO ──
        actor0_log = self._update_actor(
            self.actor0, self.actor0_opt, obs0_t, actions0_t,
            old_lp0_t, advantages_t)

        actor1_log = self._update_actor(
            self.actor1, self.actor1_opt, obs1_t, actions1_t,
            old_lp1_t, advantages_t)

        # ── Update centralized critic with global state ──
        critic_log = self._update_critic(global_t, returns_t)

        self._clear_buffers()

        return {
            'loss': actor0_log['loss'],
            'actor0_loss': actor0_log['loss'],
            'actor1_loss': actor1_log['loss'],
            'critic_loss': critic_log['loss'],
            'mean_return': returns_t.mean().item(),
            'entropy': actor0_log['entropy'],
            'entropy0': actor0_log['entropy'],
            'entropy1': actor1_log['entropy'],
            'grad_norm': actor0_log['grad_norm'],
            'a0_grad_norm': actor0_log['grad_norm'],
            'a1_grad_norm': actor1_log['grad_norm'],
            'approx_kl': actor0_log['approx_kl'],
            'a0_approx_kl': actor0_log['approx_kl'],
            'a1_approx_kl': actor1_log['approx_kl'],
            'value_loss': critic_log['loss'],
            'a0_value_loss': critic_log['loss'],
            'a1_value_loss': critic_log['loss'],
            'value_mean': value_mean,
            'advantage_mean': advantage_mean,
        }

    def _update_actor(self, actor, optimizer, obs, actions, old_log_probs,
                      advantages):
        """PPO clipped update for a single actor (no critic head update)."""
        total_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_grad_norm = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), 64):
                end = start + 64
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_act = actions[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]

                new_lp, _, entropy = actor.evaluate(mb_obs, mb_act)

                # PPO clip loss (actor only, no value term)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                loss = policy_loss - self.ent_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), self.max_grad_norm)
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()
                optimizer.step()

                total_loss += loss.item()
                total_entropy += entropy.mean().item()
                with torch.no_grad():
                    approx_kl = 0.5 * ((new_lp - mb_old_lp) ** 2).mean().item()
                total_approx_kl += approx_kl
                total_grad_norm += grad_norm

        n_updates = max(1, self.ppo_epochs * max(1, (len(obs) // 64)))
        return {
            'loss': total_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'grad_norm': total_grad_norm / n_updates,
            'approx_kl': total_approx_kl / n_updates,
        }

    def _update_critic(self, obs, returns):
        """MSE regression for centralized critic."""
        total_loss = 0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), 64):
                end = start + 64
                mb_idx = indices[start:end]
                mb_obs = obs[mb_idx]
                mb_ret = returns[mb_idx]

                values = self.critic(mb_obs)
                loss = 0.5 * (values - mb_ret).pow(2).mean()

                self.critic_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm)
                self.critic_opt.step()

                total_loss += loss.item()
                n_updates += 1

        return {'loss': total_loss / max(1, n_updates)}

    def _compute_gae(self, rewards, values, dones):
        T = len(rewards)
        gae = 0
        advantages = np.zeros(T, dtype=np.float32)

        for t in reversed(range(T)):
            if dones[t]:
                next_value = 0
            else:
                next_value = values[t + 1] if t + 1 < T else 0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - int(dones[t]))
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def _clear_buffers(self):
        self._obs0, self._obs1 = [], []
        self._global_obs = []
        self._actions0, self._actions1 = [], []
        self._log_probs0, self._log_probs1 = [], []
        self._rewards = []
        self._dones = []

    @staticmethod
    def _empty_log():
        return {
            'loss': 0, 'actor0_loss': 0, 'actor1_loss': 0,
            'critic_loss': 0, 'mean_return': 0,
            'entropy': 0, 'entropy0': 0, 'entropy1': 0,
            'grad_norm': 0, 'a0_grad_norm': 0, 'a1_grad_norm': 0,
            'approx_kl': 0, 'a0_approx_kl': 0, 'a1_approx_kl': 0,
            'value_loss': 0, 'a0_value_loss': 0, 'a1_value_loss': 0,
            'value_mean': 0, 'advantage_mean': 0,
        }
