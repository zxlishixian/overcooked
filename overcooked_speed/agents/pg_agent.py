"""PGAgent: PPO (Proximal Policy Optimization) agent.

Independent learner — each agent has its own Actor-Critic network.
Uses GAE for advantage estimation, PPO clip for stable policy updates,
and multi-epoch replay of collected trajectories.

Supports conditioned critic via critic_extra_dim > 0:
  V_i(obs_i, extra) where extra can be teammate action probs,
  intention vector, or concatenation of both.
"""
import numpy as np
from .policy import ActorCritic


class PGAgent:
    """PPO agent with GAE and clipped surrogate objective."""

    def __init__(self, agent_id, obs_dim, n_actions=6, lr=1e-3, gamma=0.99,
                 hidden_dim=256, device='cpu',
                 ppo_epochs=10, ppo_clip=0.2, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 critic_extra_dim=0, **_kwargs):
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.gamma = gamma
        self.device = device
        self.critic_extra_dim = critic_extra_dim

        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.policy = ActorCritic(obs_dim, hidden_dim, n_actions, device=device,
                                  critic_extra_dim=critic_extra_dim)
        self.optimizer = self._make_optimizer(lr)

        # Episode buffer
        self._obs = []
        self._actions = []
        self._log_probs = []
        self._values = []
        self._rewards = []
        self._dones = []
        self._critic_extra = []

    def _make_optimizer(self, lr):
        import torch
        return torch.optim.Adam(self.policy.parameters(), lr=lr)

    def get_action_probs(self, obs):
        """Return action probs (6-dim numpy) via forward_actor only.

        Pure read — no buffer side effects, no critic computation,
        torch.no_grad() ensures detachment from any computation graph.
        """
        return self.policy.get_probs(obs)

    def act(self, obs, deterministic=False, critic_extra=None):
        """Choose action, return action index. Stores log_prob, value, and
        optionally critic_extra to buffer.

        When critic_extra_dim > 0, critic_extra is required.
        """
        if critic_extra is not None:
            self._critic_extra.append(critic_extra.copy())
        elif self.policy.critic_extra_proj is not None:
            raise ValueError(
                "critic_extra required in conditioned mode "
                f"(critic_extra_dim={self.critic_extra_dim}). "
                "Provide critic_extra to act().")

        action, log_prob, value = self.policy.act(
            obs, deterministic, critic_extra=critic_extra)
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

    def end_episode(self, aux_adv=None, aux_coef=0.0,
                    shuffle_critic_extra=False, **_kwargs):
        """Compute GAE and run PPO updates.

        Args:
            aux_adv: Optional scalar episode-level auxiliary advantage.
            aux_coef: Weight for the auxiliary actor loss.
            shuffle_critic_extra: If True, permute critic_extra before PPO
                update (for shuffled ablation mode). Diagnostics are
                recomputed with shuffled extras to reflect what the critic
                actually sees during the update.

        Returns dict with loss, value diagnostics, and conditioned-critic fields.
        """
        import torch

        empty_return = {
            'loss': 0, 'mean_return': 0, 'entropy': 0, 'grad_norm': 0,
            'approx_kl': 0, 'value_loss': 0, 'aux_loss': 0,
            'value_mean': 0.0, 'explained_variance': 0.0,
            'critic_extra_entropy': 0.0,
            'value_extra_sensitivity': 0.0,
            'sensitivity_note': None,
            'value_mean_shuffled': None,
            'explained_variance_shuffled': None,
            'shuffled_diag_note': None,
        }

        if len(self._rewards) == 0:
            self._clear_buffer()
            return empty_return

        # Mark last step as terminal for GAE
        self._dones[-1] = True

        # ── Diagnostics: pre-update from rollout-time stored values ──
        rollout_values = np.array(self._values, dtype=np.float32)

        # ── Compute GAE advantages ──
        advantages, returns = self._compute_gae(
            self._rewards, self._values, self._dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return_mean = float(returns.mean())
        value_mean = float(rollout_values.mean())

        # explained_variance = 1 - Var(returns - rollout_values) / Var(returns)
        returns_var = float(returns.var())
        if returns_var > 1e-8:
            explained_variance = float(
                1.0 - ((returns - rollout_values).var() / returns_var))
        else:
            explained_variance = 0.0

        # ── Critic extra entropy (pre-update, pre-shuffle) ──
        sensitivity_note = None
        if self._critic_extra:
            extra_arr = np.stack(self._critic_extra)
            eps = 1e-8
            entropy_per_step = -np.sum(extra_arr * np.log(extra_arr + eps), axis=-1)
            critic_extra_entropy = float(entropy_per_step.mean())
        else:
            critic_extra_entropy = 0.0

        # ── Value sensitivity to critic_extra (pre-update, pre-shuffle) ──
        value_extra_sensitivity = 0.0
        if self.policy.critic_extra_proj is not None and len(self._critic_extra) >= 2:
            try:
                obs_t = torch.as_tensor(np.stack(self._obs),
                                        dtype=torch.float32, device=self.device)
                extra_t = torch.as_tensor(np.stack(self._critic_extra),
                                          dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    _, v_correct = self.policy.forward(obs_t, critic_extra=extra_t)
                    # Shuffle critic_extra across batch
                    idx_shuf = torch.randperm(len(extra_t), device=self.device)
                    extra_shuf = extra_t[idx_shuf]
                    _, v_shuffled = self.policy.forward(obs_t, critic_extra=extra_shuf)
                    value_extra_sensitivity = float(
                        (v_correct - v_shuffled).abs().mean().item())
            except Exception:
                value_extra_sensitivity = 0.0
                sensitivity_note = "computation_failed"
        elif self.policy.critic_extra_proj is not None and len(self._critic_extra) < 2:
            value_extra_sensitivity = 0.0
            sensitivity_note = "single_timestep"

        # ── Convert buffer to tensors ──
        obs_t = torch.as_tensor(np.stack(self._obs), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(self._actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(self._log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        critic_extra_t = None
        if self._critic_extra:
            critic_extra_t = torch.as_tensor(
                np.stack(self._critic_extra), dtype=torch.float32, device=self.device)
            assert len(self._critic_extra) == len(self._obs), \
                f"critic_extra buffer ({len(self._critic_extra)}) != obs buffer ({len(self._obs)})"

        # ── Shuffle critic_extra for ablation (after diagnostics) ──
        shuffled_diag_note = None
        value_mean_shuffled = None
        explained_variance_shuffled = None
        if shuffle_critic_extra and critic_extra_t is not None:
            critic_extra_t = critic_extra_t[torch.randperm(len(critic_extra_t), device=self.device)]
            # Recompute diagnostics with shuffled extras to reflect what
            # the critic actually sees during the PPO update
            try:
                with torch.no_grad():
                    _, v_shuf = self.policy.forward(obs_t, critic_extra=critic_extra_t)
                    v_shuf_np = v_shuf.cpu().numpy()
                    value_mean_shuffled = float(v_shuf_np.mean())
                    returns_np = returns  # already numpy
                    returns_var_shuf = float(returns_np.var())
                    if returns_var_shuf > 1e-8:
                        explained_variance_shuffled = float(
                            1.0 - ((returns_np - v_shuf_np).var() / returns_var_shuf))
                    else:
                        explained_variance_shuffled = 0.0
            except Exception:
                shuffled_diag_note = "shuffled_diag_computation_failed"
        elif shuffle_critic_extra:
            shuffled_diag_note = "no_critic_extra_to_shuffle"

        # ── PPO update loop ──
        total_loss = 0
        total_entropy = 0
        total_value_loss = 0
        total_approx_kl = 0
        total_grad_norm = 0
        total_aux_loss = 0
        n_updates = 0
        use_aux = aux_adv is not None and aux_coef != 0.0
        aux_adv_t = None
        if use_aux:
            aux_adv_t = torch.as_tensor(float(aux_adv), dtype=torch.float32,
                                        device=self.device).detach()

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(len(obs_t), device=self.device)
            for start in range(0, len(obs_t), 64):  # minibatch size 64
                end = start + 64
                mb_idx = indices[start:end]

                mb_obs = obs_t[mb_idx]
                mb_act = actions_t[mb_idx]
                mb_old_lp = old_log_probs_t[mb_idx]
                mb_adv = advantages_t[mb_idx]
                mb_ret = returns_t[mb_idx]
                mb_extra = (critic_extra_t[mb_idx]
                            if critic_extra_t is not None else None)

                new_lp, values, entropy = self.policy.evaluate(
                    mb_obs, mb_act, critic_extra=mb_extra)

                # PPO clip loss
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values - mb_ret).pow(2).mean()

                if use_aux:
                    aux_loss = -(new_lp * aux_adv_t).mean()
                else:
                    aux_loss = torch.zeros((), dtype=torch.float32, device=self.device)

                # Total loss
                loss = (policy_loss + self.vf_coef * value_loss -
                        self.ent_coef * entropy.mean() + aux_coef * aux_loss)

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
                total_aux_loss += aux_loss.item()

                with torch.no_grad():
                    approx_kl = 0.5 * ((new_lp - mb_old_lp) ** 2).mean().item()
                total_approx_kl += approx_kl
                total_grad_norm += grad_norm
                n_updates += 1

        n_updates = max(1, n_updates)
        self._clear_buffer()

        return {
            'loss': total_loss / n_updates,
            'mean_return': return_mean,
            'entropy': total_entropy / n_updates,
            'grad_norm': total_grad_norm / n_updates,
            'approx_kl': total_approx_kl / n_updates,
            'value_loss': total_value_loss / n_updates,
            'aux_loss': total_aux_loss / n_updates,
            'value_mean': value_mean,
            'explained_variance': explained_variance,
            'critic_extra_entropy': critic_extra_entropy,
            'value_extra_sensitivity': value_extra_sensitivity,
            'sensitivity_note': sensitivity_note,
            'value_mean_shuffled': value_mean_shuffled,
            'explained_variance_shuffled': explained_variance_shuffled,
            'shuffled_diag_note': shuffled_diag_note,
        }

    def _compute_gae(self, rewards, values, dones):
        """Compute GAE advantages and returns."""
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
        self._critic_extra = []

    def get_parameters(self):
        return np.concatenate([p.data.cpu().numpy().ravel()
                              for p in self.policy.parameters()])

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()
