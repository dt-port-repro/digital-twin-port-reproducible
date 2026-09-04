"""
PPO智能体 — Actor-Critic架构（论文 §5.3.2）

Implements the Proximal Policy Optimization (PPO) agent used as the
prediction-optimization coordinator in the yard allocation system.

MDP Formulation (§5.3.2):
    State:  14+5 dimensions
            (prediction confidence, prediction error, future arrivals,
             yard occupancy, avg equip util, queue length, congestion,
             reshuffle bias, equip bias + 5 execution feedback dims)

    Action: 6 continuous dimensions (weight scaling factors)
            [berth_dist, virtual_reserve_match, virtual_reserve_miss,
             space_util, rehandle_prob, conflict]
            Each squashed to [0.5, 1.5] via tanh.

    Reward: reshuffle_reduction + equip_util_improvement
            - weight_variance_penalty

    Discount: γ = 0.99

Architecture:
    Actor:  2-layer MLP (64 → 64 → 6) with tanh output squashed
    Critic: 2-layer MLP (64 → 64 → 1)

Training (§5.3.2):
    Clip:       0.2
    LR:         3e-4
    Epochs:     10 (per update)
    Episodes:   50 per window, 2 windows walk-forward

Results from paper:
    W1: training reward  -757.7 → -719.1 (converged)
        test reward       -318.7  (vs baseline -404.2, +21.1%)
    W2: training reward -1459.1 → -1124.1
        test reward       -427.7  (vs baseline -584.0, +26.7%)

Usage:
    agent = PPOAgent(state_dim=19, action_dim=6)
    action = agent.select_action(state)           # inference
    agent.learn(memory)                           # training update
    torch.save(agent.actor.state_dict(), "actor.pt")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)

# Device detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════
#  Paper constants (§5.3.2)
# ══════════════════════════════════════════════════════════════════

# MDP dimensions
STATE_DIM = 19                # 14 + 5 execution feedback dims
ACTION_DIM = 6                # 6 weight scaling factors

# PPO hyperparameters
GAMMA = 0.99                  # discount factor (§5.3.2)
CLIP_EPSILON = 0.2            # PPO clipping range (§5.3.2)
LR = 3e-4                     # learning rate (§5.3.2)
PPO_EPOCHS = 10               # training epochs per update (§5.3.2)
EPS = 1e-8                    # numerical stability

# Network architecture
ACTOR_HIDDEN = 64             # hidden layer size (§5.3.2)
CRITIC_HIDDEN = 64

# Action bounds
ACTION_LOW = 0.5              # scaling factor lower bound
ACTION_HIGH = 1.5             # scaling factor upper bound

# Reward components (§5.3.2 reward formulation)
REWARD_RESHUF_COEF = 1.0      # reshuffle reduction coefficient
REWARD_EQUIP_COEF = 1.0       # equipment utilization improvement coefficient
REWARD_VAR_PENALTY = 0.1      # weight variance penalty coefficient


# ══════════════════════════════════════════════════════════════════
#  Actor Network  (§5.3.2 Architecture)
# ══════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """
    PPO Actor network (论文 §5.3.2 Actor 架构).

    Maps state → mean of Gaussian action distribution.
    Architecture: 2-layer MLP (64 → 64 → 6).
    Output is squashed via tanh to [0.5, 1.5] range.

    Input:  state_dim  (default 19)
    Output: action_dim (default 6), in [0.5, 1.5]
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = ACTOR_HIDDEN,
    ):
        """
        Args:
            state_dim:  Dimension of the state vector.
            action_dim: Dimension of the action vector.
            hidden_dim: Hidden layer size.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)

        # Learnable log-std for each action dimension
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialization for all linear layers."""
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mean.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.mean.bias)

    def forward(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute action mean and log-std.

        Args:
            state: (batch, state_dim) tensor.

        Returns:
            (mean, log_std) tuples, each (batch, action_dim).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)

        # Squash mean to [ACTION_LOW, ACTION_HIGH] using tanh
        mean = torch.tanh(mean)  # [-1, 1]
        mean = (mean + 1.0) / 2.0  # [0, 1]
        mean = mean * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW  # [0.5, 1.5]

        # Log-std is a learnable parameter
        log_std = self.log_std.expand_as(mean)
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, min=math.log(1e-6), max=math.log(1.0))

        return mean, log_std

    def get_action_and_log_prob(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the Gaussian policy and return log-prob.

        Args:
            state: (batch, state_dim) tensor.

        Returns:
            (action, log_prob, mean) — each (batch, action_dim).
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()  # differentiable sampling

        # Clamp to action bounds
        action = torch.clamp(z, min=ACTION_LOW, max=ACTION_HIGH)

        # Compute log-prob (accounting for clamping)
        log_prob = dist.log_prob(z)
        # Adjust for clamping
        log_prob -= torch.log(1.0 - torch.tanh(z / 2.0).pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, mean

    def evaluate(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log-prob and entropy for given state-action pairs.

        Args:
            state:  (batch, state_dim) tensor.
            action: (batch, action_dim) tensor.

        Returns:
            (log_prob, entropy, mean) tensors.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy, mean


# ══════════════════════════════════════════════════════════════════
#  Critic Network  (§5.3.2 Architecture)
# ══════════════════════════════════════════════════════════════════

class CriticNetwork(nn.Module):
    """
    PPO Critic network (论文 §5.3.2 Critic 架构).

    Maps state → scalar value (state-value function V(s)).
    Architecture: 2-layer MLP (64 → 64 → 1).

    Input:  state_dim (default 19)
    Output: scalar value
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = CRITIC_HIDDEN,
    ):
        """
        Args:
            state_dim:  Dimension of the state vector.
            hidden_dim: Hidden layer size.
        """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.value.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute state value.

        Args:
            state: (batch, state_dim) tensor.

        Returns:
            (batch, 1) value tensor.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value


# ══════════════════════════════════════════════════════════════════
#  PPOMemory  — 经验回放缓冲区
# ══════════════════════════════════════════════════════════════════

@dataclass
class PPOMemory:
    """
    PPO经验回放缓冲区（论文 §5.3.2 经验池）.

    Stores trajectories from one or more episodes for batch update.
    Cleared after each policy update.
    """

    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    next_state: Optional[np.ndarray] = None

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        """
        Store a single transition.

        Args:
            state:   Current state vector.
            action:  Action vector.
            reward:  Reward scalar.
            done:    Whether episode terminated.
            log_prob: Log-probability of the action.
            value:   State value from Critic.
        """
        self.states.append(state.copy() if isinstance(state, np.ndarray) else state)
        self.actions.append(action.copy() if isinstance(action, np.ndarray) else action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        """Clear all stored transitions."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.next_state = None

    def __len__(self) -> int:
        return len(self.states)

    def to_tensors(self, device: torch.device = DEVICE) -> dict:
        """
        Convert stored data to PyTorch tensors on the given device.

        Returns:
            Dict with keys: states, actions, rewards, dones,
            log_probs, old_values.
        """
        return {
            "states": torch.FloatTensor(np.array(self.states)).to(device),
            "actions": torch.FloatTensor(np.array(self.actions)).to(device),
            "rewards": torch.FloatTensor(self.rewards).unsqueeze(-1).to(device),
            "dones": torch.FloatTensor(self.dones).unsqueeze(-1).to(device),
            "log_probs": torch.FloatTensor(self.log_probs).unsqueeze(-1).to(device),
            "old_values": torch.FloatTensor(self.values).unsqueeze(-1).to(device),
        }


# ══════════════════════════════════════════════════════════════════
#  PPOAgent  — 主智能体类
# ══════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    PPO协调器智能体（论文 §5.3.2 预测-优化协同）.

    The agent learns to output 6 weight scaling factors for the
    Stage 2 multi-objective penalty evaluation.  It receives a
    19-dim state (14 from prediction + 5 execution feedback) and
    produces a 6-dim action in [0.5, 1.5].

    Usage:
        agent = PPOAgent(state_dim=19, action_dim=6)
        memory = PPOMemory()

        # Inference
        action, log_prob, value = agent.select_action(state)
        memory.store(state, action, reward, done, log_prob, value)

        # Training
        loss = agent.learn(memory)
        memory.clear()
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        lr: float = LR,
        gamma: float = GAMMA,
        clip_epsilon: float = CLIP_EPSILON,
        ppo_epochs: int = PPO_EPOCHS,
        actor_hidden: int = ACTOR_HIDDEN,
        critic_hidden: int = CRITIC_HIDDEN,
        device: torch.device = DEVICE,
    ):
        """
        Args:
            state_dim:     State dimension (default: 19).
            action_dim:    Action dimension (default: 6).
            lr:            Learning rate (default: 3e-4).
            gamma:         Discount factor (default: 0.99).
            clip_epsilon:  PPO clipping range (default: 0.2).
            ppo_epochs:    Training epochs per update (default: 10).
            actor_hidden:  Actor hidden layer size (default: 64).
            critic_hidden: Critic hidden layer size (default: 64).
            device:        Torch device.
        """
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, actor_hidden).to(device)
        self.critic = CriticNetwork(state_dim, critic_hidden).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Loss tracking
        self.last_actor_loss: float = 0.0
        self.last_critic_loss: float = 0.0

        logger.info(
            "PPOAgent initialized: state_dim=%d action_dim=%d lr=%.0e γ=%.2f "
            "clip=%.1f epochs=%d device=%s",
            state_dim, action_dim, lr, gamma, clip_epsilon, ppo_epochs, device,
        )

    # ──────────────────────────────────────────────────────────────
    #  Inference
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select an action for the given state (inference mode).

        Args:
            state: 19-dim state vector as numpy array.

        Returns:
            (action, log_prob, value) tuple:
                action:   6-dim weight scaling factors in [0.5, 1.5].
                log_prob: Log-probability of the sampled action.
                value:    Critic's state value estimate.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action, log_prob, _ = self.actor.get_action_and_log_prob(state_t)
        value = self.critic(state_t)

        return (
            action.cpu().numpy().squeeze(),
            log_prob.cpu().item(),
            value.cpu().item(),
        )

    @torch.no_grad()
    def deterministic_action(self, state: np.ndarray) -> np.ndarray:
        """
        Deterministic action (mean of policy, no sampling).

        Use this for evaluation / deployment after training.

        Args:
            state: 19-dim state vector.

        Returns:
            6-dim action in [0.5, 1.5].
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, _ = self.actor.forward(state_t)
        return mean.cpu().numpy().squeeze()

    # ──────────────────────────────────────────────────────────────
    #  Training
    # ──────────────────────────────────────────────────────────────

    def learn(self, memory: PPOMemory, batch_size: int = 64) -> Dict[str, float]:
        """
        Perform PPO policy update using stored memory.

        Implements the clipped surrogate objective (§5.3.2):
            L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

        With GAE-lambda advantage estimation.

        Args:
            memory:    PPOMemory with experience from recent episodes.
            batch_size: Minibatch size (default: 64).

        Returns:
            Dict with 'actor_loss', 'critic_loss', 'entropy', 'kl_div'.
        """
        data = memory.to_tensors(self.device)

        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        old_log_probs = data["log_probs"]
        old_values = data["old_values"]

        # ── Compute returns and advantages ──
        returns, advantages = self._compute_gae(
            rewards, dones, old_values
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + EPS)

        n_samples = states.size(0)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        # ── PPO epochs ──
        for _epoch in range(self.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # ── Actor loss ──
                log_probs, entropy, _ = self.actor.evaluate(
                    batch_states, batch_actions
                )
                ratio = (log_probs - batch_old_log_probs).exp()

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    )
                    * batch_advantages
                )
                actor_loss = -torch.min(surr1, surr2).mean()

                # ── Critic loss ──
                values_pred = self.critic(batch_states)
                critic_loss = F.mse_loss(values_pred, batch_returns)

                # ── Total loss ──
                # Entropy bonus for exploration
                entropy_bonus = entropy.mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

                # ── Optimize ──
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=0.5
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=0.5
                )

                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # Track stats
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_bonus.item()

                # Approximate KL divergence
                with torch.no_grad():
                    kl = (log_probs - batch_old_log_probs).mean().item()
                    total_kl += kl

                n_updates += 1

        avg_actor_loss = total_actor_loss / max(n_updates, 1)
        avg_critic_loss = total_critic_loss / max(n_updates, 1)
        avg_entropy = total_entropy / max(n_updates, 1)
        avg_kl = total_kl / max(n_updates, 1)

        self.last_actor_loss = avg_actor_loss
        self.last_critic_loss = avg_critic_loss

        logger.debug(
            "PPO update: actor_loss=%.4f critic_loss=%.4f entropy=%.4f kl=%.4f",
            avg_actor_loss, avg_critic_loss, avg_entropy, avg_kl,
        )

        return {
            "actor_loss": avg_actor_loss,
            "critic_loss": avg_critic_loss,
            "entropy": avg_entropy,
            "kl_div": avg_kl,
        }

    # ──────────────────────────────────────────────────────────────
    #  GAE computation
    # ──────────────────────────────────────────────────────────────

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: (N, 1) reward tensor.
            dones:   (N, 1) done flag tensor.
            values:  (N, 1) value tensor.
            lam:     GAE lambda parameter (default: 0.95).

        Returns:
            (returns, advantages) tuple, each (N, 1).
        """
        n = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        gae = 0.0

        # GAE: iterate backwards
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1] * (1.0 - dones[t])

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    # ──────────────────────────────────────────────────────────────
    #  Save / Load
    # ──────────────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Save actor and critic state dicts to a checkpoint file.

        Args:
            path: Path to save the .pt checkpoint.
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": {
                "gamma": self.gamma,
                "clip_epsilon": self.clip_epsilon,
                "ppo_epochs": self.ppo_epochs,
            },
        }
        torch.save(checkpoint, path)
        logger.info("PPOAgent saved to %s", path)

    def load(self, path: str):
        """
        Load actor and critic state dicts from a checkpoint file.

        Args:
            path: Path to the .pt checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        logger.info("PPOAgent loaded from %s", path)

    def train_mode(self):
        """Set networks to training mode."""
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        """Set networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
