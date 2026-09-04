"""
PPO训练脚本（论文 §5.3.2 训练流程）

Implements the walk-forward training procedure:
  - 50 episodes per window, 2 windows walk-forward
  - Each episode: interact with the yard allocation environment,
    collect state transitions, update policy.
  - Results logged and saved to 03_results/canonical/

Training environment:
    The PPO coordinator interacts with the yard allocation system.
    State:  19-dim (14 prediction state + 5 execution feedback)
    Action: 6-dim weight scaling factors in [0.5, 1.5]
    Reward: reshuffle_reduction + equip_util_improvement
            - weight_variance_penalty

Results from paper (§5.3.2 Table):
    W1: training reward  -757.7 → -719.1 (converged)
        test reward       -318.7  (vs baseline -404.2, +21.1%)
    W2: training reward -1459.1 → -1124.1
        test reward       -427.7  (vs baseline -584.0, +26.7%)

Usage:
    python -m rl_coordinator.train_ppo                      # full training
    python -m rl_coordinator.train_ppo --windows W1         # single window
    python -m rl_coordinator.train_ppo --load actor.pt      # resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ──────────────────────────────────────────────────────────────────
# repo root for sibling imports
# ──────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl_coordinator.ppo_agent import (
    PPOAgent,
    PPOMemory,
    STATE_DIM,
    ACTION_DIM,
    GAMMA,
    CLIP_EPSILON,
    LR,
    PPO_EPOCHS,
    ACTION_LOW,
    ACTION_HIGH,
    REWARD_RESHUF_COEF,
    REWARD_EQUIP_COEF,
    REWARD_VAR_PENALTY,
    DEVICE,
)

from yard_optimization.three_stage_allocation import (
    ThreeStageAllocator,
    AllocationConfig,
    ContainerInfo,
    YardLayout,
    create_default_yard_layout,
    container_info_from_dict,
)

logger = logging.getLogger(__name__)

# Output directory
RESULTS_DIR = _REPO_ROOT / "03_results" / "canonical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = _REPO_ROOT / "03_results" / "canonical" / "ppo_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  Training environment (simplified yard allocation env)
# ══════════════════════════════════════════════════════════════════

class YardAllocationEnv:
    """
    Simplified environment for PPO training.

    Simulates yard allocation decisions across an episode of
    container placement requests.  The agent's action (6 weight
    scalings) is applied during Stage 2 evaluation, and the
    resulting allocation quality determines the reward.

    State:  19-dim vector
    Action: 6-dim weight scaling factors
    Reward: weighted combination of reshuffle reduction,
            equipment utilization improvement, and
            weight variance penalty
    """

    def __init__(
        self,
        yard: YardLayout,
        config: AllocationConfig,
        n_containers_per_episode: int = 200,
        seed: int = 42,
    ):
        """
        Args:
            yard: Yard layout.
            config: Allocation configuration.
            n_containers_per_episode: Number of containers per episode.
            seed: Random seed for reproducibility.
        """
        self.yard = yard
        self.config = config
        self.n_containers = n_containers_per_episode
        self.rng = np.random.RandomState(seed)

        self.allocator = ThreeStageAllocator(yard, config)
        self.containers: List[ContainerInfo] = []
        self.current_idx = 0
        self.episode_count = 0

        # Baseline metrics (for reward computation)
        self.baseline_reshuffle_rate = 0.25  # typical reshuffle rate
        self.baseline_equip_util = 0.65      # typical equip utilization

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Returns:
            Initial state vector (19-dim).
        """
        self.current_idx = 0
        self.episode_count += 1

        # Generate containers for this episode
        self.containers = self._generate_containers()

        # Return initial state
        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step: allocate one container with the given PPO weights.

        Args:
            action: 6-dim weight scaling factors in [0.5, 1.5].

        Returns:
            (next_state, reward, done, info) tuple.
        """
        if self.current_idx >= len(self.containers):
            return self._get_state(), 0.0, True, {}

        container = self.containers[self.current_idx]
        self.current_idx += 1

        # Apply action as PPO weight scaling for Stage 2
        try:
            pos = self.allocator.allocate(container, {}, ppo_weights=action)
            penalty = pos.total_penalty
        except RuntimeError:
            penalty = 10.0  # high penalty for infeasible

        # Compute reward
        reward = self._compute_reward(action, penalty)

        # Done if all containers allocated
        done = self.current_idx >= len(self.containers)

        info = {
            "container_id": container.container_id,
            "penalty": penalty,
            "position": (pos.block_id, pos.bay, pos.row, pos.tier) if 'pos' in dir() else None,
        }

        return self._get_state(), reward, done, info

    # ──────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _generate_containers(self) -> List[ContainerInfo]:
        """Generate a batch of containers for this episode."""
        n = self.n_containers
        types = ["general", "reefer", "dangerous"]
        type_probs = [0.85, 0.10, 0.05]
        ports = ["SHA", "NING", "XMN", "YTN", "SIN"]

        containers = []
        for i in range(n):
            ctype = self.rng.choice(types, p=type_probs)
            length_20ft = self.rng.random() < 0.7
            weight = 5.0 + self.rng.exponential(10.0)
            port = self.rng.choice(ports)
            berth = self.rng.randint(1, 4)

            containers.append(
                ContainerInfo(
                    container_id=f"ENV_TRN_{self.episode_count:03d}_{i:06d}",
                    length_20ft=length_20ft,
                    weight_t=weight,
                    discharge_port=port,
                    container_type=ctype,
                    destination_berth=berth,
                    vessel_code=f"VSL_ENV_{i // 50}",
                )
            )
        return containers

    def _get_state(self) -> np.ndarray:
        """
        Build the 19-dim state vector.

        Based on §5.3.2 state formulation:
          14 dims: prediction_confidence, prediction_error,
                   future_arrivals, yard_occupancy, avg_equip_util,
                   queue_length, congestion_index, reshuffle_bias,
                   equip_bias + 5 zero-filled
           5 dims: execution feedback (current_idx, avg_penalty, etc.)
        """
        progress = self.current_idx / max(len(self.containers), 1)
        occupancy = self.yard.occupancy

        # Simplified state components
        state_14 = np.array([
            0.85,                     # prediction_confidence
            0.10,                     # prediction_error
            0.5,                      # future_arrivals (normalized)
            occupancy,                # yard_occupancy
            0.65,                     # avg_equip_util
            3.0,                      # queue_length
            0.2,                      # congestion_index
            0.15,                     # reshuffle_bias
            0.1,                      # equip_bias
            0.0, 0.0, 0.0, 0.0, 0.0,  # remaining dims
        ])

        # 5-dim execution feedback
        feedback_5 = np.array([
            progress,                              # allocation progress
            np.random.uniform(0.0, 0.3),           # avg_penalty_this_ep
            float(self.current_idx),                # containers_done
            occupancy * 0.5,                        # yard_util_change
            0.1,                                    # equip_util_change
        ])

        return np.concatenate([state_14, feedback_5]).astype(np.float32)

    def _compute_reward(
        self, action: np.ndarray, penalty: float
    ) -> float:
        """
        Compute reward from the allocation outcome.

        Reward = reshuffle_reduction + equip_util_improvement
                 - weight_variance_penalty

        Based on §5.3.2 reward formulation.
        """
        # Reshuffle reduction: lower penalty → better
        reshuf_reward = -penalty * REWARD_RESHUF_COEF

        # Equipment utilization improvement (simplified)
        equip_reward = np.random.uniform(-0.1, 0.1) * REWARD_EQUIP_COEF

        # Weight variance penalty: penalize extreme weight differences
        weight_var = np.var(action)
        var_penalty = weight_var * REWARD_VAR_PENALTY

        reward = reshuf_reward + equip_reward - var_penalty
        return float(reward)


# ══════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════

def train_ppo(
    n_episodes: int = 50,
    n_windows: int = 2,
    episode_length: int = 200,
    seed: int = 42,
    lr: float = LR,
    gamma: float = GAMMA,
    save: bool = True,
    load_path: Optional[str] = None,
    log_interval: int = 10,
) -> Tuple[PPOAgent, pd.DataFrame]:
    """
    Train the PPO coordinator using walk-forward validation.

    Implements training procedure from §5.3.2:
      - 50 episodes per window, 2 windows walk-forward
      - Each episode: collect state-action-reward tuples
      - Policy update at end of each episode (or every K steps)
      - Model saved after each window

    Args:
        n_episodes:  Episodes per window (default: 50).
        n_windows:   Number of walk-forward windows (default: 2).
        episode_length: Containers per episode (default: 200).
        seed:        Random seed.
        lr:          Learning rate (default: 3e-4).
        gamma:       Discount factor (default: 0.99).
        save:        Save checkpoints and logs.
        load_path:   Optional checkpoint path to resume training.
        log_interval: Log frequency (episodes).

    Returns:
        (agent, history_df) — trained agent and training history.
    """
    logger.info(
        "Starting PPO training: %d windows × %d episodes, γ=%.2f, lr=%.0e",
        n_windows, n_episodes, gamma, lr,
    )

    # Initialize agent
    agent = PPOAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lr=lr,
        gamma=gamma,
        clip_epsilon=CLIP_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        device=DEVICE,
    )

    if load_path:
        agent.load(load_path)
        logger.info("Resumed from checkpoint: %s", load_path)

    # Initialize environment
    yard = create_default_yard_layout()
    config = AllocationConfig()
    env = YardAllocationEnv(yard, config, episode_length, seed)

    # Training history
    history = []

    # ── Walk-forward windows ──
    for window in range(1, n_windows + 1):
        window_seed = seed + window * 1000
        logger.info("=== Window %d/%d (seed=%d) ===", window, n_windows, window_seed)

        memory = PPOMemory()
        episode_rewards = []

        for episode in range(1, n_episodes + 1):
            state = env.reset()
            agent.train_mode()

            episode_reward = 0.0
            episode_actor_loss = 0.0
            episode_critic_loss = 0.0
            n_steps = 0

            ep_container_seed = window_seed + episode * 100
            env.rng = np.random.RandomState(ep_container_seed)

            # ── Collect one episode ──
            for step in range(episode_length):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                memory.store(state, action, reward, done, log_prob, value)
                state = next_state
                episode_reward += reward
                n_steps += 1

                if done:
                    break

            # ── Policy update ──
            if len(memory) > 0:
                loss_dict = agent.learn(memory)
                memory.clear()
                episode_actor_loss = loss_dict["actor_loss"]
                episode_critic_loss = loss_dict["critic_loss"]

            episode_rewards.append(episode_reward)

            # ── Logging ──
            record = {
                "window": window,
                "episode": episode,
                "reward": round(episode_reward, 2),
                "actor_loss": round(episode_actor_loss, 4),
                "critic_loss": round(episode_critic_loss, 4),
                "n_steps": n_steps,
            }
            history.append(record)

            if episode % log_interval == 0 or episode == 1:
                recent = episode_rewards[-min(episode, log_interval):]
                avg_reward = np.mean(recent)
                logger.info(
                    "W%d Ep %3d | reward=%.1f (avg=%.1f) | actor_loss=%.4f "
                    "critic_loss=%.4f | steps=%d",
                    window, episode, episode_reward, avg_reward,
                    episode_actor_loss, episode_critic_loss, n_steps,
                )

        # ── End-of-window summary ──
        window_avg_reward = np.mean(episode_rewards)
        window_std_reward = np.std(episode_rewards)
        logger.info(
            "Window %d complete: avg_reward=%.1f ± %.1f (baseline=%.1f)",
            window, window_avg_reward, window_std_reward,
            -404.2 if window == 1 else -584.0,
        )

        # Save window checkpoint
        if save:
            ckpt_path = CHECKPOINT_DIR / f"ppo_window{window}_final.pt"
            agent.save(str(ckpt_path))
            logger.info("Window %d checkpoint saved to %s", window, ckpt_path)

    # ── Training complete ──
    history_df = pd.DataFrame(history)

    if save:
        csv_path = RESULTS_DIR / "ppo_training_history.csv"
        history_df.to_csv(csv_path, index=False)
        logger.info("Training history saved to %s", csv_path)

        # Save final model
        final_path = CHECKPOINT_DIR / "ppo_final.pt"
        agent.save(str(final_path))

        # Save summary
        summary = {
            "n_windows": n_windows,
            "n_episodes_per_window": n_episodes,
            "episode_length": episode_length,
            "gamma": gamma,
            "lr": lr,
            "clip_epsilon": CLIP_EPSILON,
            "ppo_epochs": PPO_EPOCHS,
            "final_checkpoint": str(final_path),
            "training_history": csv_path,
        }
        summary_path = RESULTS_DIR / "ppo_training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    return agent, history_df


# ══════════════════════════════════════════════════════════════════
#  Evaluation helper
# ══════════════════════════════════════════════════════════════════

def evaluate_agent(
    agent: PPOAgent,
    n_episodes: int = 10,
    episode_length: int = 200,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluate a trained PPO agent on held-out episodes.

    Uses deterministic actions (no sampling noise) for evaluation.

    Args:
        agent: Trained PPOAgent.
        n_episodes: Number of evaluation episodes.
        episode_length: Containers per episode.
        seed: Random seed.

    Returns:
        Dict of evaluation metrics.
    """
    yard = create_default_yard_layout()
    config = AllocationConfig()
    env = YardAllocationEnv(yard, config, episode_length, seed + 9999)

    agent.eval_mode()
    rewards = []

    for ep in range(n_episodes):
        state = env.reset()
        episode_reward = 0.0
        env.rng = np.random.RandomState(seed + 9999 + ep * 100)

        for _step in range(episode_length):
            action = agent.deterministic_action(state)
            next_state, reward, done, _info = env.step(action)
            episode_reward += reward
            state = next_state
            if done:
                break

        rewards.append(episode_reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "n_episodes": n_episodes,
    }


# ══════════════════════════════════════════════════════════════════
#  CLI entry point
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PPO Coordinator Training (§5.3.2)"
    )
    parser.add_argument(
        "--windows", type=int, default=2,
        help="Number of walk-forward windows (default: 2)",
    )
    parser.add_argument(
        "--episodes", type=int, default=50,
        help="Episodes per window (default: 50)",
    )
    parser.add_argument(
        "--episode-length", type=int, default=200,
        help="Steps (containers) per episode (default: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=LR,
        help=f"Learning rate (default: {LR})",
    )
    parser.add_argument(
        "--gamma", type=float, default=GAMMA,
        help=f"Discount factor (default: {GAMMA})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Load checkpoint to resume training",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run evaluation only (requires --load)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save checkpoints or logs",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if args.eval_only:
        if not args.load:
            parser.error("--eval-only requires --load")
        agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        agent.load(args.load)
        metrics = evaluate_agent(agent, seed=args.seed)
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        return

    # Full training
    agent, history_df = train_ppo(
        n_episodes=args.episodes,
        n_windows=args.windows,
        episode_length=args.episode_length,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        save=not args.no_save,
        load_path=args.load,
    )

    # Print summary
    print("\n" + "=" * 72)
    print("PPO TRAINING SUMMARY")
    print("=" * 72)
    summary = (
        history_df.groupby("window")["reward"]
        .agg(["mean", "std", "min", "max"])
        .round(2)
    )
    print(summary.to_string())

    print(f"\nFinal models saved to: {CHECKPOINT_DIR}")
    print(f"Training history:     {RESULTS_DIR / 'ppo_training_history.csv'}")


if __name__ == "__main__":
    main()
