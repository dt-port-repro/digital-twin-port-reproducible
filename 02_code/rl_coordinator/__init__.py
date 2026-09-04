"""
rl_coordinator — 预测-优化协同强化学习模块（论文第五章 §5.3）
实现PPO协调器，为堆场选位优化提供动态权重调整。

Modules:
    ppo_agent   — PPO智能体 (Actor-Critic架构, §5.3.2)
    train_ppo   — PPO训练脚本 (§5.3.2 训练流程)

Exports:
    PPOAgent        — PPO智能体主类
    PPOMemory       — 经验回放缓冲区
    ActorNetwork    — Actor网络 (64→64→6)
    CriticNetwork   — Critic网络 (64→64→1)
    train_ppo       — 训练入口函数
"""

from .ppo_agent import PPOAgent, PPOMemory, ActorNetwork, CriticNetwork
from .train_ppo import train_ppo

__all__ = [
    "PPOAgent",
    "PPOMemory",
    "ActorNetwork",
    "CriticNetwork",
    "train_ppo",
]
