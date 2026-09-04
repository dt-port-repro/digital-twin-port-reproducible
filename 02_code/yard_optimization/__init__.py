"""
yard_optimization — 堆场选位优化模块（论文第五章 §5.2）
包含三阶段堆场分配算法核心实现。

Modules:
    three_stage_allocation  — 三阶段堆场选位分配算法（§5.2.4）
    run_experiments         — 实验运行器与对比基线（§5.4）

Exports:
    ThreeStageAllocator     — 三阶段分配器主类
    run_experiments         — 实验入口函数
"""

from .three_stage_allocation import ThreeStageAllocator
from .run_experiments import run_experiments

__all__ = [
    "ThreeStageAllocator",
    "run_experiments",
]
