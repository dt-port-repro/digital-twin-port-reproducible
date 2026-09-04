"""
数据模型（论文§6.2 仿真平台核心数据结构）
支持 des_engine.py、run_scenarios.py、verify_replication.py
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import numpy as np


# ═══════════════════════════════════════════════════════════
# 事件类型枚举（论文§6.2.1 MAS-DES事件驱动）
# ═══════════════════════════════════════════════════════════

class EventType(Enum):
    VESSEL_ARRIVAL = 'VESSEL_ARRIVAL'
    BERTH_ALLOCATED = 'BERTH_ALLOCATED'
    BERTH_RELEASED = 'BERTH_RELEASED'
    STOWAGE_START = 'STOWAGE_START'
    STOWAGE_DONE = 'STOWAGE_DONE'
    QC_CYCLE_DONE = 'QC_CYCLE_DONE'
    QC_ALL_DONE = 'QC_ALL_DONE'
    YC_CYCLE_DONE = 'YC_CYCLE_DONE'
    YC_ALL_DONE = 'YC_ALL_DONE'
    TRUCK_DELIVERY_DONE = 'TRUCK_DELIVERY_DONE'
    VESSEL_DEPART = 'VESSEL_DEPART'
    EQUIP_FAILURE = 'EQUIP_FAILURE'
    EQUIP_REPAIR = 'EQUIP_REPAIR'
    BIDI_NEGOTIATE = 'BIDI_NEGOTIATE'


# ═══════════════════════════════════════════════════════════
# 仿真事件（论文§6.2.1 事件调度机制）
# ═══════════════════════════════════════════════════════════

@dataclass(order=True)
class SimEvent:
    time: float
    type: str
    data: dict = field(default_factory=dict, compare=False)
    priority: int = field(default=0, compare=False)


# ═══════════════════════════════════════════════════════════
# 港口状态（论文§6.2 仿真环境状态跟踪）
# ═══════════════════════════════════════════════════════════

@dataclass
class PortState:
    """港口实时状态"""
    berth_occupancy: float = 0.0
    yard_utilization: float = 0.65
    equip_utilization: float = 0.0
    queue_len: int = 0
    berth_waiting_count: int = 0
    qc_count: int = 8
    yc_count: int = 12
    truck_count: int = 20
    qc_busy: int = 0
    yc_busy: int = 0
    vessels_at_berth: List[str] = field(default_factory=list)
    berth_queue: List[str] = field(default_factory=list)
    total_vessels_served: int = 0
    total_containers_processed: int = 0
    equip_available: float = 1.0  # 设备可用率


# ═══════════════════════════════════════════════════════════
# 仿真配置（论文§6.2.2 三场景配置参数）
# ═══════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """仿真配置参数"""
    scenario: str = '常规作业'
    config: str = 'A'
    n_days: int = 30
    n_runs: int = 10
    seeds: List[int] = field(default_factory=lambda: list(range(100, 110)))
    ships_per_day: float = 3.1
    yard_util_init: float = 0.65
    equip_avail: float = 1.0
    qc_count: int = 8
    yc_count: int = 12
    truck_count: int = 20
    n_berths: int = 3
    container_per_ship_mean: float = 3500
    container_per_ship_std: float = 1500
    # 设备参数（论文§6.2.1）
    qc_cycle_min: float = 1.5
    qc_cycle_max: float = 3.0
    yc_cycle_min: float = 2.0
    yc_cycle_max: float = 4.0
    truck_cycle_min: float = 5.0
    truck_cycle_max: float = 10.0
    # 故障参数
    failure_rate: float = 0.015  # 设备故障率
    repair_time_mean: float = 1.0  # 维修时间均值(小时)
    # 配载缓存
    use_stowage_cache: bool = True
    # 双向协议参数
    bidi_negotiate: bool = False  # 配置C/D启用
    bidi_rounds: int = 3


# ═══════════════════════════════════════════════════════════
# 仿真结果（论文§6.3 实验输出格式）
# ═══════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """单次仿真运行结果"""
    scenario: str
    seed: int
    n_days: int
    config: str = 'A'
    n_vessels: int = 0
    total_containers: int = 0
    turnaround_h: float = 0.0
    reshuffle_pct: float = 0.0
    equip_util_pct: float = 0.0
    turnaround_improvement_pct: float = 0.0
    reshuffle_improvement_pct: float = 0.0
    equip_util_improvement_pct: float = 0.0
    vessel_log: List[dict] = field(default_factory=list)
    time_series: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# 第五章数据模型
# ═══════════════════════════════════════════════════════════

@dataclass
class YardState:
    """堆场状态（论文§5.2.1 问题形式化）"""
    occupancy: float = 0.0
    reshuffle_rate: float = 0.0
    avg_penalty: float = 0.0
    n_containers: int = 0
    n_bays: int = 0
    utilization_by_zone: Dict[str, float] = field(default_factory=dict)

@dataclass
class PredictionOutput:
    """预测模型输出（论文§5.1.2）"""
    mean_arrivals: float = 0.0
    std_arrivals: float = 0.0
    confidence: float = 1.0
    type_distribution: Dict[str, float] = field(default_factory=dict)

@dataclass
class PPOState:
    """PPO协调器状态（论文§5.3.2 MDP五元组）"""
    prediction_confidence: float = 0.0
    prediction_error: float = 0.0
    future_arrivals: float = 0.0
    yard_occupancy: float = 0.0
    avg_equip_util: float = 0.0
    queue_length: int = 0
    congestion_index: float = 0.0
    reshuffle_bias: float = 0.0
    equip_bias: float = 0.0

    def to_array(self) -> np.ndarray:
        """转为14维状态向量"""
        return np.array([
            self.prediction_confidence, self.prediction_error, self.future_arrivals,
            self.yard_occupancy, self.avg_equip_util, float(self.queue_length),
            self.congestion_index, self.reshuffle_bias, self.equip_bias,
            0.0, 0.0, 0.0, 0.0, 0.0
        ])
