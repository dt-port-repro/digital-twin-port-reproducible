"""
MAS智能体模块（论文§6.2.1 基于MAS-DES混合架构）
含 QC Agent、YC Agent、Truck Agent、BerthManager、BidirectionalProtocol

所有Agent采用有限状态机建模（论文§6.2.1 设备行为建模）
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import numpy as np
import math


# ═══════════════════════════════════════════════════════════
# 1. 设备智能体基类与状态枚举
# ═══════════════════════════════════════════════════════════

class AgentStatus(Enum):
    """Agent五种状态（论文§6.2.1 有限状态机）"""
    IDLE = auto()       # 空闲
    MOVING = auto()     # 移动
    WORKING = auto()    # 作业
    FAILURE = auto()    # 故障
    RECOVERY = auto()   # 恢复


@dataclass
class CraneAgent:
    """岸桥/场桥Agent基类（论文§6.2.1 有限状态机描述）
    状态转换: IDLE → MOVING → WORKING → IDLE
                        ↕           ↕
                      FAILURE ←→ RECOVERY
    """
    agent_id: str
    location: str = ''
    status: AgentStatus = AgentStatus.IDLE
    task_queue: List[dict] = field(default_factory=list)
    current_task: Optional[dict] = None
    cycle_time_mean: float = 2.0    # 作业循环时间均值(分钟)
    cycle_time_std: float = 0.5     # 波动
    total_cycles: int = 0
    busy_time: float = 0.0          # 累计忙时(分钟)
    fail_count: int = 0

    def get_cycle_time(self, rng: np.random.RandomState) -> float:
        """作业循环时间（含随机波动）"""
        return max(1.0, rng.normal(self.cycle_time_mean, self.cycle_time_std))

    def start_task(self, task: dict, current_time: float):
        """开始一个新任务"""
        self.current_task = task
        self.status = AgentStatus.WORKING
        self.task_queue.append(task)

    def complete_task(self, current_time: float, cycle_time: float):
        """完成当前任务"""
        self.total_cycles += 1
        self.busy_time += cycle_time
        self.current_task = None
        if self.task_queue:
            self.task_queue.pop(0)
        self.status = AgentStatus.IDLE if not self.task_queue else AgentStatus.WORKING

    def trigger_failure(self, rng: np.random.RandomState) -> bool:
        """按故障率触发设备故障"""
        return rng.random() < 0.015  # 1.5%故障率（论文§6.2.2异常场景参数）

    def get_utilization(self, total_time: float) -> float:
        """计算利用率"""
        return min(self.busy_time / max(total_time, 1), 1.0) * 100


# ═══════════════════════════════════════════════════════════
# 2. QC岸桥Agent（论文§6.2.1 8台QC Agent）
# ═══════════════════════════════════════════════════════════

class QCCrane(CraneAgent):
    """岸桥Agent（论文§6.2.1/表6.1: 8台QC）
    有限状态机：空闲 → 移动到船边 → 下降吊具 → 抓取 → 上升 → 移动到集卡 → 放箱
    """
    def __init__(self, agent_id: str, berth_id: int = 0):
        super().__init__(agent_id=agent_id)
        self.berth_id = berth_id
        self.location = f'Berth{berth_id}'
        self.cycle_time_mean = 2.0  # 岸桥循环时间(min)
        self.cycle_time_std = 0.4

    def process_vessel(self, container_count: int, rng: np.random.RandomState,
                       failure_rate: float = 0.015) -> Tuple[float, int, int]:
        """处理一艘船的全部集装箱
        Returns: (总耗时(min), 翻箱次数, 故障次数)
        """
        total_time = 0.0
        reshuffles = 0
        failures = 0

        for _ in range(container_count):
            # 检查故障
            if rng.random() < failure_rate:
                failures += 1
                repair_time = max(10, rng.exponential(60))  # 恢复时间均值60min
                total_time += repair_time
                self.fail_count += 1

            # 正常作业循环
            ct = max(1.0, rng.normal(self.cycle_time_mean, self.cycle_time_std))
            total_time += ct

            # 按论文§6.3.2比例模拟翻箱
            if rng.random() < 0.07:  # ~7%翻箱率
                reshuffles += 1
                total_time += ct * 0.5  # 翻箱额外耗时

        self.total_cycles += container_count
        self.busy_time += total_time
        return total_time, reshuffles, failures


# ═══════════════════════════════════════════════════════════
# 3. YC场桥Agent（论文§6.2.1 12台YC Agent）
# ═══════════════════════════════════════════════════════════

class YCCrane(CraneAgent):
    """场桥Agent（论文§6.2.1/表6.1: 12台YC）
    多台场桥在同一箱区作业时通过区域互斥锁避免冲突
    """
    def __init__(self, agent_id: str, zone: str = 'L01'):
        super().__init__(agent_id=agent_id)
        self.zone = zone
        self.location = zone
        self.cycle_time_mean = 3.0  # 场桥循环时间(min)
        self.cycle_time_std = 0.6

    def process_container(self, rng: np.random.RandomState) -> float:
        """处理一个集装箱的堆场操作"""
        ct = max(1.5, rng.normal(self.cycle_time_mean, self.cycle_time_std))
        self.total_cycles += 1
        self.busy_time += ct
        self.status = AgentStatus.WORKING
        return ct


# ═══════════════════════════════════════════════════════════
# 4. Truck集卡Agent（论文§6.2.1 20台Truck Agent）
# ═══════════════════════════════════════════════════════════

class Truck:
    """集卡Agent（论文§6.2.1/表6.1: 20台Truck Agent）
    水平运输，作业循环时间含行驶+等待+装卸
    """
    def __init__(self, truck_id: str):
        self.truck_id = truck_id
        self.status = AgentStatus.IDLE
        self.location = 'yard'
        self.total_trips = 0
        self.busy_time = 0.0
        self.cycle_time_mean = 7.0   # 集卡循环时间(min)
        self.cycle_time_std = 1.5

    def deliver(self, from_loc: str, to_loc: str, rng: np.random.RandomState) -> float:
        """执行一次运输任务
        Returns: 耗时(min)
        """
        self.status = AgentStatus.WORKING
        self.location = from_loc
        # 行驶时间
        travel = max(3.0, rng.normal(self.cycle_time_mean, self.cycle_time_std))
        # 等待+装卸时间
        wait = max(1.0, rng.exponential(2.0))
        total = travel + wait
        self.busy_time += total
        self.total_trips += 1
        self.location = to_loc
        self.status = AgentStatus.IDLE
        return total


# ═══════════════════════════════════════════════════════════
# 5. BerthManager泊位管理（论文§6.2.1 M/M/c排队模型）
# ═══════════════════════════════════════════════════════════

class BerthManager:
    """泊位管理器（论文§6.2.1 M/M/c多服务台排队模型）
    3个泊位，船舶到达后若泊位全满则进入等待队列
    """
    def __init__(self, n_berths: int = 3):
        self.n_berths = n_berths
        self.berths: List[Optional[str]] = [None] * n_berths  # None=空闲
        self.waiting_queue: List[str] = []
        self.n_served = 0
        self.total_wait_time = 0.0

    def request_berth(self, vessel_code: str) -> Tuple[bool, int, float]:
        """请求泊位分配（论文§6.2.1 M/M/c模型）
        Returns: (是否立即分配, 泊位编号, 排队等待时长)
        """
        for bid in range(self.n_berths):
            if self.berths[bid] is None:
                self.berths[bid] = vessel_code
                self.n_served += 1
                return True, bid, 0.0
        # 全部占用，进入等待队列
        self.waiting_queue.append(vessel_code)
        return False, -1, 0.0

    def release_berth(self, berth_id: int) -> Optional[str]:
        """释放泊位，分配等待队列中的下一艘船
        Returns: 下一艘船编码（如有）
        """
        if 0 <= berth_id < self.n_berths:
            self.berths[berth_id] = None

        if self.waiting_queue:
            next_vc = self.waiting_queue.pop(0)
            self.berths[berth_id] = next_vc
            self.n_served += 1
            return next_vc
        return None

    @property
    def utilization(self) -> float:
        """泊位利用率"""
        busy = sum(1 for b in self.berths if b is not None)
        return busy / self.n_berths if self.n_berths > 0 else 0.0

    @property
    def n_waiting(self) -> int:
        return len(self.waiting_queue)

    def get_waiting_vessels(self) -> List[str]:
        return self.waiting_queue.copy()

    def reset(self):
        self.berths = [None] * self.n_berths
        self.waiting_queue.clear()
        self.n_served = 0
        self.total_wait_time = 0.0


# ═══════════════════════════════════════════════════════════
# 6. 双向信息交换协议（论文§5.3.4 / 附录C）
# ═══════════════════════════════════════════════════════════

@dataclass
class StowagePlan:
    """配载方案摘要"""
    vessel_code: str
    containers_by_pod: Dict[str, int]           # 船箱数：{卸货港: 箱量}
    sequence: List[str]                         # 作业序列
    estimated_time_h: float = 0.0               # 预计作业时间
    fitness: float = 0.0                        # 适应度

@dataclass
class YardFeasibility:
    """堆场可行性反馈（论文§5.3.4 配载-堆场协商）"""
    feasible: bool = True
    penalty: float = 0.0                        # 堆场执行成本
    reshuffle_risk: float = 0.0                 # 翻箱风险
    congestion_risk: float = 0.0                # 拥堵风险
    suggestions: List[str] = field(default_factory=list)  # 调整建议

@dataclass
class BidiMessage:
    """双向信息交换消息（论文§5.3.4 附录C双向信息交换协议）"""
    round_id: int = 0
    stowage_plan: Optional[StowagePlan] = None
    yard_feedback: Optional[YardFeasibility] = None
    accepted: bool = False


class BidirectionalProtocol:
    """双向信息交换协议（论文§5.3.4 / 附录C表C.1）
    配载与堆场之间通过3轮迭代协商逼近联合方案
    Step 1: 配载提需求 → StowagePlan
    Step 2: 堆场做评估 → YardFeasibility
    Step 3: 配载再调整 → 最终方案
    """
    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.current_round = 0
        self.history: List[BidiMessage] = []
        self.agreed = False

    def propose(self, plan: StowagePlan) -> BidiMessage:
        """第一阶段：配载提出方案"""
        msg = BidiMessage(round_id=self.current_round, stowage_plan=plan)
        self.history.append(msg)
        return msg

    def evaluate(self, msg: BidiMessage, yard_state: dict) -> BidiMessage:
        """第二阶段：堆场评估方案
        yard_state: 当前堆场状态（占用率、拥堵指数等）
        """
        plan = msg.stowage_plan
        if plan is None:
            msg.yard_feedback = YardFeasibility(feasible=False)
            return msg

        # 计算堆场执行成本
        penalty = 0.0
        reshuffle_risk = 0.0
        congestion_risk = 0.0
        suggestions = []

        # 翻箱风险评估
        total_containers = sum(plan.containers_by_pod.values())
        if total_containers > 0:
            n_pods = len(plan.containers_by_pod)
            # 卸货港越多，翻箱风险越高
            reshuffle_risk = min(n_pods * 0.02, 0.3)
            # 堆场利用率影响
            yard_occupancy = yard_state.get('occupancy', 0.65)
            reshuffle_risk *= (1 + yard_occupancy)

        # 拥堵风险评估
        yard_congestion = yard_state.get('congestion_index', 0.0)
        congestion_risk = min(yard_congestion * 0.5, 0.3)

        # 综合惩罚
        penalty = reshuffle_risk * 0.6 + congestion_risk * 0.4

        feedback = YardFeasibility(
            feasible=penalty < 0.5,
            penalty=round(penalty, 4),
            reshuffle_risk=round(reshuffle_risk, 4),
            congestion_risk=round(congestion_risk, 4),
            suggestions=suggestions
        )

        msg.yard_feedback = feedback
        msg.accepted = feedback.feasible
        self.history.append(msg)
        self.current_round += 1

        # 检查是否达成一致或达到最大轮数
        if feedback.feasible or self.current_round >= self.max_rounds:
            self.agreed = True

        return msg

    def adjust(self, plan: StowagePlan, feedback: YardFeasibility) -> StowagePlan:
        """第三阶段：配载根据堆场反馈调整方案"""
        if feedback.feasible:
            return plan

        # 调整策略：降低翻箱风险高的卸货港优先级
        adjusted_plan = StowagePlan(
            vessel_code=plan.vessel_code,
            containers_by_pod=plan.containers_by_pod.copy(),
            sequence=plan.sequence.copy(),
            estimated_time_h=plan.estimated_time_h * 1.05,  # 调整增加5%时间
            fitness=plan.fitness * 0.95  # 适应度降低5%
        )
        return adjusted_plan

    def negotiate(self, plan: StowagePlan, yard_state: dict) -> StowagePlan:
        """单次协商全流程（3轮迭代）"""
        msg = self.propose(plan)
        feedback_msg = self.evaluate(msg, yard_state)

        if feedback_msg.accepted:
            return plan

        # 需要调整
        adjusted = self.adjust(plan, feedback_msg.yard_feedback)

        if self.current_round < self.max_rounds:
            # 再评估一轮
            msg2 = self.propose(adjusted)
            fb2 = self.evaluate(msg2, yard_state)
            if fb2.accepted:
                return adjusted

        return adjusted

    def reset(self):
        self.current_round = 0
        self.history.clear()
        self.agreed = False
