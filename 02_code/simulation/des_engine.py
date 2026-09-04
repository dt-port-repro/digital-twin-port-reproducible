"""
DES仿真引擎 — 事件调度+时间推进
论文§6.2 高保真数字孪生仿真环境
"""
import heapq, time, numpy as np, sys
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
_SIM_DIR = Path(__file__).resolve().parent
_CODE_DIR = _SIM_DIR.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from data.models import *
from modules.agents import QCCrane, YCCrane, Truck, BerthManager, BidirectionalProtocol

class SimClock:
    """仿真时钟"""
    def __init__(self):
        self.current_time = 0.0
        self.end_time = 0.0
    
    def reset(self, end_hours: float):
        self.current_time = 0.0
        self.end_time = end_hours

class MetricsCollector:
    """仿真指标收集器"""
    def __init__(self):
        self.vessel_log = []
        self.time_series = []
    
    def log_vessel(self, vessel_code: str, arrival_h: float, departure_h: float,
                   n_containers: int, n_reshuffles: int, turnaround_h: float,
                   reshuffle_pct: float, equip_util_pct: float, config: str,
                   **kwargs):
        entry = {
            'vessel_code': vessel_code,
            'arrival_h': round(arrival_h, 1),
            'departure_h': round(departure_h, 1),
            'n_containers': n_containers,
            'n_reshuffles': n_reshuffles,
            'turnaround_h': round(turnaround_h, 1),
            'reshuffle_pct': round(reshuffle_pct, 1),
            'equip_util_pct': round(equip_util_pct, 1),
            'config': config,
        }
        # 额外字段（garh_fitness, yard_penalty, ppo_model等）
        entry.update({k: v for k, v in kwargs.items() if v is not None})
        self.vessel_log.append(entry)
    
    def snapshot(self, time_h: float, state: PortState):
        """记录时间序列快照"""
        self.time_series.append({
            'time_h': round(time_h, 1),
            'berth_occ': state.berth_occupancy,
            'yard_util': state.yard_utilization,
            'equip_util': state.equip_utilization,
            'queue_len': len(state.berth_queue),
            'containers_done': state.total_containers_processed,
        })
    
    def get_summary(self, base_turnaround: float, base_reshuffle: float,
                    base_equip: float) -> dict:
        """与基线对比的改善汇总"""
        if not self.vessel_log:
            return {}
        opt_t = np.mean([v['turnaround_h'] for v in self.vessel_log])
        opt_r = np.mean([v['reshuffle_pct'] for v in self.vessel_log])
        opt_e = np.mean([v['equip_util_pct'] for v in self.vessel_log])
        return {
            'n_vessels': len(self.vessel_log),
            'total_containers': sum(v['n_containers'] for v in self.vessel_log),
            'baseline_turnaround_h': round(base_turnaround, 1),
            'optimized_turnaround_h': round(opt_t, 1),
            'baseline_reshuffle_pct': round(base_reshuffle, 1),
            'optimized_reshuffle_pct': round(opt_r, 1),
            'baseline_equip_util_pct': round(base_equip * 100, 1),
            'optimized_equip_util_pct': round(opt_e, 1),
            'turnaround_improvement_pct': round((base_turnaround-opt_t)/base_turnaround*100, 1),
            'reshuffle_improvement_pct': round((base_reshuffle-opt_r)/base_reshuffle*100, 1),
            'equip_util_improvement_pct': round((opt_e-base_equip*100)/(base_equip*100)*100, 1),
        }

class SimEngine:
    """
    离散事件仿真引擎
    支持事件调度、模块注册、进度回调
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.clock = SimClock()
        self.event_queue = []           # 优先队列（heapq）
        self.modules: Dict[str, object] = {}
        self.state = PortState()
        self.metrics = MetricsCollector()
        self.rng = np.random.RandomState(42)
        self._running = False
        # MAS组件
        self.berth_manager = BerthManager(n_berths=4)
        self.qc_cranes: List[QCCrane] = []
        self.yc_cranes: List[YCCrane] = []
        self.trucks: List[Truck] = []
        self.vessel_log: Dict[str, dict] = {}
        self.bidi_protocol = None
        self.on_event_callbacks: List[Callable] = []

    def register_module(self, name: str, module):
        """注册算法模块"""
        self.modules[name] = module
        if hasattr(module, 'initialize'):
            module.initialize({'config': self.config})
    
    def init_agents(self, n_qc=8, n_yc=12, n_trucks=20):
        """初始化MAS智能体"""
        self.qc_cranes = [QCCrane(f'QC{i}', i%4) for i in range(n_qc)]
        self.yc_cranes = [YCCrane(f'YC{i}', f'L{i%30+1:02d}') for i in range(n_yc)]
        self.trucks = [Truck(f'Truck{i}') for i in range(n_trucks)]
        self.state.qc_count, self.state.yc_count = n_qc, n_yc
        self.state.truck_count = n_trucks
    
    def on_event(self, callback: Callable):
        """注册事件回调（用于日志/UI更新）"""
        self.on_event_callbacks.append(callback)
    
    def schedule_event(self, time_h: float, event_type: str, data: dict = None, priority: int = 0):
        """调度一个未来事件"""
        heapq.heappush(self.event_queue, SimEvent(time_h, event_type, data or {}, priority))
    
    def _fire_callbacks(self, event: SimEvent):
        for cb in self.on_event_callbacks:
            cb(event)
    
    def reset(self, seed: int):
        """重置仿真状态"""
        self.rng = np.random.RandomState(seed)
        self.event_queue = []
        self.clock.reset(self.config.n_days * 24)
        self.state = PortState()
        self.metrics = MetricsCollector()
        
        # 初始化堆场状态
        self.state.yard_utilization = self.config.yard_util_init
        self.state.equip_available = self.config.equip_avail
        
        # 调度初始船舶到达事件
        self._schedule_initial_arrivals()
    
    def _schedule_initial_arrivals(self):
        """调度第一天0点的初始船舶到达"""
        ships_per_hour = self.config.ships_per_day / 24
        # 生成24小时内的事件
        for _ in range(max(1, int(self.config.ships_per_day))):
            t = self.rng.exponential(1 / ships_per_hour)
            if t < 24:
                self.schedule_event(t, EventType.VESSEL_ARRIVAL.value, {
                    'arrival_time': t,
                })
    
    def run(self) -> SimulationResult:
        """运行仿真"""
        self.reset(self.config.seeds[0] if hasattr(self.config, 'seeds') else 42)
        self._running = True
        
        while self.event_queue and self.clock.current_time < self.clock.end_time:
            event = heapq.heappop(self.event_queue)
            self.clock.current_time = event.time
            
            # 更新港口状态（调用模块）
            self._update_state(event)
            
            # 派发事件到对应模块
            self._dispatch_event(event)
            
            # 回调
            self._fire_callbacks(event)
            
            # 每24小时记录快照
            if int(self.clock.current_time) % 24 == 0 and self.clock.current_time > 0:
                self.metrics.snapshot(self.clock.current_time, self.state)
        
        self._running = False
        return self._build_result()
    
    def _update_state(self, event: SimEvent):
        """更新港口状态（含MAS智能体状态）"""
        etype = event.type
        data = event.data
        
        # 船舶到港 → 请求泊位
        if etype == EventType.VESSEL_ARRIVAL.value:
            vc = data.get('vessel_code', f'V{self.state.total_vessels_served+1}_{int(self.clock.current_time)}')
            data['vessel_code'] = vc
            success, bid, _ = self.berth_manager.request_berth(vc)
            if success:
                # 立即分配泊位
                self.vessel_log[vc] = {'arrival': self.clock.current_time, 'berth': bid}
                self.schedule_event(self.clock.current_time + 0.01, 
                    EventType.BERTH_ALLOCATED.value, {'vessel_code': vc, 'berth_id': bid})
            else:
                # 进入队列（先创建vessel_log，等泊位释放后补berth信息）
                self.vessel_log[vc] = {'arrival': self.clock.current_time, 'berth': -1}
                self.state.berth_waiting_count = self.berth_manager.n_waiting
        
        # 泊位分配
        elif etype == EventType.BERTH_ALLOCATED.value:
            vc = data['vessel_code']
            if vc not in self.state.vessels_at_berth:
                self.state.vessels_at_berth.append(vc)
            # 开始配载
            self.schedule_event(self.clock.current_time + 0.5,
                EventType.STOWAGE_START.value, {'vessel_code': vc})
        
        # 泊位释放 → 分配下一艘
        elif etype == EventType.BERTH_RELEASED.value:
            bid = data.get('berth_id', 0)
            next_vc = self.berth_manager.release_berth(bid)
            if next_vc:
                self.vessel_log[next_vc]['berth'] = bid
                self.vessel_log[next_vc]['waiting_h'] = self.clock.current_time - self.vessel_log[next_vc]['arrival']
                self.schedule_event(self.clock.current_time + 0.01,
                    EventType.BERTH_ALLOCATED.value, {'vessel_code': next_vc, 'berth_id': bid})
        
        # 船舶离港
        elif etype == EventType.VESSEL_DEPART.value:
            vc = data.get('vessel_code', '')
            bid = self.vessel_log.get(vc, {}).get('berth', 0)
            if vc in self.state.vessels_at_berth:
                self.state.vessels_at_berth.remove(vc)
            # 释放泊位（触发下一艘分配）
            self.schedule_event(self.clock.current_time + 0.01,
                EventType.BERTH_RELEASED.value, {'berth_id': bid})
        
        # QC事件
        elif etype == EventType.QC_CYCLE_DONE.value:
            self.state.total_containers_processed += 1
        elif etype == EventType.QC_ALL_DONE.value:
            pass  # 后续事件由模块处理
        
        # YC事件
        elif etype == EventType.YC_CYCLE_DONE.value:
            pass
        elif etype == EventType.YC_ALL_DONE.value:
            pass
        
        # 更新统计
        self.state.berth_occupancy = self.berth_manager.utilization
        self.state.berth_waiting_count = self.berth_manager.n_waiting
        busy_qc = sum(1 for q in self.qc_cranes if q.status.name == 'WORKING')
        busy_yc = sum(1 for y in self.yc_cranes if y.status.name == 'WORKING')
        self.state.qc_busy = busy_qc
        self.state.yc_busy = busy_yc
        self.state.equip_utilization = min(
            (busy_qc + busy_yc) / max(1, self.state.qc_count + self.state.yc_count),
            0.95
        )
    
    def _dispatch_event(self, event: SimEvent):
        """派发事件到对应处理模块"""
        handler_name = f'_handle_{event.type}'
        for name, module in self.modules.items():
            handler = getattr(module, handler_name, None)
            if handler:
                result = handler(event, self.state, self.clock, self.rng)
                if result:
                    # 记录船舶日志（如果有）
                    if 'vessel_log' in result:
                        self.metrics.log_vessel(**result['vessel_log'])
                    # 调度新事件
                    if 'new_events' in result:
                        for e in result['new_events']:
                            self.schedule_event(e['time'], e['type'], e.get('data', {}), e.get('priority', 0))
    
    def _build_result(self) -> SimulationResult:
        """构建仿真结果（记录绝对指标，不做基线对比）"""
        vessel_log = self.metrics.vessel_log
        
        if not vessel_log:
            return SimulationResult(
                scenario=self.config.scenario,
                seed=self.config.seeds[0] if hasattr(self.config, 'seeds') else 42,
                n_days=self.config.n_days,
            )
        
        opt_t = np.mean([v['turnaround_h'] for v in vessel_log])
        opt_r = np.mean([v['reshuffle_pct'] for v in vessel_log])
        opt_e = np.mean([v['equip_util_pct'] for v in vessel_log])
        
        result = SimulationResult(
            scenario=self.config.scenario,
            seed=self.config.seeds[0] if hasattr(self.config, 'seeds') else 42,
            n_days=self.config.n_days,
            n_vessels=len(vessel_log),
            total_containers=sum(v['n_containers'] for v in vessel_log),
            turnaround_h=round(opt_t, 1),
            reshuffle_pct=round(opt_r, 1),
            equip_util_pct=round(opt_e, 1),
            turnaround_improvement_pct=0.0,  # 由外部run.py计算
            reshuffle_improvement_pct=0.0,
            equip_util_improvement_pct=0.0,
        )
        result.vessel_log = vessel_log
        return result
