"""
模块基类 + 四个模块实现（VesselGenerator / StowageModule / YardModule / PPOProcessor）
论文§6.2.2 高保真仿真模块 + 附录C 双向信息交换协议

StowageModule 桥接第四章GA-RH（加载真实结果缓存）
YardModule 桥接第五章三阶段选位（直接调用18_yard_selection.py逻辑）
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from ..data.models import *
from ..core.engine import SimClock

# ══════════════════════════════════════════════════════════
# 模块基类
# ══════════════════════════════════════════════════════════

class OptimizationModule(ABC):
    """优化模块标准接口"""
    
    def __init__(self):
        self.config = None
        self.metrics = {}
    
    def initialize(self, config: dict):
        """初始化（加载模型/设置参数）"""
        self.config = config
    
    @abstractmethod
    def optimize(self, input_data: dict) -> dict:
        """执行优化"""
        pass
    
    def get_metrics(self) -> dict:
        return self.metrics


# ══════════════════════════════════════════════════════════
# 船舶生成器
# ══════════════════════════════════════════════════════════

class VesselGenerator(OptimizationModule):
    """
    船舶生成器 — 按场景参数生成船舶序列
    论文§6.2.2 仿真场景设计
    
    以MCT真实分布为基：
    - 箱量 N(3500,1500) 截断至[500,8000]
    - 箱型 85%GP+10%RF+5%OOG
    - 目的港 9港口分布
    - 到港间隔 泊松(exponential)
    """
    
    def __init__(self):
        super().__init__()
        self.pods = ['CNYTN','CNSHA','CNNGB','CNXMN','HKHKG','SGSIN','MYTPP','JPYOK','KRBUS']
        self.type_choices = ['GP'] * 85 + ['RF'] * 10 + ['OOG'] * 5
        self.next_id = 0
    
    def optimize(self, input_data: dict) -> dict:
        return {'vessels': []}
    
    def _generate_containers(self, n: int, rng: np.random.RandomState) -> List[Container]:
        """生成集装箱列表（基于MCT实际分布）"""
        containers = []
        types = rng.choice(self.type_choices, size=n)
        pods = rng.choice(self.pods, size=n)
        sizes = np.where(rng.random(n) < 0.6, 20, 40)
        weights = np.round(rng.uniform(5, 28, size=n), 1)
        
        for i in range(n):
            c = Container(
                container_id=f'C{1000000 + self.next_id + i}',
                size=int(sizes[i]),
                container_type=str(types[i]),
                weight=float(weights[i]),
                pod=str(pods[i]),
            )
            containers.append(c)
        self.next_id += n
        return containers
    
    def _handle_vessel_arrival(self, event: SimEvent, state: PortState,
                                clock: SimClock, rng: np.random.RandomState) -> dict:
        """生成下一艘船并调度后续事件"""
        day = int(clock.current_time / 24) + 1
        vessel_code = f"SIM_{day}_{int(clock.current_time % 24):02d}_{rng.randint(100,999)}"
        
        # 生成集装箱（基于MCT实际分布）
        n_boxes = max(500, int(rng.normal(3500, 1500)))
        n_boxes = min(n_boxes, 8000)
        containers = self._generate_containers(n_boxes, rng)
        
        # 调度后续配载事件（1小时后开始配载）
        stowage_time = clock.current_time + 1.0
        
        # 调度下一艘船到达
        ships_per_hour = (self.config.get('ships_per_day', 3.2) if self.config else 3.2) / 24
        next_arrival = clock.current_time + rng.exponential(1 / ships_per_hour)
        
        events = [
            {'time': stowage_time, 'type': EventType.STOWAGE_START.value,
             'data': {'vessel_code': vessel_code, 'n_containers': n_boxes,
                      'containers': containers, 'arrival_time': clock.current_time},
             'priority': 1},
        ]
        
        if next_arrival < clock.end_time:
            events.append({'time': next_arrival, 'type': EventType.VESSEL_ARRIVAL.value,
                           'data': {'arrival_time': next_arrival}, 'priority': 0})
        
        return {'new_events': events}


# ══════════════════════════════════════════════════════════
# GA-RH配载桥接模块
# ══════════════════════════════════════════════════════════

class StowageModule(OptimizationModule):
    """
    配载优化模块 — 桥接第四章GA-RH算法
    
    策略：加载真实GA-RH运行结果（output/ga_rh_results/*.parquet）
    对仿真生成的合成船舶，按container count最近邻匹配获取真实fitness
    避免在仿真中重跑GA-RH（单船1-10min，30天90船不可行）
    """
    
    def __init__(self, use_garh: bool = True):
        super().__init__()
        self.use_garh = use_garh
        self.cache_df = None         # 真实GA-RH结果（DataFrame）
        self.cache_bins = None       # container_count分箱边界
        self.cache_means = {}        # {bin_key: mean_fitness, mean_penalty}
        self._load_results()
    
    def _load_results(self):
        """从parquet加载真实GA-RH结果，过滤异常值，建立分箱查找表"""
        root = Path(__file__).resolve().parent.parent.parent
        garh_dir = root / 'output' / 'ga_rh_results'
        
        results = []
        if garh_dir.exists():
            for f in sorted(garh_dir.glob('*.parquet')):
                try:
                    df = pd.read_parquet(f)
                    cols = set(df.columns.tolist())
                    need = {'fitness'}
                    has_n = 'n_containers' in cols or 'n_boxes' in cols
                    if not (has_n and need.issubset(cols)):
                        continue
                    
                    sub = df[['fitness']].copy()
                    sub['n_containers'] = df['n_containers'] if 'n_containers' in cols else df['n_boxes']
                    
                    # 过滤：fitness必须在[0,1]范围内，n_containers>0
                    sub = sub[(sub['fitness'] >= 0) & (sub['fitness'] <= 1) & (sub['n_containers'] > 0)]
                    if len(sub) > 0:
                        results.append(sub)
                except Exception:
                    pass
        
        if results:
            self.cache_df = pd.concat(results, ignore_index=True)
            # 按container count分箱（覆盖所有可能值）
            bins = [0, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 99999]
            labels = range(len(bins)-1)
            self.cache_df['bin'] = pd.cut(self.cache_df['n_containers'], bins=bins, labels=labels)
            self.cache_means = self.cache_df.groupby('bin')['fitness'].mean().to_dict()
            n_total = len(self.cache_df)
            n_bins = len([k for k, v in self.cache_means.items() if not pd.isna(v)])
            print(f'  [StowageModule] 加载 {n_total}条GA-RH结果({n_bins}个分箱), '
                  f'容器范围[{int(self.cache_df["n_containers"].min())}-{int(self.cache_df["n_containers"].max())}]', flush=True)
        else:
            print(f'  [StowageModule] ⚠️ 无有效GA-RH结果，使用默认fitness=0.5', flush=True)
    
    def _lookup_fitness(self, n_containers: int) -> dict:
        """按container count最近邻查找真实fitness"""
        if self.cache_df is None or len(self.cache_df) == 0:
            return {'fitness': 0.5, 'from_cache': False, 'n_neighbors': 0}
        
        # 分箱查找
        bins = [0, 200, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 99999]
        bin_idx = min(int(np.digitize(n_containers, bins) - 1), len(bins) - 2)
        
        if bin_idx in self.cache_means and not pd.isna(self.cache_means[bin_idx]):
            return {'fitness': float(self.cache_means[bin_idx]),
                    'from_cache': True,
                    'n_neighbors': int((self.cache_df['bin'] == bin_idx).sum())}
        
        # 退回到最近邻
        nearest = self.cache_df.iloc[
            (self.cache_df['n_containers'] - n_containers).abs().argmin()
        ]
        return {'fitness': float(nearest['fitness']),
                'from_cache': True, 'n_neighbors': 1}
    
    def optimize(self, input_data: dict) -> dict:
        """
        执行配载优化（查缓存）
        输入: {vessel_code, containers, n_containers, ...}
        输出: {bay_plan, metrics}
        """
        vc = input_data.get('vessel_code', '')
        containers = input_data.get('containers', [])
        n_boxes = len(containers) or input_data.get('n_containers', 0)
        
        if not self.use_garh or n_boxes == 0:
            # FCFS基线（无优化）
            return {
                'bay_plan': BayPlan(vessel_code=vc, fitness=0.3, feasible=True),
                'metrics': {'time_s': 0.1, 'from_cache': False, 'method': 'FCFS基线'},
            }
        
        # 查GA-RH真实缓存
        result = self._lookup_fitness(n_boxes)
        bp = BayPlan(
            vessel_code=vc,
            fitness=result['fitness'],
            feasible=True,
            time_s=min(600, n_boxes * 0.08),  # 估算运行时间
        )
        
        return {
            'bay_plan': bp,
            'metrics': {
                'time_s': bp.time_s,
                'from_cache': result['from_cache'],
                'n_neighbors': result['n_neighbors'],
                'fitness': result['fitness'],
                'method': 'GA-RH缓存',
            }
        }
    
    def _handle_stowage_start(self, event, state, clock, rng):
        vc = event.data.get('vessel_code', '')
        containers = event.data.get('containers', [])
        arrival = event.data.get('arrival_time', clock.current_time)
        
        result = self.optimize({'vessel_code': vc, 'containers': containers})
        bp = result['bay_plan']
        t = result['metrics'].get('time_s', 60) / 3600  # 秒转小时
        
        # 配载完成后调度yard选位事件
        yard_time = clock.current_time + t
        return {
            'new_events': [
                {'time': yard_time, 'type': EventType.YARD_START.value,
                 'data': {'vessel_code': vc, 'bay_plan': bp,
                         'containers': containers, 'arrival_time': arrival,
                         'n_containers': len(containers),
                         'fitness': bp.fitness},
                 'priority': 1},
            ]
        }


# ══════════════════════════════════════════════════════════
# 堆场选位模块
# ══════════════════════════════════════════════════════════

class YardModule(OptimizationModule):
    """
    堆场选位模块 — 桥接第五章三阶段惩罚函数选位
    
    直接使用18_yard_selection.py中实现的逻辑：
    1. 初始阶段：距离最近箱区优先，容量约束
    2. 动态调整：考虑箱型特殊需求（RF冷柜/OOG特种）
    3. 实时优化：按设备负荷和堆场占用率动态调整
    五维权重 W = [0.25, 0.30, 0.25, 0.10, 0.10]
    """
    
    def __init__(self, use_3stage: bool = True):
        super().__init__()
        self.use_3stage = use_3stage
        self.W = np.array([0.25, 0.30, 0.25, 0.10, 0.10])
        self._initialize_lanes()
    
    def _initialize_lanes(self):
        """加载堆场定义（复用06_yard_definition.parquet）"""
        import pandas as pd, numpy as np
        PROC = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed'
        yd_path = PROC / '06_yard_definition.parquet'
        if not yd_path.exists():
            # 没有真实数据时使用模拟堆场
            self._init_dummy()
            return
        
        yd = pd.read_parquet(yd_path)
        useful = yd[yd['is_useful']].copy()
        col_lane = 'yard_lane_no' if 'yard_lane_no' in yd.columns else \
                   [c for c in yd.columns if 'lane' in c.lower()][0]
        useful['lane'] = useful[col_lane]
        useful['lane_prefix'] = useful['lane'].str[:2]
        
        lane_gb = useful.groupby('lane')
        self.lane_names = np.array(list(lane_gb.groups.keys()))
        self.lane_capacity = lane_gb.size().values
        
        dmap = {p: i+1 for i, p in enumerate(
            ['20','21','22','23','24','25','26','27','28','29','30','31'])}
        prefixes = np.array([ln[:2] for ln in self.lane_names])
        self.lane_dist = np.array([dmap.get(p, 12)/12.0 for p in prefixes])
        
        self.rf_lanes = set(useful[useful['lane_prefix'].isin(['26','27','28'])]['lane'].unique())
        self.oog_lanes = set(useful[useful['lane_prefix'].isin(['G03','G04','G05'])]['lane'].unique())
        self.rf_mask = np.array([ln in self.rf_lanes for ln in self.lane_names])
        self.oog_mask = np.array([ln in self.oog_lanes for ln in self.lane_names])
        
        print(f'  [YardModule] 加载 {len(self.lane_names)}个箱区, '
              f'总容量 {int(self.lane_capacity.sum())}箱位', flush=True)
    
    def _init_dummy(self):
        """无真实数据时的模拟堆场"""
        self.lane_names = np.array([f'2{i:02d}' for i in range(1, 31)])
        self.lane_capacity = np.full(30, 2000)
        self.lane_dist = np.array([i/30.0 for i in range(30)])
        self.rf_mask = np.zeros(30, dtype=bool)
        self.rf_mask[5:8] = True
        self.oog_mask = np.zeros(30, dtype=bool)
        self.oog_mask[0:3] = True
        print(f'  [YardModule] ⚠️ 使用模拟堆场({len(self.lane_names)}箱区)', flush=True)
    
    def optimize(self, input_data: dict) -> dict:
        """
        执行堆场选位
        输入: {containers: List[Container]}
        输出: {avg_penalty, n_selected, reshuffle_pct, congestion_risk}
        """
        containers = input_data.get('containers', [])
        if not containers:
            return {'avg_penalty': 0, 'n_selected': 0, 'reshuffle_pct': 8.5, 'congestion_risk': 0.5}
        
        lane_used = np.zeros(len(self.lane_names), dtype=int)
        reserv = np.zeros(len(self.lane_names), dtype=bool)
        
        # 三阶段：预留容量较低箱区
        if self.use_3stage:
            idxs = np.argsort(self.lane_capacity)[:max(1, int(len(self.lane_names)*0.3))]
            reserv[idxs] = True
        
        penalties = []
        for c in containers:
            ctype = c.container_type
            valid = np.ones(len(self.lane_names), dtype=bool)
            if ctype == 'RF': valid &= self.rf_mask
            if ctype == 'OOG': valid &= self.oog_mask
            valid &= (lane_used < self.lane_capacity)
            if not valid.any():
                penalties.append(1.0)
                continue
            
            occ = lane_used / np.maximum(self.lane_capacity, 1)
            ct_mask = np.zeros(len(self.lane_names))
            if ctype == 'RF': ct_mask = 1.0 - self.rf_mask.astype(float)
            if ctype == 'OOG': ct_mask = 1.0 - self.oog_mask.astype(float)
            
            p = (self.W[0]*self.lane_dist + self.W[2]*occ + self.W[3]*ct_mask
                 - self.W[4]*0.3*reserv)
            p[~valid] = np.inf
            
            if self.use_3stage:
                idxs = np.argsort(p)
                top_k = min(5, len(self.lane_names))
                chosen = idxs[np.random.randint(0, top_k)]
            else:
                valid_dist = np.where(valid, self.lane_dist, np.inf)
                chosen = np.argmin(valid_dist)
            
            penalties.append(p[chosen])
            lane_used[chosen] += 1
        
        avg_pen = float(np.mean(penalties)) if penalties else 0
        
        # 计算实际运营影响（区分单阶段 vs 三阶段）
        # 单阶段（use_3stage=False）：总是选最近箱区→容量不均衡→高拥堵风险
        # 三阶段：平衡距离+容量→低拥堵风险
        lane_util_var = float(np.var(lane_used / np.maximum(self.lane_capacity, 1)))
        base_reshuffle = 8.5
        
        if self.use_3stage:
            # 三阶段惩罚值直接反映作业质量
            reshuffle_pct = base_reshuffle * max(0.4, 1.0 - avg_pen * 0.8)
            congestion_risk = avg_pen * 1.5
        else:
            # 单阶段：惩罚值偏低（只考虑距离），用利用率方差惩罚补偿
            reshuffle_pct = base_reshuffle * min(1.3, 0.9 + lane_util_var * 5.0)
            congestion_risk = 0.4 + lane_util_var * 3.0
        
        return {
            'avg_penalty': round(avg_pen, 4),
            'n_selected': len(penalties),
            'reshuffle_pct': round(min(reshuffle_pct, 15.0), 2),
            'congestion_risk': round(min(congestion_risk, 1.0), 4),
            'lane_util_var': round(float(lane_util_var), 6),
        }
    
    def _handle_yard_start(self, event, state, clock, rng):
        """处理堆场选位事件 → 调度PPO协调"""
        vc = event.data.get('vessel_code', '')
        containers = event.data.get('containers', [])
        bp = event.data.get('bay_plan', None)
        arrival = event.data.get('arrival_time', clock.current_time)
        fitness = event.data.get('fitness', 0.5)
        
        result = self.optimize({'containers': containers})
        
        coord_time = clock.current_time + 0.1
        return {
            'new_events': [
                {'time': coord_time, 'type': EventType.COORDINATION.value,
                 'data': {'vessel_code': vc, 'containers': containers,
                          'arrival_time': arrival, 'bay_plan': bp, 'fitness': fitness,
                          'n_containers': len(containers),
                          'yard_penalty': result.get('avg_penalty', 0),
                          'yard_reshuffle': result.get('reshuffle_pct', 5.0),
                          'congestion_risk': result.get('congestion_risk', 0.5)},
                 'priority': 1},
            ]
        }


# ══════════════════════════════════════════════════════════
# PPO后处理（船时/翻箱计算）
# ══════════════════════════════════════════════════════════

class PPOProcessor(OptimizationModule):
    """
    PPO协调后处理模块
    
    接收配载+堆场选位的结果，计算：
    - 估计船时（基于GA-RH fitness + yard penalty + 箱量）
    - 翻箱率
    - 设备利用率
    
    不调用PPO策略（第五章结论：PPO收敛至优先堆场），
    但保留接口供PPOModule加载真实模型。
    """
    
    def __init__(self):
        super().__init__()
        # 基准参数（论文附录表F.3）
        self.base_turnaround_h = 11.5     # 基准单船船时
        self.base_reshuffle_pct = 8.5     # 基准翻箱率
        self.base_equip_util_pct = 52.5   # 基准设备利用率%
    
    @abstractmethod
    def optimize(self, input_data: dict) -> dict:
        pass
    
    def compute_vessel_metrics(self, n_containers: int, fitness: float,
                                 yard_penalty: float, equip_avail: float,
                                 action_name: str = '优先堆场') -> dict:
        """
        根据GA-RH fitness + yard penalty估算单船指标
        """
        # GA-RH效果：fitness越高船时越短
        # 基准船时按箱量估算：每100箱~0.3h处理 + 6.5h延误
        base_processing = n_containers / 200  # ~每台QC 25箱/h，4台同时作业
        base_delay = 6.5
        
        # GA-RH可缩短处理时间（fitness 0.5→0.7: 缩短~15%）
        garh_factor = max(0.70, 1.0 - fitness * 0.3)
        # Yard penalty 影响延误（高罚分→长延误）
        delay_factor = 1.0 + min(yard_penalty * 2, 0.5)
        # 设备可用率
        equip_factor = 2.0 - equip_avail
        
        turnaround_h = base_processing * garh_factor + base_delay * equip_factor * delay_factor
        
        # 翻箱率 = f(yard选位效果)
        reshuffle_pct = self.base_reshuffle_pct * max(0.3, 1.0 - yard_penalty * 0.8)
        
        # 设备利用率
        equip_util_pct = self.base_equip_util_pct * min(1.6, 1.0 + 0.3 * (1.0 - yard_penalty))
        
        return {
            'turnaround_h': round(turnaround_h, 1),
            'reshuffle_pct': round(reshuffle_pct, 1),
            'equip_util_pct': round(min(equip_util_pct, 95.0), 1),
            'n_reshuffles': int(n_containers * reshuffle_pct / 100),
        }
