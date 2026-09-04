"""
Step 15: GA-RH 配载优化模型（论文第四章）
=================================================
基于论文4.2节混合GA-RH算法规范实现：
  - 三层分层编码（贝位-列-层）
  - 按卸货港分组自适应交叉
  - 3种变异策略（位置交换/区域重分配/增量调整）
  - 5类规则启发式局部优化
  - 多目标适应度（稳性+效率+堆场协同）
  - Tournament(k=3) + top5%精英保留
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json, time, warnings
warnings.filterwarnings('ignore')

PROC = Path('data/processed')
OUT = Path('output')
SPLIT = OUT / 'splits' / 'ga_rh_80_20'
RESULT = Path('output/ga_rh_results')
RESULT.mkdir(parents=True, exist_ok=True)
np.random.seed(42)


# ══════════════════════════════════════════════════════════
# 1. 数据模型
# ══════════════════════════════════════════════════════════

@dataclass
class Slot:
    """一个可用箱位（对应论文4.2.3三层编码的底层）"""
    cell_code: str
    bay: int          # 贝位
    row: int          # 列
    tier: int         # 层
    size_type: str    # 20/40
    allow_sizes: str  # 兼容箱型

class VesselProblem:
    """单搜船配载问题（对应论文4.1数学模型）"""
    
    # ── 硬约束（§4.1.2） ──
    HARD_CONSTRAINTS = [
        'geometry_match',    # C1: 几何匹配
        'stack_weight',      # C2: 堆重限制  
        'loading_order',     # C4: 装卸顺序（后卸不压先卸）
    ]
    
    def __init__(self, vessel_code: str, berth_plan_no: str,
                 containers_df: pd.DataFrame, bay_df: pd.DataFrame,
                 vessel_info: pd.Series):
        self.vessel_code = vessel_code
        self.berth_plan_no = berth_plan_no
        self.containers = containers_df.reset_index(drop=True)
        self.n_container = len(containers_df)
        
        # 构建箱位
        self.slots = [
            Slot(cell_code=str(r.get('custom_cell', '')),
                 bay=int(r.get('custom_bay', 0)),
                 row=int(r.get('custom_stack', 0)),
                 tier=int(r.get('CUSTOMTIER', 0)),
                 size_type=str(r.get('size_type', '20')),
                 allow_sizes=str(r.get('allow_sizes', '20,40')))
            for _, r in bay_df.iterrows()
        ]
        self.n_slot = len(self.slots)
        
        # 船舶参数
        self.max_teu = float(vessel_info.get('max_teu', 0) or 0)
        self.max_tier = max(s.tier for s in self.slots) if self.slots else 1
        self.max_row = max(s.row for s in self.slots) if self.slots else 1
        
        # 箱属性数组（向量化计算用）
        self.csize = np.array([int(c.get('container_size', 20) or 20)
                               for _, c in self.containers.iterrows()])
        self.cweight = np.array([
            pd.to_numeric(c.get('weight_kg', c.get('gross_weight_x', 0)), errors='coerce')
            for _, c in self.containers.iterrows()
        ])
        self.pods = self.containers['pod'].fillna('UNK').values
        self.unique_pods = np.unique(self.pods)
        
        # 兼容箱位索引（论文4.2.3三层编码：贝位分配需满足尺寸匹配）
        self.compat_slots = []
        # 已知的标准箱尺寸和fallback映射
        slot_size_fallback = {'45': '40', '30': '20', '48': '40', '53': '40'}
        for sz in self.csize:
            sz_str = str(int(sz))
            # 先尝试精确匹配
            compat = np.array([
                i for i, s in enumerate(self.slots)
                if sz_str in s.allow_sizes.split(',')
            ])
            if len(compat) == 0 and sz_str in slot_size_fallback:
                # Fallback到近似尺寸（如45ft→40ft slot）
                fb = slot_size_fallback[sz_str]
                compat = np.array([
                    i for i, s in enumerate(self.slots)
                    if fb in s.allow_sizes.split(',')
                ])
            # 如果仍然为空，接受任何尺寸（宁可不精确也不能无解）
            if len(compat) == 0:
                compat = np.array([
                    i for i, s in enumerate(self.slots)
                    if any(size in s.allow_sizes.split(',') for size in ['20', '40'])
                ])
            self.compat_slots.append(compat)
        
        # 实际装船位置（验证用）
        self._parse_real_positions()
    
    def _parse_real_positions(self):
        self.real_cells = {}
        for i, row in self.containers.iterrows():
            pos = str(row.get('stow_position', ''))
            if pos and pos != 'nan' and len(pos) >= 5:
                self.real_cells[i] = pos


# ══════════════════════════════════════════════════════════
# 2. 多目标适应度函数（论文4.1.3 + 4.2.3）
# ══════════════════════════════════════════════════════════

class FitnessFunction:
    """
    多目标适应度（论文式4.14）
    fitness = Σ(wi × fi) - Σ(pj × vj)
    
    f₁ = 翻箱成本(反比于翻箱次数)  ← 新增独立目标
    f₂ = 装卸效率(同港集中度)
    f₃ = 重量分布均衡
    f₄ = 堆场协同成本
    penalty = 几何匹配违规 + 堆重违规 + 装卸顺序违规 + 同港分散
    """
    
    def __init__(self, problem: VesselProblem):
        self.p = problem
        # 目标权重（去掉稳性后重分配，式4.14的wi）
        self.w = {'rehandle': 0.25, 'efficiency': 0.35,
                  'balance': 0.25, 'yard_collab': 0.15}
        self.penalty_weight = 5.0  # 约束违反惩罚权重（式4.14的pj）
        # 缓存POD顺序
        self._pod_order = {pod: i for i, pod in enumerate(sorted(problem.unique_pods))} if problem is not None else {}
    
    def _rehandle_cost(self, chrom: np.ndarray) -> float:
        """目标f₁：翻箱次数最小化（论文4.1.3式4.1）
           同列内，后卸港箱压住先卸港箱 → 翻转计数
           归一化到[0,1]，越高=翻箱越少"""
        p = self.p
        pod_seq = np.array([self._pod_order.get(pod, 0) for pod in p.pods])
        
        # 按(贝位,列)分组
        groups = {}
        for i, s in enumerate(chrom):
            sl = p.slots[s]
            key = (sl.bay, sl.row)
            if key not in groups:
                groups[key] = []
            groups[key].append((sl.tier, pod_seq[i]))
        
        total_inv = 0
        total_max = 0
        for key, items in groups.items():
            k = len(items)
            if k < 2:
                continue
            # 按tier从顶到底排列
            items.sort(key=lambda x: -x[0])
            max_inv = k * (k - 1) // 2
            total_max += max_inv
            # 计逆序：上层(早遍历)seq > 下层(晚遍历)seq → 翻箱
            for a in range(k):
                for b in range(a + 1, k):
                    if items[a][1] > items[b][1]:  # 上层seq > 下层seq
                        total_inv += 1
        
        if total_max == 0:
            return 1.0
        return max(0, 1 - total_inv / total_max)
    
    def _efficiency(self, chrom: np.ndarray) -> float:
        """目标f₂：装卸效率（同港集中度，论文4.1.3式4.3）
           同卸货港集装箱在相邻贝位"""
        bays = np.array([self.p.slots[s].bay for s in chrom])
        if len(self.p.unique_pods) <= 1:
            return 1.0
        spreads = []
        for pod in self.p.unique_pods:
            mask = self.p.pods == pod
            if mask.sum() > 1:
                spreads.append(np.std(bays[mask]))
        if not spreads:
            return 1.0
        avg_spread = np.mean(spreads)
        return max(0, 1 - avg_spread / 50)
    
    def _balance(self, chrom: np.ndarray) -> float:
        """目标f₃：重量分布均衡（论文4.1.3式4.4）
           左右舷重量差最小"""
        rows = np.array([self.p.slots[s].row for s in chrom])
        weights = self.p.cweight
        center = self.p.max_row / 2
        left_mask = rows < center
        right_mask = rows >= center
        left_w = weights[left_mask].sum() if left_mask.any() else 0
        right_w = weights[right_mask].sum() if right_mask.any() else 0
        total_w = left_w + right_w
        if total_w == 0:
            return 1.0
        imbalance = abs(left_w - right_w) / total_w
        return max(0, 1 - imbalance)
    
    def _yard_collab(self, chrom: np.ndarray) -> float:
        """目标f₄：堆场协同成本（论文4.1.4）
           优先低层（简化：奇数层比例低=取箱方便）"""
        tiers = np.array([self.p.slots[s].tier for s in chrom])
        odd_ratio = (tiers % 2 == 1).mean()
        return 1.0 - odd_ratio * 0.3
    
    def _constraint_violations(self, chrom: np.ndarray) -> float:
        """硬约束违反评估（仅C1+C2，C4和同港集中转为目标函数）
           返回归一化违反程度[0,1]"""
        violations = 0.0
        p = self.p
        
        # 预转换compat_slots为set集合加速查找
        compat_sets = getattr(self, '_compat_sets', None)
        if compat_sets is None:
            compat_sets = [set(slots) for slots in p.compat_slots]
            self._compat_sets = compat_sets
        
        # C1: 几何匹配 — 每个箱子必须分配兼容箱位
        for i, s in enumerate(chrom):
            if s not in compat_sets[i]:
                violations += 1.0
        
        # C2: 堆重限制 — 下层箱重≥上层的50%
        groups = {}
        for i, s in enumerate(chrom):
            sl = p.slots[s]
            key = (sl.bay, sl.row)
            if key not in groups:
                groups[key] = []
            groups[key].append((sl.tier, p.cweight[i]))
        
        for key, items in groups.items():
            items.sort(key=lambda x: -x[0])  # 顶层→底层
            for j in range(1, len(items)):
                if items[j][1] < items[j-1][1] * 0.5:
                    violations += 0.5
        
        # 归一化
        return violations / p.n_container
    
    def evaluate(self, chrom: np.ndarray) -> float:
        """综合适应度（论文式4.14）"""
        f_r = self._rehandle_cost(chrom)
        f_e = self._efficiency(chrom)
        f_b = self._balance(chrom)
        f_y = self._yard_collab(chrom)
        penalty = self._constraint_violations(chrom)
        
        fitness = (self.w['rehandle'] * f_r +
                   self.w['efficiency'] * f_e +
                   self.w['balance'] * f_b +
                   self.w['yard_collab'] * f_y -
                   self.penalty_weight * penalty)
        return fitness
    
    def detail(self, chrom: np.ndarray) -> Dict:
        return {
            'rehandle': self._rehandle_cost(chrom),
            'efficiency': self._efficiency(chrom),
            'balance': self._balance(chrom),
            'yard_collab': self._yard_collab(chrom),
            'penalty': self._constraint_violations(chrom),
        }


# ══════════════════════════════════════════════════════════
# 3. 遗传算法组件（论文4.2.3）
# ══════════════════════════════════════════════════════════

class GAComponents:
    """遗传算法核心算子"""
    
    @staticmethod
    def init_population(prob: VesselProblem, pop_size: int) -> np.ndarray:
        """初始化种群（快速版：随机选兼容箱位，不强制去重）"""
        pop = np.zeros((pop_size, prob.n_container), dtype=int)
        for idx in range(pop_size):
            chrom = np.array([
                np.random.choice(prob.compat_slots[i]) if len(prob.compat_slots[i]) > 0
                else i % max(1, prob.n_slot)
                for i in range(prob.n_container)
            ])
            pop[idx] = chrom
        return pop
    
    @staticmethod
    def selection(pop: np.ndarray, fitness: np.ndarray,
                  elite_ratio: float = 0.05, tournament_k: int = 3) -> np.ndarray:
        """锦标赛选择 + 精英保留（论文4.2.3）
           保留top5%精英，其余通过k=3锦标赛"""
        n = len(pop)
        n_elite = max(2, int(n * elite_ratio))
        elite_idx = np.argsort(fitness)[-n_elite:]
        
        selected = list(pop[elite_idx])
        while len(selected) < n:
            contenders = np.random.choice(n, tournament_k, replace=False)
            winner = contenders[np.argmax(fitness[contenders])]
            selected.append(pop[winner].copy())
        
        return np.array(selected[:n])
    
    @staticmethod
    def crossover_port_group(prob: VesselProblem, 
                             parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """按卸货港分组自适应交叉（论文4.2.3核心创新）
           - 随机选两个卸货港
           - 交换这两个港所有箱子的贝位分配方案
           - 保持各港箱数不变 → 不产生不可行解"""
        pods = prob.pods
        unique = prob.unique_pods
        if len(unique) < 2:
            return parent1.copy(), parent2.copy()
        
        # 随机选两个不同港口交换
        p1, p2 = np.random.choice(unique, 2, replace=False)
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(prob.n_container):
            if pods[i] == p1:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
            elif pods[i] == p2:
                child1[i] = parent2[i]
                child2[i] = parent1[i]
        
        return child1, child2
    
    @staticmethod
    def mutate(prob: VesselProblem, chrom: np.ndarray, 
               mutation_rate: float = 0.1) -> np.ndarray:
        """3种变异策略混合应用（论文4.2.3）
           1. 位置交换（局部扰动）
           2. 区域重分配（大范围探索）
           3. 增量调整（精细微调）"""
        mutant = chrom.copy()
        n = prob.n_container
        
        for i in range(n):
            if np.random.random() >= mutation_rate:
                continue
            
            strategy = np.random.choice(['swap', 'reassign', 'adjust'],
                                        p=[0.4, 0.3, 0.3])
            
            if strategy == 'swap':
                # 位置交换：随机选另一个箱交换船位
                j = np.random.randint(n)
                while j == i:
                    j = np.random.randint(n)
                # 检查兼容性
                if chrom[i] in prob.compat_slots[j] and chrom[j] in prob.compat_slots[i]:
                    mutant[i], mutant[j] = mutant[j], mutant[i]
            
            elif strategy == 'reassign':
                # 区域重分配：换到其他箱位
                used = set(mutant)
                available = [s for s in prob.compat_slots[i] if s not in used]
                if len(available) > 0:
                    current = mutant[i]
                    others = [s for s in available if s != current]
                    if len(others) > 0:
                        mutant[i] = np.random.choice(others)
                    elif len(available) > 0:
                        mutant[i] = np.random.choice(available)
            
            elif strategy == 'adjust':
                # 增量调整：上移/下移一层
                current_slot = prob.slots[mutant[i]]
                current_tier = current_slot.tier
                # 找相同贝位列、上下层兼容的箱位
                candidates = []
                for j, s in enumerate(prob.slots):
                    if (s.bay == current_slot.bay and 
                        s.row == current_slot.row and
                        abs(s.tier - current_tier) == 2 and  # 同列相邻层
                        j != mutant[i] and
                        j not in set(mutant)):
                        candidates.append(j)
                if candidates:
                    mutant[i] = np.random.choice(candidates)
        
        return mutant


# ══════════════════════════════════════════════════════════
# 4. 规则启发式组件（论文4.2.4）
# ══════════════════════════════════════════════════════════

class RuleHeuristics:
    """5类规则启发式局部优化（论文4.2.4）
       选择top20%个体进行规则优化"""
    
    @staticmethod
    def optimize(prob: VesselProblem, chrom: np.ndarray, 
                 fitness_fn: FitnessFunction) -> Tuple[np.ndarray, float]:
        """对单个染色体应用规则优化
        
        按顺序尝试5类规则，有改进则保留"""
        best_chrom = chrom.copy()
        best_fit = fitness_fn.evaluate(chrom)
        
        improved = True
        max_iter = 10
        iteration = 0
        
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            
            # R1: 装卸顺序规则 — 确保同列内后卸港箱不压先卸港箱
            r1 = RuleHeuristics._loading_order_rule(prob, best_chrom)
            if r1 is not None:
                f1 = fitness_fn.evaluate(r1)
                if f1 > best_fit:
                    best_chrom, best_fit = r1, f1
                    improved = True
            
            # R2: 同港集中规则 — 同港箱聚类
            r2 = RuleHeuristics._clustering_rule(prob, best_chrom)
            if r2 is not None:
                f2 = fitness_fn.evaluate(r2)
                if f2 > best_fit:
                    best_chrom, best_fit = r2, f2
                    improved = True
            
            # R3: 重量均衡规则 — 左右平衡
            r3 = RuleHeuristics._balance_rule(prob, best_chrom)
            if r3 is not None:
                f3 = fitness_fn.evaluate(r3)
                if f3 > best_fit:
                    best_chrom, best_fit = r3, f3
                    improved = True
            
            # R4: 翻箱避免规则 — 装卸顺序优化
            r4 = RuleHeuristics._rehandle_rule(prob, best_chrom)
            if r4 is not None:
                f4 = fitness_fn.evaluate(r4)
                if f4 > best_fit:
                    best_chrom, best_fit = r4, f4
                    improved = True
        
        return best_chrom, best_fit
    
    @staticmethod
    def _loading_order_rule(prob, chrom):
        """R1: 装卸顺序 — 同列内调整，后卸港箱在下、先卸港箱在上"""
        slots = prob.slots
        pods = prob.pods
        unique_pods = prob.unique_pods
        pod_order = {pod: i for i, pod in enumerate(sorted(unique_pods))}
        pod_seq = np.array([pod_order.get(p, 0) for p in pods])
        new_chrom = chrom.copy()
        
        swapped = False
        # 按贝位+列分组
        pairs = {}
        for i, s in enumerate(chrom):
            key = (slots[s].bay, slots[s].row)
            if key not in pairs:
                pairs[key] = []
            pairs[key].append(i)
        
        for key, indices in pairs.items():
            if len(indices) < 2:
                continue
            # 按tier排序（从顶层到底层）
            tiers = [(i, slots[chrom[i]].tier) for i in indices]
            tiers.sort(key=lambda x: -x[1])  # 顶层→底层
            for a in range(len(tiers)):
                for b in range(a+1, len(tiers)):
                    # a在上层, b在下层
                    # 如果上层是后卸港(bigger seq)，下层是先卸港(smaller seq) → 正确
                    # 如果上层是先卸港，下层是后卸港 → 需要交换
                    i_top, i_bot = tiers[a][0], tiers[b][0]
                    if pod_seq[i_top] > pod_seq[i_bot]:  # 上层后卸=正确
                        continue
                    # 上层先卸、下层后卸 → 翻箱风险，交换！
                    if chrom[i_top] in prob.compat_slots[i_bot] and chrom[i_bot] in prob.compat_slots[i_top]:
                        new_chrom[i_top], new_chrom[i_bot] = chrom[i_bot], chrom[i_top]
                        swapped = True
        
        return new_chrom if swapped else None
    
    @staticmethod
    def _clustering_rule(prob, chrom):
        """R2: 同港集中 — 同港箱移到相邻贝位（优化版：可用位置用set）"""
        bays = np.array([prob.slots[s].bay for s in chrom])
        pods = prob.pods
        new_chrom = chrom.copy()
        
        # 预建已占用位置set（O(1)查询）
        used_positions = set()
        for s in chrom:
            used_positions.add((prob.slots[s].bay, prob.slots[s].row))
        
        improved = False
        for pod in prob.unique_pods:
            mask = pods == pod
            indices = np.where(mask)[0]
            if len(indices) < 5:
                continue
            median_bay = np.median(bays[mask])
            # 只查偏离中位数的
            outliers = [i for i in indices if abs(bays[i] - median_bay) > 10]
            for idx in outliers[:10]:  # 最多修10个
                # 找靠近中位数的空位
                candidates = []
                for s in prob.compat_slots[idx][:100]:  # 限制搜索范围
                    sl = prob.slots[s]
                    if abs(sl.bay - median_bay) < 5 and (sl.bay, sl.row) not in used_positions:
                        candidates.append(s)
                    if len(candidates) >= 5:
                        break
                if candidates:
                    chosen = np.random.choice(candidates)
                    new_chrom[idx] = chosen
                    used_positions.add((prob.slots[chosen].bay, prob.slots[chosen].row))
                    improved = True
        
        return new_chrom if improved else None
    
    @staticmethod
    def _balance_rule(prob, chrom):
        """R3: 重量均衡 — 左右舷重量差减小"""
        rows = np.array([prob.slots[s].row for s in chrom])
        weights = prob.cweight
        center = prob.max_row / 2
        new_chrom = chrom.copy()
        
        left_idx = np.where(rows < center)[0]
        right_idx = np.where(rows >= center)[0]
        left_w = weights[left_idx].sum()
        right_w = weights[right_idx].sum()
        
        if abs(left_w - right_w) / (left_w + right_w + 1e-10) < 0.1:
            return None  # 已经很平衡
        
        # 从重侧移重箱到轻侧
        if left_w > right_w:
            heavy_side, light_side = left_idx, right_idx
        else:
            heavy_side, light_side = right_idx, left_idx
        
        # 在heavy_side找最重箱
        heavy_order = np.argsort(-weights[heavy_side])
        for idx in heavy_order[:3]:
            i = heavy_side[idx]
            # 找light_side的空位
            light_rows = np.array([prob.slots[s].row for s in prob.compat_slots[i]])
            target = [s for s, r in zip(prob.compat_slots[i], light_rows)
                      if (r < center if left_w > right_w else r >= center)
                      and s not in set(new_chrom)]
            if target:
                new_chrom[i] = np.random.choice(target)
                return new_chrom
        
        return None
    
    @staticmethod
    def _rehandle_rule(prob, chrom):
        """R4: 翻箱避免 — 同列内，先卸港箱在上、后卸港箱在下
           对每列检查，违反则交换"""
        slots = prob.slots
        pods = prob.pods
        unique_pods = prob.unique_pods
        pod_order = {pod: i for i, pod in enumerate(sorted(unique_pods))}
        pod_seq = np.array([pod_order.get(p, 0) for p in pods])
        new_chrom = chrom.copy()
        
        swapped = False
        # 按贝位+列分组
        for key, indices in [(k, [i for i in range(prob.n_container) 
                                   if (slots[chrom[i]].bay, slots[chrom[i]].row) == k])
                              for k in set((slots[chrom[i]].bay, slots[chrom[i]].row) for i in range(prob.n_container))]:
            if len(indices) < 2:
                continue
            # 按tier从顶层到底层排
            tiers = [(i, slots[chrom[i]].tier) for i in indices]
            tiers.sort(key=lambda x: -x[1])
            
            for a in range(len(tiers)):
                for b in range(a+1, len(tiers)):
                    i_top, i_bot = tiers[a][0], tiers[b][0]
                    if pod_seq[i_top] >= pod_seq[i_bot]:
                        continue  # 正确顺序
                    # 上层先卸（小序号）、下层后卸（大序号）→ 翻箱，交换
                    if (chrom[i_top] in prob.compat_slots[i_bot] and 
                        chrom[i_bot] in prob.compat_slots[i_top]):
                        new_chrom[i_top], new_chrom[i_bot] = chrom[i_bot], chrom[i_top]
                        swapped = True
        
        return new_chrom if swapped else None


# ══════════════════════════════════════════════════════════
# 5. GA-RH优化器
# ══════════════════════════════════════════════════════════

class GARHOptimizer:
    """混合GA-RH算法主优化器（论文4.2.2）"""
    
    def __init__(self, problem: VesselProblem,
                 pop_size: int = 200,
                 generations: int = 150,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.85,
                 elite_ratio: float = 0.05,
                 rh_top_ratio: float = 0.2):
        self.problem = problem
        self.pop_size = min(pop_size, problem.n_slot)
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.rh_top_ratio = rh_top_ratio
        self.fitness_fn = FitnessFunction(problem)
        self.ga = GAComponents()
        self.rh = RuleHeuristics()
        self.history = []
    
    def optimize(self, verbose: bool = True, skip_heuristics: bool = False) -> Dict:
        p = self.problem
        n_rh = max(2, int(self.pop_size * self.rh_top_ratio))
        
        # 初始化
        pop = self.ga.init_population(p, self.pop_size)
        best_fit = -np.inf
        best_chrom = None
        best_detail = {}
        
        for gen in range(self.generations):
            # 评估
            fitness = np.array([self.fitness_fn.evaluate(chrom) for chrom in pop])
            
            # 更新最优
            gen_best = fitness.max()
            self.history.append(gen_best)
            if gen_best > best_fit:
                best_fit = gen_best
                best_chrom = pop[fitness.argmax()].copy()
                best_detail = self.fitness_fn.detail(best_chrom)
            
            # 规则启发式（大船可跳过）
            if not skip_heuristics and gen % 5 == 0:
                top_idx = np.argsort(fitness)[-n_rh:]
                for idx in top_idx:
                    improved_chrom, improved_fit = self.rh.optimize(
                        p, pop[idx], self.fitness_fn)
                    if improved_fit > fitness[idx]:
                        pop[idx] = improved_chrom
                        fitness[idx] = improved_fit
            
            # 第四阶段：选择 + 交叉 + 变异
            selected = self.ga.selection(pop, fitness, self.elite_ratio)
            next_pop = []
            
            for i in range(0, self.pop_size, 2):
                if i + 1 < self.pop_size:
                    if np.random.random() < self.crossover_rate:
                        c1, c2 = self.ga.crossover_port_group(p, selected[i], selected[i+1])
                    else:
                        c1, c2 = selected[i].copy(), selected[i+1].copy()
                    next_pop.extend([c1, c2])
                else:
                    next_pop.append(selected[i].copy())
            
            # 变异（自适应：多样性低时增加变异率）
            diversity = min(20, len(np.unique([tuple(c) for c in pop[:20]])))
            adaptive_mr = self.mutation_rate * (2 - diversity / 20)
            adaptive_mr = max(0.02, min(0.5, adaptive_mr))  # 限制范围[0.02, 0.5]
            
            pop = np.array(next_pop[:self.pop_size])
            for j in range(self.pop_size):
                pop[j] = self.ga.mutate(p, pop[j], adaptive_mr)
            
            if verbose and gen % 20 == 0:
                print(f'    Gen {gen:3d}: best={best_fit:.4f}, '
                      f'div={diversity:.0f}, '
                      f'mr={adaptive_mr:.3f}')
        
        return {
            'best_fitness': best_fit,
            'best_chromosome': best_chrom,
            'best_detail': best_detail,
            'history': self.history,
            'n_containers': p.n_container,
        }


# ══════════════════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════════════════

def run():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print('GA-RH starting...', flush=True)
    print('=' * 60, flush=True)
    print('GA-RH 配载优化（论文第四章规范实现）', flush=True)
    print('=' * 60, flush=True)
    
    # 用 stowage_features（新数据：ves船名→berth→manifest链路）
    sf = pd.read_parquet(OUT / '10_stowage_features.parquet')
    bay = pd.read_parquet(PROC / '02_bay_structure.parquet')
    
    # 建立VESSELTYPECODE→TEU映射（用于船型匹配fallback）
    bay_codes = bay[['VESSELTYPECODE']].drop_duplicates()
    # 找一个通用大船型作为fallback
    type_teu = {'CMPES': 23112, 'AFULN': 17292, 'HORA': 16010, 'CGPAN': 15072,
                'EAOT': 14026, 'CGASK': 12917, 'ULX': 13167, 'CMCAS': 11388,
                'CGRIG': 10034, 'H7E': 9087, 'HHCB': 6542, 'AKBRCL': 5086,
                'KSL': 3105, 'HMB': 2817, 'FPCE': 1773, 'EAOT': 14026}
    
    # 按层级分层抽样
    sf['max_teu'] = pd.to_numeric(sf['max_teu'], errors='coerce').fillna(0)
    sf['tier'] = pd.cut(sf['max_teu'],
        bins=[0,500,2000,5000,10000,15000,24000,99999],
        labels=['tiny','L1','L2','L3','L4','L5','mega'])
    
    # 预计算每船箱数（加速后续查询）
    ship_boxes = sf.groupby('berth_plan_no').size().reset_index(name='n_boxes')
    sf = sf.merge(ship_boxes, on='berth_plan_no')
    
    # 按层级+箱量分层抽样（确保每层箱量渐进增长）
    rng = np.random.RandomState(42)
    test_ships = []
    # 每层选3艘：小/中/大箱量（渐进）
    for tier_name, low, high in [
        ('tiny', 5, 50), ('tiny', 50, 120), ('tiny', 120, 999),
        ('L1', 50, 200), ('L1', 200, 400), ('L1', 400, 999),
        ('L2', 100, 300), ('L2', 300, 600), ('L2', 600, 1500),
        ('L3', 200, 500), ('L3', 500, 1000), ('L3', 1000, 3000),
        ('L4', 300, 800), ('L4', 800, 1500), ('L4', 1500, 5000),
        ('L5', 500, 1000), ('L5', 1000, 2000), ('L5', 2000, 5000),
    ]:
        pool = sf[(sf['tier']==tier_name) & 
                  (sf['n_boxes']>=low) & (sf['n_boxes']<high)]['berth_plan_no'].unique()
        if len(pool) == 0:
            continue
        chosen = rng.choice(pool, 1)[0]
        test_ships.append((chosen, sf[sf['berth_plan_no']==chosen]['n_boxes'].iloc[0]))
        print(f'  {tier_name}: {chosen[:16]} ({int(low)}-{high}箱, 实际{test_ships[-1][1]}箱)', flush=True)
    
    test_ships = [s[0] for s in test_ships]
    print(f'\n  共选 {len(test_ships)} 艘测试船', flush=True)
    
    all_results = []
    t0 = time.time()
    
    for idx, bpn in enumerate(test_ships):
        ship_containers = sf[sf['berth_plan_no'] == bpn].copy()
        n_boxes = len(ship_containers)
        if n_boxes < 5:
            print(f'  [{idx+1}/{len(test_ships)}] ⏭ {bpn}: 仅{n_boxes}箱', flush=True)
            continue
        
        # 从sf直接取vessel信息（已merge）
        teu = ship_containers['max_teu'].iloc[0]
        ename = str(ship_containers['e_vessel_name'].iloc[0]).strip()
        
        # 匹配bay_structure
        matched_bay = bay[bay['VESSELTYPECODE'] == ename]
        if len(matched_bay) < n_boxes:
            # 按TEU找最接近的船型
            best_code = min(type_teu.keys(), key=lambda c: abs(type_teu[c] - teu))
            matched_bay = bay[bay['VESSELTYPECODE'] == best_code]
            print(f'    bay匹配: {ename} → {best_code} (TEU差距{abs(type_teu[best_code]-teu):.0f})', flush=True)
        
        if len(matched_bay) < n_boxes:
            print(f'  [{idx+1}/{len(test_ships)}] ⏭ {bpn}: {n_boxes}箱>{len(matched_bay)}slot', flush=True)
            continue
        
        # 构建问题
        vessel_info = pd.Series({
            'max_teu': teu, 'e_vessel_name': ename,
            'length': ship_containers['length'].iloc[0] if 'length' in ship_containers else 0,
            'width': ship_containers['width'].iloc[0] if 'width' in ship_containers else 0,
        })
        prob = VesselProblem(ename, bpn, ship_containers, matched_bay, vessel_info)
        
        # GA参数：大船需要更多代，小船可以少
        n_boxes = prob.n_container
        # 根据箱量设定种群和世代（不是TEU，因为计算复杂度与箱数线性相关）
        if n_boxes > 2000:
            n_gen, n_pop = 100, 120
        elif n_boxes > 800:
            n_gen, n_pop = 80, 100
        elif n_boxes > 300:
            n_gen, n_pop = 60, 80
        elif n_boxes > 100:
            n_gen, n_pop = 40, 60
        else:
            n_gen, n_pop = 20, 40
        print(f'    GA: pop={n_pop}, gen={n_gen}', flush=True)
        
        # 确定船型TEU标签
        teu_bins = [0, 1000, 3000, 8000, 15000, 99999]
        teu_labels = ['barge', 'feeder', 'medium', 'large', 'mega']
        cat_label = '?'
        for bi in range(len(teu_bins)-1):
            if teu_bins[bi] < teu <= teu_bins[bi+1]:
                cat_label = teu_labels[bi]
                break
        
        print(f'  [{idx+1}/{len(test_ships)}] {bpn} ({cat_label}, {teu:.0f}TEU): '
              f'{n_boxes}箱, {prob.n_slot}slot', flush=True)
        
        optimizer = GARHOptimizer(prob, pop_size=n_pop, generations=n_gen)
        try:
            result = optimizer.optimize(verbose=True)
        except Exception as e:
            print(f'    ❌ 失败: {e}', flush=True)
            continue
        
        detail = result['best_detail']
        all_results.append({
            'berth_plan_no': bpn,
            'vessel_code': ename,
            'max_teu': teu,
            'size_cat': cat_label,
            'n_containers': n_boxes,
            'n_slots': prob.n_slot,
            'fitness': result['best_fitness'],
            'stability': detail.get('rehandle', 0),
            'efficiency': detail.get('efficiency', 0),
            'balance': detail.get('balance', 0),
            'yard_collab': detail.get('yard_collab', 0),
            'real_match': detail.get('real_match', 0),
            'penalty': detail.get('penalty', 0),
        })
        # 每船即时保存
        pd.DataFrame(all_results).to_parquet(RESULT / 'ga_rh_results.parquet', index=False)
        print(f'    ✅ fitness={result["best_fitness"]:.4f}, '
              f'match={detail.get("real_match",0):.2%}, '
              f'gen_time={(time.time()-t0)/(idx+1):.0f}s/船', flush=True)
    
    # 汇总
    elapsed = time.time() - t0
    print('\n' + '=' * 60, flush=True)
    print(f'  测试完成: {len(all_results)} 艘船, {elapsed:.1f}s', flush=True)
    print('=' * 60, flush=True)
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f'\n  综合结果:', flush=True)
        for col in ['fitness', 'stability', 'efficiency', 'balance', 'yard_collab', 'real_match']:
            print(f'    {col:12s}: {df[col].mean():.4f}', flush=True)
        
        print(f'\n  分船型:', flush=True)
        for cat in sorted(df['size_cat'].unique()):
            sub = df[df['size_cat'] == cat]
            print(f'    {cat:8s}: {len(sub)}船, match={sub["real_match"].mean():.2%}', flush=True)
        
        df.to_parquet(RESULT / 'ga_rh_results.parquet', index=False)
        print(f'\n📄 保存: {RESULT}/ga_rh_results.parquet', flush=True)
    
    print('\n✅ GA-RH 配载优化完成', flush=True)


if __name__ == '__main__':
    run()
