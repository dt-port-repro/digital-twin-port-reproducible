"""
§5.2: 堆场选位优化实验（论文忠实版 + 高效numpy实现）
三阶段递进：硬约束筛选 → 多目标惩罚评估 → 排序指派
"""
import pandas as pd, numpy as np, time, json
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

OUT = Path('output'); PROC = Path('data/processed')
RESULT = OUT / 'yard_selection_results'
RESULT.mkdir(exist_ok=True)
np.random.seed(42)

print("=== 加载数据 ===", flush=True)
yd = pd.read_parquet(PROC/'06_yard_definition.parquet')
useful = yd[yd['is_useful']].copy()
useful['lane'] = useful['yard_lane_no']
useful['tier'] = useful['YARDTIERNO']
useful['cell_id'] = useful['yard_cell']
useful['lane_prefix'] = useful['lane'].str[:2]
print(f"  有效箱位: {len(useful)}", flush=True)

# Lane级预计算 (numpy)
lane_dist_map = {p: i+1 for i, p in enumerate(['20','21','22','23','24','25','26','27','28','29','30','31'])}
lane_gb = useful.groupby('lane')
lane_names = np.array(list(lane_gb.groups.keys()))
lane_prefixes = np.array([ln[:2] for ln in lane_names])
lane_dist = np.array([lane_dist_map.get(p, 12)/12.0 for p in lane_prefixes])
lane_capacity = lane_gb.size().values
lane_tier1_count = lane_gb['tier'].apply(lambda x: (x==1).sum()).values
lane_cell_idx = {lane: i for i, lane in enumerate(lane_names)}  # lane -> index

# 箱型兼容
rf_lanes = set(useful[useful['lane_prefix'].isin(['26','27','28'])]['lane'].unique())
oog_lanes = set(useful[useful['lane_prefix'].isin(['G03','G04','G05'])]['lane'].unique())
rf_mask = np.array([ln in rf_lanes for ln in lane_names])
oog_mask = np.array([ln in oog_lanes for ln in lane_names])

# 每个lane的cell信息 (tier列表)
lane_cells_info = {}
for ln in lane_names:
    cells = useful[useful['lane']==ln][['cell_id','tier']].values
    lane_cells_info[ln] = (cells, len(cells))

# 入箱
me = pd.read_parquet(PROC/'06_movement_events.parquet')
incoming = me[(me['op_type']=='L') & (me['target_area']=='MCT')].copy()
if len(incoming) > 2000:  # 减到2000箱提升速度
    incoming = incoming.sample(2000, random_state=42).reset_index(drop=True)

cts = ['GP']*1600 + ['RF']*200 + ['OOG']*200
np.random.shuffle(cts)
incoming['ctype'] = cts[:len(incoming)]
incoming['csize'] = incoming['container_size'].fillna(20).astype(int)
n_test = len(incoming)
print(f"  测试: {n_test}箱 (GP:{cts[:n_test].count('GP')} RF:{cts[:n_test].count('RF')} OOG:{cts[:n_test].count('OOG')})", flush=True)

# ════════════════════════════════════════════
# 惩罚函数 (numpy)
# ════════════════════════════════════════════
W = np.array([0.25, 0.30, 0.25, 0.10, 0.10])  # dist, height, occupy, type, reserv

def lane_penalties(ctype_mask, occ_ratios, reserv_mask):
    """计算所有lane的惩罚 (numpy向量化)"""
    p = np.zeros(len(lane_names))
    p += W[0] * lane_dist                # 距离
    p += W[2] * occ_ratios              # 占用
    p += W[3] * ctype_mask              # 箱型
    p -= W[4] * 0.3 * reserv_mask       # 虚拟占位奖励
    return p

# ════════════════════════════════════════════
# 选位器
# ════════════════════════════════════════════

class FastSelector:
    def __init__(self, use_virtual=True):
        self.lane_used = np.zeros(len(lane_names), dtype=int)
        self.reserv_mask = np.zeros(len(lane_names), dtype=bool)
        if use_virtual:
            # 容量最小的30% lane作为保留区
            n_res = max(1, int(len(lane_names)*0.3))
            idx = np.argsort(lane_capacity)[:n_res]
            self.reserv_mask[idx] = True
    
    def select(self, ctype, csize):
        # 阶段1: 硬约束筛选
        valid = np.ones(len(lane_names), dtype=bool)
        if ctype == 'RF': valid &= rf_mask
        if ctype == 'OOG': valid &= oog_mask
        # 剔除满lane
        valid &= (self.lane_used < lane_capacity)
        if not valid.any():
            return None
        
        # 阶段2: lane级惩罚
        occ = self.lane_used / np.maximum(lane_capacity, 1)
        ctype_mask = np.zeros(len(lane_names))
        if ctype == 'RF': ctype_mask = 1.0 - rf_mask.astype(float)  # 非RF区=惩罚
        if ctype == 'OOG': ctype_mask = 1.0 - oog_mask.astype(float)
        p = lane_penalties(ctype_mask, occ, self.reserv_mask)
        p[~valid] = np.inf  # 无效lane
        
        best_idx = np.argmin(p)
        best_lane = lane_names[best_idx]
        
        # 阶段3: cell级
        cells, n_cells = lane_cells_info[best_lane]
        u = self.lane_used[best_idx]
        if u < n_cells:
            ch = cells[u]
            self.lane_used[best_idx] += 1
            # cell级惩罚 = lane基础 + 高度增量 + 占用增量
            occ_here = self.lane_used[best_idx] / max(lane_capacity[best_idx], 1)
            cell_pen = (W[0]*lane_dist[best_idx] + W[2]*occ_here +
                        W[1]*(ch[1]/15.0) + ctype_mask[best_idx]*W[3] -
                        W[4]*0.3*self.reserv_mask[best_idx])
            return {'cell_id': ch[0], 'tier': ch[1], 'penalty': float(cell_pen)}
        return None


class FcfsSelector:
    """FCFS: 按lane前缀顺序"""
    def __init__(self):
        self.lane_used = np.zeros(len(lane_names), dtype=int)
        # 按前缀排序的lane索引
        self.order = np.argsort(lane_dist)  # 近的优先
    
    def select(self, ctype, csize):
        for idx in self.order:
            if self.lane_used[idx] >= lane_capacity[idx]:
                continue
            if ctype=='RF' and not rf_mask[idx]: continue
            if ctype=='OOG' and not oog_mask[idx]: continue
            cells, n_cells = lane_cells_info[lane_names[idx]]
            u = self.lane_used[idx]
            if u < n_cells:
                ch = cells[u]
                self.lane_used[idx] += 1
                occ_here = self.lane_used[idx] / max(lane_capacity[idx], 1)
                # 统一惩罚公式
                cm = 0.0
                if ctype=='RF': cm = 0.0 if rf_mask[idx] else 1.0
                if ctype=='OOG': cm = 0.0 if oog_mask[idx] else 1.0
                pen = (W[0]*lane_dist[idx] + W[2]*occ_here +
                       W[1]*(ch[1]/15.0) + cm*W[3])
                return {'cell_id': ch[0], 'tier': ch[1], 'penalty': float(pen)}
        return None


# ════════════════════════════════════════════
# 运行
# ════════════════════════════════════════════

def run(label, sel_class, **kw):
    sel = sel_class(**kw)
    t0 = time.time()
    pens = []; n = 0
    for _, row in incoming.iterrows():
        r = sel.select(row['ctype'], row['csize'])
        if r: pens.append(r['penalty']); n += 1
    dt = time.time()-t0
    avg = float(np.mean(pens)) if pens else 0
    std = float(np.std(pens)) if pens else 0
    print(f"  {label}: pen={avg:.4f}+-{std:.4f}, {n}/{n_test}箱, {dt:.1f}s ({dt/max(n,1)*1000:.2f}ms/箱)", flush=True)
    return {'method':label, 'avg_penalty':round(avg,4), 'std_penalty':round(std,4),
            'n_selected':n, 'total_time_s':round(dt,2)}

print("\n=== 运行选位实验 ===", flush=True)
results = []
results.append(run("FCFS先到先得", FcfsSelector))
results.append(run("三阶段惩罚选位", FastSelector, use_virtual=False))
results.append(run("三阶段+虚拟占位", FastSelector, use_virtual=True))

with open(RESULT/'selection_results_v2.json','w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n📄 {RESULT/'selection_results_v2.json'}")
for r in results:
    ms = r['total_time_s']/max(r['n_selected'],1)*1000
    print(f"  {r['method']}: pen={r['avg_penalty']:.4f}, {ms:.2f}ms/箱")

if len(results)>=2:
    i1 = (results[0]['avg_penalty']-results[1]['avg_penalty'])/results[0]['avg_penalty']*100
    print(f"\n✅ 三阶段 vs FCFS: 惩罚降低 {i1:.1f}%")
if len(results)>=3:
    i2 = (results[0]['avg_penalty']-results[2]['avg_penalty'])/results[0]['avg_penalty']*100
    i3 = (results[1]['avg_penalty']-results[2]['avg_penalty'])/results[1]['avg_penalty']*100
    print(f"✅ 含虚拟占位 vs FCFS: 惩罚降低 {i2:.1f}%")
    print(f"✅ 虚拟占位边际效果: 额外 {i3:.1f}%")
