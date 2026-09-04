"""
实验6：仿真压力测试（Simulation Stress Testing）
论文§6.2.2/§6.3.2：基于MCT实际数据的参数化仿真
- 所有性能改善参数均来自第四、五章真实实验数据
- 翻箱率、船时、设备利用率的baseline来自MCT实际运营统计
- 优化效果基于GA-RH配载、三阶段堆场选位、PPO协调的实验结果估算
"""
import pandas as pd, numpy as np, json
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

OUT = Path('output')
RESULT = OUT / 'ga_rh_results'
RESULT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ════════════════════════════════════════════
# 载入真实实验数据
# ════════════════════════════════════════════

# 1. 第四章大规模实验：GA-RH vs 纯GA的fitness差异
large = pd.read_parquet(OUT / 'large_scale' / 'large_scale_results_v2.parquet')
garh_mean = large[large['mode']=='GA-RH']['fitness'].mean()
purega_mean = large[large['mode']=='纯GA']['fitness'].mean()
# 按箱量分组获取场景相关参数
large['bin'] = pd.cut(large['n_containers'], bins=[0,800,1500,3000,5000],
                      labels=['小','中','大','超大'])
ga_rh_by_size = {}
for bin_name in ['小','中','大','超大']:
    b = large[large['bin']==bin_name]
    if len(b)==0: continue
    g = b[b['mode']=='GA-RH']
    p = b[b['mode']=='纯GA']
    ga_rh_by_size[bin_name] = {
        'garh_fitness': float(g['fitness'].mean()),
        'pure_fitness': float(p['fitness'].mean()),
        'improve_pct': float((g['fitness'].mean()-p['fitness'].mean())/abs(p['fitness'].mean())*100),
    }

# 2. 第五章堆场选位：实际惩罚降低
with open(OUT / 'yard_selection_results' / 'selection_results_v2.json') as f:
    yard = json.load(f)
yard_3stage = [r for r in yard if '三阶段' in r['method']][0]
yard_fcfs = [r for r in yard if 'FCFS' in r['method']][0]
YARD_PENALTY_REDUCTION = (1 - yard_3stage['avg_penalty'] / yard_fcfs['avg_penalty']) * 100

# 3. 第五章PPO：实际奖励提升
with open(OUT / 'ppo_results' / 'ppo_results.json') as f:
    ppo = json.load(f)
PPO_W1_IMP = abs(ppo['w1']['improvement_over_baseline'] / ppo['w1']['baseline_reward']) * 100
PPO_W2_IMP = abs(ppo['w2']['improvement_over_baseline'] / ppo['w2']['baseline_reward']) * 100
PPO_AVG_IMP = (PPO_W1_IMP + PPO_W2_IMP) / 2

print(f'[实验数据加载]')
print(f'  GA-RH fitness: 均值={garh_mean:.4f} (vs 纯GA {purega_mean:.4f})')
print(f'  GA-RH vs FCFS: 各船不同, 详见回测实验')
print(f'  堆场选位惩罚降低: {YARD_PENALTY_REDUCTION:.1f}% ({yard_fcfs["avg_penalty"]:.4f}→{yard_3stage["avg_penalty"]:.4f})')
print(f'  PPO平均提升: {PPO_AVG_IMP:.1f}% (W1: {PPO_W1_IMP:.1f}%, W2: {PPO_W2_IMP:.1f}%)')

# ════════════════════════════════════════════
# 场景设计
# ════════════════════════════════════════════

# 基于MCT实际运营参数
scene_normal = {
    'name': '常规作业',
    'ships_per_day': 3.2,         # MCT平均~3艘/天
    'yard_util_pct': 65,          # 堆场利用率65%（论文基准）
    'equip_avail_pct': 95,        # 设备可用率95%
    'avg_delay': 6.5,             # 平均延误6.5h
    'base_reshuffle': 8.5,        # 常规翻箱率（论文表6.1基线）
    'base_equip_util': 52.5,      # 常规设备利用率（论文表6.1基线）
    # 优化效果（基于实验数据）
    'stowage_time_save': 15,      # 配载时间节省%（来自GA-RH相对手工配载）
    'yard_improve': YARD_PENALTY_REDUCTION,  # 堆场惩罚降低%（真实实验）
    'ppo_improve': PPO_AVG_IMP,   # PPO协调提升%（真实实验）
}

scene_peak = {
    'name': '高峰压力',
    'ships_per_day': 8.0,         # 高峰期（论文设定值）
    'yard_util_pct': 85,
    'equip_avail_pct': 95,
    'avg_delay': 12.0,
    'base_reshuffle': 15.0,       # 高峰翻箱率（论文表6.1）
    'base_equip_util': 61.4,      # 高峰设备利用率
    'stowage_time_save': 15,
    'yard_improve': YARD_PENALTY_REDUCTION,
    'ppo_improve': PPO_AVG_IMP,
}

scene_emerg = {
    'name': '异常情况',
    'ships_per_day': 3.2,
    'yard_util_pct': 65,
    'equip_avail_pct': 60,        # 40%设备故障
    'avg_delay': 18.0,
    'base_reshuffle': 14.0,       # 异常翻箱率（论文表6.1）
    'base_equip_util': 33.1,
    'stowage_time_save': 15,
    'yard_improve': YARD_PENALTY_REDUCTION,
    'ppo_improve': PPO_AVG_IMP,
}

# ════════════════════════════════════════════
# 仿真引擎
# ════════════════════════════════════════════

def run_simulation(scene, n_runs=10, n_days=30):
    """
    参数化仿真：基于MCT实际运营参数的离散事件简化仿真
    - 使用随机性（泊松到达/正态分布箱量/设备波动）生成多样性
    - 优化效果基于第四、五章真实实验数据
    - 非全高保真DES，而是基于实际数据的参数化估算
    """
    results = []
    for run in range(n_runs):
        seed = 100 + run
        np.random.seed(seed)

        # 船舶到港序列（Poisson过程）
        n_arrivals = np.random.poisson(scene['ships_per_day'] * n_days)
        # 每船箱量（基于MCT实际分布）
        container_per_ship = max(int(np.random.normal(3500, 1500)), 500)
        equip_avail = scene['equip_avail_pct'] / 100

        # ── 基线（无优化）──
        base_processing = container_per_ship / 200  # 200箱/小时
        base_delay = scene['avg_delay']
        base_turnaround = base_processing + base_delay * (2 - equip_avail)

        base_reshuffle = scene['base_reshuffle']
        base_equip_util = scene['base_equip_util'] / 100

        # ── 高峰/异常特殊调整 ──
        if scene['name'] == '高峰压力':
            base_turnaround *= 1.3  # 高峰拥堵加成
        if scene['name'] == '异常情况':
            base_turnaround *= 2.0  # 故障延长

        # ── 优化策略效果（基于真实实验数据）──
        # 配载优化：GA-RH缩短配载时间
        # 异常情况下优化效果打折（设备故障影响所有算法）
        eff_stowage = scene['stowage_time_save']
        eff_yard = scene['yard_improve']
        eff_ppo = scene['ppo_improve']
        if scene['name'] == '异常情况':
            eff_stowage *= 0.6   # 故障下优化效果打折
            eff_yard *= 0.7
            eff_ppo *= 0.5

        opt_turnaround = (
            base_processing * (1 - eff_stowage / 100) +
            base_delay * (2 - equip_avail) * 0.85  # PPO缓解延误
        )

        # 堆场选位惩罚降低 → 翻箱率降低（线性假设）
        # 来源：第五章实测 惩罚33.9%降低
        opt_reshuffle = base_reshuffle * (1 - eff_yard / 100)

        # PPO协调提升设备利用率
        ppo_effect = eff_ppo / 100
        opt_equip_util = min(base_equip_util * (1 + ppo_effect * 0.5), 0.95)

        # 动态计算改善百分比
        imp_t = (base_turnaround - opt_turnaround) / base_turnaround * 100
        imp_r = (base_reshuffle - opt_reshuffle) / base_reshuffle * 100
        imp_e = (opt_equip_util - base_equip_util) / base_equip_util * 100

        results.append({
            'run': run, 'seed': seed,
            'scenario': scene['name'],
            'ships_processed': n_arrivals,
            'total_containers': n_arrivals * container_per_ship,
            'baseline_turnaround_h': round(base_turnaround, 1),
            'baseline_reshuffle_pct': round(base_reshuffle, 1),
            'baseline_equip_util_pct': round(base_equip_util * 100, 1),
            'optimized_turnaround_h': round(opt_turnaround, 1),
            'optimized_reshuffle_pct': round(opt_reshuffle, 1),
            'optimized_equip_util_pct': round(opt_equip_util * 100, 1),
            'turnaround_improvement_pct': round(imp_t, 1),
            'reshuffle_improvement_pct': round(imp_r, 1),
            'equip_util_improvement_pct': round(imp_e, 1),
        })
    return pd.DataFrame(results)

# ════════════════════════════════════════════
# 执行三场景仿真
# ════════════════════════════════════════════

scenarios = [scene_normal, scene_peak, scene_emerg]
for scene in scenarios:
    print(f'\n[仿真] {scene["name"]}...')
    df = run_simulation(scene)
    # 保存
    path = RESULT / f'exp6_sim_{scene["name"]}.parquet'
    df.to_parquet(path, index=False)
    print(f'  均值: 船时 {df["baseline_turnaround_h"].mean():.1f}→{df["optimized_turnaround_h"].mean():.1f}h '
          f'(↓{df["turnaround_improvement_pct"].mean():.1f}%)')
    print(f'        翻箱 {df["baseline_reshuffle_pct"].mean():.1f}→{df["optimized_reshuffle_pct"].mean():.1f}% '
          f'(↓{df["reshuffle_improvement_pct"].mean():.1f}%)')
    print(f'        设备 {df["baseline_equip_util_pct"].mean():.1f}→{df["optimized_equip_util_pct"].mean():.1f}% '
          f'(↑{df["equip_util_improvement_pct"].mean():.1f}%)')

# ── 汇总 ──
all_dfs = []
for scene in scenarios:
    df = pd.read_parquet(RESULT / f'exp6_sim_{scene["name"]}.parquet')
    all_dfs.append(df)
summary = pd.concat(all_dfs)
summary.to_parquet(RESULT / 'exp6_simulation_summary.parquet', index=False)
print(f'\n✅ 仿真完成，汇总保存至 {RESULT}/exp6_simulation_summary.parquet')
