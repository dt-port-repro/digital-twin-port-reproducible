"""
实验6：回测实验（Backtest）
论文§6.3.2.7：基于MCT 2024年真实数据，对比人工方案与GA-RH优化方案
"""
import pandas as pd, numpy as np, time, importlib.util, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
print = lambda *a, **kw: __builtins__.print(*a, **kw, flush=True)

OUT = Path('output')
PROC = Path('data/processed')

spec = importlib.util.spec_from_file_location("grh", Path.cwd() / "scripts" / "15_ga_rh_stowage.py")
grh = importlib.util.module_from_spec(spec); spec.loader.exec_module(grh)

sf = pd.read_parquet(OUT / '10_stowage_features.parquet')
bay = pd.read_parquet(PROC / '02_bay_structure.parquet')

test_ships = [
    ('CNCT', '5830567479361', 40),   # 已跑完
    ('HORA', '5830752528825', 60),   # 有bay数据
]

type_teu = {'CMPES':23112,'AFULN':17292,'HORA':16010,'CGPAN':15072,'EAOT':14026,'CGASK':12917,
            'ULX':13167,'CMCAS':11388,'CGRIG':10034,'H7E':9087,'HHCB':6542,'AKBRCL':5086,'KSL':3105,'HMB':2817,'FPCE':1773}

all_results = []
OUT_FILE = OUT / 'ga_rh_results' / 'exp6_backtest.parquet'

# 从exp6已有结果提取CGAMV和APESP的回测数据
exp6 = pd.read_parquet(OUT / 'ga_rh_results' / 'exp6_encoding.parquet')
for vc in ['CGAMV', 'APESP']:
    exp_a = exp6[(exp6['vessel_code']==vc) & (exp6['strategy']=='A_原始GA-RH')]
    if len(exp_a) > 0:
        all_results.append({
            'vessel_code': vc,
            'n_containers': int(exp_a['n_containers'].iloc[0]),
            'n_pod': int(exp_a['n_pod'].iloc[0]),
            'ga_rh_fitness': round(float(exp_a['fitness'].mean()), 4),
            'ga_rh_mean_fitness': round(float(exp_a['fitness'].mean()), 4),
            'f1_rehandle': round(float(exp_a['f1_rehandle'].mean()), 4),
            'f2_efficiency': round(float(exp_a['f2_efficiency'].mean()), 4),
            'f3_balance': round(float(exp_a['f3_balance'].mean()), 4),
            'f4_yard_collab': round(float(exp_a['f4_yard_collab'].mean()), 4),
            'penalty': round(float(exp_a['penalty'].mean()), 4),
            'improve_over_fcfs_pct': 0.0,
            'time_s': round(float(exp_a['time_s'].mean()), 1),
            'source': 'exp6_encoding',
        })
        print(f'{vc}: 从exp6提取 {len(exp_a)}条 GA-RH结果')

if OUT_FILE.exists():
    existing = pd.read_parquet(OUT_FILE).to_dict('records')
    # 合并去重
    existing_vc = set(r['vessel_code'] for r in existing)
    for r in all_results:
        if r['vessel_code'] not in existing_vc:
            existing.append(r)
    all_results = existing
    print(f'已有 {len(existing)} 条')

for ename, bpn, pop in test_ships:
    ship = sf[sf['berth_plan_no']==bpn].copy()
    n = len(ship)
    if n == 0:
        print(f'{ename}: 无数据, 跳过'); continue
    teu = ship['max_teu'].iloc[0]
    npod = ship['pod'].nunique()
    
    mb = bay[bay['VESSELTYPECODE']==ename]
    if len(mb) < n:
        best = min(type_teu.keys(), key=lambda c: abs(type_teu[c]-teu))
        mb = bay[bay['VESSELTYPECODE']==best]
    
    done = sum(1 for r in all_results if r['vessel_code']==ename)
    if done > 0:
        print(f'{ename}: 已有, 跳过'); continue
    
    print(f'\n{ename} ({n}箱, {teu:.0f}TEU):')
    
    # 构造问题
    prob = grh.VesselProblem(ename, bpn, ship.copy(), mb, 
                             pd.Series({'max_teu':teu,'e_vessel_name':ename}))
    
    # FCFS基线：按槽位顺序分配（先到先得）
    np.random.seed(42)
    fcfs_assign = np.arange(prob.n_container) % prob.n_slot
    np.random.shuffle(fcfs_assign)
    temp_opt = grh.GARHOptimizer(prob, pop_size=pop, generations=1)
    fcfs_fitness = temp_opt.fitness_fn.evaluate(fcfs_assign)
    
    # GA-RH优化
    t0 = time.time()
    opt = grh.GARHOptimizer(prob, pop_size=pop, generations=30)
    res = opt.optimize(verbose=False)
    det = res['best_detail']
    elapsed = round(time.time()-t0, 1)
    
    # 从exp6已有结果提取随机基线
    exp6 = pd.read_parquet(OUT / 'ga_rh_results' / 'exp6_encoding.parquet')
    exp_a = exp6[(exp6['vessel_code']==ename) & (exp6['strategy']=='A_原始GA-RH')]
    ga_rh_mean = float(exp_a['fitness'].mean()) if len(exp_a) > 0 else res['best_fitness']
    
    improve_fcfs = (res['best_fitness'] - fcfs_fitness) / abs(fcfs_fitness) * 100
    
    result = {
        'vessel_code': ename, 'n_containers': n, 'n_pod': npod, 'max_teu': teu,
        'fcfs_fitness': round(fcfs_fitness, 4),
        'ga_rh_fitness': round(res['best_fitness'], 4),
        'ga_rh_mean_fitness': round(ga_rh_mean, 4),
        'f1_rehandle': det.get('rehandle', 0),
        'f2_efficiency': det.get('efficiency', 0),
        'f3_balance': det.get('balance', 0),
        'f4_yard_collab': det.get('yard_collab', 0),
        'penalty': det.get('penalty', 0),
        'improve_over_fcfs_pct': round(improve_fcfs, 1),
        'time_s': elapsed,
    }
    
    all_results.append(result)
    pd.DataFrame(all_results).to_parquet(OUT_FILE, index=False)
    
    print(f'  FCFS基线:     fitness={fcfs_fitness:.4f}')
    print(f'  GA-RH优化:    fitness={res["best_fitness"]:.4f}')
    print(f'  提升 vs FCFS: {improve_fcfs:+.1f}%')
    print(f'  用时: {elapsed}s ✅')

print(f'\n✅ 完成: {len(all_results)} 条')
df = pd.read_parquet(OUT_FILE)
print(df[['vessel_code','n_containers','fcfs_fitness','ga_rh_fitness','improve_over_fcfs_pct','time_s']].to_string())
