"""实验6：可扩展性测试 — 不同船舶规模的计算时间 & 收敛表现"""
import pandas as pd, numpy as np, time, warnings, importlib.util, sys
from pathlib import Path
warnings.filterwarnings('ignore')
print = lambda *a, **kw: __builtins__.print(*a, **kw, flush=True)

PROC, OUT = Path('data/processed'), Path('output')
OUT_FILE = OUT / 'ga_rh_results' / 'exp6_scalability.parquet'
spec = importlib.util.spec_from_file_location("grh", Path.cwd() / "scripts" / "15_ga_rh_stowage.py")
grh = importlib.util.module_from_spec(spec); spec.loader.exec_module(grh)
sf = pd.read_parquet(OUT / '10_stowage_features.parquet')
bay = pd.read_parquet(PROC / '02_bay_structure.parquet')

# Representative ships: KSL(345) FPCE(1489) HMB(2530) CMCAS(2636) HORA(3664)
test_ships = [
    ('KSL',    '5831513753386', 30),   # 345箱, 2,530 TEU — 小船
    ('FPCE',   '5831383552187', 40),   # 1,489箱, 3,006 TEU — 中船
    ('HMB',    '5832222763556', 50),   # 2,530箱, 7,092 TEU — 大船
    ('CMCAS',  '5831779755029', 50),   # 2,636箱, 11,388 TEU — 中大船
    ('ULX',    '5831201859587', 60),   # 4,190箱, 13,167 TEU — 超大船
]

# Load existing
results = []
if OUT_FILE.exists():
    results = pd.read_parquet(OUT_FILE).to_dict('records')
    print(f'已有 {len(results)} 条\n')

for ename, bpn, pop in test_ships:
    ship = sf[sf['berth_plan_no']==bpn].copy()
    n = len(ship)
    if n == 0:
        print(f'{ename}: 无数据, 跳过')
        continue
    teu = ship['max_teu'].iloc[0]
    npod = ship['pod'].nunique()
    mb = bay[bay['VESSELTYPECODE']==ename]
    if len(mb) < n:
        # TEU-based fallback
        type_teu = bay.groupby('VESSELTYPECODE').apply(lambda x: len(x)).to_dict()
        best_vc = min(bay['VESSELTYPECODE'].unique(), 
                      key=lambda v: abs(len(bay[bay['VESSELTYPECODE']==v]) - n))
        mb = bay[bay['VESSELTYPECODE']==best_vc]
    
    done = sum(1 for r in results if r['vessel_code']==ename)
    if done > 0:
        print(f'{ename} ({n}箱, {teu:.0f}TEU): 已有, 跳过')
        continue
    
    print(f'{ename} ({n}箱, {teu:.0f}TEU): 启动...')
    t0 = time.time()
    opt = grh.GARHOptimizer(grh.VesselProblem(ename, bpn, ship.copy(), mb, 
                                              pd.Series({'max_teu':teu,'e_vessel_name':ename})),
                            pop_size=pop, generations=30)
    res = opt.optimize(verbose=False)
    det = res['best_detail']
    elapsed = round(time.time()-t0, 1)
    results.append({
        'vessel_code': ename, 'n_containers': n, 'n_pod': npod,
        'max_teu': teu, 'pop_size': pop,
        'fitness': res['best_fitness'],
        'f1_rehandle': det.get('rehandle',0), 'f2_efficiency': det.get('efficiency',0),
        'f3_balance': det.get('balance',0), 'f4_yard_collab': det.get('yard_collab',0),
        'penalty': det.get('penalty',0), 'time_s': elapsed, 'convergence': res.get('convergence', []),
    })
    pd.DataFrame(results).to_parquet(OUT_FILE, index=False)
    print(f'  ✅ fitness={res["best_fitness"]:.4f}, {elapsed}s')

print(f'\n完成 {len(results)} 条')
df = pd.read_parquet(OUT_FILE)
print(df[['vessel_code','n_containers','max_teu','fitness','time_s']].to_string())
