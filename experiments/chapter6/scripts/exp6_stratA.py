"""补跑实验6策略A（原始GA-RH）×3条船×3次"""
import pandas as pd, numpy as np, time, warnings, importlib.util, sys
from pathlib import Path
warnings.filterwarnings('ignore')
print = lambda *a, **kw: __builtins__.print(*a, **kw, flush=True)

PROC, OUT = Path('data/processed'), Path('output')
OUT_FILE = OUT / 'ga_rh_results' / 'exp6_encoding.parquet'
spec = importlib.util.spec_from_file_location("grh", Path.cwd() / "scripts" / "15_ga_rh_stowage.py")
grh = importlib.util.module_from_spec(spec); spec.loader.exec_module(grh)
sf = pd.read_parquet(OUT / '10_stowage_features.parquet')
bay = pd.read_parquet(PROC / '02_bay_structure.parquet')
type_teu = {'CMPES':23112,'AFULN':17292,'HORA':16010,'CGPAN':15072,'EAOT':14026,'CGASK':12917,
            'ULX':13167,'CMCAS':11388,'CGRIG':10034,'H7E':9087,'HHCB':6542,'AKBRCL':5086,'KSL':3105,'HMB':2817,'FPCE':1773}

# 加载当前结果
all_results = []
if OUT_FILE.exists():
    all_results = pd.read_parquet(OUT_FILE).to_dict('records')
print(f'加载 {len(all_results)} 条已有结果')

test_ships = [('CNCT','5830567479361',40),('CGAMV','5831575061746',60),('APESP','5830653078367',60)]

for ename, bpn, pop in test_ships:
    ship = sf[sf['berth_plan_no']==bpn].copy()
    n = len(ship); teu = ship['max_teu'].iloc[0]; npod = ship['pod'].nunique()
    mb = bay[bay['VESSELTYPECODE']==ename]
    if len(mb)<n:
        best = min(type_teu.keys(), key=lambda c: abs(type_teu[c]-teu))
        mb = bay[bay['VESSELTYPECODE']==best]
    
    # ── 策略A: 原始GA-RH ──
    done_a = sum(1 for r in all_results if r['vessel_code']==ename and r['strategy']=='A_原始GA-RH')
    print(f'{ename} 策略A: {done_a}/3 已有')
    for run in range(3):
        if done_a > run:
            continue
        print(f'  启动 run {run} (pop=60, gen=30)...', end=' ')
        t0 = time.time()
        opt_a = grh.GARHOptimizer(grh.VesselProblem(ename, bpn, ship.copy(), mb, pd.Series({'max_teu':teu,'e_vessel_name':ename})),
                                  pop_size=60, generations=30)
        res_a = opt_a.optimize(verbose=False)
        det_a = res_a['best_detail']
        all_results.append({
            'vessel_code': ename, 'n_containers': n, 'n_pod': npod, 'run': run,
            'strategy': 'A_原始GA-RH',
            'fitness': res_a['best_fitness'],
            'f1_rehandle': det_a.get('rehandle',0), 'f2_efficiency': det_a.get('efficiency',0),
            'f3_balance': det_a.get('balance',0), 'f4_yard_collab': det_a.get('yard_collab',0),
            'penalty': det_a.get('penalty',0), 'time_s': round(time.time()-t0, 1),
        })
        pd.DataFrame(all_results).to_parquet(OUT_FILE, index=False)
        print(f'✅ fitness={res_a["best_fitness"]:.4f} ({round(time.time()-t0,1)}s)')

print(f'\n✅ 完成: {len(all_results)} 条')
# 汇总
df = pd.read_parquet(OUT_FILE)
print(df.groupby(['vessel_code','strategy']).size().to_string())
