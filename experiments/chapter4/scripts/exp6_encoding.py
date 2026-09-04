"""实验6：编码策略权衡分析 — 中间保存版"""
import pandas as pd, numpy as np, time, warnings, importlib.util
from pathlib import Path
warnings.filterwarnings('ignore')

PROC, OUT = Path('data/processed'), Path('output')
OUT_FILE = OUT / 'ga_rh_results' / 'exp6_encoding.parquet'
spec = importlib.util.spec_from_file_location("grh", Path.cwd() / "scripts" / "15_ga_rh_stowage.py")
grh = importlib.util.module_from_spec(spec); spec.loader.exec_module(grh)
sf = pd.read_parquet(OUT / '10_stowage_features.parquet')
bay = pd.read_parquet(PROC / '02_bay_structure.parquet')
type_teu = {'CMPES':23112,'AFULN':17292,'HORA':16010,'CGPAN':15072,'EAOT':14026,'CGASK':12917,
            'ULX':13167,'CMCAS':11388,'CGRIG':10034,'H7E':9087,'HHCB':6542,'AKBRCL':5086,'KSL':3105,'HMB':2817,'FPCE':1773}

all_results = []
if OUT_FILE.exists():
    all_results = pd.read_parquet(OUT_FILE).to_dict('records')
    print(f'已有 {len(all_results)} 条结果')

test_ships = [('CNCT','5830567479361',40),('CGAMV','5831575061746',60),('APESP','5830653078367',60)]

for ename, bpn, pop in test_ships:
    ship = sf[sf['berth_plan_no']==bpn].copy()
    n = len(ship); teu = ship['max_teu'].iloc[0]; npod = ship['pod'].nunique()
    mb = bay[bay['VESSELTYPECODE']==ename]
    if len(mb)<n:
        best = min(type_teu.keys(), key=lambda c: abs(type_teu[c]-teu))
        mb = bay[bay['VESSELTYPECODE']==best]
    
    # 策略B: 扁平+惩罚
    done_b = sum(1 for r in all_results if r['vessel_code']==ename and r['strategy']=='B_扁平+惩罚')
    print(f'\n{ename} 策略B: {done_b}/3 已完成')
    for run in range(3):
        if done_b > run:
            continue
        ship_b = ship.copy()
        prob_b = grh.VesselProblem(ename, bpn, ship_b, mb, pd.Series({'max_teu':teu,'e_vessel_name':ename}))
        for i in range(prob_b.n_container):
            prob_b.compat_slots[i] = list(range(prob_b.n_slot))
        
        t0 = time.time()
        opt_b = grh.GARHOptimizer(prob_b, pop_size=pop, generations=30)
        opt_b.fitness_fn.penalty_weight = 50.0
        res_b = opt_b.optimize(verbose=False, skip_heuristics=True)
        det_b = res_b['best_detail']
        
        all_results.append({
            'vessel_code': ename, 'n_containers': n, 'n_pod': npod, 'run': run,
            'strategy': 'B_扁平+惩罚',
            'fitness': res_b['best_fitness'],
            'f1_rehandle': det_b.get('rehandle',0), 'f2_efficiency': det_b.get('efficiency',0),
            'f3_balance': det_b.get('balance',0), 'f4_yard_collab': det_b.get('yard_collab',0),
            'penalty': det_b.get('penalty',0), 'time_s': round(time.time()-t0, 1),
        })
        pd.DataFrame(all_results).to_parquet(OUT_FILE, index=False)
        print(f'  run {run}: fitness={res_b["best_fitness"]:.4f} pen={det_b.get("penalty",0):.4f} ✅')
    
    # 策略C: 近似最优
    done_c = sum(1 for r in all_results if r['vessel_code']==ename and r['strategy']=='C_近似最优(100gen)')
    if done_c > 0:
        print(f'{ename} 策略C: 已有, 跳过')
    else:
        print(f'{ename} 策略C: 近似最优(gen=100, pop=80)...')
        prob_c = grh.VesselProblem(ename, bpn, ship.copy(), mb, pd.Series({'max_teu':teu,'e_vessel_name':ename}))
        t0 = time.time()
        opt_c = grh.GARHOptimizer(prob_c, pop_size=80, generations=100)
        res_c = opt_c.optimize(verbose=False)
        det_c = res_c['best_detail']
        
        all_results.append({
            'vessel_code': ename, 'n_containers': n, 'n_pod': npod, 'run': 0,
            'strategy': 'C_近似最优(100gen)',
            'fitness': res_c['best_fitness'],
            'f1_rehandle': det_c.get('rehandle',0), 'f2_efficiency': det_c.get('efficiency',0),
            'f3_balance': det_c.get('balance',0), 'f4_yard_collab': det_c.get('yard_collab',0),
            'penalty': det_c.get('penalty',0), 'time_s': round(time.time()-t0, 1),
        })
        pd.DataFrame(all_results).to_parquet(OUT_FILE, index=False)
        print(f'  ✅ fitness={res_c["best_fitness"]:.4f}')

print(f'\n✅ 实验6完成: {len(all_results)}条')
print('策略A数据需从Phase 1合并')
