"""实验4：鲁棒性测试 — 重量扰动（中间保存版）"""
import pandas as pd, numpy as np, time, warnings, importlib.util, os
from pathlib import Path
warnings.filterwarnings('ignore')

PROC, OUT = Path('data/processed'), Path('output')
OUT_FILE = OUT / 'ga_rh_results' / 'exp4_robustness.parquet'
spec = importlib.util.spec_from_file_location("grh", Path.cwd() / "scripts" / "15_ga_rh_stowage.py")
grh = importlib.util.module_from_spec(spec); spec.loader.exec_module(grh)
sf = pd.read_parquet(OUT / '10_stowage_features.parquet')
bay = pd.read_parquet(PROC / '02_bay_structure.parquet')
type_teu = {'CMPES':23112,'AFULN':17292,'HORA':16010,'CGPAN':15072,'EAOT':14026,'CGASK':12917,
            'ULX':13167,'CMCAS':11388,'CGRIG':10034,'H7E':9087,'HHCB':6542,'AKBRCL':5086,'KSL':3105,'HMB':2817,'FPCE':1773}

# 读取已有的中间结果
all_results = []
if OUT_FILE.exists():
    all_results = pd.read_parquet(OUT_FILE).to_dict('records')
    print(f'已有 {len(all_results)} 条结果')

test_ships = [('CNCT','5830567479361'),('CGAMV','5831575061746'),('APESP','5830653078367')]
perturbations = [0.0, 0.10, 0.20]
n_runs = 3

for ename, bpn in test_ships:
    ship_base = sf[sf['berth_plan_no']==bpn].copy()
    n = len(ship_base); teu = ship_base['max_teu'].iloc[0]; npod = ship_base['pod'].nunique()
    mb = bay[bay['VESSELTYPECODE']==ename]
    if len(mb)<n:
        best = min(type_teu.keys(), key=lambda c: abs(type_teu[c]-teu))
        mb = bay[bay['VESSELTYPECODE']==best]
    
    for pert_level in perturbations:
        label = f'±{pert_level*100:.0f}%'
        done = sum(1 for r in all_results if r['vessel_code']==ename and r['perturbation']==label)
        if done >= n_runs:
            print(f'{ename} {label}: {done}/{n_runs} 已有, 跳过')
            continue
        
        for run in range(n_runs):
            ship = ship_base.copy()
            if pert_level > 0:
                noise = np.random.normal(1.0, pert_level, size=n)
                noise = np.clip(noise, 1.0-2*pert_level, 1.0+2*pert_level)
                ship['gross_weight_x'] = ship_base['gross_weight_x'] * noise
                ship['weight_kg'] = ship_base['weight_kg'] * noise
            
            vi = pd.Series({'max_teu':teu,'e_vessel_name':ename})
            prob = grh.VesselProblem(ename,bpn,ship,mb,vi)
            t0 = time.time()
            opt = grh.GARHOptimizer(prob, pop_size=40, generations=30)
            res = opt.optimize(verbose=False)
            det = res['best_detail']
            
            all_results.append({
                'vessel_code': ename, 'n_containers': n, 'n_pod': npod,
                'perturbation': label, 'run': run,
                'fitness': res['best_fitness'],
                'f1_rehandle': det.get('rehandle',0), 'f2_efficiency': det.get('efficiency',0),
                'f3_balance': det.get('balance',0), 'f4_yard_collab': det.get('yard_collab',0),
                'penalty': det.get('penalty',0), 'time_s': round(time.time()-t0, 1),
            })
            # 每次跑完保存
            pd.DataFrame(all_results).to_parquet(OUT_FILE, index=False)
            print(f'{ename} {label} run {run}: fitness={res["best_fitness"]:.4f} ✅ 已保存')

print(f'\n✅ 实验4完成: {len(all_results)} runs')
