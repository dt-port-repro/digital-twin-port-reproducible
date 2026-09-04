"""实验5：消融实验 — 中间保存版"""
import pandas as pd, numpy as np, time, warnings, importlib.util
from pathlib import Path
warnings.filterwarnings('ignore')

PROC, OUT = Path('data/processed'), Path('output')
OUT_FILE = OUT / 'ga_rh_results' / 'exp5_ablation.parquet'
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

test_ships = [('CNCT','5830567479361'),('CGAMV','5831575061746'),('APESP','5830653078367')]
n_gen = 30

for ename, bpn in test_ships:
    ship = sf[sf['berth_plan_no']==bpn].copy()
    n = len(ship); teu = ship['max_teu'].iloc[0]; npod = ship['pod'].nunique()
    mb = bay[bay['VESSELTYPECODE']==ename]
    if len(mb)<n:
        best = min(type_teu.keys(), key=lambda c: abs(type_teu[c]-teu))
        mb = bay[bay['VESSELTYPECODE']==best]
    
    for run in range(3):  # 3次独立运行
        done = sum(1 for r in all_results if r['vessel_code']==ename and r['run']==run and r['mode']=='B_GA+后处理')
        if done > 0:
            print(f'{ename} run {run}: 已有, 跳过')
            continue
        
        vi = pd.Series({'max_teu':teu,'e_vessel_name':ename})
        prob = grh.VesselProblem(ename,bpn,ship.copy(),mb,vi)
        opt = grh.GARHOptimizer(prob, pop_size=40, generations=n_gen)
        
        t0 = time.time()
        res_ga = opt.optimize(verbose=False, skip_heuristics=True)
        best_chrom = res_ga['best_chromosome']
        best_detail_before = res_ga['best_detail']
        fitness_before = res_ga['best_fitness']
        
        rh = grh.RuleHeuristics()
        improved_chrom, improved_fit = rh.optimize(prob, best_chrom, opt.fitness_fn)
        improved_detail = opt.fitness_fn.detail(improved_chrom)
        elapsed = time.time() - t0
        
        # 模式B
        all_results.append({
            'vessel_code': ename, 'n_containers': n, 'n_pod': npod, 'run': run,
            'mode': 'B_GA+后处理',
            'fitness_before': fitness_before, 'fitness_after': improved_fit,
            'f1_rehandle': improved_detail.get('rehandle',0),
            'f2_efficiency': improved_detail.get('efficiency',0),
            'f3_balance': improved_detail.get('balance',0),
            'f4_yard_collab': improved_detail.get('yard_collab',0),
            'penalty': improved_detail.get('penalty',0),
            'improvement_pct': (improved_fit/fitness_before-1)*100 if fitness_before>0 else 0,
            'time_s': round(elapsed, 1),
        })
        # 模式A
        all_results.append({
            'vessel_code': ename, 'n_containers': n, 'n_pod': npod, 'run': run,
            'mode': 'A_纯GA', 'fitness_before': fitness_before, 'fitness_after': fitness_before,
            'f1_rehandle': best_detail_before.get('rehandle',0),
            'f2_efficiency': best_detail_before.get('efficiency',0),
            'f3_balance': best_detail_before.get('balance',0),
            'f4_yard_collab': best_detail_before.get('yard_collab',0),
            'penalty': best_detail_before.get('penalty',0),
            'improvement_pct': 0, 'time_s': round(elapsed, 1),
        })
        
        pd.DataFrame(all_results).to_parquet(OUT_FILE, index=False)
        print(f'{ename} run {run}: A={fitness_before:.4f} → B={improved_fit:.4f} (+{(improved_fit/fitness_before-1)*100:.1f}%) ✅')

print(f'\n✅ 实验5完成: {len(all_results)}条')
print('模式C数据需从Phase 1合并')
